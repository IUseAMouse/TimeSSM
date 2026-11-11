"""
Multivariate items of a corpus: which stored series share a time axis.

A corpus stores every variate as its own series, cut to its observed span. groups.npz
(written by scripts/build_groups.py) puts them back together:

    names          [S]      subset names
    item_offsets   [I + 1]  item i is members item_offsets[i]:item_offsets[i + 1]
    member_subset  [M]      index into names
    member_series  [M]      series index in that subset
    member_shift   [M]      steps from the item's time origin to the member's first point

A corpus without groups.npz has no items; everything else reads it as before.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

GROUPS_FILE = "groups.npz"
Member = Tuple[object, int, int]          # (Subset, series index, shift)


def load_groups(root, subsets: Dict[str, object], min_members: int = 2) -> List[List[Member]]:
    """The items whose members are in `subsets` (a held-out subset drops its members); an
    item keeps at least `min_members`. [] when the corpus has no groups file."""
    path = Path(root) / GROUPS_FILE
    if not path.exists():
        return []
    g = np.load(path, allow_pickle=False)
    names = [str(n) for n in g["names"]]
    off, sub, ser, shift = (g[k] for k in ("item_offsets", "member_subset", "member_series",
                                           "member_shift"))
    items = []
    for i in range(len(off) - 1):
        members = [(subsets[names[sub[j]]], int(ser[j]), int(shift[j]))
                   for j in range(off[i], off[i + 1]) if names[sub[j]] in subsets]
        if len(members) >= min_members:
            items.append(members)
    return items


def span(members: List[Member]) -> Tuple[int, int]:
    """[start, end) of the time axis every member covers (empty when start >= end)."""
    start = max(shift for _, _, shift in members)
    end = min(shift + int(s.lengths[i]) for s, i, shift in members)
    return start, end


def aligned(members: List[Member], start: Optional[int] = None,
            end: Optional[int] = None) -> np.ndarray:
    """[V, end - start] float32 on the item's time axis, NaN where a member has no value;
    by default the union of the members' spans."""
    lo = min(shift for _, _, shift in members) if start is None else start
    hi = max(shift + int(s.lengths[i]) for s, i, shift in members) if end is None else end
    out = np.full((len(members), hi - lo), np.nan, dtype=np.float32)
    for v, (s, i, shift) in enumerate(members):
        a, b = max(lo, shift), min(hi, shift + int(s.lengths[i]))
        if a < b:
            out[v, a - lo:b - lo] = s.series(i)[a - shift:b - shift]
    return out


def save_groups(path, names: List[str], items: List[List[Tuple[str, int, int]]]) -> None:
    """items: lists of (subset name, series index, shift)."""
    index = {n: k for k, n in enumerate(names)}
    flat = [m for item in items for m in item]
    np.savez(path, names=np.array(names),
             item_offsets=np.concatenate([[0], np.cumsum([len(it) for it in items])]).astype(np.int64),
             member_subset=np.array([index[n] for n, _, _ in flat], dtype=np.int32),
             member_series=np.array([i for _, i, _ in flat], dtype=np.int64),
             member_shift=np.array([s for _, _, s in flat], dtype=np.int64))
