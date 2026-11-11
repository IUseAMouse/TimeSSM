"""
Multivariate items of a corpus, found back from the raw sources: a corpus with groups.npz.

    python scripts/build_groups.py --corpus /workspace/data/v10 --out /workspace/data/v10m \
        --raw /workspace/data/raw --caches /workspace/data/raw_d

The new corpus links every file of `--corpus` (no value is rewritten) and adds groups.npz
(see timessm.groups) and groups.json, the matching report.

Each variate of a raw item (an arrow item of the LOTSA download, a station grid of the ISD
or NDBC caches) is cut to its observed span, as the corpus builders cut it, and found among
the stored series of its subset by the hash of its values. A series removed since (GIFT
overlap, too short, constant) is simply not found: its item keeps the others. A hash held by
several stored series (duplicates, constants) matches nothing. The shift of a member is the
index of its first observed point in the raw item. Wind farm subsets (kelmarsh, penmanshiel)
are grouped by turbine index across their four channels, when the four have the same series
count and lengths.
"""

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from timessm.groups import GROUPS_FILE, save_groups

ISD_CHANNELS = ("temp", "dew", "pressure", "wind")
NDBC_CHANNELS = ("wspd", "gst", "wvht", "dpd", "apd", "pres", "atmp", "wtmp", "dewp")
WIND_FARMS = ("kelmarsh", "penmanshiel")
WIND_CHANNELS = ("power", "rpm", "temp", "wind")
# Below this share of variates found, a subset whose series count says nothing was dropped
# points to a bug in the cut rule, not to removed series.
MIN_MATCH = 0.95
# The corpus builders drop a variate with fewer observed points (build_corpus_v4.MIN_LENGTH);
# such a variate is not counted as missing.
MIN_OBSERVED = 16
# ... and stop a subset at this many points (build_corpus_v4.MAX_OBS): the later items of a
# subset that reached it were never stored.
MAX_OBS = 400_000_000


def key(values: np.ndarray) -> bytes:
    """Hash of the values, every non-finite one written as the same NaN."""
    v = np.asarray(values, dtype=np.float32)
    v = np.where(np.isfinite(v), v, np.float32("nan"))
    return hashlib.blake2b(np.ascontiguousarray(v).tobytes(), digest_size=16).digest()


def cut(channel: np.ndarray) -> Optional[Tuple[np.ndarray, int]]:
    """Observed span of a raw variate and its first index; None with nothing observed."""
    channel = np.where(np.isfinite(channel), channel, np.nan).astype(np.float32)
    obs = np.flatnonzero(np.isfinite(channel))
    if not len(obs):
        return None
    return channel[obs[0]:obs[-1] + 1], int(obs[0])


class Index:
    """Hash of every stored series of some subsets -> (subset, index); ambiguous hashes
    are kept apart."""

    def __init__(self, corpus: Path, names: List[str]):
        self.table: Dict[bytes, Tuple[str, int]] = {}
        self.ambiguous = set()
        for name in names:
            values = np.load(corpus / f"{name}.values.npy", mmap_mode="r")
            off = np.load(corpus / f"{name}.offsets.npy")
            for i in range(len(off) - 1):
                k = key(values[off[i]:off[i + 1]])
                if k in self.table or k in self.ambiguous:
                    self.table.pop(k, None)
                    self.ambiguous.add(k)
                else:
                    self.table[k] = (name, i)

    def find(self, channel: np.ndarray) -> Optional[Tuple[str, int, int]]:
        c = cut(channel)
        if c is None:
            return None
        hit = self.table.get(key(c[0]))
        return None if hit is None else (hit[0], hit[1], c[1])

    def is_ambiguous(self, channel: np.ndarray) -> bool:
        c = cut(channel)
        return c is not None and key(c[0]) in self.ambiguous


def arrow_items(subset_dir: Path) -> Iterator[np.ndarray]:
    """[V, T] targets of the multivariate items of a LOTSA subset (none if univariate)."""
    import pyarrow as pa
    import pyarrow.ipc as ipc
    for shard in sorted(subset_dir.glob("*.arrow")):
        with pa.memory_map(str(shard)) as source:
            for batch in ipc.open_stream(source):
                for target in batch.column("target").to_pylist():
                    arr = np.asarray(target, dtype=np.float32)
                    if arr.ndim == 1:
                        return
                    yield arr.reshape(-1, arr.shape[-1])


def cache_items(cache_dir: Path) -> Iterator[np.ndarray]:
    """[channels, hours] station grids of the ISD or NDBC caches."""
    for path in sorted(cache_dir.glob("*.npy")):
        yield np.load(path)


def match_items(raw_items, index: Index, channel_subsets=None) -> Tuple[list, dict]:
    """Items as lists of (subset, series, shift) with >= 2 members found, and the counts.
    `channel_subsets` [c] restricts channel c to those subsets (one subset per channel)."""
    items, stats = [], {"variates": 0, "found": 0, "ambiguous": 0}
    for grid in raw_items:
        members = []
        for c, channel in enumerate(grid):
            if np.isfinite(channel).sum() < MIN_OBSERVED:
                continue
            stats["variates"] += 1
            if index.is_ambiguous(channel):
                stats["ambiguous"] += 1           # a duplicate series: left out on purpose
                continue
            hit = index.find(channel)
            if hit is not None and (channel_subsets is None or hit[0] in channel_subsets[c]):
                members.append(hit)
        stats["found"] += len(members)
        if len(members) >= 2:
            items.append(members)
    stats["items"] = len(items)
    return items, stats


def wind_items(corpus: Path, manifest: dict, farm: str) -> Tuple[list, dict]:
    names = [f"{farm}_{c}" for c in WIND_CHANNELS]
    if not all(n in manifest for n in names):
        return [], {"items": 0}
    offs = [np.load(corpus / f"{n}.offsets.npy") for n in names]
    lengths = [np.diff(o) for o in offs]
    if len({len(x) for x in lengths}) > 1 or any((x != lengths[0]).any() for x in lengths):
        print(f"{farm}: channels differ in series or lengths, not grouped", flush=True)
        return [], {"items": 0}
    items = [[(n, i, 0) for n in names] for i in range(len(lengths[0]))]
    return items, {"items": len(items), "variates": 4 * len(items), "found": 4 * len(items)}


def build(corpus: Path, out: Path, raw: Optional[Path], caches: Optional[Path]) -> dict:
    manifest = json.loads((corpus / "manifest.json").read_text())
    items, report = [], {}

    def add(source, found, stats, expected_whole=False):
        items.extend(found)
        report[source] = stats
        rate = stats["found"] / max(1, stats.get("variates", 0) - stats.get("ambiguous", 0))
        print(f"{source}: {stats}  ({rate:.1%} of the unambiguous variates found)", flush=True)
        if expected_whole and stats.get("variates") and rate < MIN_MATCH:
            raise SystemExit(f"{source}: {rate:.1%} of the variates found, below {MIN_MATCH:.0%} "
                             "for a subset that dropped nothing: the cut rule is wrong")

    if raw is not None:
        for subset_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
            name = subset_dir.name
            if name not in manifest or not manifest[name].get("n_obs"):
                continue
            grids = arrow_items(subset_dir)
            first = next(grids, None)
            if first is None:
                continue                          # univariate subset
            index = Index(corpus, [name])
            found, stats = match_items(_chain(first, grids), index)
            whole = not manifest[name].get("dropped") and manifest[name]["n_obs"] < MAX_OBS
            add(name, found, stats, expected_whole=whole)
    if caches is not None:
        for source, channels in (("isd", ISD_CHANNELS), ("ndbc", NDBC_CHANNELS)):
            if not (caches / source).is_dir():
                continue
            per_channel = [[n for n in manifest if re.fullmatch(f"{source}_{c}_H(_s\\d+)?", n)]
                           for c in channels]
            index = Index(corpus, sorted({n for names in per_channel for n in names}))
            found, stats = match_items(cache_items(caches / source), index, per_channel)
            add(source, found, stats)
    for farm in WIND_FARMS:
        found, stats = wind_items(corpus, manifest, farm)
        if found:
            add(farm, found, stats)

    names = sorted({n for item in items for n, _, _ in item})
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(corpus.iterdir()):
        if f.name.endswith(".npy") and not (out / f.name).exists():
            os.symlink(f.resolve(), out / f.name)
    (out / "manifest.json").write_text((corpus / "manifest.json").read_text())
    save_groups(out / GROUPS_FILE, names, items)
    summary = {"corpus": str(corpus), "items": len(items),
               "members": sum(len(it) for it in items), "sources": report}
    (out / "groups.json").write_text(json.dumps(summary, indent=1))
    print(f"{len(items)} items, {summary['members']} members over {len(names)} subsets", flush=True)
    return summary


def _chain(first, rest):
    yield first
    yield from rest


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--raw", type=Path, help="the LOTSA arrow download")
    ap.add_argument("--caches", type=Path, help="the directory holding isd/ and ndbc/")
    a = ap.parse_args()
    build(a.corpus, a.out, a.raw, a.caches)


if __name__ == "__main__":
    main()
