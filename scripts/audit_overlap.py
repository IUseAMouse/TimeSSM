"""
Search the corpus for series of the GIFT-Eval datasets, by value.

    python scripts/audit_overlap.py --gift /workspace/data/gift_eval --corpus /workspace/data/v5

Names cannot catch an anonymized dataset such as M4. A window of 8 consecutive
growth ratios, rounded, is unchanged by a rescaling, so a shared stretch of
history gives the same hash whatever the units. Windows with repeated ratios
are skipped, they match by chance. Reports, per corpus subset and GIFT dataset,
the corpus series sharing at least `--min-hits` windows.
"""

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import datasets
import numpy as np

K = 8


def hashes(x: np.ndarray, positions: bool = False):
    """Hashes of the ratio windows, unique, or every window with its start if `positions`."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    empty = np.empty(0, dtype=np.uint64)
    if len(x) <= K:
        return (empty, np.empty(0, np.int64)) if positions else empty
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.round(x[1:] / x[:-1], 3)
    w = np.lib.stride_tricks.sliding_window_view(r, K)
    pos = np.arange(len(w))
    keep = np.isfinite(w).all(1) & (w != 1.0).all(1) & (np.abs(w) < 1e6).all(1)
    w, pos = w[keep], pos[keep]
    if len(w):
        keep = np.array([len(set(row)) >= K - 1 for row in w])
        w, pos = w[keep], pos[keep]
    if len(w) == 0:
        return (empty, np.empty(0, np.int64)) if positions else empty
    v = (w * 1000).astype(np.int64).astype(np.uint64)
    h = np.zeros(len(v), dtype=np.uint64)
    for j in range(K):
        h = (h ^ v[:, j]) * np.uint64(1099511628211)
    return (h, pos) if positions else np.unique(h)


def gift_index(root: Path):
    index = {}
    for d in sorted(p for p in root.iterdir() if (p / "state.json").exists()):
        ds = datasets.load_from_disk(str(d))
        hs = []
        for row in ds:
            t = np.asarray(row["target"], dtype=np.float64)
            for channel in t.reshape(-1, t.shape[-1]):
                hs.append(hashes(channel))
        index[d.name] = np.unique(np.concatenate(hs)) if hs else np.empty(0, np.uint64)
        print(d.name, len(index[d.name]), flush=True)
    return index


INDEX = {}


def scan(args):
    corpus, name, max_series, min_hits = args
    values = np.load(corpus / f"{name}.values.npy", mmap_mode="r")
    offsets = np.load(corpus / f"{name}.offsets.npy")
    picks = np.arange(len(offsets) - 1)
    if len(picks) > max_series:
        picks = np.random.default_rng(0).choice(picks, max_series, replace=False)
    found = Counter()
    for i in picks:
        h = hashes(values[offsets[i]:offsets[i + 1]][:20000])
        for g, ref in INDEX.items():
            if len(h) and np.isin(h, ref, assume_unique=True).sum() >= min_hits:
                found[g] += 1
    return name, len(picks), dict(found)


def gift_positions(root: Path, names):
    """Every window hash of the named GIFT datasets with (dataset, series, start), sorted by hash."""
    hs, ids = [], []
    for g, name in enumerate(names):
        ds = datasets.load_from_disk(str(root / name))
        series = 0
        for row in ds:
            t = np.asarray(row["target"], dtype=np.float64)
            for channel in t.reshape(-1, t.shape[-1]):
                h, pos = hashes(channel, positions=True)
                hs.append(h)
                ids.append(np.stack([np.full(len(h), g), np.full(len(h), series), pos], 1))
                series += 1
    h, ids = np.concatenate(hs), np.concatenate(ids)
    order = np.argsort(h, kind="stable")
    return h[order], ids[order]


def verify(a):
    """A shared stretch gives many hits at one offset between the two series; chance does not."""
    rows = json.loads(a.out.read_text())
    # the first pass samples at most --max-series per subset: re-check every series
    flagged = [r for r in rows if r["found"]]
    names = sorted({g for r in flagged for g in r["found"]})
    ref_h, ref_ids = gift_positions(a.gift, names)
    report = []
    for r in flagged:
        values = np.load(a.corpus / f"{r['subset']}.values.npy", mmap_mode="r")
        offsets = np.load(a.corpus / f"{r['subset']}.offsets.npy")
        leaks = Counter()
        examples, series = [], []
        for i in range(len(offsets) - 1):
            h, pos = hashes(values[offsets[i]:offsets[i + 1]], positions=True)
            if len(h) == 0:
                continue
            lo = np.searchsorted(ref_h, h, "left")
            hi = np.searchsorted(ref_h, h, "right")
            m = hi > lo
            if m.sum() < a.min_hits:
                continue
            idx = np.concatenate([np.arange(l, u) for l, u in zip(lo[m], hi[m])])
            rep = np.repeat(pos[m], hi[m] - lo[m])
            key = Counter(zip(ref_ids[idx, 0], ref_ids[idx, 1], ref_ids[idx, 2] - rep))
            (g, s, off), n = key.most_common(1)[0]
            if n >= a.min_aligned:
                leaks[names[g]] += 1
                series.append(i)
                if len(examples) < 3:
                    examples.append({"series": i, "gift": names[g], "gift_series": int(s),
                                     "offset": int(off), "aligned_windows": int(n)})
        out = {"subset": r["subset"], "series": len(offsets) - 1, "leaks": dict(leaks),
               "examples": examples}
        print(json.dumps(out), flush=True)
        report.append({**out, "leaking_series": series})
    a.out.with_name(f"{a.out.stem}_verified_{a.min_aligned}.json").write_text(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gift", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--max-series", type=int, default=20000)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", type=Path, default=Path("overlap.json"))
    ap.add_argument("--verify", action="store_true", help="re-check the flagged subsets by alignment")
    ap.add_argument("--min-aligned", type=int, default=20)
    a = ap.parse_args()
    if a.verify:
        return verify(a)
    INDEX.update(gift_index(a.gift))
    names = sorted(json.loads((a.corpus / "manifest.json").read_text()))
    names = [n for n in names if not n.startswith("synth")]
    with ProcessPoolExecutor(a.workers) as pool:
        rows = []
        for name, n, found in pool.map(scan, [(a.corpus, n, a.max_series, a.min_hits) for n in names]):
            rows.append({"subset": name, "scanned": n, "found": found})
            if found:
                print(name, n, found, flush=True)
    a.out.write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} subsets scanned, {sum(bool(r['found']) for r in rows)} with shared history")


if __name__ == "__main__":
    main()
