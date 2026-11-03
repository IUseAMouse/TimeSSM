"""
Corpus v5: the converted roots linked into one directory, with a coverage report.

    python scripts/merge_corpus.py --out /workspace/data/v5 \
        /workspace/data/v4 /workspace/data/v4_ext /workspace/data/v4_ext_b /workspace/data/v4_syn

A subset name present in two roots stops the script. `--drop` takes the verified
overlap report of audit_overlap.py: the series sharing history with GIFT-Eval are
left out, the subset is written filtered instead of linked. `--drop-subset`
leaves a whole subset out. The sources are never modified. The report gives, per rate
class, the observations and the share of drawn windows that WindowBatches would
produce with the configured weights.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from timessm.corpus import RATE_WEIGHT, family, rate_class


def write_filtered(root: Path, out: Path, name: str, drop) -> dict:
    """Copy a subset without the series in `drop`, returns the new counts."""
    for kind in ("values", "offsets"):
        if (out / f"{name}.{kind}.npy").exists() or (out / f"{name}.{kind}.npy").is_symlink():
            raise SystemExit(f"{out / name} exists, merge a filtered corpus into a new directory")
    values = np.load(root / f"{name}.values.npy", mmap_mode="r")
    offsets = np.load(root / f"{name}.offsets.npy")
    keep = np.setdiff1d(np.arange(len(offsets) - 1), np.asarray(sorted(drop), dtype=np.int64))
    parts = [values[offsets[i]:offsets[i + 1]] for i in keep]
    lengths = np.array([len(p) for p in parts], dtype=offsets.dtype)
    np.save(out / f"{name}.values.npy", np.concatenate(parts) if parts else values[:0])
    np.save(out / f"{name}.offsets.npy", np.concatenate([[0], np.cumsum(lengths)]).astype(offsets.dtype))
    return {"n_series": int(len(keep)), "n_obs": int(lengths.sum()), "dropped": int(len(offsets) - 1 - len(keep))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--drop", type=Path, help="verified overlap report")
    ap.add_argument("--drop-subset", nargs="*", default=[])
    a = ap.parse_args()
    drop = {}
    if a.drop:
        drop = {r["subset"]: r["leaking_series"] for r in json.loads(a.drop.read_text())
                if r["leaking_series"]}
    a.out.mkdir(parents=True, exist_ok=True)
    merged = {}
    for root in a.roots:
        for name, row in json.loads((root / "manifest.json").read_text()).items():
            if name in merged:
                raise SystemExit(f"{name} appears twice")
            if not row["n_obs"] or name in a.drop_subset:
                continue
            merged[name] = row
            if name in drop:
                merged[name] = {**row, **write_filtered(root, a.out, name, drop[name])}
                print(f"{name}: {merged[name]['dropped']} series left out")
                continue
            for kind in ("values", "offsets"):
                link = a.out / f"{name}.{kind}.npy"
                if not link.exists():
                    os.symlink(root / f"{name}.{kind}.npy", link)
    (a.out / "manifest.json").write_text(json.dumps(merged, indent=1))
    total = sum(RATE_WEIGHT.values())
    print(f"{len(merged)} subsets, {sum(r['n_obs'] for r in merged.values()) / 1e9:.2f} B")
    for rate, weight in RATE_WEIGHT.items():
        rows = [(n, r) for n, r in merged.items() if rate_class(r["freq"]) == rate]
        real = [r for n, r in rows if not n.startswith("synth")]
        print(f"{rate:4s} draw {100 * weight / total:5.1f} %  real {sum(r['n_obs'] for r in real) / 1e9:7.3f} B "
              f"in {len({family(n) for n, r in rows if not n.startswith('synth')}):3d} families, "
              f"{sum(r['n_series'] for r in real):9,d} series; synth "
              f"{sum(r['n_obs'] for n, r in rows if n.startswith('synth')) / 1e9:6.3f} B")


if __name__ == "__main__":
    main()
