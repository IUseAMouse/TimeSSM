"""
Corpus v5: the converted roots linked into one directory, with a coverage report.

    python scripts/merge_corpus.py --out /workspace/data/v5 \
        /workspace/data/v4 /workspace/data/v4_ext /workspace/data/v4_ext_b /workspace/data/v4_syn

A subset name present in two roots stops the script. The report gives, per rate
class, the observations and the share of drawn windows that WindowBatches would
produce with the configured weights.
"""

import argparse
import json
import os
from pathlib import Path

from timessm.corpus import RATE_WEIGHT, family, rate_class


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("roots", nargs="+", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    merged = {}
    for root in a.roots:
        for name, row in json.loads((root / "manifest.json").read_text()).items():
            if name in merged:
                raise SystemExit(f"{name} appears twice")
            if not row["n_obs"]:
                continue
            merged[name] = row
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
