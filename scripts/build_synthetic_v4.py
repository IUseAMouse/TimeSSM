"""
Synthetic part of corpus v5: one subset per rate and generator.

    python scripts/build_synthetic_v4.py --out /workspace/data/v4_syn

For each sampling frequency of the grid, the reference cycle is the season that
frequency implies: 10S cycles every 360 steps, monthly every 12. `tight` keeps
every period near that cycle, `wide` spreads them over two octaves below and
three above. `ops` (bursty) covers the fast rates, `intermittent` the slow ones.
Short rates get short series, as in the real data.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from timejepa.data.frequency import season_length
from timejepa.data.synthetic import SyntheticSpec, sample_series

# freq -> (series length, number of series)
GRID = {"10S": (8192, 36000), "1T": (8192, 24000), "5T": (8192, 36000), "10T": (8192, 24000),
        "15T": (8192, 24000), "30T": (8192, 18000), "H": (8192, 36000), "D": (4096, 48000),
        "W": (1024, 96000), "M": (512, 144000), "Q": (256, 144000), "A": (128, 144000)}


def specs(freq: str):
    season = season_length(freq, weekly=True)
    length, n = GRID[freq]
    tight = (max(2.0, season / 1.15), season * 1.15)
    wide = (max(2.0, season / 4), min(season * 8, length / 2))
    out = [SyntheticSpec(f"synth_tight_{freq}", length, tight),
           SyntheticSpec(f"synth_wide_{freq}", length, wide)]
    if season >= 24:
        out.append(SyntheticSpec(f"synth_ops_{freq}", length, tight, kind="ops"))
    if freq in ("D", "W", "M", "Q"):
        out.append(SyntheticSpec(f"synth_intermittent_{freq}", length, tight, kind="intermittent"))
    return [(spec, n // len(out), freq) for spec in out]


def build(args):
    (spec, n, freq), out, seed = args
    rng = np.random.default_rng(seed)
    rows = np.stack([sample_series(spec, rng) for _ in range(n)]).astype(np.float32)
    np.save(out / f"{spec.name}.values.npy", rows.ravel())
    np.save(out / f"{spec.name}.offsets.npy", np.arange(n + 1, dtype=np.int64) * spec.chunk_length)
    return {"subset": spec.name, "n_series": n, "n_obs": int(rows.size), "freq": freq,
            "median_length": spec.chunk_length}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, a.out, seed) for seed, s in enumerate(x for f in GRID for x in specs(f))]
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(build, jobs))
    (a.out / "manifest.json").write_text(json.dumps({r["subset"]: r for r in rows}, indent=1))
    print(f"{len(rows)} subsets, {sum(r['n_obs'] for r in rows) / 1e9:.2f} B observations")


if __name__ == "__main__":
    main()
