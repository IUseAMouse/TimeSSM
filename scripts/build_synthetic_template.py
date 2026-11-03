"""
Synthetic family of repeated shapes: one subset per rate, `synth_template_<freq>`.

    python scripts/build_synthetic_template.py --out /workspace/data/syn_template

The kernel generator draws seasonality as a few smooth harmonics. Regular real series
repeat a shape with corners instead: a weekday step, a peak hour, a flat night. Here a
cycle is a random shape over its phase, piecewise constant or piecewise linear with 3-12
knots, sometimes a spike, near the reference cycle of the rate; it may carry a second
shape at 7 cycles (52 for daily data), a slow change of amplitude, a trend, a level shift,
and a small noise, log-uniform between 0.3 % and 15 % of the signal.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from timejepa.data.frequency import season_length

from build_synthetic_v4 import GRID

SHARE = 4          # a quarter of the series count of one v4 generator per rate


def shape(n: int, period: float, rng: np.random.Generator) -> np.ndarray:
    """A random shape over the phase, repeated every `period` steps (period need not be
    an integer), unit variance."""
    k = int(rng.integers(3, 13))
    knots = np.sort(rng.random(k))
    values = rng.normal(size=k)
    if rng.random() < 0.3:
        values[rng.integers(k)] += rng.choice([-1.0, 1.0]) * rng.uniform(2.0, 5.0)
    phase = (np.arange(n) / period + rng.random()) % 1.0
    if rng.random() < 0.5:
        out = values[np.searchsorted(knots, phase, side="right") - 1]   # -1 wraps to the last
    else:
        out = np.interp(phase, np.r_[knots - 1, knots, knots + 1], np.tile(values, 3))
    return (out - out.mean()) / max(out.std(), 1e-8)


def sample_template(n: int, season: float, freq: str, rng: np.random.Generator) -> np.ndarray:
    period = season if rng.random() < 0.5 else season * np.exp(rng.uniform(-0.1, 0.1))
    period = max(period, 2.0)
    x = shape(n, period, rng)
    m = 365.25 / 7 if freq.endswith("D") else 7.0
    if rng.random() < 0.5 and period * m <= n / 3:
        second = rng.uniform(0.3, 1.0) * shape(n, period * m, rng)
        x = x * (1 + 0.5 * second) if rng.random() < 0.5 else x + second
    t = np.arange(n) / n
    if rng.random() < 0.5:
        x = x * (1 + rng.uniform(0.0, 0.3) * np.sin(2 * np.pi * rng.uniform(0.5, 3) * t
                                                     + rng.uniform(0, 2 * np.pi)))
    if rng.random() < 0.5:
        x = x + rng.normal(0, 1.0) * t
    if rng.random() < 0.2:
        x = x + rng.normal(0, 1.0) * (t > rng.uniform(0.2, 0.9))
    x = x + np.exp(rng.uniform(np.log(0.003), np.log(0.15))) * x.std() * rng.normal(size=n)
    if rng.random() < 0.3:
        x = x - x.min()                     # non-negative, as counts and loads are
    return (x * np.exp(rng.uniform(-1.0, 4.0)) + rng.uniform(-10.0, 1000.0)).astype(np.float32)


def build(args):
    freq, out, seed = args
    length, count = GRID[freq]
    n = count // SHARE
    season = season_length(freq, weekly=True)
    rng = np.random.default_rng([seed, 7])
    rows = np.stack([sample_template(length, season, freq, rng) for _ in range(n)])
    name = f"synth_template_{freq}"
    np.save(out / f"{name}.values.npy", rows.ravel())
    np.save(out / f"{name}.offsets.npy", np.arange(n + 1, dtype=np.int64) * length)
    return {"subset": name, "n_series": n, "n_obs": int(rows.size), "freq": freq,
            "median_length": length}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(build, [(f, a.out, i) for i, f in enumerate(GRID)]))
    (a.out / "manifest.json").write_text(json.dumps({r["subset"]: r for r in rows}, indent=1))
    print(f"{len(rows)} subsets, {sum(r['n_obs'] for r in rows) / 1e9:.2f} B observations")


if __name__ == "__main__":
    main()
