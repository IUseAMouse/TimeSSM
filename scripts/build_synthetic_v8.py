"""
Synthetic families of corpus v8: one subset per rate and generator, `synth_<kind>_<freq>`.

    python scripts/build_synthetic_v8.py --out /workspace/data/v8_syn

The v4 and template generators are sums of cycles, smooth noise and white noise: every
series they draw is stationary around its trend and its noise has no memory. Four
families add what they miss:

- `arima`: ARMA(p<=3, q<=2) with random stable roots, integrated once half of the time
  (random walk, drift), optional seasonal AR term, optional GARCH(1,1) variance,
  sometimes in logs (multiplicative). Macro and market series behave so.
- `regime`: a Markov chain over 2-3 regimes, each with its own level, cycle amplitude,
  slope and noise, geometric dwell times, sometimes a short ramp between regimes.
- `calendar`: load and traffic built from a clock: a day shape that differs between
  workdays and weekends, a weekday factor, a yearly envelope, a temperature-like driver
  with a U-shaped response, holidays drawn as weekend days, AR(1) multiplicative noise.
  Rates from 1 min to daily.
- `growth`: a positive level with a stochastic local trend, a multiplicative cycle whose
  shape drifts, log-normal noise, sometimes a sharp drop and a slow recovery. Monthly,
  quarterly, weekly and yearly series of tourism and retail behave so.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

from timejepa.data.frequency import parse_frequency, season_length

from build_synthetic_v4 import GRID

SHARE = 4          # a quarter of the series count of one v4 generator per rate
SECONDS = {"S": 1, "T": 60, "H": 3600, "D": 86400}


def _finish(x: np.ndarray, rng, positive: bool) -> np.ndarray:
    """Random scale, and a random offset unless the series is positive by construction."""
    x = x * np.exp(rng.uniform(-1.0, 4.0))
    if not positive:
        x = x + rng.uniform(-10.0, 1000.0)
    return x.astype(np.float32)


def _stable_poly(order: int, rng) -> np.ndarray:
    """Coefficients of 1 - a1 z - ... with roots inside the unit circle (moduli 0.2-0.98),
    complex roots in conjugate pairs."""
    roots = []
    while len(roots) < order:
        r = rng.uniform(0.2, 0.98)
        if order - len(roots) >= 2 and rng.random() < 0.5:
            th = rng.uniform(0.1, np.pi - 0.1)
            roots += [r * np.exp(1j * th), r * np.exp(-1j * th)]
        else:
            roots.append(r * rng.choice([-1.0, 1.0]))
    return np.real(np.poly(roots))


def _innovations(n: int, rng) -> np.ndarray:
    e = rng.standard_t(int(rng.integers(3, 8)), n) if rng.random() < 0.3 else rng.normal(size=n)
    if rng.random() < 0.3:                                    # GARCH(1,1) variance
        a, b = rng.uniform(0.03, 0.15), rng.uniform(0.75, 0.95)
        b = min(b, 0.99 - a)
        h, out = 1.0, np.empty(n)
        for t in range(n):
            out[t] = np.sqrt(h) * e[t]
            h = (1 - a - b) + a * out[t] ** 2 + b * h
        e = out
    return e


def sample_arima(n: int, season: float, rng) -> np.ndarray:
    ar = _stable_poly(int(rng.integers(0, 4)), rng)
    ma = np.r_[1.0, rng.uniform(-0.8, 0.8, int(rng.integers(0, 3)))]
    if season >= 2 and rng.random() < 0.5:                    # seasonal AR, integer lag
        s = int(round(season))
        if s < n // 4:
            sar = np.zeros(s + 1)
            sar[0], sar[s] = 1.0, -rng.uniform(0.3, 0.9)
            ar = np.convolve(ar, sar)
    x = lfilter(ma, ar, _innovations(n + 200, rng))[200:]
    if rng.random() < 0.5:                                    # integrated: random walk, drift
        x = np.cumsum(x * rng.uniform(0.05, 1.0) + rng.normal(0, 0.05))
    x = (x - x.mean()) / max(x.std(), 1e-8)
    if rng.random() < 0.3:                                    # in logs
        return _finish(np.exp(rng.uniform(0.1, 0.8) * x), rng, positive=True)
    return _finish(x, rng, positive=False)


def _cycle(n: int, period: float, rng) -> np.ndarray:
    t = np.arange(n) / period + rng.random()
    out = sum(rng.uniform(0.3, 1.0) / h * np.sin(2 * np.pi * h * t + rng.uniform(0, 2 * np.pi))
              for h in range(1, int(rng.integers(1, 5)) + 1))
    return out / max(np.std(out), 1e-8)


def sample_regime(n: int, season: float, rng) -> np.ndarray:
    k = int(rng.integers(2, 4))
    dwell = n / rng.uniform(4, 40)
    states, t = [], 0
    s = int(rng.integers(k))
    while t < n:
        d = int(rng.geometric(1 / dwell))
        states.append((t, min(n, t + d), s))
        t += d
        s = (s + int(rng.integers(1, k))) % k                 # always a different regime
    level, amp = rng.normal(0, 2, k), rng.uniform(0, 1.5, k)
    slope, noise = rng.normal(0, 2 / n, k), np.exp(rng.uniform(np.log(0.05), np.log(1.0), k))
    cyc = _cycle(n, max(season * np.exp(rng.uniform(-0.1, 0.1)), 2.0), rng) if season >= 2 \
        else np.zeros(n)
    idx = np.empty(n, dtype=int)
    for a, b, s in states:
        idx[a:b] = s
    lv = level[idx] + np.concatenate([slope[s] * np.arange(b - a) for a, b, s in states])
    if rng.random() < 0.5:                                    # short ramps between regimes
        w = max(2, int(rng.integers(2, max(3, int(season) + 3))))
        lv = np.convolve(np.pad(lv, (w - 1, 0), mode="edge"), np.ones(w) / w, mode="valid")
    x = lv + amp[idx] * cyc + noise[idx] * rng.normal(size=n)
    return _finish(x, rng, positive=False)


def _day_shape(steps: int, rng) -> np.ndarray:
    """A smooth positive shape over one day, peaks where the harmonics put them."""
    ph = np.arange(steps) / steps
    out = sum(rng.uniform(0.2, 1.0) / h * np.cos(2 * np.pi * h * ph + rng.uniform(0, 2 * np.pi))
              for h in range(1, int(rng.integers(2, 5)) + 1))
    out = (out - out.min()) / max(np.ptp(out), 1e-8)
    return rng.uniform(0.1, 0.6) + out


def sample_calendar(n: int, freq: str, rng) -> np.ndarray:
    unit, m = parse_frequency(freq)
    steps = max(1, int(round(86400 / (SECONDS[unit] * m))))   # steps per day
    day = np.arange(n) // steps + int(rng.integers(0, 7))
    weekday = day % 7
    weekend = weekday >= 5
    n_days = int(day[-1] - day[0]) + 1
    holiday = np.zeros(int(day[-1]) + 1, dtype=bool)
    holiday[rng.integers(day[0], day[-1] + 1, rng.poisson(n_days * rng.uniform(0, 12) / 365))] = True
    off = weekend | holiday[day]
    work, rest = _day_shape(steps, rng), _day_shape(steps, rng)
    if rng.random() < 0.5:
        rest = 0.5 * (work + rest)
    phase = np.arange(n) % steps
    x = np.where(off, rest[phase], work[phase])
    factor = np.exp(rng.normal(0, 0.1, 7))
    factor[5:] *= rng.uniform(0.4, 1.2)
    x = x * factor[weekday]
    year = 2 * np.pi * (day + rng.random() * 365.25) / 365.25
    x = x * (1 + rng.uniform(0, 0.3) * np.sin(year))
    if rng.random() < 0.6:                                    # temperature-like driver
        temp = np.sin(year + rng.uniform(-0.5, 0.5)) + 0.3 * np.cumsum(rng.normal(0, 0.05, n_days))[
            day - day[0]]
        x = x + rng.uniform(0.1, 0.8) * np.abs(temp - rng.uniform(-0.3, 0.3)) ** rng.uniform(1.0, 2.0)
    if rng.random() < 0.5:
        x = x * (1 + rng.normal(0, 0.3) * np.arange(n) / n)  # slow growth or decline
    rho, sd = rng.uniform(0.5, 0.98), np.exp(rng.uniform(np.log(0.01), np.log(0.1)))
    x = x * np.exp(lfilter([1.0], [1.0, -rho], rng.normal(0, sd * np.sqrt(1 - rho ** 2), n)))
    return _finish(np.maximum(x, 0.0), rng, positive=True)


def sample_growth(n: int, season: float, rng) -> np.ndarray:
    slope = np.cumsum(rng.normal(0, rng.uniform(0.0005, 0.005), n)) + rng.normal(0.003, 0.01)
    level = np.cumsum(slope + rng.normal(0, rng.uniform(0.002, 0.03), n))
    level = level / max(np.ptp(level) / rng.uniform(0.3, 3.0), 1.0)   # at most x20 over the series
    if season >= 2:
        s = season * (1 if rng.random() < 0.7 else np.exp(rng.uniform(-0.05, 0.05)))
        a, b = _cycle(n, s, rng), _cycle(n, s, rng)
        mix = np.clip(0.5 + np.cumsum(rng.normal(0, 0.02, n)), 0, 1)   # the shape drifts
        level = level + rng.uniform(0.05, 0.6) * ((1 - mix) * a + mix * b)
    if rng.random() < 0.2:                                    # sharp drop, slow recovery
        c = int(rng.integers(n // 4, n - 2))
        tail = np.arange(n - c)
        level[c:] += np.log(rng.uniform(0.1, 0.7)) * np.exp(-tail / rng.uniform(2, max(3, n / 8)))
    x = np.exp(level - level.mean() + rng.normal(0, rng.uniform(0.01, 0.15), n))
    return _finish(x, rng, positive=True)


def kinds(freq: str):
    unit, m = parse_frequency(freq)
    out = ["arima", "regime"]
    if unit in ("T", "H", "D") and 86400 / (SECONDS[unit] * m) <= 1440:
        out.append("calendar")
    if unit in ("D", "W", "M", "Q", "A"):
        out.append("growth")
    return out


def sample(kind: str, n: int, freq: str, season: float, rng) -> np.ndarray:
    if kind == "calendar":
        return sample_calendar(n, freq, rng)
    return {"arima": sample_arima, "regime": sample_regime, "growth": sample_growth}[kind](
        n, season, rng)


def build(args):
    kind, freq, out, seed = args
    length, count = GRID[freq]
    n = count // SHARE
    season = season_length(freq, weekly=True) or 0.0
    rng = np.random.default_rng([seed, 8])
    rows = np.stack([sample(kind, length, freq, season, rng) for _ in range(n)])
    assert np.isfinite(rows).all(), f"{kind} {freq}: non-finite values"
    name = f"synth_{kind}_{freq}"
    np.save(out / f"{name}.values.npy", rows.ravel())
    np.save(out / f"{name}.offsets.npy", np.arange(n + 1, dtype=np.int64) * length)
    return {"subset": name, "n_series": n, "n_obs": int(rows.size), "freq": freq,
            "median_length": length}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(k, f, a.out, i) for i, (k, f) in enumerate((k, f) for f in GRID for k in kinds(f))]
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(build, jobs))
    (a.out / "manifest.json").write_text(json.dumps({r["subset"]: r for r in rows}, indent=1))
    print(f"{len(rows)} subsets, {sum(r['n_obs'] for r in rows) / 1e9:.2f} B observations")


if __name__ == "__main__":
    main()
