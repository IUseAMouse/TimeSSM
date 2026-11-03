"""
Batches from corpus v4 (scripts/build_corpus_v4.py).

Each batch is drawn in a worker: rate class, real or synthetic, family, subset,
series weighted by length, forecast origin. Rate classes are drawn evenly enough
that a low-frequency series is not drowned by billions of 5-minute points.
Series shorter than the window are left padded with NaN and read through the
mask. With probability `p_decimate` a window is decimated by block means at a
random factor and its Delta scale follows.

A batch holds `groups` draws of a subset and a decimation factor, each giving
batch / groups windows. The S4D layers compute one kernel per distinct Delta
scale in a batch, so the number of groups bounds that cost.
"""

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset

from timejepa.data.frequency import BASE_SEASON, parse_frequency, season_length

# Daily subsets with a human weekly rhythm (FlowState's rule).
WEEKLY = ("favorita", "m5", "nn5", "uber", "rideshare", "vehicle_trips", "covid_mobility",
          "pedestrian", "taxi", "bitcoin", "synth")
# Whole subsets kept out of training: model selection reads them, never GIFT-Eval.
DEV_SUBSETS = ("favorita_sales", "favorita_transactions", "fred_md", "gfc12_load", "gfc14_load",
               "gfc17_load", "kdd2022", "m5", "tourism_monthly", "tourism_quarterly",
               "tourism_yearly", "cif_2016_12", "alibaba_cluster_trace_2018", "LOS_LOOP")
DECIMATION = (2, 3, 4, 6, 8, 12)
# Draw weight of each rate class.
RATE_WEIGHT = {"S": 2.5, "5T": 3.5, "15T": 4.5, "H": 5.5, "D": 3.9, "W": 2.8, "M": 2.2,
               "Q": 1.0, "A": 1.0}


def family(subset: str) -> str:
    name = re.sub(r"_(18|19|20)\d{2}$", "", subset)
    return re.sub(r"(_s\d+|_dec\d+)$", "", name)


def rate_class(freq) -> str:
    unit, n = parse_frequency(freq)
    if unit == "T":
        return "5T" if n <= 5 else "15T"
    return unit


def subset_scale(name: str, freq) -> float:
    """w = 24 / season; 0.0 when the subset has no declared frequency."""
    season = season_length(freq, weekly=any(name.startswith(p) for p in WEEKLY))
    return BASE_SEASON / season if season else 0.0


class Subset:
    def __init__(self, root: Path, name: str, freq):
        self.name, self.scale = name, subset_scale(name, freq)
        self.rate = rate_class(freq)
        self.synthetic = name.startswith("synth")
        self.values = np.load(root / f"{name}.values.npy", mmap_mode="r")
        self.offsets = np.load(root / f"{name}.offsets.npy")
        self.lengths = np.diff(self.offsets)

    def series(self, i: int) -> np.ndarray:
        return self.values[self.offsets[i]:self.offsets[i + 1]]


def load_subsets(root, names: Sequence[str]) -> Dict[str, Subset]:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    return {n: Subset(root, n, manifest[n]["freq"]) for n in names if manifest[n]["n_obs"]}


def train_subsets(root) -> List[str]:
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    return sorted(n for n, r in manifest.items() if r["n_obs"] and n not in DEV_SUBSETS)


def block_mean(x: np.ndarray, k: int) -> np.ndarray:
    """Means over blocks of k ending at the last point; NaN-aware, a partial block at the
    start is dropped."""
    n = len(x) // k
    seg = x[len(x) - n * k:].reshape(n, k)
    count = np.isfinite(seg).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nansum(seg, 1) / np.where(count, count, np.nan)


class WindowBatches(IterableDataset):
    def __init__(self, root, batch_size: int, context: int, max_horizon: int = 720,
                 patch: int = 16, p_decimate: float = 0.5, p_flip: float = 0.5, seed: int = 0,
                 temperature: float = 0.3, p_synth: float = 0.5, groups: int = 12,
                 names: Sequence[str] = None, p_full: float = 0.0, p_mix: float = 0.0):
        self.root, self.batch_size, self.context = root, batch_size, context
        self.max_horizon, self.patch = max_horizon, patch
        self.p_decimate, self.p_flip, self.seed = p_decimate, p_flip, seed
        self.temperature, self.p_synth, self.n_groups = temperature, p_synth, groups
        # Share of windows with all the history the series has (up to `context`); the
        # others draw their length log-uniformly, so long contexts are seen as at test time.
        self.p_full = p_full
        # Share of windows replaced by a convex mix of 2-3 windows of the same rate, each
        # divided by its mean absolute context value: new series from the real ones.
        self.p_mix = p_mix
        self.names = list(names) if names is not None else train_subsets(root)

    def _setup(self):
        self.subsets = load_subsets(self.root, self.names)
        self.pools = {}
        for s in self.subsets.values():
            self.pools.setdefault((s.rate, s.synthetic), {}).setdefault(family(s.name), []).append(s)
        self.rates = sorted({rate for rate, _ in self.pools})
        weight = np.array([RATE_WEIGHT[r] for r in self.rates])
        self.rate_p = weight / weight.sum()
        self.groups, self.group_p = {}, {}
        for key, groups in self.pools.items():
            self.groups[key] = list(groups.values())
            obs = np.array([sum(int(s.lengths.sum()) for s in g) for g in groups.values()], float)
            self.group_p[key] = obs ** self.temperature / (obs ** self.temperature).sum()
        self.series_p = {s.name: s.lengths / s.lengths.sum() for s in self.subsets.values()}

    def _subset(self, rng) -> Subset:
        rate = self.rates[rng.choice(len(self.rates), p=self.rate_p)]
        synthetic = rng.random() < self.p_synth
        key = (rate, synthetic) if (rate, synthetic) in self.pools else (rate, not synthetic)
        group = self.groups[key][rng.choice(len(self.groups[key]), p=self.group_p[key])]
        return group[rng.integers(len(group))]

    def _draw(self, rng):
        """A subset and a decimation factor."""
        for _ in range(100):
            s = self._subset(rng)
            if s.lengths.max() >= 2 * self.patch:
                break
        k = 1
        if rng.random() < self.p_decimate:
            k = int(rng.choice(DECIMATION))
            if s.lengths.max() // k < 2 * self.patch or s.scale * k > 8:
                k = 1
        return s, k

    def _window(self, rng, s: Subset, k: int, horizon: int):
        for _ in range(100):
            i = rng.choice(len(s.lengths), p=self.series_p[s.name]) if len(s.lengths) > 1 else 0
            n = int(s.lengths[i])
            if n // k >= 2 * self.patch:
                break
        steps = n // k
        h = min(horizon, max(1, steps // 4)) if steps < 2 * horizon else horizon
        ctx_min = min(steps - h, self.patch)
        if self.p_full > 0 and rng.random() < self.p_full:
            ctx = min(self.context, steps - h)
            origin = int(rng.integers(ctx, steps - h + 1))
        else:
            ctx = int(math.exp(rng.uniform(math.log(ctx_min),
                                           math.log(max(self.context, ctx_min + 1)))))
            ctx = min(ctx, steps - h)
            origin = int(rng.integers(ctx, steps - h + 1))
        lo = max(0, n - (steps - origin + ctx) * k)
        hi = n - (steps - origin - h) * k
        raw = np.asarray(s.series(i)[lo:hi], dtype=np.float32)
        y = raw if k == 1 else block_mean(raw, k)
        return y[:-h], y[-h:], s.scale * k

    def _mix(self, rng, s: Subset, k: int, horizon: int, c, t):
        """c, t mixed with 1-2 windows of subsets of the same rate, pool and scale, aligned on
        the forecast origin; lengths cut to the shortest. Dirichlet(1.5) weights."""
        key = (s.rate, s.synthetic)
        parts = [(c, t)]
        for _ in range(int(rng.integers(1, 3))):
            g = self.groups[key][rng.choice(len(self.groups[key]), p=self.group_p[key])]
            other = g[rng.integers(len(g))]
            if other.scale != s.scale or other.lengths.max() // k < 2 * self.patch:
                continue
            parts.append(self._window(rng, other, k, horizon)[:2])
        if len(parts) == 1:
            return c, t
        nc, nt = min(len(p[0]) for p in parts), min(len(p[1]) for p in parts)
        a = rng.dirichlet(np.full(len(parts), 1.5))
        mc, mt = np.zeros(nc, np.float32), np.zeros(nt, np.float32)
        for weight, (pc, pt) in zip(a, parts):
            scale = np.nanmean(np.abs(pc[-nc:])) if np.isfinite(pc[-nc:]).any() else 0.0
            scale = scale if np.isfinite(scale) and scale > 0 else 1.0
            mc += np.float32(weight / scale) * pc[-nc:]
            mt += np.float32(weight / scale) * pt[:nt]
        return mc, mt

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0])
        self._setup()
        L = self.context
        log_max = math.log(self.max_horizon)
        while True:
            H = int(math.exp(rng.uniform(math.log(self.patch), log_max)))
            x = np.full((self.batch_size, L), np.nan, dtype=np.float32)
            y = np.full((self.batch_size, H), np.nan, dtype=np.float32)
            w = np.ones(self.batch_size, dtype=np.float32)
            s, k = None, 1
            for b in range(self.batch_size):
                if b % max(1, self.batch_size // self.n_groups) == 0:
                    s, k = self._draw(rng)
                c, t, w[b] = self._window(rng, s, k, H)
                if self.p_mix > 0 and rng.random() < self.p_mix:
                    c, t = self._mix(rng, s, k, H, c, t)
                sign = -1.0 if rng.random() < self.p_flip else 1.0
                c = c[-L:]
                x[b, L - len(c):] = sign * c
                y[b, :len(t)] = sign * t
            yield torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w)
