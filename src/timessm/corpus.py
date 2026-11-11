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

Multivariate batches (`p_multi` > 0, a corpus with groups.npz): with probability
`p_multi` a batch holds items of V variates instead, (x [B, V, L], y [B, V, H], w [B],
roles [B, V], valid [B, V]) with B * V close to the batch size. An item is either a real
multivariate item of the corpus (its variates on one time axis, one origin, one
decimation, one sign) or, with probability `p_cov_synth`, a synthetic target built from
a real window and 1-4 covariates (see covariate_target). Roles: at least one target;
the other variates of a real item are targets, past covariates or future covariates
(their known horizon is y); synthetic covariates are future (p 0.7) or past ones.
"""

import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset

from timejepa.data.frequency import BASE_SEASON, parse_frequency, season_length

from .groups import aligned, load_groups, span
from .patched import FUTURE_COVARIATE, PAST_COVARIATE, TARGET

# Daily subsets with a human weekly rhythm (FlowState's rule).
WEEKLY = ("favorita", "m5", "nn5", "uber", "rideshare", "vehicle_trips", "covid_mobility",
          "pedestrian", "taxi", "bitcoin", "synth")
# Whole subsets kept out of training: model selection reads them, never GIFT-Eval.
DEV_SUBSETS = ("favorita_sales", "favorita_transactions", "fred_md", "gfc12_load", "gfc14_load",
               "gfc17_load", "kdd2022", "m5", "tourism_monthly", "tourism_quarterly",
               "tourism_yearly", "cif_2016_12", "alibaba_cluster_trace_2018", "LOS_LOOP")
# Multivariate subsets kept out of training as well when the run reads items: the
# multivariate dev (kelmarsh: power as target, wind as a future covariate).
DEV_MULTI = ("beijing_air_quality", "PEMS08", "kelmarsh_power", "kelmarsh_rpm",
             "kelmarsh_temp", "kelmarsh_wind")
DECIMATION = (2, 3, 4, 6, 8, 12)
# Draw weight of each rate class.
RATE_WEIGHT = {"S": 2.5, "5T": 3.5, "15T": 4.5, "H": 5.5, "D": 3.9, "W": 2.8, "M": 2.2,
               "Q": 1.0, "A": 1.0}


def family(subset: str) -> str:
    name = re.sub(r"_(18|19|20)\d{2}$", "", subset)
    return re.sub(r"(_s\d+|_dec\d+)$", "", name)


def item_family(members) -> str:
    """Family of a multivariate item: its subset's, or the common prefix of its subsets
    (kelmarsh_power, kelmarsh_wind... -> kelmarsh)."""
    names = sorted({m[0].name for m in members})
    if len(names) == 1:
        return family(names[0])
    return os.path.commonprefix(names).rstrip("_") or family(names[0])


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


def train_subsets(root, multivariate: bool = False) -> List[str]:
    """Every non-empty subset but the dev ones (and the multivariate dev ones for a
    multivariate run)."""
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    dev = set(DEV_SUBSETS) | (set(DEV_MULTI) if multivariate else set())
    return sorted(n for n, r in manifest.items() if r["n_obs"] and n not in dev)


def covariate_target(rng, base: np.ndarray, covariates: np.ndarray, n_context: int,
                     max_lag: int):
    """A target driven by covariates: y = b + sd(b) * sum_j a_j g_j(c_j delayed by l_j).

    base [T] (context then horizon), covariates [C, T] on the same axis; each c_j is
    standardized on the first `n_context` steps (missing read as 0), delayed by l_j in
    [0, max_lag] (the first value repeated), passed through g_j in {identity, tanh(2x),
    step x > 0, x * standardized base} and weighted by a_j ~ N(0, s^2), s ~ U(0.5, 2).
    Returns (y [T], spec), spec the draws (kinds, lags, weights) for the record."""
    def standard(x):
        ref = x[:n_context]
        mu = np.nanmean(ref) if np.isfinite(ref).any() else 0.0
        sd = np.nanstd(ref) if np.isfinite(ref).any() else 1.0
        return np.nan_to_num((x - mu) / (sd if sd > 0 else 1.0))
    b_hat = standard(base)
    sd_b = np.nanstd(base[:n_context]) if np.isfinite(base[:n_context]).any() else 1.0
    sd_b = sd_b if sd_b > 0 else 1.0
    effect = np.zeros_like(base, dtype=np.float64)
    spec = []
    scale = rng.uniform(0.5, 2.0)
    for c in covariates:
        kind = int(rng.integers(4))
        lag = int(rng.integers(0, max_lag + 1))
        a = float(rng.normal(0.0, scale))
        z = standard(c)
        z = np.concatenate([np.full(lag, z[0]), z[:len(z) - lag]]) if lag else z
        g = (z, np.tanh(2 * z), (z > 0).astype(np.float64), z * b_hat)[kind]
        effect += a * g
        spec.append((kind, lag, a))
    return (base + sd_b * effect).astype(np.float32), spec


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
                 names: Sequence[str] = None, p_full: float = 0.0, p_mix: float = 0.0,
                 series_frac: float = 1.0, p_multi: float = 0.0, max_variates: int = 16,
                 p_cov_synth: float = 0.5):
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
        # Share of the series of each subset kept for training (a fixed draw, at least one
        # per subset); the subset weights stay those of the full corpus.
        self.series_frac = series_frac
        # Share of multivariate batches, their width cap and their share of synthetic
        # covariate items (the rest are the corpus' items).
        self.p_multi, self.max_variates, self.p_cov_synth = p_multi, max_variates, p_cov_synth
        self.names = (list(names) if names is not None
                      else train_subsets(root, multivariate=p_multi > 0))

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
        self.series_p = {}
        for j, s in enumerate(sorted(self.subsets.values(), key=lambda s: s.name)):
            w = s.lengths.astype(float)
            if self.series_frac < 1:
                keep = np.random.default_rng(j).random(len(w)) < self.series_frac
                keep[np.argmax(w * keep) if keep.any() else 0] = True
                w = w * keep
            self.series_p[s.name] = w / w.sum()
        self.items = load_groups(self.root, self.subsets) if self.p_multi > 0 else []
        self.item_pools, self.item_p = {}, {}
        for it in self.items:
            start, end = span(it)
            if end - start >= 4 * self.patch:
                self.item_pools.setdefault(item_family(it), []).append(it)
        if self.item_pools:
            self.item_families = sorted(self.item_pools)
            obs = np.array([sum(span(it)[1] - span(it)[0] for it in self.item_pools[f])
                            for f in self.item_families], float)
            self.item_family_p = obs ** self.temperature / (obs ** self.temperature).sum()

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
        lo, hi, h = self._bounds(rng, n, k, horizon)
        raw = np.asarray(s.series(i)[lo:hi], dtype=np.float32)
        y = raw if k == 1 else block_mean(raw, k)
        return y[:-h], y[-h:], s.scale * k

    def _bounds(self, rng, n: int, k: int, horizon: int):
        """A window of a series of n steps decimated by k: raw slice [lo, hi) and horizon h
        (context length and origin drawn as in `_window`)."""
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
        return lo, hi, h

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

    def _item(self, rng, fam: str, k: int, V: int, horizon: int):
        """A real item of family `fam`: up to V of its variates (padded to V), one window
        on their common span decimated by k (1 when the span or the scale does not allow
        it) -> context [V, c], target [V, h], valid [V], w."""
        pool = self.item_pools[fam]
        for _ in range(100):
            members = pool[rng.integers(len(pool))]
            if len(members) > V:
                members = [members[j] for j in sorted(rng.choice(len(members), V, replace=False))]
            start, end = span(members)
            if (end - start) // k >= 2 * self.patch:
                break
        scale = members[0][0].scale
        if (end - start) // k < 2 * self.patch or scale * k > 8:
            k = 1
        lo, hi, h = self._bounds(rng, end - start, k, horizon)
        vals = aligned(members, start + lo, start + hi)
        if k > 1:
            vals = np.stack([block_mean(v, k) for v in vals])
        pad = V - len(members)
        valid = np.array([True] * len(members) + [False] * pad)
        vals = np.concatenate([vals, np.full((pad, vals.shape[1]), np.nan, np.float32)])
        return vals[:, :-h], vals[:, -h:], valid, scale * k

    def _covariate_item(self, rng, s: Subset, k: int, V: int, horizon: int):
        """A synthetic item: row 0 the target built by covariate_target from a window of s,
        rows 1.. its V - 1 covariates (windows of s or sparse pulses) on the same axis."""
        c, t, _ = self._window(rng, s, k, horizon)
        n_ctx, h = len(c), len(t)
        base = np.concatenate([c, t])
        covs = []
        for _ in range(V - 1):
            if rng.random() < 0.6:
                cc, ct, _ = self._window(rng, s, k, h)
                full = np.full(len(base), np.nan, np.float32)
                cc, ct = cc[-n_ctx:], ct[:h]
                full[n_ctx - len(cc):n_ctx] = cc
                full[n_ctx:n_ctx + len(ct)] = ct
            else:
                # Promotion or holiday flags: pulses of a few steps at a low rate.
                full = np.zeros(len(base), np.float32)
                for start in np.flatnonzero(rng.random(len(base)) < rng.uniform(0.005, 0.05)):
                    full[start:start + int(rng.integers(1, 6))] = 1.0
            covs.append(full)
        y, _ = covariate_target(rng, base, np.stack(covs), n_ctx, min(h, 24))
        vals = np.concatenate([y[None], np.stack(covs)])
        return vals[:, :n_ctx], vals[:, n_ctx:], np.ones(V, bool)

    def _multi_batch(self, rng, H: int):
        L = self.context
        synth = not self.item_pools or rng.random() < self.p_cov_synth
        if synth:
            V = int(rng.integers(2, 6))
        else:
            fam0 = self.item_families[rng.choice(len(self.item_families), p=self.item_family_p)]
            V = min(self.max_variates, max(len(it) for it in self.item_pools[fam0]))
        B = max(1, self.batch_size // V)
        x = np.full((B, V, L), np.nan, dtype=np.float32)
        y = np.full((B, V, H), np.nan, dtype=np.float32)
        w = np.ones(B, dtype=np.float32)
        roles = np.full((B, V), TARGET, dtype=np.int64)
        valid = np.zeros((B, V), dtype=bool)
        for b in range(B):
            if b % max(1, B // self.n_groups) == 0:
                # A group of items per draw, as for single windows: one family (or subset)
                # and one decimation factor.
                if synth:
                    s, k = self._draw(rng)
                else:
                    fam = fam0 if b == 0 else self.item_families[
                        rng.choice(len(self.item_families), p=self.item_family_p)]
                    k = int(rng.choice(DECIMATION)) if rng.random() < self.p_decimate else 1
            if synth:
                c, t, ok = self._covariate_item(rng, s, k, V, H)
                w[b] = s.scale * k
                roles[b, 1:] = np.where(rng.random(V - 1) < 0.7, FUTURE_COVARIATE, PAST_COVARIATE)
            else:
                c, t, ok, w[b] = self._item(rng, fam, k, V, H)
                n = int(ok.sum())
                r = rng.choice([TARGET, PAST_COVARIATE, FUTURE_COVARIATE], size=n,
                               p=[0.5, 0.25, 0.25])
                r[rng.integers(n)] = TARGET
                roles[b, :n] = r
            sign = -1.0 if rng.random() < self.p_flip else 1.0
            c = c[:, -L:]
            x[b, :, L - c.shape[1]:] = sign * c
            y[b, :, :t.shape[1]] = sign * t
            valid[b] = ok
        return tuple(torch.from_numpy(a) for a in (x, y, w, roles, valid))

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0])
        self._setup()
        L = self.context
        log_max = math.log(self.max_horizon)
        while True:
            H = int(math.exp(rng.uniform(math.log(self.patch), log_max)))
            if self.p_multi > 0 and rng.random() < self.p_multi:
                yield self._multi_batch(rng, H)
                continue
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
