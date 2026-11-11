"""
Model selection on the subsets held out of training (timessm.corpus.DEV_SUBSETS).

Forecasts from the last `origins` origins of each series, one horizon apart, so
that one shock shared by every series at the end of a subset does not decide its
score. The score of a subset is
the weighted quantile loss over the seasonal naive's, the same ratio as on the
leaderboard; the summary is their geometric mean. GIFT-Eval is never read here.

The multivariate dev (dev_multi_score) reads the items of the multivariate dev subsets
(timessm.corpus.DEV_MULTI, plus the dev subset alibaba_cluster_trace_2018) the same way,
on their target variates, twice: jointly (forecast_multivariate, covariates read) and
variate by variate (forecast); the gap is what reading across variates brings.
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from timejepa.data.frequency import parse_frequency
from timessm.corpus import DEV_MULTI, DEV_SUBSETS, block_mean, item_family, load_subsets
from timessm.groups import aligned, load_groups, span
from timessm.patched import FUTURE_COVARIATE, PAST_COVARIATE, TARGET

HORIZON = {"S": 360, "T": 288, "H": 48, "D": 30, "W": 8, "M": 18, "Q": 8, "A": 4}
SEASON = {"S": 360, "T": 288, "H": 24, "D": 7, "W": 52, "M": 12, "Q": 4, "A": 1}
MAX_SERIES = 512
ORIGINS = 4
MAX_ITEMS = 64
# How the variates of a multivariate dev family are read (by subset suffix); all targets
# otherwise. Kelmarsh: the power of a turbine with its wind known ahead.
DEV_MULTI_ROLES = {"kelmarsh": {"power": TARGET, "wind": FUTURE_COVARIATE,
                                "rpm": PAST_COVARIATE, "temp": PAST_COVARIATE}}


def wql(q: np.ndarray, y: np.ndarray, levels: np.ndarray) -> float:
    e = y[..., None] - q
    return 2 * np.nanmean(np.nansum(np.maximum(levels * e, (levels - 1) * e), axis=(0, 1))) \
        / max(np.nansum(np.abs(y)), 1e-9)


def rate_factor(scale: float, cycle: float) -> int:
    """Block size bringing a series of `24 / scale` steps per cycle near `cycle` steps per cycle,
    1 when it is already slower. The fixed rule of RateIN."""
    if not cycle or scale <= 0:
        return 1
    return max(1, int(round(24.0 / scale / cycle)))


def dev_subsets(root: str):
    """Per dev subset: (name, subset, horizon h, season m, indices of the sampled series),
    the sample every dev score reads."""
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    for name, s in load_subsets(root, DEV_SUBSETS).items():
        unit, step = parse_frequency(manifest[name]["freq"])
        h = max(1, HORIZON[unit] // step)
        m = max(1, SEASON[unit] // step)
        keep = np.flatnonzero(s.lengths >= h + max(m + 1, 8))
        rng = np.random.default_rng(0)
        yield name, s, h, m, rng.choice(keep, min(MAX_SERIES, len(keep)), replace=False)


@torch.no_grad()
def dev_forecasts(model, root: str, batch: int = 128, origins: int = ORIGINS, cycle: float = 0.0,
                  flip: bool = False, clip: bool = False):
    """Per dev subset: (name, quantiles [n, h, Q], targets [n, h], seasonal naive [n, h]).
    `cycle` > 0 forecasts from block means of the context, at about `cycle` steps per
    season, each coarse quantile held over its k native steps. `flip` averages the forecast
    with the mirrored forecast of the negated context; `clip` sets the quantiles of a series
    whose context is non-negative at 0 at least (after the flip average)."""
    device = next(model.parameters()).device
    for name, s, h, m, keep in dev_subsets(root):
        cuts = [(j, o) for j in keep for o in range(origins)
                if s.lengths[j] >= (o + 1) * h + max(m + 1, 8)]
        w = s.scale if s.scale > 0 else 1.0
        k = rate_factor(s.scale, cycle)
        hk = -(-h // k)
        ours, naive, targets = [], [], []
        for i in range(0, len(cuts), batch):
            ctx, tgt, nai = [], [], []
            for j, o in cuts[i:i + batch]:
                x = np.asarray(s.series(j), dtype=np.float32)
                x = x[:len(x) - o * h]
                c, t = x[:-h], x[-h:]
                ctx.append((block_mean(c, k) if k > 1 else c)[-model.input_length:])
                tgt.append(t)
                nai.append(np.resize(c[-m:], h))
            L = max(map(len, ctx))
            pad = np.stack([np.pad(c, (L - len(c), 0), constant_values=np.nan) for c in ctx])
            x = torch.from_numpy(pad).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fan = model.forecast(x, hk, w * k)["quantiles_denorm"].float()
                if flip:
                    fan = (fan - model.forecast(-x, hk, w * k)["quantiles_denorm"].float()
                           .flip(-1)) / 2
            fan = fan.cpu().numpy()
            if clip:
                nonneg = ~(np.nan_to_num(pad, nan=0.0) < 0).any(1)
                fan = np.where(nonneg[:, None, None], np.maximum(fan, 0), fan)
            ours.append(np.repeat(fan, k, axis=1)[:, :h])
            naive.append(np.stack(nai))
            targets.append(np.stack(tgt))
        yield name, *map(np.concatenate, (ours, targets, naive))


def dev_score(model, root: str, batch: int = 128, origins: int = ORIGINS,
              cycle: float = 0.0, flip: bool = False, clip: bool = False) -> Dict[str, float]:
    levels = np.array(model.quantile_levels)
    scores = {}
    for name, q, y, sn in dev_forecasts(model, root, batch, origins, cycle, flip, clip):
        base = wql(sn[..., None].repeat(len(levels), -1), y, levels)
        scores[name] = float(wql(q, y, levels) / max(base, 1e-9))
    scores["all"] = math.exp(np.mean([math.log(max(v, 1e-6)) for v in scores.values()]))
    return scores


def dev_items(root: str, origins: int = ORIGINS):
    """Per multivariate dev family: (family, items, roles per item [V], h, m), at most
    MAX_ITEMS items long enough for `origins` windows, a fixed sample."""
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    names = [n for n in DEV_MULTI + ("alibaba_cluster_trace_2018",) if n in manifest]
    by_family = defaultdict(list)
    for it in load_groups(root, load_subsets(root, names)):
        by_family[item_family(it)].append(it)
    for fam, items in sorted(by_family.items()):
        unit, step = parse_frequency(manifest[items[0][0][0].name]["freq"])
        h = max(1, HORIZON[unit] // step)
        m = max(1, SEASON[unit] // step)
        keep = [it for it in items if span(it)[1] - span(it)[0] >= origins * h + max(m + 1, 8)]
        if not keep:
            continue
        pick = np.random.default_rng(0).choice(len(keep), min(MAX_ITEMS, len(keep)), replace=False)
        keep = [keep[i] for i in sorted(pick)]
        table = DEV_MULTI_ROLES.get(fam, {})
        roles = [np.array([table.get(s.name.rsplit("_", 1)[-1], TARGET) for s, _, _ in it])
                 for it in keep]
        yield fam, keep, roles, h, m


@torch.no_grad()
def dev_multi_score(model, root: str, origins: int = ORIGINS) -> Dict[str, Dict[str, float]]:
    """{"joint": per family and "all", "single": the same model variate by variate}, the
    score of dev_score (wQL over the seasonal naive's) on the target variates."""
    device = next(model.parameters()).device
    levels = np.array(model.quantile_levels)
    out = {"joint": {}, "single": {}}
    for fam, items, roles, h, m in dev_items(root, origins):
        acc = defaultdict(list)
        for it, r in zip(items, roles):
            start, end = span(it)
            vals = aligned(it, start, end)
            w = it[0][0].scale if it[0][0].scale > 0 else 1.0
            tgt = r == TARGET
            for o in range(origins):
                cut = vals.shape[1] - o * h
                c, t = vals[:, :cut - h], vals[:, cut - h:cut]
                x = torch.from_numpy(np.ascontiguousarray(c[:, -model.input_length:])).to(device)
                with torch.autocast(device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    joint = model.forecast_multivariate(
                        x[None], h, w, torch.from_numpy(r)[None].to(device),
                        future=torch.from_numpy(t)[None].to(device))["quantiles_denorm"][0]
                    single = model.forecast(x, h, w)["quantiles_denorm"]
                acc["joint"].append(joint.float().cpu().numpy()[tgt])
                acc["single"].append(single.float().cpu().numpy()[tgt])
                acc["y"].append(t[tgt])
                acc["naive"].append(np.stack([np.resize(row[-m:], h) for row in c[tgt]]))
        y, naive = np.concatenate(acc["y"]), np.concatenate(acc["naive"])
        base = wql(naive[..., None].repeat(len(levels), -1), y, levels)
        for mode in ("joint", "single"):
            out[mode][fam] = float(wql(np.concatenate(acc[mode]), y, levels) / max(base, 1e-9))
    for mode in out:
        vals = list(out[mode].values())
        out[mode]["all"] = (math.exp(np.mean([math.log(max(v, 1e-6)) for v in vals]))
                            if vals else float("nan"))
    return out
