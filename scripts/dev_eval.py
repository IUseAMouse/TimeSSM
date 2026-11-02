"""
Model selection on the subsets held out of training (timessm.corpus.DEV_SUBSETS).

One forecast per series, from the end of its history. The score of a subset is
the weighted quantile loss over the seasonal naive's, the same ratio as on the
leaderboard; the summary is their geometric mean. GIFT-Eval is never read here.
"""

import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from timejepa.data.frequency import parse_frequency
from timessm.corpus import DEV_SUBSETS, load_subsets

HORIZON = {"S": 360, "T": 288, "H": 48, "D": 30, "W": 8, "M": 18, "Q": 8, "A": 4}
SEASON = {"S": 360, "T": 288, "H": 24, "D": 7, "W": 52, "M": 12, "Q": 4, "A": 1}
MAX_SERIES = 512


def wql(q: np.ndarray, y: np.ndarray, levels: np.ndarray) -> float:
    e = y[..., None] - q
    return 2 * np.nanmean(np.nansum(np.maximum(levels * e, (levels - 1) * e), axis=(0, 1))) \
        / max(np.nansum(np.abs(y)), 1e-9)


@torch.no_grad()
def dev_score(model, root: str, batch: int = 128) -> Dict[str, float]:
    device = next(model.parameters()).device
    levels = np.array(model.quantile_levels)
    scores = {}
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    for name, s in load_subsets(root, DEV_SUBSETS).items():
        unit, step = parse_frequency(manifest[name]["freq"])
        h = max(1, HORIZON[unit] // step)
        m = max(1, SEASON[unit] // step)
        keep = np.flatnonzero(s.lengths >= h + max(m + 1, 8))
        rng = np.random.default_rng(0)
        keep = rng.choice(keep, min(MAX_SERIES, len(keep)), replace=False)
        ours, naive, targets = [], [], []
        for i in range(0, len(keep), batch):
            ctx, tgt, nai = [], [], []
            for j in keep[i:i + batch]:
                x = np.asarray(s.series(j), dtype=np.float32)
                c, t = x[:-h], x[-h:]
                ctx.append(c[-model.input_length:])
                tgt.append(t)
                nai.append(np.resize(c[-m:], h))
            L = max(map(len, ctx))
            pad = np.stack([np.pad(c, (L - len(c), 0), constant_values=np.nan) for c in ctx])
            w = s.scale if s.scale > 0 else 1.0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.forecast(torch.from_numpy(pad).to(device), h, w)
            ours.append(out["quantiles_denorm"].float().cpu().numpy())
            naive.append(np.stack(nai))
            targets.append(np.stack(tgt))
        q, y, sn = map(np.concatenate, (ours, targets, naive))
        base = wql(sn[..., None].repeat(len(levels), -1), y, levels)
        scores[name] = float(wql(q, y, levels) / max(base, 1e-9))
    scores["all"] = math.exp(np.mean([math.log(max(v, 1e-6)) for v in scores.values()]))
    return scores
