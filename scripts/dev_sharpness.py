"""
Where the dev loss comes from, by how regular the series are.

    python scripts/dev_sharpness.py /workspace/runs/full0 [/workspace/runs/...] [--root ...]

Each forecast cut of the dev subsets gets a regularity, the mean absolute error of the
seasonal naive over the mean absolute target. Cuts are pooled over subsets and split in
terciles of regularity; per tercile: the quantile loss over the seasonal naive's (
a geometric mean of the per-cut ratios, cuts with a flat or zero target left out), the coverage of the
10-90 % interval (0.8 nominal) and its width over the mean absolute error of the median.
A diagnostic: it reads the targets to sort the cuts, so it is never used to select.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from timessm.patched import build_from_config

from dev_eval import dev_forecasts


def pinball(q, y, levels):
    """Mean quantile loss per cut, [n]."""
    e = y[..., None] - q
    return np.nanmean(np.maximum(levels * e, (levels - 1) * e), axis=(1, 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--root", help="corpus, default the one of the run")
    a = ap.parse_args()
    for run in a.runs:
        cfg = OmegaConf.load(run / "config.yaml")
        model = build_from_config(cfg).cuda().eval()
        state = torch.load(run / f"{cfg.run.name}.ckpt", map_location="cuda", weights_only=False)
        model.load_state_dict({k[len("model."):]: v for k, v in state["state_dict"].items()})
        levels = np.array(model.quantile_levels)
        lo, mid, hi = 0, len(levels) // 2, len(levels) - 1
        cols = {k: [] for k in ("reg", "ours", "naive", "cover", "width")}
        for _, q, y, sn in dev_forecasts(model, a.root or cfg.data.root):
            scale = np.maximum(np.nanmean(np.abs(y), 1), 1e-9)
            ok = np.isfinite(y).any(1)
            cols["reg"].append(np.nanmean(np.abs(y - sn), 1) / scale)
            cols["ours"].append(pinball(q, y, levels) / scale)
            cols["naive"].append(pinball(sn[..., None].repeat(len(levels), -1), y, levels) / scale)
            inside = (y >= q[..., lo]) & (y <= q[..., hi])
            cols["cover"].append(np.nanmean(np.where(np.isfinite(y), inside, np.nan), 1))
            err = np.nanmean(np.abs(y - q[..., mid]), 1)
            cols["width"].append(np.nanmean(q[..., hi] - q[..., lo], 1) / np.maximum(err, 1e-9))
            for k in cols:
                cols[k][-1] = np.where(ok, cols[k][-1], np.nan)
        c = {k: np.concatenate(v) for k, v in cols.items()}
        # Flat or all-zero targets have no defined ratio: the naive is exact or the scale is 0.
        keep = np.isfinite(c["reg"]) & np.isfinite(c["ours"]) & (c["naive"] > 1e-6) \
            & (c["reg"] > 1e-6) & (c["reg"] < 10)
        c = {k: v[keep] for k, v in c.items()}
        edges = np.quantile(c["reg"], [0, 1 / 3, 2 / 3, 1])
        out = {"run": run.name, "step": state["step"], "terciles": []}
        for i in range(3):
            sel = (c["reg"] >= edges[i]) & (c["reg"] <= edges[i + 1])
            out["terciles"].append({
                "regularity": [round(float(edges[i]), 4), round(float(edges[i + 1]), 4)],
                "cuts": int(sel.sum()),
                "loss_vs_naive": round(float(np.exp(np.mean(np.log(np.clip(
                    c["ours"][sel] / c["naive"][sel], 1e-3, 1e3))))), 4),
                "coverage_10_90": round(float(np.nanmean(c["cover"][sel])), 3),
                "width_over_error": round(float(np.nanmedian(c["width"][sel])), 2)})
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
