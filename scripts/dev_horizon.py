"""
Dev score by part of the horizon: is a gain made on the first steps or on the whole horizon?

    python scripts/dev_horizon.py /workspace/runs/full0 /workspace/runs/causal0 --parts 3

The horizon of each subset is cut into `parts` equal slices; a slice scores as the dev
does, WQL over the seasonal naive's on the same steps, geometric mean over the subsets.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from timessm.patched import build_from_config

from dev_eval import ORIGINS, dev_forecasts, wql


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--parts", type=int, default=3)
    ap.add_argument("--origins", type=int, default=ORIGINS)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    for run in a.runs:
        cfg = OmegaConf.load(run / "config.yaml")
        model = build_from_config(cfg).cuda().eval()
        state = torch.load(run / f"{cfg.run.name}.ckpt", map_location="cuda", weights_only=False)
        model.load_state_dict({k[len("model."):]: v for k, v in state["state_dict"].items()})
        levels = np.array(model.quantile_levels)
        logs = [[] for _ in range(a.parts)]
        for name, q, y, sn in dev_forecasts(model, cfg.data.root, a.batch, a.origins):
            edges = np.linspace(0, y.shape[1], a.parts + 1).round().astype(int)
            for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                if hi <= lo:
                    continue
                base = wql(sn[:, lo:hi, None].repeat(len(levels), -1), y[:, lo:hi], levels)
                logs[i].append(math.log(max(wql(q[:, lo:hi], y[:, lo:hi], levels)
                                            / max(base, 1e-9), 1e-6)))
        print(json.dumps({"run": run.name, "step": state["step"],
                          "parts": [math.exp(np.mean(v)) for v in logs]}), flush=True)


if __name__ == "__main__":
    main()
