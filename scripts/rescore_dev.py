"""
Score finished runs again on the held-out subsets.

    python scripts/rescore_dev.py /workspace/runs/base0 /workspace/runs/base1 --origins 1 4
"""

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from timessm.patched import build_from_config

from dev_eval import dev_score


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--origins", nargs="+", type=int, default=[4])
    ap.add_argument("--root", help="corpus, default the one of the run")
    ap.add_argument("--cycle", nargs="+", type=float, default=[0.0],
                    help="steps per season of the fixed-rate rule, 0 = native")
    ap.add_argument("--flip", action="store_true", help="average with the negated context")
    ap.add_argument("--clip", action="store_true", help="non-negative series: quantiles >= 0")
    a = ap.parse_args()
    for run in a.runs:
        cfg = OmegaConf.load(run / "config.yaml")
        model = build_from_config(cfg).cuda().eval()
        state = torch.load(run / f"{cfg.run.name}.ckpt", map_location="cuda", weights_only=False)
        model.load_state_dict({k[len("model."):]: v for k, v in state["state_dict"].items()})
        for o in a.origins:
            for c in a.cycle:
                score = dev_score(model, a.root or cfg.data.root, origins=o, cycle=c,
                                  flip=a.flip, clip=a.clip)
                print(json.dumps({"run": run.name, "step": state["step"], "origins": o,
                                  "cycle": c, "flip": a.flip, "clip": a.clip, "dev": score["all"], "per_subset": score}),
                      flush=True)


if __name__ == "__main__":
    main()
