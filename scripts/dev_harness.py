"""
The dev subsets through the GIFT harness procedure (evaluate_gift.evaluate_config): the
inference options (flip, the RateIN stack) are measured on held-out data with the code that
serves them on GIFT, never on GIFT.

    python scripts/dev_harness.py /workspace/runs/timessm_10m_v9 --flip --ratein

Same sample as dev_eval.py (dev_subsets), each series forecast at its last ORIGINS windows
of h steps, scored as the weighted quantile loss over the seasonal naive's; the summary is
their geometric mean. Unlike dev_eval.py, a window whose context is shorter than a season is
kept (the harness splits as GIFT does), so the two scores are close, not equal.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from timessm.patched import build_from_config

from dev_eval import ORIGINS, dev_subsets

TIMEJEPA = Path(os.environ.get("TIMEJEPA", Path(__file__).resolve().parents[2] / "TimeJEPA"))
sys.path.insert(0, str(TIMEJEPA / "scripts"))
import evaluate_gift as EG  # noqa: E402

# The published RateIN stack (+ratein=mix +ratein_pool=true +ratein_k_up=2x3x4
# +ratein_min_bt=4 +ratein_bt_windows=4), with the flip in tta_flip.
RATEIN_STACK = dict(ratein_mode="mix", ratein_pool=True, ratein_k_up=[2, 3, 4],
                    ratein_min_bt=4, ratein_bt_windows=4)


def harness_score(model, root: str, flip: bool = False, ratein: bool = False,
                  batch: int = 128, origins: int = ORIGINS) -> dict:
    device = next(model.parameters()).device
    kw = dict(tta_flip=flip, **(RATEIN_STACK if ratein else {}))
    scores, diags = {}, {}
    for name, s, h, m, keep in dev_subsets(root):
        series = [np.asarray(s.series(j), dtype=np.float32) for j in keep]
        data = EG.gift.ConfigData(series, h, m, origins)
        # +freq_delta: the season the corpus declares sets Delta, as in training
        season = 24.0 / s.scale if s.scale > 0 else None
        res = EG.evaluate_config(model, name, Path("."), device, batch, data=data,
                                 freq_season=season, **kw)
        scores[name] = res["model"]["CRPS"] / max(res["seasonal_naive_local"]["CRPS"], 1e-9)
        if "ratein" in res:
            diags[name] = res["ratein"]["mix"]["weights"]
    scores["all"] = math.exp(np.mean([math.log(max(v, 1e-6)) for v in scores.values()]))
    return {"dev": scores["all"], "per_subset": scores, "ratein": diags}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--ckpt", type=Path, help="default: the run's last checkpoint")
    ap.add_argument("--root", help="corpus, default the one of the run")
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--ratein", action="store_true", help="the published RateIN stack")
    a = ap.parse_args()
    cfg = OmegaConf.load(a.run / "config.yaml")
    model = build_from_config(cfg).cuda().eval()
    ckpt = a.ckpt or a.run / f"{cfg.run.name}.ckpt"
    state = torch.load(ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict({k[len("model."):]: v for k, v in state["state_dict"].items()})
    out = harness_score(model, a.root or cfg.data.root, a.flip, a.ratein)
    print(json.dumps({"run": a.run.name, "ckpt": ckpt.name, "step": state["step"], "flip": a.flip,
                      "ratein": a.ratein, **out}), flush=True)


if __name__ == "__main__":
    main()
