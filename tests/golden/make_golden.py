"""
Reference outputs of the model and of the training module, recorded by the code
as it stood BEFORE the frequency-tied Delta (2026-10-05). tests/test_golden.py
compares the current code against this file with every new mode off.

    python tests/golden/make_golden.py        # writes tests/golden/ssm_golden.pt

Regenerate only for a change that is MEANT to move the default numbers, and say
so in the commit; never to make a failing test pass.
"""

import platform
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from timessm.model import SSMForecaster  # noqa: E402
from timessm.training import SSMFinetuneModule  # noqa: E402


def small_model():
    torch.manual_seed(0)
    return SSMForecaster(input_length=128, prediction_length=32, d_model=16, n_layers=2,
                         d_state=8, expand=2, dropout=0.0, quantile_hidden_dim=32).eval()


def series(B=4, L=128, seed=1):
    torch.manual_seed(seed)
    t = torch.arange(L).float()
    base = torch.sin(2 * torch.pi * t / 24)[None, :, None]
    return 50.0 + 10.0 * base + torch.randn(B, L, 1) + 5.0 * torch.arange(B)[:, None, None]


def compute() -> dict:
    """Everything the golden file pins, from the current code."""
    model = small_model()
    x, y = series(), series(L=32, seed=2)
    out = {}
    with torch.no_grad():
        for name, w in (("bare", None), ("w_float", 0.5), ("w_uniform", torch.full((4,), 0.5)),
                        ("w_per_item", torch.tensor([0.25, 0.5, 1.0, 2.0]))):
            res = model.forecast(x, n=48) if w is None else model.forecast(x, n=48, w=w)
            out[f"forecast/{name}"] = res["quantiles_denorm"].clone()
    # the training module: three consecutive losses in train mode pin the random
    # stream of the Delta draw, one in eval pins the Delta = 1 path
    mod = SSMFinetuneModule(small_model(), finetune_mode="full_finetune", loss_type="huber",
                            learning_rate=1e-3, encoder_lr_multiplier=1.0, lr_scheduler="constant",
                            delta_scales=[0.25, 0.5, 2.0], p_delta_scale=0.7)
    mod.train()
    random.seed(7)
    torch.manual_seed(7)
    losses, scales = [], []
    for _ in range(3):
        loss, _, _ = mod._forward_and_loss(x, y)
        losses.append(loss.detach())
        scales.append(mod._last_delta_scale)
    out["train/losses"] = torch.stack(losses)
    out["train/scales"] = torch.tensor(scales)
    mod.eval()
    with torch.no_grad():
        out["eval/loss"] = mod._forward_and_loss(x, y)[0].detach()
    out["state_dict_keys"] = sorted(mod.model.state_dict())
    return out


if __name__ == "__main__":
    golden = compute()
    golden["meta"] = {"torch": torch.__version__, "machine": platform.machine(), "date": "2026-10-05"}
    torch.save(golden, HERE / "ssm_golden.pt")
    print({k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in golden.items() if k != "state_dict_keys"})
