"""
Reference outputs of PatchedSSM, of its training loss and of the training batches with every
optional mode off. tests/test_patched_golden.py compares the current code against this file.

    python tests/golden/make_patched_golden.py   # writes tests/golden/patched_golden.pt

Regenerate only for a change that is meant to move the default numbers, and say so in the
commit; never to make a failing test pass.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parents[1] / "scripts"))

from timessm.corpus import WindowBatches  # noqa: E402
from timessm.patched import PatchedSSM  # noqa: E402
from train_patched import step_loss  # noqa: E402


def small(**kw):
    torch.manual_seed(0)
    return PatchedSSM(input_length=256, patch=16, d_model=32, n_layers=2, d_state=4,
                      n_heads=4, **kw)


def series(B=4, L=200, seed=1):
    """Seasonal series at different levels, a NaN gap and one short history (left NaN)."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(L).float()
    x = (50.0 + 10.0 * torch.sin(2 * torch.pi * t / 24)[None] + torch.randn(B, L, generator=g)
         + 5.0 * torch.arange(B)[:, None])
    x[1, 60:75] = float("nan")
    x[2, :150] = float("nan")
    return x


def corpus(root: Path):
    rng = np.random.default_rng(0)
    subsets = {"short_m": ("MS", [40, 60]), "long_h": ("H", [3000]),
               "synth_tight_H": ("H", [2000, 2000]), "mid_d": ("D", [500, 900])}
    manifest = {}
    for name, (freq, lengths) in subsets.items():
        values = np.concatenate([rng.normal(10, 3, n).astype(np.float32) for n in lengths])
        np.save(root / f"{name}.values.npy", values)
        np.save(root / f"{name}.offsets.npy", np.concatenate([[0], np.cumsum(lengths)]))
        manifest[name] = {"subset": name, "n_obs": int(values.size), "freq": freq}
    (root / "manifest.json").write_text(json.dumps(manifest))


def compute() -> dict:
    out = {}
    x = series()
    with torch.no_grad():
        for kind, kw in (("plain", {}), ("causal", dict(causal=True, loops=2))):
            model = small(**kw).eval()
            out[f"keys/{kind}"] = sorted(model.state_dict())
            out[f"params/{kind}"] = sum(p.numel() for p in model.parameters())
            for name, w in (("bare", None), ("w_float", 0.5),
                            ("w_per_item", torch.tensor([0.25, 0.5, 1.0, 2.0]))):
                res = model.forecast(x, n=40) if w is None else model.forecast(x, n=40, w=w)
                out[f"forecast/{kind}/{name}"] = res["quantiles_denorm"].clone()
        model = small(causal=True, loops=2).eval()
        xp = model._pad(x)
        mask = torch.isfinite(xp)
        z, _, _ = model.normalize_causal(xp, mask, 16)
        q, qc = model.forward_normalized(z, mask, 40, 1.0, return_context=True)
        out["context/q"], out["context/qc"] = q.clone(), qc.clone()
    t = OmegaConf.create({"loss": "asinh", "context_loss": 1.0, "context_skip": 2})
    y = series(L=48, seed=2)
    w = torch.tensor([1.0, 0.5, 2.0, 1.0])
    model = small(causal=True, loops=2).train()
    levels = torch.tensor(model.levels)
    loss = step_loss(model, x, y, w, levels, t, 16)
    loss.backward()
    out["train/loss"] = loss.detach().clone()
    out["train/grad_norm"] = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()
                                            if p.grad is not None)).detach()
    with tempfile.TemporaryDirectory() as tmp:
        corpus(Path(tmp))
        it = iter(WindowBatches(tmp, 8, 256, max_horizon=64, patch=16, p_decimate=0.5,
                                p_flip=0.5, p_synth=0.5, groups=4, seed=3, p_full=0.5))
        for i in range(3):
            bx, by, bw = next(it)
            out[f"batch/{i}/x"], out[f"batch/{i}/y"], out[f"batch/{i}/w"] = bx, by, bw
    return out


if __name__ == "__main__":
    torch.save(compute(), HERE / "patched_golden.pt")
    print("wrote", HERE / "patched_golden.pt")
