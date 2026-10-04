"""refit_last_layer.py: the L-BFGS refit of the head's final projection (2026-10-04)."""

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))

import refit_last_layer as rl  # noqa: E402
from timejepa.models.decoders.quantile_head import QuantileHead  # noqa: E402
from timessm.model import SSMForecaster  # noqa: E402


def _problem(n=4000, d=8):
    torch.manual_seed(0)
    head = QuantileHead(d_model=d, patch_size=1, stride=1, prediction_length=4, use_context=False)
    phi = torch.randn(n, d)
    y = phi @ torch.randn(d) + 0.3 * torch.randn(n)          # a median the random init does not know
    proj = head.unpatching.projection
    return head, phi, y, proj.weight.detach().clone(), proj.bias.detach().clone()


def test_refit_lowers_the_pinball_and_keeps_the_fan_sorted():
    head, phi, y, w0, b0 = _problem()
    w, b, before, after = rl.refit(head, [(phi, y)], w0, b0)
    assert after < 0.5 * before
    fan = head._make_monotone(phi @ w.T + b)
    assert bool((fan[:, 1:] >= fan[:, :-1]).all())
    assert abs(float((fan[:, 0] > y).float().mean()) - 0.1) < 0.03      # the 10% level covers ~10%


def test_median_mode_moves_the_median_row_only():
    head, phi, y, w0, b0 = _problem()
    w, b, before, after = rl.refit(head, [(phi, y)], w0, b0, rows="median")
    mid = head.median_idx
    others = [i for i in range(w.shape[0]) if i != mid]
    assert torch.equal(w[others], w0[others]) and torch.equal(b[others], b0[others])
    assert not torch.equal(w[mid], w0[mid]) and after < before


def test_ridge_penalty_pulls_toward_the_trained_weights():
    head, phi, y, w0, b0 = _problem()
    free = rl.refit(head, [(phi, y)], w0, b0)[0]
    tied = rl.refit(head, [(phi, y)], w0, b0, lam=10.0)[0]
    assert (tied - w0).norm() < 0.2 * (free - w0).norm()


def test_chunks_and_slices_give_the_full_batch_fit(monkeypatch):
    import numpy as np
    head, phi, y, w0, b0 = _problem()
    w = w0.clone().requires_grad_()
    b = b0.clone().requires_grad_()
    full = rl.head_loss(head, phi, y, w, b)
    full.backward()
    grad_w, grad_b = w.grad.clone(), b.grad.clone()
    w.grad = None; b.grad = None
    monkeypatch.setattr(rl, "SLICE_ROWS", 700)                    # uneven slices inside uneven chunks
    chunks = rl.spread(phi, y, ["cpu", "cpu", "cpu"], rows_per_device=10_000, rng=np.random.default_rng(0))
    assert [len(c[1]) for c in chunks] == [1333, 1333, 1334]
    value = rl.chunked_loss(head, chunks, w, b, backward=True)
    assert abs(value - float(full)) < 1e-6
    assert torch.allclose(w.grad, grad_w, atol=1e-6) and torch.allclose(b.grad, grad_b, atol=1e-6)
    whole = rl.refit(head, [(phi, y)], w0, b0, max_iter=40)
    split = rl.refit(head, chunks, w0, b0, max_iter=40)
    assert abs(whole[2] - split[2]) < 1e-6 and abs(whole[3] - split[3]) < 1e-3
    capped = rl.spread(phi, y, ["cpu", "cpu"], rows_per_device=500, rng=np.random.default_rng(0))
    assert sum(len(c[1]) for c in capped) == 1000


def test_collect_reads_the_projection_input_in_the_loss_frame():
    torch.manual_seed(0)
    model = SSMForecaster(input_length=64, prediction_length=16, d_model=16, n_layers=1, d_state=4,
                          quantile_hidden_dim=32).eval()
    data = [{"context": torch.randn(64).cumsum(0), "target": torch.randn(16)} for _ in range(24)]
    import numpy as np
    phi, y = rl.collect(model, data, [24], per_dataset=24, steps=4, batch_size=8, device="cpu",
                        context_lengths=[32], p_context=0.5, rng=np.random.default_rng(0))
    assert phi.shape == (24 * 4, 16) and y.shape == (24 * 4,)
    head = model.decoder.decoder
    proj = head.unpatching.projection
    # the collected features reproduce the model's own fan on one window
    ctx = data[0]["context"].reshape(1, -1, 1)
    out = model.forecast(ctx, n=16)
    grabbed = {}
    hook = proj.register_forward_pre_hook(lambda _m, inp: grabbed.update(phi=inp[0]))
    model.forecast(ctx, n=16); hook.remove()
    fan = head._make_monotone(grabbed["phi"][0] @ proj.weight.T + proj.bias)
    assert torch.allclose(fan, out["quantiles"][0], atol=1e-5)
