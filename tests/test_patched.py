import math

import numpy as np
import pytest
import torch

from timessm.corpus import block_mean, family, rate_class, subset_scale
from timessm.patched import PatchedSSM, rotate
from timessm.ssm import S4DLayer


def small(**kw):
    args = dict(input_length=256, patch=16, d_model=32, n_layers=2, d_state=4, n_heads=4)
    return PatchedSSM(**(args | kw)).eval()


def test_forecast_shapes_and_sorted():
    out = small().forecast(torch.randn(3, 200), 50)
    q = out["quantiles_denorm"]
    assert q.shape == (3, 50, 9)
    assert (q[..., 1:] >= q[..., :-1]).all()


def test_missing_values_do_not_poison_the_forecast():
    x = torch.randn(2, 160)
    x[0, 20:60] = float("nan")
    assert torch.isfinite(small().forecast(x, 32)["quantiles_denorm"]).all()


def test_masked_values_are_ignored():
    m = small()
    x = torch.randn(1, 160)
    y = x.clone()
    x[0, 40], y[0, 40] = float("nan"), float("nan")
    a = m.forecast(x, 16)["quantiles_denorm"]
    b = m.forecast(y, 16)["quantiles_denorm"]
    assert torch.equal(a, b)


def test_huge_values_stay_finite():
    x = torch.randn(2, 160) * 1e25
    assert torch.isfinite(small().forecast(x, 16)["quantiles"]).all()


def test_constant_series():
    assert torch.isfinite(small().forecast(torch.ones(2, 160) * 7, 16)["quantiles_denorm"]).all()


def test_scale_and_shift_equivariance():
    m = small()
    x = torch.randn(1, 160).double().float()
    a = m.forecast(x, 16)["quantiles_denorm"]
    b = m.forecast(3 * x + 5, 16)["quantiles_denorm"]
    assert torch.allclose(b, 3 * a + 5, atol=2e-2, rtol=2e-2)


def test_left_padding_is_read_as_absent():
    m = small()
    x = torch.randn(1, 160)
    pad = torch.cat([torch.full((1, 96), float("nan")), x], 1)
    a = m.forecast(x, 16)["quantiles_denorm"]
    b = m.forecast(pad, 16)["quantiles_denorm"]
    assert a.shape == b.shape and torch.isfinite(b).all()


def test_horizon_is_free():
    m = small()
    x = torch.randn(1, 160)
    long = m.forecast(x, 64)["quantiles_denorm"]
    assert long.shape[1] == 64 and m.forecast(x, 7)["quantiles_denorm"].shape[1] == 7


def test_per_item_rate():
    w = torch.tensor([0.5, 2.0])
    assert torch.isfinite(small().forecast(torch.randn(2, 160), 16, w)["quantiles"]).all()


def test_rate_reaches_ssm_and_attention():
    m, x = small(), torch.randn(1, 160)
    a = m.forecast(x, 16, 1.0)["quantiles"]
    b = m.forecast(x, 16, 2.0)["quantiles"]
    assert not torch.allclose(a, b)


def test_rotation_preserves_norm_and_keeps_relative_inner_product():
    x, y = torch.randn(1, 1, 3, 8), torch.randn(1, 1, 3, 8)
    inv = 0.1 * torch.arange(1, 5).float()
    p = torch.tensor([[0.0, 1.0, 2.0]])
    assert torch.allclose(rotate(x, p, inv).norm(dim=-1), x.norm(dim=-1), atol=1e-5)
    a = (rotate(x, p, inv) * rotate(y, p, inv)).sum(-1)
    b = (rotate(x, p + 5, inv) * rotate(y, p + 5, inv)).sum(-1)
    assert torch.allclose(a, b, atol=1e-5)


def test_batched_kernels_match_the_loop():
    torch.manual_seed(0)
    layer = S4DLayer(8, d_state=4)
    u, w = torch.randn(4, 20, 8), torch.tensor([1.0, 2.0, 1.0, 0.5])
    a = layer(u, w)
    layer.batched_kernels = True
    assert torch.allclose(a, layer(u, w), atol=1e-5)


def test_gradient_reaches_all_parameters():
    m = small().train()
    x = torch.randn(4, 160)
    mask = torch.isfinite(x)
    q = m.forward_normalized(x, mask, 16, torch.tensor([1.0, 2.0, 0.5, 1.0]))
    q.sum().backward()
    assert all(p.grad is not None for p in m.parameters())


def test_block_mean_ends_at_the_last_point():
    x = np.arange(1.0, 8.0, dtype=np.float32)
    assert np.allclose(block_mean(x, 3), [3.0, 6.0])


def test_block_mean_with_missing():
    x = np.array([np.nan, 2.0, np.nan, np.nan], dtype=np.float32)
    out = block_mean(x, 2)
    assert out[0] == 2.0 and math.isnan(out[1])


@pytest.mark.parametrize("freq,cls", [("10S", "S"), ("5T", "5T"), ("15T", "15T"), ("30T", "15T"),
                                      ("6H", "H"), ("D", "D"), ("W-SUN", "W"), ("MS", "M"),
                                      ("Q-DEC", "Q"), ("A-DEC", "A")])
def test_rate_class(freq, cls):
    assert rate_class(freq) == cls


def test_scale_follows_the_rule():
    assert subset_scale("x", "15T") == 0.25 and subset_scale("x", "H") == 1.0
    assert subset_scale("m5", "D") == pytest.approx(24 / 7)
    assert subset_scale("subseasonal", "D") == pytest.approx(24 / 365)


def test_family_groups_slices():
    assert family("era5_1999") == family("era5_2000") == "era5"
