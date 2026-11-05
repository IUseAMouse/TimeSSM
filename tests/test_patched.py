import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from timessm.corpus import block_mean, family, rate_class, subset_scale
from timessm.patched import QUANTILES, QUANTILES_21, PatchedSSM, rotate
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


def test_robust_norm_ignores_a_spike():
    m = small(robust_norm=True)
    x = torch.randn(1, 160)
    spiky = x.clone()
    spiky[0, 80] = 1e6
    mask = torch.isfinite(x)
    _, _, a = m.normalize(x, mask)
    _, _, b = m.normalize(spiky, mask)
    assert b.item() < 2 * a.item()


def test_robust_norm_falls_back_on_std_when_mostly_zero():
    m = small(robust_norm=True)
    x = torch.zeros(1, 160)
    x[0, ::40] = 5.0
    _, _, scale = m.normalize(x, torch.isfinite(x))
    assert scale.item() > 0.1


def test_robust_norm_equivariance():
    m = small(robust_norm=True)
    x = torch.randn(1, 160)
    a = m.forecast(x, 16)["quantiles_denorm"]
    b = m.forecast(3 * x + 5, 16)["quantiles_denorm"]
    assert torch.allclose(b, 3 * a + 5, atol=2e-2, rtol=2e-2)


def test_stats_model_starts_as_the_plain_one():
    torch.manual_seed(0)
    plain = small()
    with_stats = small(stats=True)
    missing = with_stats.load_state_dict(plain.state_dict(), strict=False)
    assert all(k.startswith("stats.") for k in missing.missing_keys)
    x = torch.randn(2, 200)
    torch.testing.assert_close(plain.forecast(x, 16)["quantiles"],
                               with_stats.forecast(x, 16)["quantiles"])


def test_context_stats_values():
    m = small(stats=True)
    x = torch.full((1, 160), float("nan"))
    x[0, 80:] = torch.tensor([0.0, 2.0] * 40)
    mask = torch.isfinite(x)
    z, c, s = m.normalize(x, mask)
    st = m.context_stats(x, mask, c, s, z)[0]
    assert st[0].item() == pytest.approx(0.5)
    assert st[1].item() == pytest.approx(0.5)
    assert st[4].item() == pytest.approx(math.log(80) / math.log(256))


def test_stats_keep_scale_equivariance():
    torch.manual_seed(0)
    m = small(stats=True)
    torch.nn.init.normal_(m.stats[-1].weight, std=0.5)
    x = torch.randn(2, 160).abs()
    x[:, ::7] = 0.0
    a = m.forecast(x, 16)["quantiles_denorm"]
    b = m.forecast(3.0 * x, 16)["quantiles_denorm"]
    torch.testing.assert_close(3.0 * a, b, rtol=1e-3, atol=1e-3)


def test_single_size_model_is_unchanged():
    torch.manual_seed(0)
    plain = small()
    torch.manual_seed(0)
    same = small(patches=[16])
    assert plain.patches == same.patches == (16,) and same.input_length == 256
    x = torch.randn(2, 200)
    torch.testing.assert_close(plain.forecast(x, 20)["quantiles"], same.forecast(x, 20)["quantiles"])


def test_multi_size_reads_the_same_number_of_tokens():
    m = small(patches=[8, 32])
    assert m.patches == (8, 16, 32) and m.max_tokens == 16 and m.input_length == 512
    mask = torch.ones(4, 512, dtype=torch.bool)
    mask[0, :512 - 100] = False      # 100 steps: 8 x 16 tokens cover them
    mask[1, :512 - 200] = False      # 200 steps: 16 x 16
    mask[2, :512 - 300] = False      # 300 steps: 32 x 16
    mask[3] = False                  # nothing observed: nothing to cover
    assert m.choose_patch(mask).tolist() == [8, 16, 32, 8]


def test_mixed_sizes_in_a_batch_match_one_item_at_a_time():
    torch.manual_seed(0)
    m = small(patches=[8, 32])
    x = torch.randn(3, 512)
    x[0, :400] = float("nan")
    x[1, :300] = float("nan")
    out = m.forecast(x, 24)["quantiles_denorm"]
    for i in range(3):
        torch.testing.assert_close(out[i:i + 1], m.forecast(x[i:i + 1], 24)["quantiles_denorm"],
                                   rtol=1e-4, atol=1e-4)
    assert torch.isfinite(out).all() and out.shape == (3, 24, 9)


def test_every_size_keeps_scale_equivariance_and_its_gradients():
    torch.manual_seed(0)
    m = small(patches=[8, 32])
    x = torch.randn(2, 512).abs() + 1
    for p in m.patches:
        a = m.forecast(x, 16, patch=p)["quantiles_denorm"]
        b = m.forecast(5.0 * x, 16, patch=p)["quantiles_denorm"]
        torch.testing.assert_close(5.0 * a, b, rtol=1e-3, atol=1e-3)
    m.train()
    xp = m._pad(x, 8)
    mask = torch.isfinite(xp)
    z, _, _ = m.normalize(xp, mask)
    m.forward_normalized(z, mask, 16, 1.0, None, 8).sum().backward()
    assert m.embed_size["8"].weight.grad is not None and m.embed.weight.grad is None


def test_causal_frame_ends_on_the_global_frame():
    m = small(causal=True)
    x = torch.randn(3, 160) * 5 + 100
    x[0, :50] = float("nan")
    mask = torch.isfinite(x)
    z, c, s = m.normalize_causal(x, mask, 16)
    zg, cg, sg = PatchedSSM.normalize(small(), x, mask)
    torch.testing.assert_close(c[:, -1:], cg)
    torch.testing.assert_close(s[:, -1:], sg)
    torch.testing.assert_close(z[:, -16:], zg[:, -16:], rtol=1e-5, atol=1e-5)


def test_causal_context_outputs_ignore_the_future():
    torch.manual_seed(0)
    m = small(causal=True, stats=True)
    torch.nn.init.normal_(m.stats[-1].weight, std=0.5)
    x = torch.randn(2, 256)
    x[1, :40] = float("nan")
    cut = 7 * 16                      # tokens 0..6 must not see the steps from here on
    y = x.clone()
    y[:, cut:] = torch.randn(2, 256 - cut) * 50 + 30
    outs = []
    for v in (x, y):
        mask = torch.isfinite(v)
        z, c, s = m.normalize_causal(v, mask, 16)
        st = m.context_stats(v, mask, c[:, -1:], s[:, -1:], z)
        _, ctx = m.forward_normalized(z, mask, 16, 1.0, st, return_context=True)
        outs.append((z[:, :cut], c[:, :7], ctx[:, :7]))
    for a, b in zip(*outs):
        torch.testing.assert_close(a, b)


def test_next_patch_targets_use_the_frame_of_the_previous_token():
    sys_path = __import__("sys").path
    from pathlib import Path
    sys_path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_patched import next_patch_targets
    m = small(causal=True)
    x = torch.randn(2, 64) * 3 + 10
    mask = torch.isfinite(x)
    _, c, s = m.normalize_causal(x, mask, 16)
    t, keep = next_patch_targets(x, mask, c, s, 16, skip=1)
    assert t.shape == keep.shape == (2, 48)
    assert not keep[:, :16].any() and keep[:, 16:].all()      # token 0 skipped
    expect = torch.asinh((x[:, 32:48].double() - c[:, 1:2]) / s[:, 1:2]).float()
    torch.testing.assert_close(t[:, 16:32], expect)


def test_causal_model_without_context_loss_still_forecasts():
    m = small(causal=True, patches=[8, 32])
    x = torch.randn(3, 512)
    x[0, :450] = float("nan")
    out = m.forecast(x, 40)["quantiles_denorm"]
    assert out.shape == (3, 40, 9) and torch.isfinite(out).all()


def test_loops_share_weights_and_deepen():
    torch.manual_seed(0)
    one, two = small(), small(loops=2)
    two.load_state_dict(one.state_dict())
    assert sum(p.numel() for p in one.parameters()) == sum(p.numel() for p in two.parameters())
    x = torch.randn(2, 256)
    a, b = one.forecast(x, 32)["quantiles"], two.forecast(x, 32)["quantiles"]
    assert not torch.allclose(a, b)
    b.sum().backward()
    assert all(p.grad is not None for p in two.ssm.parameters() if p.requires_grad)


def test_delta_rule_chunks_match_the_recurrence():
    from timessm.delta import delta_rule
    torch.manual_seed(0)
    B, H, L, dk, dv = 2, 3, 37, 8, 5
    q, k = torch.randn(B, H, L, dk), F.normalize(torch.randn(B, H, L, dk), dim=-1)
    v, beta = torch.randn(B, H, L, dv), torch.rand(B, H, L)
    s, ref = torch.zeros(B, H, dk, dv), []
    for t in range(L):
        kt, vt = k[:, :, t], v[:, :, t]
        old = (s * kt[..., None]).sum(-2)
        s = s + beta[:, :, t, None, None] * kt[..., None] * (vt - old)[..., None, :]
        ref.append((s * q[:, :, t, :, None]).sum(-2))
    torch.testing.assert_close(delta_rule(q, k, v, beta, chunk=8), torch.stack(ref, 2),
                               atol=1e-5, rtol=1e-4)


def test_delta_mixers_forecast_and_stay_equivariant():
    m = small(mixers=["delta", "ssm"])
    x = torch.randn(3, 200)
    a = m.forecast(x, 40)["quantiles_denorm"]
    b = m.forecast(5 * x + 2, 40)["quantiles_denorm"]
    torch.testing.assert_close(b, 5 * a + 2, atol=1e-3, rtol=1e-3)
    with pytest.raises(ValueError):
        small(mixers=["delta"])


def test_selective_scan_chunks_match_the_recurrence():
    from timessm.selective import selective_scan
    torch.manual_seed(0)
    B, L, H, P, N = 2, 37, 3, 4, 5
    x, b, c = torch.randn(B, L, H, P), torch.randn(B, L, H, N), torch.randn(B, L, H, N)
    log_a = -torch.rand(B, L, H)
    h, ref = torch.zeros(B, H, P, N), []
    for t in range(L):
        h = log_a[:, t, :, None, None].exp() * h + x[:, t, :, :, None] * b[:, t, :, None, :]
        ref.append((h * c[:, t, :, None, :]).sum(-1))
    torch.testing.assert_close(selective_scan(x, log_a, b, c, chunk=8), torch.stack(ref, 1),
                               atol=1e-5, rtol=1e-4)


def test_selective_block_is_causal_and_keeps_the_step_invariance():
    from timessm.selective import SelectiveSSMBlock
    torch.manual_seed(0)
    blk = SelectiveSSMBlock(16, d_state=4, head_dim=8, chunk=8).double()
    x = torch.randn(2, 30, 16, dtype=torch.float64)
    y = blk(x)
    x2 = x.clone()
    x2[:, 20:] = torch.randn(2, 10, 16, dtype=torch.float64)
    torch.testing.assert_close(blk(x2)[:, :20], y[:, :20])
    # constant input: two steps at w equal one step at 2w (exact zero-order hold)
    const = torch.randn(2, 1, 16, dtype=torch.float64)
    fine = blk(const.expand(2, 20, 16), delta_scale=torch.tensor([1.0, 0.5]))
    coarse = blk(const.expand(2, 10, 16), delta_scale=torch.tensor([2.0, 1.0]))
    torch.testing.assert_close(fine[:, 1::2], coarse)


def test_selective_mixers_forecast_and_stay_equivariant():
    m = small(mixers=["selective", "ssm"])
    x = torch.randn(3, 200)
    a = m.forecast(x, 40)["quantiles_denorm"]
    b = m.forecast(5 * x + 2, 40)["quantiles_denorm"]
    torch.testing.assert_close(b, 5 * a + 2, atol=1e-3, rtol=1e-3)
    a.sum().backward()
    assert all(p.grad is not None for p in m.ssm[0].parameters())


def test_head_of_21_levels_returns_the_9():
    assert len(QUANTILES_21) == 21 and set(QUANTILES) <= set(QUANTILES_21)
    m = small(head_levels=QUANTILES_21, causal=True).eval()
    x = torch.randn(2, 160)
    out = m.forecast(x, 16)
    assert out["quantiles_denorm"].shape[-1] == 9 and out["quantile_levels"] == QUANTILES
    assert (out["quantiles_denorm"].diff(dim=-1) >= 0).all()
    z, c, s = m.normalize_causal(x, torch.isfinite(x), m.patch)
    full = m.forward_normalized(z, torch.isfinite(x), 16)
    assert full.shape[-1] == 21
    assert torch.allclose(full[..., m.out_index], out["quantiles"])
    with pytest.raises(ValueError):
        small(head_levels=(0.1, 0.5, 0.9))
