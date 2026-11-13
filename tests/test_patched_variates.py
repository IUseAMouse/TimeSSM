"""
PatchedSSM with variate layers: a one-variate item is the univariate model, variates are
exchangeable, padding variates are never read, the model stays causal across variates,
future covariates are read on the horizon only, and the release size stays under 10M.
"""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from timessm.patched import (FUTURE_COVARIATE, PAST_COVARIATE, TARGET, PatchedSSM,
                             build_from_config)

D160K = Path.home() / "archive" / "timessm_ckpts" / "timessm_10m_d160k.ckpt"


def small(**kw):
    torch.manual_seed(0)
    kw = {"causal": True, "loops": 2, "variate_layers": (1,), "variate_qk": 16, **kw}
    model = PatchedSSM(input_length=256, patch=16, d_model=32, n_layers=2, d_state=4,
                       n_heads=4, **kw).eval()
    # Role vectors start apart, as after training, so that a role swap shows.
    torch.nn.init.normal_(model.role_embed.weight) if model.role_embed is not None else None
    return model


def items(B=3, V=3, L=256, seed=1):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(L).float()
    x = (20.0 + 5.0 * torch.sin(2 * torch.pi * t / 24) + torch.randn(B, V, L, generator=g)
         + 3.0 * torch.arange(V)[None, :, None])
    x[0, -1, :100] = float("nan")
    return x


def context_outputs(model, x, roles=None, future=None, n=32):
    xr, mask, z, _, _, _, _, w, layout, fut = model.prepare_variates(x, None, roles, future)
    q, qc = model.forward_normalized(z, mask, n, w, layout=layout, future=fut,
                                     return_context=True)
    B, V = x.shape[:2]
    return q.reshape(B, V, *q.shape[1:]), qc.reshape(B, V, *qc.shape[1:])


def test_off_by_default_no_module_no_key():
    model = PatchedSSM(input_length=256, patch=16, d_model=32, n_layers=2, d_state=4, n_heads=4)
    assert not any(k.startswith(("var_att", "role_embed")) for k in model.state_dict())
    assert model.role_embed is None and "var_att." not in model.core_prefixes
    with pytest.raises(ValueError, match="variate_layers"):
        model.forecast_multivariate(items())


def test_refusals():
    with pytest.raises(ValueError, match="not in"):
        small(variate_layers=(2,))
    with pytest.raises(ValueError, match="halting"):
        small(halt=True)
    with pytest.raises(ValueError, match="halting"):
        small(patches=(32,))
    with pytest.raises(ValueError, match="divide"):
        small(variate_qk=10)


@torch.no_grad()
def test_one_variate_item_is_the_univariate_forecast():
    model, x = small(), items()[:, 0]
    w = torch.tensor([0.5, 1.0, 2.0])
    for kw in ({}, {"w": w}):
        uni = model.forecast(x, n=40, **kw)
        multi = model.forecast_multivariate(x[:, None], n=40, **kw)
        for key in ("quantiles", "quantiles_denorm", "forecast_denorm"):
            assert torch.equal(uni[key], multi[key][:, 0]), key


@torch.no_grad()
def test_variates_are_exchangeable():
    model, x = small(), items()
    roles = torch.tensor([TARGET, PAST_COVARIATE, TARGET])
    perm = torch.tensor([2, 0, 1])
    a = model.forecast_multivariate(x, n=40, roles=roles)["quantiles_denorm"]
    b = model.forecast_multivariate(x[:, perm], n=40, roles=roles[perm])["quantiles_denorm"]
    torch.testing.assert_close(a[:, perm], b, rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_padding_variates_are_never_read():
    model, x = small(), items()
    pad = torch.randn(3, 2, 256) * 100
    valid = torch.tensor([[True] * 3 + [False] * 2] * 3)
    a = model.forecast_multivariate(x, n=40)["quantiles_denorm"]
    b = model.forecast_multivariate(torch.cat([x, pad], 1), n=40, valid=valid)["quantiles_denorm"]
    torch.testing.assert_close(a, b[:, :3], rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_variates_mixing_is_used():
    model, x = small(), items()
    a = model.forecast_multivariate(x, n=40)["quantiles_denorm"][:, 0]
    y = x.clone()
    y[:, 2] = y[:, 2] * 3 - 7
    b = model.forecast_multivariate(y, n=40)["quantiles_denorm"][:, 0]
    assert not torch.allclose(a, b)


@torch.no_grad()
def test_causal_across_variates():
    """Changing any variate from token 7 on leaves the context outputs of tokens < 7 of every
    variate unchanged (up to the FFT round-off of the S4D convolution, as in test_patched)."""
    model, x = small(), items()
    _, qc = context_outputs(model, x)
    y = x.clone()
    y[:, 2, 7 * 16:] = y[:, 2, 7 * 16:] * 5 + 50
    _, qc2 = context_outputs(model, y)
    torch.testing.assert_close(qc[:, :, :7], qc2[:, :, :7])
    assert not torch.allclose(qc[:, 0, 8:], qc2[:, 0, 8:])


@torch.no_grad()
def test_future_covariates_are_read_on_the_horizon_only():
    model, x = small(), items()
    n = 32
    roles = torch.tensor([TARGET, FUTURE_COVARIATE, PAST_COVARIATE])
    future = torch.full((3, 3, n), float("nan"))
    future[:, 1] = torch.linspace(0, 40, n)
    q0, qc0 = context_outputs(model, x, roles, None, n)
    q1, qc1 = context_outputs(model, x, roles, future, n)
    torch.testing.assert_close(qc0, qc1)                          # context outputs untouched
    assert not torch.allclose(q0[:, 0], q1[:, 0])                 # the target reads them
    # Future values on a past covariate or a target row are ignored.
    leak = future.clone()
    leak[:, 0] = 1e3
    leak[:, 2] = -1e3
    q2, _ = context_outputs(model, x, roles, leak, n)
    assert torch.equal(q1, q2)


def test_future_covariate_gradient_reaches_the_target_horizon_only():
    model, x = small(), items()
    n = 32
    roles = torch.tensor([TARGET, FUTURE_COVARIATE, TARGET])
    future = torch.full((3, 3, n), float("nan"))
    vals = torch.linspace(0, 40, n).repeat(3, 1).requires_grad_(True)
    future = torch.cat([future[:, :1], vals[:, None], future[:, 2:]], 1)
    q, qc = context_outputs(model, x, roles, future, n)
    (g_q,) = torch.autograd.grad(q[:, 0].sum(), vals, retain_graph=True)
    assert g_q.abs().sum() > 0
    (g_c,) = torch.autograd.grad(qc.sum(), vals, allow_unused=True)
    assert g_c is None or g_c.abs().max() < 1e-6 * g_q.abs().max()   # FFT round-off only


def test_roles_change_the_forecast():
    model, x = small(), items()
    with torch.no_grad():
        a = model.forecast_multivariate(x, n=40)["quantiles_denorm"]
        b = model.forecast_multivariate(x, n=40, roles=torch.tensor([0, 1, 1]))["quantiles_denorm"]
    assert not torch.allclose(a[:, 0], b[:, 0])


def release_config(**model):
    return OmegaConf.create({"model": {
        "name": "timessm_10m_mv", "seq_length": 4096, "prediction_length": 256, "patch": 16,
        "d_model": 336, "n_layers": 4, "d_state": 32, "expand": 2, "n_heads": 8,
        "attention": True, "dropout": 0.0, "causal": True, "loops": 2,
        "variate_layers": [1, 2], "variate_qk": 168, **model}})


def test_release_size_is_under_ten_million():
    model = build_from_config(release_config())
    n = sum(p.numel() for p in model.parameters())
    assert n == 9_807_312 and n < 10_000_000
    assert set(model.variate_layers) == {1, 2} and "role_embed." in model.core_prefixes
    univariate = build_from_config(release_config(d_model=352, variate_layers=None,
                                                  variate_qk=None))
    assert sum(p.numel() for p in univariate.parameters()) == 9_982_512


@pytest.mark.skipif(not D160K.exists(), reason="the d160k checkpoint is not on this machine")
def test_the_univariate_release_checkpoint_loads_strictly():
    model = build_from_config(release_config(d_model=352, variate_layers=None, variate_qk=None))
    state = torch.load(D160K, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({k[len("model."):]: v for k, v in state.items()})
    with torch.no_grad():
        out = model.forecast(items(V=1, L=512)[:, 0], n=48)
    assert torch.isfinite(out["quantiles_denorm"]).all()


def test_from_pretrained_builds_the_patched_model(tmp_path):
    from timessm import from_pretrained
    from timessm.hub import export
    cfg = OmegaConf.create({"model": {
        "name": "tiny", "builder": "timessm.patched:build_from_config", "seq_length": 256,
        "prediction_length": 64, "patch": 16, "d_model": 32, "n_layers": 2, "d_state": 4,
        "expand": 2, "n_heads": 4, "attention": True, "dropout": 0.0, "causal": True, "loops": 2}})
    torch.manual_seed(0)
    model = build_from_config(cfg).eval()
    back = from_pretrained(str(export(model, cfg.model, tmp_path)))
    assert isinstance(back, PatchedSSM)
    x = items(V=1)[:, 0]
    with torch.no_grad():
        assert torch.equal(model.forecast(x, n=40)["quantiles_denorm"],
                           back.forecast(x, n=40)["quantiles_denorm"])
    (tmp_path / "config.json").write_text('{"builder": "os:system"}')
    with pytest.raises(ValueError, match="outside"):
        from_pretrained(str(tmp_path))
