"""
SSMForecaster contract (2026-09-09).

5. forecast: the JEPATST keys, shapes [B, n, Q] for n in {8, 128, 256, 900}
   (free horizon, no rollout), quantiles increasing along Q, denormalized fan
   = exact inverse of the normalized one; FinetuneModule on the model:
   finite loss, backward reaches the blocks and the head, optimizer groups
   'encoder' / 'decoder'; size in [2M, 4M] at the spike's dimensions.
6. w changes the fan, w = 1 is the identity, per-item w == per-item loop;
   selective_readout composes; the multi-rate module draws w in train only
   and logs its witnesses.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timessm.model import SSMForecaster  # noqa: E402
from timessm.training import SSMFinetuneModule  # noqa: E402
from timejepa.training.finetune_module import FinetuneModule  # noqa: E402

KEYS = ("forecast", "forecast_denorm", "quantiles", "quantiles_denorm", "quantile_levels")


def _small(**kw):
    torch.manual_seed(0)
    args = dict(input_length=128, prediction_length=32, d_model=16, n_layers=2,
                d_state=8, expand=2, dropout=0.0, quantile_hidden_dim=32)
    args.update(kw)
    return SSMForecaster(**args).eval()


def _ctx(B=3, L=128, seed=1):
    torch.manual_seed(seed)
    t = torch.arange(L).float()
    base = torch.sin(2 * torch.pi * t / 24)[None, :, None]
    return 50.0 + 10.0 * base + torch.randn(B, L, 1) + 5.0 * torch.arange(B)[:, None, None]


# ------------------------------------------------------------- 5. contract
@pytest.mark.parametrize("n", [8, 128, 256, 900])
def test_forecast_contract_any_horizon(n):
    m = _small()
    x = _ctx()
    with torch.no_grad():
        out = m.forecast(x, n=n, return_representations=True)
    for k in KEYS:
        assert k in out
    Q = len(out["quantile_levels"])
    assert out["quantiles"].shape == (3, n, Q)
    assert out["quantiles_denorm"].shape == (3, n, Q)
    assert out["forecast"].shape == (3, n, 1)
    assert out["forecast_denorm"].shape == (3, n, 1)
    assert (out["quantiles"].diff(dim=-1) >= 0).all()
    assert out["context_embeddings"].shape == (3, 128, 16)
    assert out["future_representations"].shape == (3, n, 16)
    assert out["context_norm"].shape == x.shape
    # denormalization is the exact inverse of the normalization chain
    q = out["quantiles"]
    back = m.robust_scaler.inverse(m.revin.denormalize_target_space(q))
    assert torch.allclose(back, out["quantiles_denorm"], atol=1e-5, rtol=1e-5)
    assert torch.isfinite(out["quantiles_denorm"]).all()
    # the denormalized fan lives near the context's level
    assert (out["forecast_denorm"].mean() - x.mean()).abs() < 3 * x.std()


def test_finetune_module_trains_the_model():
    m = _small()
    mod = SSMFinetuneModule(
        m, finetune_mode="full_finetune", loss_type="huber", learning_rate=1e-3,
        encoder_lr_multiplier=1.0, lr_scheduler="constant",
        delta_scales=[0.5, 2.0], p_delta_scale=1.0,
    )
    mod.train()
    x = _ctx(B=4)
    y = _ctx(B=4, L=32, seed=2)
    loss, results, target = mod._forward_and_loss(x, y)
    assert torch.isfinite(loss)
    assert mod._last_delta_scale in (0.5, 2.0)          # drawn in train
    loss.backward()
    assert m.blocks[0].ssm.log_dt.grad is not None
    assert m.blocks[0].ssm.C_re.grad is not None
    assert m.decoder.decoder.mlp[0].weight.grad is not None
    assert m.future_token.grad is not None
    assert m.patching.projection.weight.grad is not None
    opt = mod.configure_optimizers()
    names = [g["name"] for g in opt.param_groups]
    assert names == ["encoder", "decoder"]
    n_dec = sum(p.numel() for g in opt.param_groups if g["name"] == "decoder" for p in g["params"])
    assert n_dec == sum(p.numel() for p in m.decoder.parameters())
    mod.eval()
    loss_eval, _, _ = mod._forward_and_loss(x, y)
    assert mod._last_delta_scale == 1.0                  # never drawn in eval


def test_spike_size_between_2m_and_4m():
    m = SSMForecaster(d_model=192, n_layers=6, d_state=32, expand=2,
                      quantile_hidden_dim=1536)
    n = m.count_parameters()
    assert 2_000_000 <= n <= 4_000_000, n
    parts = m.get_num_params()
    assert parts["decoder"] < 0.4 * parts["total"]


def test_linear_probe_and_pretrain_refusals():
    m = _small()
    with pytest.raises(ValueError):
        m.set_pretrain_mode(True)
    with pytest.raises(NotImplementedError):
        m.forward_pretrain(None, None)
    m.freeze_encoder(); m.freeze_patching()
    assert not any(p.requires_grad for p in m.blocks.parameters())
    assert all(p.requires_grad for p in m.decoder.parameters())
    m.unfreeze_encoder(); m.unfreeze_patching()
    assert all(p.requires_grad for p in m.blocks.parameters())
    assert m.predictor.w_film is None and m.rate_knob == "delta"


# ------------------------------------------------------------- 6. the knob
def test_w_changes_the_fan_and_w1_is_identity():
    m = _small()
    x = _ctx()
    with torch.no_grad():
        base = m.forecast(x, n=32)["quantiles"]
        one = m.forecast(x, n=32, w=1.0)["quantiles"]
        vec1 = m.forecast(x, n=32, w=torch.ones(3))["quantiles"]
        half = m.forecast(x, n=32, w=0.5)["quantiles"]
        mixed = m.forecast(x, n=32, w=torch.tensor([0.5, 1.0, 0.25]))["quantiles"]
        q_third = m.forecast(x[2:3], n=32, w=0.25)["quantiles"]
    assert torch.allclose(base, one) and torch.allclose(base, vec1)
    assert not torch.allclose(base, half, atol=1e-4)
    assert torch.allclose(mixed[0:1], half[0:1], atol=1e-5, rtol=1e-5)
    assert torch.allclose(mixed[1:2], base[1:2], atol=1e-5, rtol=1e-5)
    assert torch.allclose(mixed[2:3], q_third, atol=1e-5, rtol=1e-5)


def test_selective_readout_and_conv_compose():
    for kw in (dict(selective_readout=True), dict(d_conv=4), dict(real=True)):
        m = _small(**kw)
        with torch.no_grad():
            out = m.forecast(_ctx(), n=16)
        assert out["quantiles"].shape == (3, 16, 9)
        assert torch.isfinite(out["quantiles_denorm"]).all()
    assert _small(selective_readout=True).blocks[0].readout_gate is not None
    assert _small().blocks[0].readout_gate is None and _small().blocks[0].conv is None


def test_multirate_module_refuses_bad_config():
    m = _small()
    with pytest.raises(ValueError):
        SSMFinetuneModule(m, finetune_mode="full_finetune", p_delta_scale=0.5)
    with pytest.raises(ValueError):
        SSMFinetuneModule(m, finetune_mode="full_finetune", delta_scales=[0.0, 1.0],
                          p_delta_scale=0.5)


# ------------------------------------------------------------- 7. scaling knobs
def test_activation_checkpointing_same_numbers_train_only():
    """Recomputing the blocks in backward must change nothing: same forward,
    same gradients (float64), and it is inert in eval."""
    m = _small(activation_checkpointing=True).double()
    ref = _small().double()
    ref.load_state_dict(m.state_dict())
    x = _ctx(B=2).double()
    m.train(); ref.train()
    # the quantile head keeps its own dropout in train mode: same seed, same masks
    torch.manual_seed(11); a = m.forecast(x, n=16)["quantiles"]
    torch.manual_seed(11); b = ref.forecast(x, n=16)["quantiles"]
    assert torch.allclose(a, b, atol=1e-10)
    a.sum().backward(); b.sum().backward()
    n_checked = 0
    for (n1, p1), (n2, p2) in zip(m.named_parameters(), ref.named_parameters()):
        assert n1 == n2
        if p1.grad is None or p2.grad is None:          # Patching.value_embedding is unused
            assert p1.grad is None and p2.grad is None, n1
            continue
        assert torch.allclose(p1.grad, p2.grad, atol=1e-8), n1
        n_checked += 1
    assert n_checked > 10
    m.eval(); ref.eval()
    with torch.no_grad():
        assert torch.allclose(m.forecast(x, n=16)["quantiles"],
                              ref.forecast(x, n=16)["quantiles"], atol=1e-10)


def test_strategy_builder_fsdp():
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "scripts"))
    from omegaconf import OmegaConf
    from train_ssm import build_strategy
    from pytorch_lightning.strategies import FSDPStrategy
    from timessm.block import GatedSSMBlock
    plain = OmegaConf.create({"trainer": {"strategy": "ddp"}, "model": {"ssm": {}}})
    assert build_strategy(plain) == "ddp"
    cfg = OmegaConf.create({"trainer": {"strategy": "fsdp"},
                            "model": {"ssm": {"activation_checkpointing": True}}})
    st = build_strategy(cfg)
    assert isinstance(st, FSDPStrategy)
    assert st.kwargs["auto_wrap_policy"]._module_classes == {GatedSSMBlock}
    assert str(st.sharding_strategy).endswith("FULL_SHARD")
    assert st._activation_checkpointing_kwargs["auto_wrap_policy"]._module_classes == {GatedSSMBlock}


def test_resume_replaces_stale_revin_batch_statistics():
    """A checkpoint saved after a forward carries RevIN's [B, 1, 1] statistics;
    the resume hook restores the fresh shapes so a strict load succeeds and
    every other tensor is loaded unchanged."""
    m = _small()
    mod = SSMFinetuneModule(m, finetune_mode="full_finetune", lr_scheduler="constant")
    with torch.no_grad():
        m.forecast(_ctx(B=5), n=8)                      # revin.mean is now [5, 1, 1]
    assert m.revin.mean.shape == (5, 1, 1)
    ckpt = {"state_dict": {k: v.clone() for k, v in mod.state_dict().items()}}
    fresh = SSMFinetuneModule(_small(), finetune_mode="full_finetune", lr_scheduler="constant")
    with torch.no_grad():
        fresh.model.blocks[0].ssm.log_dt.add_(1.0)      # differs before the load
    with pytest.raises(RuntimeError):
        fresh.load_state_dict(ckpt["state_dict"], strict=True)
    fresh.on_load_checkpoint(ckpt)
    fresh.load_state_dict(ckpt["state_dict"], strict=True)
    assert fresh.model.revin.mean.shape == (1,)
    assert torch.equal(fresh.model.blocks[0].ssm.log_dt, m.blocks[0].ssm.log_dt)


def test_check_schedule_refuses_a_warmup_longer_than_the_run():
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "scripts"))
    from omegaconf import OmegaConf
    from train_ssm import check_schedule
    from hydra import initialize_config_dir, compose
    import os
    cfg_dir = os.path.abspath(str(_P(__file__).resolve().parents[1] / "configs"))
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        for name in ("ssm_mini_v3", "ssm_mini_v3_wide"):
            check_schedule(compose(config_name=name))          # every shipped config is sane
        with pytest.raises(ValueError, match="audit_batch_sizes"):   # fraction left MISSING
            check_schedule(compose(config_name="ssm_mid_v3"))
        check_schedule(compose(config_name="ssm_mid_v3", overrides=["training.schedule_fraction=0.05"]))
        bad = compose(config_name="ssm_mid_v3", overrides=["training.lr_scheduler.warmup_epochs=0.1",
                                                            "training.schedule_fraction=0.06667"])
    with pytest.raises(ValueError, match="warmup_epochs"):
        check_schedule(bad)
    check_schedule(OmegaConf.create({"training": {"lr_scheduler": {"type": "constant"}}}))


# ------------------------------------------------------------- 8. random horizon (B1)
def _batch(B=4, L=128, P=32, mask=None):
    b = {"context": _ctx(B=B, L=L, seed=3), "target": _ctx(B=B, L=P, seed=4)}
    if mask is not None:
        b["target_mask"] = mask
    return b


def _mod(m, **kw):
    args = dict(finetune_mode="full_finetune", loss_type="huber", learning_rate=1e-3,
                encoder_lr_multiplier=1.0, lr_scheduler="constant")
    args.update(kw)
    return SSMFinetuneModule(m, **args)


def test_resplit_conserves_the_window_and_never_runs_in_eval():
    mod = _mod(_small(input_length=128, prediction_length=32),
               horizon_lengths=[16, 48], p_random_horizon=1.0, horizon_min_context=64)
    b = _batch()
    full = torch.cat([b["context"], b["target"]], 1)
    mod.train()
    out = mod._maybe_resplit_horizon(b)
    h = out["target"].shape[1]
    assert h in (16, 48) and out["context"].shape[1] == 160 - h
    assert torch.equal(torch.cat([out["context"], out["target"]], 1), full)
    assert mod._last_horizon == h
    assert b["target"].shape[1] == 32                       # caller's batch untouched
    mod.eval()
    out = mod._maybe_resplit_horizon(b)
    assert out["target"].shape[1] == 32 and mod._last_horizon == 32


def test_forward_and_loss_matches_the_parent_at_native_horizon():
    m = _small(input_length=128, prediction_length=32)
    mod = _mod(m)
    mod.eval()
    x, y = _ctx(B=3), _ctx(B=3, L=32, seed=2)
    torch.manual_seed(5); loss_a, res_a, tgt_a = mod._forward_and_loss(x, y)
    torch.manual_seed(5); loss_b, res_b, tgt_b = FinetuneModule._forward_and_loss(mod, x, y)
    assert torch.allclose(loss_a, loss_b) and torch.equal(tgt_a, tgt_b)
    assert torch.allclose(res_a["quantiles"], res_b["quantiles"])


def test_random_horizon_training_step_trains_the_head_and_the_future_token():
    m = _small(input_length=128, prediction_length=32)
    mod = _mod(m, horizon_lengths=[16, 48], p_random_horizon=1.0, horizon_min_context=64)
    mod.log = lambda *a, **k: None
    mod.train()
    loss = mod.training_step(_batch(), 0)
    assert torch.isfinite(loss) and mod._last_horizon in (16, 48)
    loss.backward()
    assert m.future_token.grad is not None and m.decoder.decoder.mlp[0].weight.grad is not None


def test_resplit_propagates_the_target_mask():
    m = _small(input_length=128, prediction_length=32)
    mod = _mod(m, horizon_lengths=[48], p_random_horizon=1.0, horizon_min_context=64)
    mod.train()
    mask = torch.ones(4, 32, dtype=torch.bool); mask[:, -4:] = False
    out = mod._maybe_resplit_horizon(_batch(mask=mask))
    assert out["target_mask"].shape == (4, 48) and out["target_mask"][:, :16].all()
    assert not out["target_mask"][:, -4:].any()
    # h < native with padding at the FRONT of the target: refused (padding would enter the context)
    mod2 = _mod(_small(input_length=128, prediction_length=32),
                horizon_lengths=[16], p_random_horizon=1.0, horizon_min_context=64)
    mod2.train()
    front = torch.ones(4, 32, dtype=torch.bool); front[:, :8] = False
    out2 = mod2._maybe_resplit_horizon(_batch(mask=front))
    assert out2["target"].shape[1] == 32
    # h < native with a clean front: accepted, mask sliced
    clean = torch.ones(4, 32, dtype=torch.bool); clean[:, -2:] = False
    out3 = mod2._maybe_resplit_horizon(_batch(mask=clean))
    assert out3["target_mask"].shape == (4, 16) and not out3["target_mask"][:, -2:].any()


def test_context_crop_applies_after_the_resplit():
    m = _small(input_length=128, prediction_length=32)
    mod = _mod(m, horizon_lengths=[48], p_random_horizon=1.0, horizon_min_context=64,
               context_lengths=[64, 96, 128], p_random_context_finetune=1.0)
    seen = []
    mod.log = lambda name, value, *a, **k: seen.append((name, float(value))) if name == "geometry/context_len" else None
    mod.train()
    for _ in range(6):
        mod.training_step(_batch(), 0)
    lens = {v for n, v in seen}
    assert lens and all(v <= 160 - 48 for v in lens) and 128.0 not in lens


def test_both_draws_active():
    m = _small(input_length=128, prediction_length=32)
    mod = _mod(m, delta_scales=[0.5, 2.0], p_delta_scale=1.0,
               horizon_lengths=[16, 48], p_random_horizon=1.0, horizon_min_context=64)
    mod.log = lambda *a, **k: None
    mod.train()
    loss = mod.training_step(_batch(), 0)
    assert torch.isfinite(loss) and mod._last_delta_scale in (0.5, 2.0) and mod._last_horizon in (16, 48)
