"""
Delta tied to the declared frequency, on the TimeSSM side (2026-10-05).

Off (default): the batch key is ignored, no random number is drawn, the loss
is the one of the code before (the golden file pins the absolute numbers).
On: w = 24 / season per item, clamped to the model's range; items without a
frequency keep the legacy draw in train and w = 1 in eval; a decimated window
has a shorter cycle; validation runs tied; no parameter is added, so a
checkpoint trained before loads in a model built with the mode on.
"""

import random
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "scripts"))

from timessm.model import SSMForecaster, build_from_config  # noqa: E402
from timessm.training import SSMFinetuneModule  # noqa: E402


def _small(**kw):
    torch.manual_seed(0)
    args = dict(input_length=128, prediction_length=32, d_model=16, n_layers=2,
                d_state=8, expand=2, dropout=0.0, quantile_hidden_dim=32)
    args.update(kw)
    return SSMForecaster(**args).eval()


def _series(B=4, L=128, seed=1):
    torch.manual_seed(seed)
    t = torch.arange(L).float()
    return 50.0 + 10.0 * torch.sin(2 * torch.pi * t / 24)[None, :] + torch.randn(B, L)


def _mod(model=None, **kw):
    args = dict(finetune_mode="full_finetune", loss_type="huber", learning_rate=1e-3,
                encoder_lr_multiplier=1.0, lr_scheduler="constant",
                delta_scales=[0.25, 0.5, 2.0], p_delta_scale=0.7)
    args.update(kw)
    return SSMFinetuneModule(model or _small(), **args)


def _batch(season=None):
    b = {"context": _series(), "target": _series(L=32, seed=2)}
    if season is not None:
        b["season"] = torch.tensor(season, dtype=torch.float32)
    return b


# ------------------------------------------------------------ off = as before
def test_off_ignores_the_key_and_draws_nothing():
    mod = _mod()
    mod.train()
    batch = _batch(season=[96.0, 24.0, 0.0, 7.0])
    random.seed(11)
    state = random.getstate()
    out = mod._tie_delta(batch)
    assert out is batch and "w" not in out and random.getstate() == state

    def losses(b):
        random.seed(5)
        torch.manual_seed(5)
        return torch.stack([mod._forward_and_loss(b["context"].unsqueeze(-1), b["target"].unsqueeze(-1))[0]
                            for _ in range(3)]).detach()
    assert torch.equal(losses(_batch()), losses(batch))


def test_no_parameter_is_added_and_an_old_checkpoint_loads():
    plain, tied = _small(), _small(expects_frequency=True, delta_range=(1 / 64, 8.0))
    assert list(plain.state_dict()) == list(tied.state_dict())
    tied.load_state_dict(plain.state_dict(), strict=True)
    x = _series().unsqueeze(-1)
    with torch.no_grad():
        assert torch.equal(plain.forecast(x, n=40)["quantiles"], tied.forecast(x, n=40)["quantiles"])
    assert plain.expects_frequency is False and plain.delta_range == (1 / 48, 4.0)
    with pytest.raises(ValueError, match="delta_range"):
        _small(delta_range=(2.0, 1.0))


# ------------------------------------------------------------ on
def test_on_sets_w_from_the_season_and_clamps():
    mod = _mod(delta_from_frequency=True)
    mod.eval()
    out = mod._tie_delta(_batch(season=[96.0, 24.0, 4.0, 8640.0]))
    assert torch.allclose(out["w"], torch.tensor([0.25, 1.0, 4.0, 1 / 48]))     # 6 and 1/360 clamped
    wide = _mod(_small(delta_range=(1 / 64, 8.0)), delta_from_frequency=True)
    wide.eval()
    assert torch.allclose(wide._tie_delta(_batch(season=[4.0, 7.0, 52.0, 12.0]))["w"],
                          torch.tensor([6.0, 24 / 7, 24 / 52, 2.0]))
    assert mod._freq_stats[0] == 1.0 and mod._freq_stats[3] == 4.0


def test_unknown_items_keep_the_legacy_draw_in_train_and_one_in_eval():
    mod = _mod(delta_from_frequency=True, p_delta_scale=1.0)
    batch = _batch(season=[96.0, 0.0, 24.0, 0.0])
    mod.train()
    random.seed(2)
    w = mod._tie_delta(batch)["w"]
    assert w[0] == 0.25 and w[2] == 1.0
    assert w[1] == w[3] and float(w[1]) in (0.25, 0.5, 2.0)
    assert mod._freq_stats[0] == 0.5
    mod.eval()
    w = mod._tie_delta(batch)["w"]
    assert torch.allclose(w, torch.tensor([0.25, 1.0, 1.0, 1.0]))
    # every item labelled: the legacy draw is never consulted
    mod.train()
    random.seed(2)
    state = random.getstate()
    mod._tie_delta(_batch(season=[96.0, 24.0, 7.0, 12.0]))
    assert random.getstate() == state


def test_a_decimated_window_has_a_shorter_cycle():
    mod = _mod(delta_from_frequency=True)
    mod.eval()
    assert torch.allclose(mod._tie_delta(_batch(season=[96.0] * 4), k=4)["w"], torch.ones(4))


def test_on_without_the_key_is_an_error():
    mod = _mod(delta_from_frequency=True)
    with pytest.raises(KeyError, match="frequency_table"):
        mod._tie_delta(_batch())


def test_training_and_validation_steps_run_tied(monkeypatch):
    mod = _mod(delta_from_frequency=True)
    monkeypatch.setattr(mod, "log", lambda *a, **k: None)
    monkeypatch.setattr(mod, "log_dict", lambda *a, **k: None, raising=False)
    batch = _batch(season=[96.0, 24.0, 7.0, 12.0])
    mod.train()
    loss = mod.training_step(batch, 0)
    assert torch.isfinite(loss)
    expected = torch.tensor([0.25, 1.0, 24 / 7, 2.0]).mean()
    assert mod._last_delta_scale == pytest.approx(float(expected), rel=1e-6)
    loss.backward()
    assert mod.model.blocks[0].ssm.log_dt.grad is not None
    mod.eval()
    with torch.no_grad():
        mod.validation_step(batch, 0)
    assert mod._last_delta_scale == pytest.approx(float(expected), rel=1e-6)      # tied in eval too
    assert "w" not in batch                                                       # the caller's dict is untouched


def test_tied_forward_equals_the_forecast_called_with_that_w():
    model = _small()
    mod = _mod(model, delta_from_frequency=True)
    mod.eval()
    batch = _batch(season=[96.0, 24.0, 7.0, 12.0])
    w = mod._tie_delta(batch)["w"]
    x = batch["context"].unsqueeze(-1)
    with torch.no_grad():
        _, res, _ = mod._forward_and_loss(x, batch["target"].unsqueeze(-1), w=w)
        direct = model.forecast(x, n=32, w=w)
    assert torch.equal(res["quantiles"], direct["quantiles"])


# ------------------------------------------------------------ configs
def _compose(name):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(version_base=None, config_dir=str((HERE / "configs").resolve())):
        return compose(config_name=name)


def test_configs_only_the_freq_arm_asks_for_the_mode():
    for name in ("ssm_mini_v3", "ssm_mini_v3_wide", "ssm_mini_v3_wide_eval", "ssm_mid_v3"):
        cfg = _compose(name)
        assert not cfg.model.ssm.get("delta_from_frequency", False)
        assert cfg.data.get("frequency_table") is None
    for name in ("ssm_mini_v3_freq", "ssm_mini_v3_freq_eval"):
        cfg = _compose(name)
        model = build_from_config(cfg)
        assert model.expects_frequency and model.delta_range == (0.015625, 8.0)
        assert cfg.data.frequency_table.endswith("corpus_v3_frequencies.yaml")
    for name in ("ssm_mini_v3_freq_syn", "ssm_mini_v3_freq_syn_eval"):
        cfg = _compose(name)
        assert build_from_config(cfg).expects_frequency
        assert cfg.data.frequency_table.endswith("corpus_v3_frequencies_syn.yaml")
        assert cfg.model.name == "timessm_mini_v3_freq_syn_zs"
    wide, freq = _compose("ssm_mini_v3_wide"), _compose("ssm_mini_v3_freq")
    for key in ("schedule_fraction", "p_delta_scale", "delta_scales"):
        assert wide.training[key] == freq.training[key]
    assert wide.data.batch_size == freq.data.batch_size
    assert list(build_from_config(wide).state_dict()) == list(build_from_config(freq).state_dict())


def test_train_script_relays_the_table_only_when_set():
    import train_ssm
    assert train_ssm._frequency_kwargs(_compose("ssm_mini_v3_wide")) == {}
    kw = train_ssm._frequency_kwargs(_compose("ssm_mini_v3_freq"))
    assert set(kw) == {"frequency_table"}


def test_release_config_carries_the_card_settings_and_the_others_carry_none():
    """timessm_2.5m_gift: one command reproduces the model card; every other
    config stays free of inference settings, so evaluating without them is
    still the default."""
    import json
    cfg = _compose("timessm_2.5m_gift")
    assert cfg.tta_flip is True and cfg.ratein == "mix" and cfg.ratein_pool is True
    assert cfg.ratein_k_up == "2x3x4" and cfg.ratein_min_bt == 4 and cfg.ratein_bt_windows == 4
    assert cfg.freq_delta is True and cfg.gift_batch_size == 32
    assert cfg.model.name == "timessm_2.5m"
    freq = _compose("ssm_mini_v3_freq_eval")
    assert cfg.model.ssm == freq.model.ssm and cfg.model.decoder == freq.model.decoder   # same architecture
    assert build_from_config(cfg).expects_frequency and "1.2814" in cfg.quantile_gamma
    gamma = json.loads((HERE / cfg.quantile_gamma).read_text())
    assert len(gamma["gamma"]) == len(gamma["levels"]) == 9 and gamma["gamma"][4] == 1.0   # the median is untouched
    keys = ("tta_flip", "ratein", "ratein_pool", "ratein_k_up", "freq_delta", "quantile_gamma")
    for name in ("ssm_mini_v3_eval", "ssm_mini_v3_wide_eval", "ssm_mini_v3_freq_eval", "ssm_mid_v3_eval"):
        other = _compose(name)
        assert all(other.get(k) is None for k in keys), name
