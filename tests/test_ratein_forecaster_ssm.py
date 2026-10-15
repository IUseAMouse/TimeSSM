"""
RateINForecaster around a frozen SSMForecaster, through the TimeSSM stack.

1. Prior k = 1 on a context of model_len points: the wrapper's fans equal the
   inner's bit for bit, with and without w.
2. SSMFinetuneModule with the frequency-tied Delta trains the wrapper: one
   training_step gives a finite loss, gradients on the selector only, the
   inner's w equals the tied w (spied), freq/* and rate/* are logged.
3. The e2e config composes with Hydra; create_model_from_config returns the
   wrapper; a Lightning-format checkpoint of it round-trips through
   load_checkpoint and a changed inner geometry is refused; build_module and
   build_strategy accept the config.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "scripts"))

from timessm.model import SSMForecaster  # noqa: E402
from timessm.training import SSMFinetuneModule  # noqa: E402
from timejepa import ratein_layer as RL  # noqa: E402
from timejepa.models.ratein_forecaster import RateINForecaster  # noqa: E402


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


def _wrap(inner=None, model_len=128, k_prior=1.0):
    inner = inner or _small()
    layer = RL.RateINLayer(RL.RateSelector(8, 4), model_len=model_len, min_blocks=8)
    return RateINForecaster(inner, layer, log_k_prior=math.log(k_prior), name="ratein_test")


# ------------------------------------------------------------- 1. identity
def test_prior_one_equals_the_inner_bit_for_bit():
    w = _wrap()
    x = _series().unsqueeze(-1)
    with torch.no_grad():
        for ww in (None, torch.tensor([0.25, 1.0, 4.0, 0.5])):
            ref = w.inner.forecast(x, n=32, w=ww)
            out = w.forecast(x, n=32, w=ww)
            for key in ("quantiles", "quantiles_denorm", "forecast", "forecast_denorm"):
                assert torch.equal(out[key], ref[key]), key
    assert w.rate_knob == "delta" and w.delta_range == w.inner.delta_range
    assert w.core_prefixes == ("inner.", "layer.")


# ---------------------------------------------------------- 2. the module
def test_module_trains_the_selector_with_the_tied_delta():
    w = _wrap()
    with torch.no_grad():
        w.layer.selector.mlp[-1].weight.fill_(0.05)
    mod = SSMFinetuneModule(w, delta_from_frequency=True, finetune_mode="full_finetune",
                            loss_type="huber", learning_rate=1e-3, lr_scheduler="constant",
                            context_lengths=[64, 96, 128], p_random_context_finetune=0.0)
    logged = {}
    mod.log = lambda name, value, *a, **k: logged.__setitem__(name, float(torch.as_tensor(value).detach()))
    seen = []
    inner_forecast = w.inner.forecast

    def spy(context, n=None, w=None, **kw):
        seen.append(None if w is None else w.clone())
        return inner_forecast(context, n=n, w=w, **kw)
    w.inner.forecast = spy
    mod.train()
    batch = {"context": _series(), "target": _series(L=32, seed=2),
             "season": torch.tensor([96.0, 24.0, 0.0, 7.0])}
    loss = mod.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in w.layer.selector.parameters())
    assert all(p.grad is None for p in w.inner.parameters())
    assert not w.inner.training and w.training
    tied = mod._tie_delta(batch)["w"]
    kq = torch.exp(torch.round(torch.log(w.last_k) * 8 / math.log(2)) * math.log(2) / 8)
    expect = (tied * kq).clamp(*w.delta_range)
    torch.testing.assert_close(seen[-1], expect)
    assert "freq/labelled_frac" in logged and "rate/k_median" in logged and "rate/n_unique_w" in logged


# ---------------------------------------------------- 3. config round trip
def _compose(overrides):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(HERE / "configs"), version_base=None):
        return compose(config_name="ratein_layer_e2e", overrides=overrides)


TINY = ["model.ssm.d_model=16", "model.ssm.n_layers=2", "model.ssm.d_state=8",
        "model.decoder.quantile_hidden_dim=32", "model.ratein_layer.model_len=128",
        "model.ratein_layer.n_modes=8", "model.ratein_layer.hidden=4", "model.ratein_layer.min_blocks=8"]


def test_config_builds_the_wrapper_and_round_trips(tmp_path):
    from timejepa.evaluation.loading import create_model_from_config, load_checkpoint
    import train_ssm
    cfg = _compose(TINY)
    assert cfg.model.builder.endswith("ratein_forecaster:build_from_config")
    assert cfg.model.ssm.delta_from_frequency and cfg.training.p_decimation == 0.0
    w = create_model_from_config(cfg)
    assert isinstance(w, RateINForecaster) and w.name == "timessm_mini_v3_freq_ratein"
    assert w.inner.input_length == 128 and w.input_length == 128 * RL.K_MAX
    assert w.expects_frequency and w.prediction_length == 256
    mod = train_ssm.build_module(cfg, w)
    assert isinstance(mod, SSMFinetuneModule) and mod.delta_from_frequency
    train_ssm.build_strategy(cfg)
    # Lightning-format checkpoint of the wrapper -> load_checkpoint
    with torch.no_grad():
        w.layer.selector.mlp[-1].weight.fill_(0.02)
    torch.save({"state_dict": {"model." + k: v for k, v in w.state_dict().items()}}, tmp_path / "w.ckpt")
    w2 = load_checkpoint(create_model_from_config(cfg), str(tmp_path / "w.ckpt"), torch.device("cpu"))
    x = _series(L=400).unsqueeze(-1)
    with torch.no_grad():
        torch.testing.assert_close(w2.forecast(x, n=32)["quantiles"], w.forecast(x, n=32)["quantiles"])
    other = create_model_from_config(_compose(TINY + ["model.ssm.d_state=4"]))
    with pytest.raises(RuntimeError, match="core components"):
        load_checkpoint(other, str(tmp_path / "w.ckpt"), torch.device("cpu"))
