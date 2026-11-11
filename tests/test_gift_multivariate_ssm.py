"""
A PatchedSSM with variate layers through the GIFT harness with +multivariate: items are
forecast whole (plain and with the RateIN stack), a univariate config runs as without the
flag, and the univariate model is refused.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "TimeJEPA" / "src"))
sys.path.insert(0, str(ROOT.parent / "TimeJEPA" / "scripts"))

EG = pytest.importorskip("evaluate_gift")

from timessm.patched import PatchedSSM  # noqa: E402

STACK = dict(ratein_mode="mix", ratein_pool=True, ratein_k_up=[2, 3, 4], ratein_min_bt=4,
             ratein_bt_windows=4)


def model(**kw):
    torch.manual_seed(0)
    return PatchedSSM(input_length=256, patch=16, d_model=32, n_layers=2, d_state=4,
                      n_heads=4, causal=True, loops=2, **kw).eval()


def items(n_items=3, V=3, length=700):
    rng = np.random.default_rng(0)
    t = np.arange(length)
    return [(rng.uniform(5, 50) + rng.uniform(1, 5) * np.sin(2 * np.pi * t / 24 + c)
             + rng.normal(scale=0.3, size=length)).astype(np.float32)
            for _ in range(n_items) for c in range(V)]


def run(m, series, V, **kw):
    data = EG.gift.ConfigData(series, h=48, m=24, windows=2, n_variates=V)
    return EG.evaluate_config(m, "mv/H/short", Path("."), torch.device("cpu"), batch_size=8,
                              data=data, freq_season=24, **kw)


@pytest.mark.parametrize("kw", [{}, STACK], ids=["plain", "stack"])
def test_items_forecast_whole(kw):
    res = run(model(variate_layers=(1,), variate_qk=16), items(), 3, multivariate=True, **kw)
    assert math.isfinite(res["model"]["CRPS"]) and res["n_instances"] == 3 * 3 * 2


def test_univariate_config_runs_as_without_the_flag():
    m, series = model(variate_layers=(1,), variate_qk=16), items(V=1)
    a = run(m, series, 1, multivariate=True, **STACK)
    b = run(m, series, 1, **STACK)
    assert a["model"] == b["model"] and a["ratein"] == b["ratein"]


def test_univariate_model_is_refused():
    with pytest.raises(ValueError, match="variate_layers"):
        EG.check_model_flags(model(), "mix", False, None, None, multivariate=True)
