import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("train_patched", ROOT / "train_patched.py")
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)
LEVELS = torch.tensor([0.1, 0.5, 0.9])


def test_naive_loss_is_zero_on_a_seasonal_series():
    x = torch.arange(96.0)[None] % 24
    y = torch.arange(96.0, 144.0)[None] % 24
    assert tp.naive_loss(x, y, torch.ones_like(y, dtype=torch.bool), torch.ones(1), LEVELS) == 0


def test_score_losses_are_scale_invariant():
    x, y = torch.randn(2, 96), torch.randn(2, 48)
    q = torch.randn(2, 48, 3).sort(-1).values
    m, c, s = torch.ones_like(y, dtype=torch.bool), torch.zeros(2, 1), torch.ones(2, 1).double()
    w = torch.ones(2)
    for kind in ("ratio", "logratio"):
        a = tp.score_loss(q, c.double(), s, y, m, x, w, LEVELS, kind)
        b = tp.score_loss(q, c.double(), 10 * s, 10 * y, m, 10 * x, w, LEVELS, kind)
        assert a == pytest.approx(b.item(), rel=1e-3) and torch.isfinite(a)


def test_schedule_is_continuous_and_ends_low():
    assert tp.schedule(0, 10, 100) == pytest.approx(0.1)
    assert tp.schedule(99, 10, 100) < 0.01
