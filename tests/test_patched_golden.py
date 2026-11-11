"""PatchedSSM with every optional mode off is pinned to a recorded file: forecasts (plain and
causal, scalar and per-item w, NaN gap, short history), the causal context outputs, one
training loss and its gradient norm (scripts/train_patched.step_loss), the parameter names
and count, and the first training batches of a seeded WindowBatches.

Tolerance as in test_golden.py: 1e-5 by default (FFT and BLAS paths depend on the CPU),
bit equality with GOLDEN_STRICT=1. Batches are exact: they are numpy draws."""

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "golden"))

import make_patched_golden  # noqa: E402

GOLDEN = torch.load(HERE / "golden" / "patched_golden.pt", weights_only=False)
STRICT = os.environ.get("GOLDEN_STRICT") == "1"


def test_defaults_reproduce_the_patched_golden_file():
    now = make_patched_golden.compute()
    assert set(now) == set(GOLDEN)
    for key in sorted(GOLDEN):
        a, b = now[key], GOLDEN[key]
        if key.startswith(("keys/", "params/")):
            assert a == b, key
        elif key.startswith("batch/"):
            assert torch.equal(a.isnan(), b.isnan()) and torch.equal(a.nan_to_num(), b.nan_to_num()), key
        else:
            worst = float(((a - b).abs() / b.abs().clamp_min(1.0)).max())
            ok = torch.equal(a, b) if STRICT else torch.allclose(a, b, atol=1e-5, rtol=1e-5)
            assert ok, f"{key} moved: largest relative difference {worst:.2e} (strict: {STRICT})"
