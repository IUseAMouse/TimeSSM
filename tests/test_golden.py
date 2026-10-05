"""The default behaviour is pinned to a file recorded before the frequency-tied
Delta (2026-10-05): model outputs, the training losses with their random Delta
draws, the eval loss and the parameter names. Strict equality on the recording
machine's torch build, 1e-5 elsewhere (FFT kernels differ in the last bits)."""

import platform
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "golden"))

import make_golden  # noqa: E402

GOLDEN = torch.load(HERE / "golden" / "ssm_golden.pt", weights_only=False)
SAME_BUILD = (GOLDEN["meta"]["torch"] == torch.__version__
              and GOLDEN["meta"]["machine"] == platform.machine())


def _same(a, b):
    return torch.equal(a, b) if SAME_BUILD else torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_defaults_reproduce_the_golden_file():
    now = make_golden.compute()
    assert now["state_dict_keys"] == GOLDEN["state_dict_keys"]          # no parameter added or renamed
    assert torch.equal(now["train/scales"], GOLDEN["train/scales"])     # same Delta draws, same order
    for key in sorted(k for k in GOLDEN if k.startswith(("forecast/", "train/losses", "eval/"))):
        assert _same(now[key], GOLDEN[key]), f"{key} moved (strict: {SAME_BUILD})"
