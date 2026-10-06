"""The default behaviour is pinned to a file recorded before the frequency-tied
Delta (2026-10-05): model outputs, the training losses with their random Delta
draws, the eval loss and the parameter names.

Tolerance. Bit equality only holds on the machine that recorded the file: the
FFT and BLAS kernels pick their code path from the CPU, so another machine
with the very same torch build differs in the last bits (seen on the pod,
2026-10-06). The default is therefore 1e-5; GOLDEN_STRICT=1 asks for bit
equality and is how the file was checked on the recording machine right after
the change. A real regression is orders of magnitude above 1e-5."""

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "golden"))

import make_golden  # noqa: E402

GOLDEN = torch.load(HERE / "golden" / "ssm_golden.pt", weights_only=False)
STRICT = os.environ.get("GOLDEN_STRICT") == "1"


def test_defaults_reproduce_the_golden_file():
    now = make_golden.compute()
    assert now["state_dict_keys"] == GOLDEN["state_dict_keys"]          # no parameter added or renamed
    assert torch.equal(now["train/scales"], GOLDEN["train/scales"])     # same Delta draws, same order
    for key in sorted(k for k in GOLDEN if k.startswith(("forecast/", "train/losses", "eval/"))):
        a, b = now[key], GOLDEN[key]
        worst = float(((a - b).abs() / b.abs().clamp_min(1.0)).max())
        ok = torch.equal(a, b) if STRICT else torch.allclose(a, b, atol=1e-5, rtol=1e-5)
        assert ok, f"{key} moved: largest relative difference {worst:.2e} (strict: {STRICT})"
