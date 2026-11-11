"""
scripts/dev_harness.py: a small patched model scored on a two-subset dev corpus through the
GIFT harness, plain, flipped and with the RateIN stack; the sample is
dev_eval.py's (dev_subsets).
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "TimeJEPA" / "src"))
sys.path.insert(0, str(ROOT.parent / "TimeJEPA" / "scripts"))

pytest.importorskip("evaluate_gift")

import dev_eval  # noqa: E402
from timessm.patched import PatchedSSM  # noqa: E402

SUBSETS = {"LOS_LOOP": "5T", "tourism_monthly": "M"}


def _corpus(tmp_path, n_series=6):
    rng = np.random.default_rng(0)
    manifest = {}
    for name, freq in SUBSETS.items():
        length = 2000 if freq == "5T" else 200
        period = 288 if freq == "5T" else 12
        t = np.arange(length)
        rows = [(rng.uniform(5, 50) + rng.uniform(1, 5) * np.sin(2 * np.pi * t / period)
                 + rng.normal(scale=0.3, size=length)).astype(np.float32) for _ in range(n_series)]
        np.save(tmp_path / f"{name}.values.npy", np.concatenate(rows))
        np.save(tmp_path / f"{name}.offsets.npy", np.arange(n_series + 1) * length)
        manifest[name] = {"freq": freq, "n_obs": n_series * length}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return str(tmp_path)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setattr(dev_eval, "DEV_SUBSETS", tuple(SUBSETS))
    import dev_harness
    torch.manual_seed(0)
    model = PatchedSSM(input_length=256, patch=16, d_model=32, n_layers=2, d_state=4,
                       n_heads=4).eval()
    return dev_harness, model, _corpus(tmp_path)


def test_dev_subsets_is_the_dev_eval_sample(harness):
    _, _, root = harness
    subsets = {name: (h, m, len(keep)) for name, _, h, m, keep in dev_eval.dev_subsets(root)}
    assert subsets == {"LOS_LOOP": (288 // 5, 288 // 5, 6), "tourism_monthly": (18, 12, 6)}


def test_scores_in_every_mode(harness):
    DH, model, root = harness
    plain = DH.harness_score(model, root, batch=8)
    assert set(plain["per_subset"]) == set(SUBSETS) | {"all"} and not plain["ratein"]
    assert math.isfinite(plain["dev"]) and plain["dev"] > 0
    flip = DH.harness_score(model, root, flip=True, batch=8)
    assert flip["dev"] != plain["dev"]
    stack = DH.harness_score(model, root, flip=True, ratein=True, batch=8)
    assert math.isfinite(stack["dev"]) and set(stack["ratein"]) == set(SUBSETS)
    assert all(math.isclose(sum(w.values()), 1.0, abs_tol=1e-3) for w in stack["ratein"].values())
