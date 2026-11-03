import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_synthetic_template import sample_template, shape  # noqa: E402


def test_shape_repeats_every_period():
    x = shape(480, 24.0, np.random.default_rng(0))
    np.testing.assert_allclose(x[24:], x[:-24], atol=1e-6)
    assert abs(x.std() - 1) < 1e-6


def test_templates_are_finite_and_mostly_periodic():
    rng = np.random.default_rng(1)
    for _ in range(50):
        x = sample_template(2048, 24.0, "H", rng).astype(np.float64)
        assert x.shape == (2048,) and np.isfinite(x).all()
    # With the period exactly the season, the seasonal naive error is small against the
    # spread for most draws: these are the regular series the family is for.
    ratios = []
    for _ in range(200):
        x = sample_template(2048, 24.0, "H", rng).astype(np.float64)
        ratios.append(np.abs(x[24:] - x[:-24]).mean() / x.std())
    assert np.median(ratios) < 0.5
