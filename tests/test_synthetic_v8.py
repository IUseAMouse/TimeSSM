import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_synthetic_v8 import GRID, kinds, sample  # noqa: E402

from timejepa.data.frequency import season_length  # noqa: E402


@pytest.mark.parametrize("freq", list(GRID))
def test_every_family_draws_finite_float32_series(freq):
    length = GRID[freq][0]
    season = season_length(freq, weekly=True) or 0.0
    for kind in kinds(freq):
        rng = np.random.default_rng(0)
        for _ in range(20):
            x = sample(kind, length, freq, season, rng)
            assert x.shape == (length,) and x.dtype == np.float32
            assert np.isfinite(x).all() and x.std() > 0
            if kind in ("calendar", "growth"):
                assert x.min() >= 0


def test_calendar_only_where_a_day_fits_and_growth_only_at_slow_rates():
    assert "calendar" not in kinds("10S") and "calendar" in kinds("5T") and "calendar" in kinds("D")
    assert "growth" in kinds("M") and "growth" not in kinds("H")


def test_same_seed_same_series():
    a = sample("arima", 512, "M", 12.0, np.random.default_rng(3))
    b = sample("arima", 512, "M", 12.0, np.random.default_rng(3))
    assert np.array_equal(a, b)
