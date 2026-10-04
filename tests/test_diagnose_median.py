"""diagnose_median.py: helpers and the `sn` table on a synthetic fixture (2026-10-04)."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))

import diagnose_median as dm  # noqa: E402

SN_COLS = ["dataset", "model", "eval_metrics/MASE[0.5]", "eval_metrics/mean_weighted_sum_quantile_loss",
           "domain", "num_variates"]


def _write_official(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SN_COLS); w.writeheader()
        w.writerows(rows)


def test_spearman_is_a_rank_correlation():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert abs(dm.spearman(x, [v ** 3 for v in x]) - 1.0) < 1e-12
    assert abs(dm.spearman(x, [-v for v in x]) + 1.0) < 1e-12
    assert np.isnan(dm.spearman(x, [1.0] * 5)) and np.isnan(dm.spearman(x[:2], x[:2]))


def test_tail_stats_counts_share_weight_and_mass():
    z = np.array([0.0, 1.0, -3.0, np.nan, 4.0])
    share, weight, mass = dm.tail_stats(z)
    assert share == 0.5                                          # two of the four finite points
    assert abs(mass - 7.0 / 8.0) < 1e-12                         # (3 + 4) / (0 + 1 + 3 + 4)
    expected = np.mean([1.0, 1 / np.sqrt(2), 1 / np.sqrt(10), 1 / np.sqrt(17)])
    assert abs(weight - expected) < 1e-12


def test_amplitude_profile_is_the_std_per_block():
    t = np.arange(48)
    x = np.stack([np.sin(2 * np.pi * t / 24), np.zeros(48)])
    amp = dm.amplitude_profile(x, 24)
    assert amp.shape == (2, 2)
    assert np.allclose(amp[0], np.sqrt(0.5), atol=1e-6) and np.allclose(amp[1], 0.0)
    assert dm.amplitude_profile(x[:, :47], 24).shape == (2, 1)   # the incomplete block is dropped


def _fixture(tmp_path, n=dm.N_CONFIGS):
    raw = tmp_path / "raw"; raw.mkdir()
    cfgs = [f"ds{i}/H/short" for i in range(n)]
    rows = lambda mase: [{"dataset": c, "model": "x", SN_COLS[2]: mase, SN_COLS[3]: 1.0,
                          "domain": "Energy", "num_variates": 1} for c in cfgs]
    _write_official(raw / "seasonal_naive.csv", rows(2.0))
    _write_official(raw / "FlowState-9.1M.csv", rows(1.0))
    run = tmp_path / "run" / "ckpt" / "gift"; (run / "per_config").mkdir(parents=True)
    for i, c in enumerate(cfgs):
        mase = 2.4 if i == 0 else 1.0                            # ratio 1.2 on one config, 0.5 elsewhere
        json.dump({"config": c, "prediction_length": 48, "model": {"MASE": mase, "CRPS": 0.5, "coverage": {}}},
                  open(run / "per_config" / (c.replace("/", "__") + ".json"), "w"))
    return raw, run


def _sn(raw, run):
    return subprocess.run([sys.executable, str(HERE / "scripts" / "diagnose_median.py"), "--snapshot", str(raw),
                           "sn", str(run)], capture_output=True, text=True)


def test_sn_lists_configs_above_the_seasonal_naive_and_the_oracle_bound(tmp_path):
    raw, run = _fixture(tmp_path)
    out = _sn(raw, run)
    assert out.returncode == 0, out.stderr
    assert "configs at or above 0.95: 1" in out.stdout and "ds0/H/short" in out.stdout and "1.200" in out.stdout
    n = dm.N_CONFIGS
    actual = np.exp((np.log(1.2) + (n - 1) * np.log(0.5)) / n)
    oracle = np.exp((n - 1) * np.log(0.5) / n)
    assert f"geomean {actual:.4f}" in out.stdout and f"{oracle:.4f}" in out.stdout


def test_sn_refuses_an_incomplete_run(tmp_path):
    raw, run = _fixture(tmp_path, n=5)
    out = _sn(raw, run)
    assert out.returncode != 0 and "97 needed" in out.stderr


def test_error_parts_separate_level_shape_and_phase():
    t = np.arange(48, dtype=float)
    truth = np.sin(2 * np.pi * t / 24)
    mase, level, shape, lag, at_lag = dm.error_parts(truth + 2.0, truth, 1.0, 6)       # pure level error
    assert abs(mase - 2.0) < 1e-12 and abs(level - 2.0) < 1e-12 and shape < 1e-12 and lag == 0
    late = np.sin(2 * np.pi * (t - 3) / 24)                                            # forecast 3 steps late
    mase, level, shape, lag, at_lag = dm.error_parts(late, truth, 1.0, 6)
    assert lag == 3 and at_lag < 1e-12 and mase > 0.3 and level < 0.05
    mase, _, _, lag, at_lag = dm.error_parts(truth, truth, 0.5, 6)                     # exact: nothing to gain
    assert mase == 0.0 and lag == 0 and at_lag < 1e-12
    nan_truth = truth.copy(); nan_truth[:5] = np.nan
    assert np.isfinite(dm.error_parts(truth + 1.0, nan_truth, 1.0, 6)[0])
