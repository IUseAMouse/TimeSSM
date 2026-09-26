"""gift_gap_ssm.py on a synthetic fixture: ratios, groups, coverage, wins (2026-09-26)."""

import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SN_COLS = ["dataset", "model", "eval_metrics/MASE[0.5]", "eval_metrics/mean_weighted_sum_quantile_loss",
           "domain", "num_variates"]


def _write_official(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SN_COLS); w.writeheader()
        for r in rows:
            w.writerow(r)


def test_gap_map_on_fixture(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    cfgs = [("solar/H/short", 48, "Energy", 1), ("ett1/H/long", 720, "Energy", 7), ("m4_monthly/M/short", 18, "Econ", 1)]
    _write_official(raw / "seasonal_naive.csv", [
        {"dataset": c, "model": "Seasonal_Naive", SN_COLS[2]: 1.0, SN_COLS[3]: 1.0, "domain": d, "num_variates": v}
        for c, _, d, v in cfgs])
    _write_official(raw / "Comp.csv", [
        {"dataset": c, "model": "Comp", SN_COLS[2]: 0.8, SN_COLS[3]: 0.5, "domain": d, "num_variates": v}
        for c, _, d, v in cfgs])
    run = tmp_path / "run" / "ckpt" / "gift_x"; (run / "per_config").mkdir(parents=True)
    ours = {"solar/H/short": (0.4, 0.7), "ett1/H/long": (0.8, 0.9), "m4_monthly/M/short": (0.5, 0.8)}
    for (c, h, _, _) in cfgs:
        crps, mase = ours[c]
        json.dump({"config": c, "prediction_length": h,
                   "model": {"MASE": mase, "CRPS": crps, "coverage": {"0.1": 0.15, "0.9": 0.85}},
                   "ratein": {"frac_k_gt1": 0.5}},
                  open(run / "per_config" / (c.replace("/", "__") + ".json"), "w"))
    corpus = tmp_path / "corpus"; corpus.mkdir(); (corpus / "solar_power.npy").touch()
    out = subprocess.run([sys.executable, str(HERE / "scripts" / "gift_gap_ssm.py"), str(run),
                          "--snapshot", str(raw), "--competitors", "Comp", "--corpus-dir", str(corpus)],
                         capture_output=True, text=True, check=True).stdout
    assert "3 configs" in out and "NOT comparable" in out
    # short group: geomean CRPS of 0.4 and 0.5 = 0.4472; long group 0.8
    assert "0.4472" in out and "0.8000" in out
    # vs Comp (0.5 everywhere): short rel = 0.4472/0.5 = 0.894, wins 2/2; long 1.6, wins 0/1
    assert "0.894" in out and "1/2" in out and "1.600" in out and "0/1" in out   # m4 ties Comp: not a win
    # horizon buckets and variates
    assert ">480" in out and "2-10" in out
    # coverage line: q10 0.150, q90 0.850, interval 0.700, k>1 0.50
    assert "0.150 0.850  0.700 0.50" in out
    # corpus cousin: solar resolved yes, ett1 no, m4_monthly no (no m1/m3 file)
    assert "yes" in out and "no" in out
