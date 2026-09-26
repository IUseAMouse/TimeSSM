"""
Where does a run lose on GIFT-Eval? (2026-09-26, plan phase A1)

Per-config CRPS / MASE ratios against the OFFICIAL Seasonal Naive, aggregated
by term, frequency, horizon bucket, number of variates, domain and "has a
cousin in the training corpus", side by side for one or more of our runs and
against the leaderboard's per-config results (TimeJEPA vendors them under
docs/assets/gift_leaderboard/<date>/raw/<model>.csv). Also the 80% coverage
by term and by horizon bucket - the two questions of the plan: is the gap
concentrated on the horizons the model never trained on (> 256), and is the
fan too narrow everywhere or only under shift.

    python scripts/gift_gap_ssm.py evaluation/timessm_mini_v3_wide_zs/<ckpt>/gift_flip_ratein-mix-pool \
        [more run dirs...] --competitors Toto-2.0-4m,FlowState-9.1M,TTM-R3-PT \
        [--corpus-manifest ../TimeJEPA/configs/corpus_v3_manifest.txt | --corpus-dir <lotsa_v3>]

Inputs: <run>/per_config/*.json written by TimeJEPA's evaluate_gift.py
(fields: config, prediction_length, model.{MASE, CRPS, coverage{...}},
ratein.frac_k_gt1). Domain and num_variates come from the official
seasonal_naive.csv (our all_results.csv leaves them empty).
"""

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

TERMS = ("short", "medium", "long")
MASE_COL = "eval_metrics/MASE[0.5]"
CRPS_COL = "eval_metrics/mean_weighted_sum_quantile_loss"

# GIFT dataset -> corpus "cousins" (the corpus excludes the 28 GIFT sources by
# construction; these are the related families that ARE in lotsa_v3, matched
# as substrings of the .npy stems). Unresolved names are printed, not guessed.
GIFT_TO_CORPUS = {
    "solar": ["solar_power"],
    "electricity": ["australian_electricity_demand"],
    "kdd_cup_2018": ["beijing_air_quality", "china_air_quality", "kdd2022"],
    "sz_taxi": ["taxi_30min"],
    "loop_seattle": ["q-traffic", "q_traffic"],
    "m4_yearly": ["m1_yearly", "monash_m3_yearly", "tourism_yearly"],
    "m4_quarterly": ["m1_quarterly", "monash_m3_quarterly", "tourism_quarterly"],
    "m4_monthly": ["m1_monthly", "monash_m3_monthly", "tourism_monthly"],
    "m4_weekly": ["nn5_weekly"],
    "m4_daily": ["nn5_daily"],
    "m4_hourly": [],
    "covid_deaths": ["covid_mobility", "covid19_energy"],
    "hospital": [],
    "car_parts": [],
    "us_births": [],
    "saugeen": [],
    "temperature_rain": [],
    "hierarchical_sales": [],
    "restaurant": [],
    "m_dense": ["q-traffic", "q_traffic"],
    "ett1": [],
    "ett2": [],
    "jena_weather": [],
    "bitbrains_fast_storage": [],
    "bitbrains_rnd": [],
    "bizitobs_l2c": [],
    "bizitobs_application": [],
    "bizitobs_service": [],
}


def geomean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x) and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def horizon_bucket(h):
    return "<=64" if h <= 64 else "<=256" if h <= 256 else "<=480" if h <= 480 else ">480"


def variates_bucket(v):
    return "1" if v <= 1 else "2-10" if v <= 10 else ">10"


def load_official(csv_path):
    out = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            out[row["dataset"]] = {
                "MASE": float(row[MASE_COL]), "CRPS": float(row[CRPS_COL]),
                "domain": row.get("domain", ""),
                "num_variates": int(float(row["num_variates"])) if row.get("num_variates") else 1,
            }
    return out


def load_ours(run_dir):
    out = {}
    for f in glob.glob(os.path.join(run_dir, "per_config", "*.json")):
        d = json.load(open(f))
        m = d.get("model") or {}
        cov = m.get("coverage") or {}
        out[d["config"]] = {
            "MASE": m.get("MASE"), "CRPS": m.get("CRPS"),
            "h": int(d.get("prediction_length", 0)),
            "q10": cov.get("0.1"), "q90": cov.get("0.9"),
            "frac_k_gt1": (d.get("ratein") or {}).get("frac_k_gt1"),
        }
    return out


def corpus_stems(manifest=None, corpus_dir=None):
    if manifest:
        return {ln.split()[0].lower() for ln in open(manifest) if ln.strip() and not ln.startswith("#")}
    if corpus_dir:
        return {Path(p).stem.lower() for p in glob.glob(os.path.join(corpus_dir, "*.npy"))}
    return None


def has_cousin(dataset, stems):
    if stems is None:
        return "?"
    cousins = GIFT_TO_CORPUS.get(dataset)
    if cousins is None:
        return "?"
    return "yes" if any(any(c in s for s in stems) for c in cousins) else "no"


def group_keys(config, ours_row, official_row, stems):
    ds, freq, term = config.split("/")
    return {
        "term": term, "freq": freq,
        "horizon": horizon_bucket(ours_row["h"]),
        "variates": variates_bucket(official_row["num_variates"]),
        "domain": official_row["domain"] or "?",
        "corpus_cousin": has_cousin(ds, stems),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run dirs (.../gift_<tag>/)")
    ap.add_argument("--snapshot", default=None,
                    help="dir of official per-config CSVs (default: TimeJEPA docs/assets/gift_leaderboard/2026-09-06/raw)")
    ap.add_argument("--competitors", default="Toto-2.0-4m,FlowState-9.1M,TTM-R3-PT")
    ap.add_argument("--corpus-manifest", default=None)
    ap.add_argument("--corpus-dir", default=None)
    ap.add_argument("--by", default="term,horizon,freq,variates,corpus_cousin,domain")
    args = ap.parse_args()

    here = Path(__file__).resolve().parents[1]
    snapshot = Path(args.snapshot) if args.snapshot else \
        here.parent / "TimeJEPA" / "docs" / "assets" / "gift_leaderboard" / "2026-09-06" / "raw"
    sn = load_official(snapshot / "seasonal_naive.csv")
    competitors = {c: load_official(snapshot / f"{c}.csv") for c in args.competitors.split(",") if c}
    stems = corpus_stems(args.corpus_manifest, args.corpus_dir)
    runs = {Path(r).parts[-3] + "/" + Path(r).parts[-2]: load_ours(r) for r in args.runs}

    # ratios vs official SN, per run and per competitor
    ratios = {name: {} for name in runs}
    for name, ours in runs.items():
        for cfg, row in ours.items():
            if cfg in sn and row["CRPS"] and row["MASE"]:
                ratios[name][cfg] = (row["CRPS"] / sn[cfg]["CRPS"], row["MASE"] / sn[cfg]["MASE"])
    comp_ratios = {c: {cfg: (r["CRPS"] / sn[cfg]["CRPS"], r["MASE"] / sn[cfg]["MASE"])
                       for cfg, r in rows.items() if cfg in sn} for c, rows in competitors.items()}

    first = next(iter(runs))
    configs = sorted(ratios[first])
    print(f"{len(configs)} configs in {first}" + (" (fewer than 97: NOT comparable)" if len(configs) < 97 else ""))
    unresolved = sorted({c.split('/')[0] for c in configs if has_cousin(c.split('/')[0], stems) == '?'})
    if stems is not None and unresolved:
        print("corpus cousin unresolved for:", ", ".join(unresolved))

    for key in args.by.split(","):
        groups = defaultdict(list)
        for cfg in configs:
            groups[group_keys(cfg, runs[first][cfg], sn[cfg], stems)[key]].append(cfg)
        order = TERMS if key == "term" else sorted(groups, key=lambda g: (-len(groups[g]), g))
        print(f"\n== by {key}")
        head = f"{'group':>14s} {'n':>3s}" + "".join(f" | {n[-22:]:>22s} CRPS  MASE" for n in runs) \
            + "".join(f" | vs {c[:12]:>12s} CRPS wins" for c in competitors)
        print(head)
        for g in order:
            cfgs = groups.get(g, [])
            if not cfgs:
                continue
            line = f"{g:>14s} {len(cfgs):>3d}"
            for n in runs:
                line += f" | {geomean([ratios[n][c][0] for c in cfgs if c in ratios[n]]):27.4f} {geomean([ratios[n][c][1] for c in cfgs if c in ratios[n]]):.4f}"
            for c, cr in comp_ratios.items():
                both = [x for x in cfgs if x in cr and x in ratios[first]]
                rel = geomean([ratios[first][x][0] / cr[x][0] for x in both])
                wins = sum(ratios[first][x][0] < cr[x][0] for x in both)
                line += f" | {rel:26.3f} {wins:>2d}/{len(both)}"
            print(line)
        # overall row
        line = f"{'ALL':>14s} {len(configs):>3d}"
        for n in runs:
            line += f" | {geomean([ratios[n][c][0] for c in configs if c in ratios[n]]):27.4f} {geomean([ratios[n][c][1] for c in configs if c in ratios[n]]):.4f}"
        for c, cr in comp_ratios.items():
            both = [x for x in configs if x in cr]
            line += f" | {geomean([ratios[first][x][0] / cr[x][0] for x in both]):26.3f} {sum(ratios[first][x][0] < cr[x][0] for x in both):>2d}/{len(both)}"
        print(line)

    # coverage and rate share by term and horizon bucket, per run
    for key in ("term", "horizon"):
        print(f"\n== coverage (q10 / q90 / 80% interval) and share of k>1 instances, by {key}")
        groups = defaultdict(list)
        for cfg in configs:
            groups[group_keys(cfg, runs[first][cfg], sn[cfg], stems)[key]].append(cfg)
        order = TERMS if key == "term" else sorted(groups)
        print(f"{'group':>14s} {'n':>3s}" + "".join(f" | {n[-22:]:>22s}   q10   q90  cov80  k>1" for n in runs))
        for g in order:
            cfgs = groups.get(g, [])
            if not cfgs:
                continue
            line = f"{g:>14s} {len(cfgs):>3d}"
            for n in runs:
                rows = [runs[n][c] for c in cfgs if c in runs[n] and runs[n][c]["q10"] is not None]
                if not rows:
                    line += " | " + " " * 22 + "   n/a"
                    continue
                q10 = sum(r["q10"] for r in rows) / len(rows); q90 = sum(r["q90"] for r in rows) / len(rows)
                ks = [r["frac_k_gt1"] for r in rows if r["frac_k_gt1"] is not None]
                k = sum(ks) / len(ks) if ks else float("nan")
                line += f" | {'':>22s} {q10:5.3f} {q90:5.3f} {q90 - q10:6.3f} {k:4.2f}"
            print(line)

    # the tail: worst 10 configs of the first run vs the first competitor
    if comp_ratios:
        c0 = next(iter(comp_ratios))
        both = [x for x in configs if x in comp_ratios[c0]]
        worst = sorted(both, key=lambda x: -ratios[first][x][0] / comp_ratios[c0][x][0])[:10]
        print(f"\n== worst 10 configs of {first} vs {c0} (our CRPS ratio / theirs)")
        for x in worst:
            print(f"  {x:40s} x{ratios[first][x][0] / comp_ratios[c0][x][0]:.2f}  ours {ratios[first][x][0]:.3f} theirs {comp_ratios[c0][x][0]:.3f}  h={runs[first][x]['h']}")


if __name__ == "__main__":
    main()
