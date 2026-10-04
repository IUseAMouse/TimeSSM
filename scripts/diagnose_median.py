"""
Why is the median weak? Three measurements, no training (2026-10-04, phase 0
of the MASE plan). Each one tests a reading of the code against the per-config
map; none of them tunes anything on GIFT.

    # B - how often is the model no better than the seasonal naive? (CPU, JSONs)
    python scripts/diagnose_median.py sn <run_dir>

    # D and E - context lengths and heavy tails of the targets (CPU, GIFT data)
    python scripts/diagnose_median.py data [--run <run_dir>] [--configs a,b]

    # A and B - does the forecast flatten along the horizon? (GPU, minutes)
    python scripts/diagnose_median.py flat --checkpoint <ckpt> --config-name ssm_mini_v3_wide_eval

Hypotheses (docs/EXPERIMENTAL_LOG.md, 2026-10-04):
  B  no seasonal copy path: configs at or above the seasonal naive (`sn`), and
     a forecast amplitude already low in the first season (`flat`);
  D  contexts shorter than the training minimum (128): share per config, and
     the MASE gap to FlowState of the short-context configs (`data`);
  E  the pinball lives in the arcsinh frame, where an error at a robust
     z-score z weighs 1 / sqrt(1 + z^2), and MASE is a raw-frame metric: share
     of the raw deviation carried by |z| > 2, and its rank correlation with
     the MASE gap (`data`);
  A  undriven rollout: amplitude and step-to-step change of the future
     representations decaying with the step (`flat`).

`<run_dir>` is a .../gift_<tag>/ directory with per_config/*.json. `sn` and
the cross tables of `data` refuse a run with fewer than 97 configs.
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
TIMEJEPA = Path(os.environ.get("TIMEJEPA", HERE.parent / "TimeJEPA"))
SNAPSHOT = TIMEJEPA / "docs" / "assets" / "gift_leaderboard" / "2026-09-06" / "raw"
sys.path.insert(0, str(HERE / "scripts"))

from gift_gap_ssm import geomean, load_official, load_ours  # noqa: E402

N_CONFIGS = 97
SHORT_CONTEXT = 128          # the smallest of training.context_lengths
TAIL_Z = 2.0
FLAT_CONFIGS = "m4_hourly/H/short,electricity/H/short,electricity/H/long,solar/10T/long"


# ------------------------------------------------------------------ helpers
def spearman(x, y) -> float:
    """Rank correlation (average ranks on ties), nan below 3 finite pairs."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")

    def rank(v):
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        for u in np.unique(v):
            r[v == u] = r[v == u].mean()
        return r

    rx, ry = rank(x[ok]), rank(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def tail_stats(z: np.ndarray) -> tuple:
    """z = (target - context median) / context scale, any shape, NaN-masked ->
    (share of points with |z| > TAIL_Z, mean arcsinh-frame weight
    1 / sqrt(1 + z^2), share of the total deviation |z| those points carry).
    The deviation is counted in scale units so every series weighs the same."""
    z = np.abs(z[np.isfinite(z)])
    if z.size == 0:
        return float("nan"), float("nan"), float("nan")
    tail = z > TAIL_Z
    mass = float(z[tail].sum() / z.sum()) if z.sum() > 0 else 0.0
    return float(tail.mean()), float((1.0 / np.sqrt(1.0 + z * z)).mean()), mass


def amplitude_profile(x: np.ndarray, width: int) -> np.ndarray:
    """x [B, T] -> [B, T // width]: standard deviation inside each consecutive
    block of `width` steps (NaN-tolerant; the tail shorter than a block is
    dropped)."""
    n = x.shape[1] // width
    return np.nanstd(x[:, :n * width].reshape(x.shape[0], n, width), axis=2)


def run_ratios(run_dir: str, snapshot: Path, competitor: str = "FlowState-9.1M") -> dict:
    """config -> (our MASE ratio to the official seasonal naive, MASE relative
    to the competitor, decoupling = MASE relative / CRPS relative)."""
    sn = load_official(Path(snapshot) / "seasonal_naive.csv")
    comp = load_official(Path(snapshot) / f"{competitor}.csv")
    ours = {c: r for c, r in load_ours(run_dir).items() if c in sn and r["MASE"] and r["CRPS"]}
    if len(ours) != N_CONFIGS:
        raise SystemExit(f"{run_dir}: {len(ours)} configs, {N_CONFIGS} needed (run still being evaluated?)")
    out = {}
    for c, r in ours.items():
        mase, crps = r["MASE"] / sn[c]["MASE"], r["CRPS"] / sn[c]["CRPS"]
        rel_m = mase / (comp[c]["MASE"] / sn[c]["MASE"])
        rel_c = crps / (comp[c]["CRPS"] / sn[c]["CRPS"])
        out[c] = (mase, rel_m, rel_m / rel_c)
    return out


# ------------------------------------------------------------------ sn
def cmd_sn(args):
    ratios = run_ratios(args.run, args.snapshot)
    mase = {c: v[0] for c, v in ratios.items()}
    near = sorted((c for c in mase if mase[c] >= args.threshold), key=lambda c: -mase[c])
    actual = geomean(mase.values())
    oracle = geomean(min(v, 1.0) for v in mase.values())
    print(f"{len(mase)} configs, MASE ratio to the seasonal naive: geomean {actual:.4f}")
    print(f"configs at or above {args.threshold:.2f}: {len(near)}")
    for c in near:
        print(f"  {c:42s} {mase[c]:.3f}   vs FlowState x{ratios[c][1]:.2f}")
    print(f"oracle 'best of model and seasonal naive per config': {oracle:.4f} "
          f"({100 * (actual - oracle):+.2f} pt) - DIAGNOSTIC, picks on the test set")


# ------------------------------------------------------------------ data
def config_data_stats(series, h: int, windows: int, max_instances: int) -> dict:
    """Context lengths (capped at 1024, the harness' cap) and tail statistics
    of the targets in the RobustScale frame of their own context."""
    import torch
    from evaluate_gift import prepare_context
    from timejepa.evaluation import gift
    from timejepa.models.components.robust_scale import RobustScale

    insts = list(gift.iter_test_instances(series, h, windows))
    if len(insts) > max_instances:
        insts = [insts[i] for i in np.linspace(0, len(insts) - 1, max_instances).astype(int)]
    lengths, by_len = [], {}
    for inst in insts:
        ctx = prepare_context(inst.context, 1024, 1, 1)
        if ctx is None:
            continue
        lengths.append(min(len(inst.context), 1024))
        by_len.setdefault(len(ctx), []).append((ctx, inst.target))
    zs = []
    scaler = RobustScale()
    for items in by_len.values():
        ctx = torch.from_numpy(np.stack([c for c, _ in items])).unsqueeze(-1)       # [B, L, 1]
        scaler.fit(ctx)
        med, scale = scaler.median[:, 0, 0].numpy(), scaler.scale[:, 0, 0].numpy()
        tgt = np.stack([t for _, t in items])                                       # [B, h]
        zs.append(((tgt - med[:, None]) / scale[:, None]).ravel())
    lengths = np.asarray(lengths)
    share, weight, mass = tail_stats(np.concatenate(zs)) if zs else (float("nan"),) * 3
    return {
        "n": len(lengths),
        "ctx_median": float(np.median(lengths)) if len(lengths) else float("nan"),
        "ctx_q10": float(np.quantile(lengths, 0.1)) if len(lengths) else float("nan"),
        "ctx_short": float((lengths < SHORT_CONTEXT).mean()) if len(lengths) else float("nan"),
        "tail_share": share, "weight": weight, "tail_mass": mass,
    }


def cmd_data(args):
    sys.path.insert(0, str(TIMEJEPA / "scripts"))
    from timejepa.evaluation import gift

    gift_root = Path(args.gift_data_dir) if args.gift_data_dir else TIMEJEPA / "data" / "gift_eval"
    only = [c.strip() for c in args.configs.split(",")] if args.configs else []
    configs = [c for c in gift.GIFT_CONFIGS if not only or c in only]
    ratios = run_ratios(args.run, args.snapshot) if args.run else {}
    cache, stats = {}, {}
    for config in configs:
        key = gift.storage_path(config)
        if key not in cache:
            cache = {key: gift.load_series(gift_root, config)}      # one dataset in memory at a time
        series = cache[key]
        h = gift.prediction_length(config)
        windows = gift.num_windows(config, min(len(s) for s in series))
        stats[config] = config_data_stats(series, h, windows, args.max_instances)
        s, r = stats[config], ratios.get(config)
        print(f"{config:42s} n {s['n']:5d} | ctx med {s['ctx_median']:6.0f} q10 {s['ctx_q10']:6.0f} "
              f"<{SHORT_CONTEXT} {s['ctx_short']:5.2f} | |z|>{TAIL_Z:g} {s['tail_share']:5.3f} "
              f"w {s['weight']:5.3f} mass {s['tail_mass']:5.3f}"
              + (f" | MASE/FlowState x{r[1]:.3f} decoupling {r[2]:.2f}" if r else ""), flush=True)
    if not ratios:
        print("\n(no --run: cross tables skipped)")
        return
    if len(stats) != N_CONFIGS:
        print(f"\n({len(stats)} configs: cross tables need the {N_CONFIGS})")
        return
    print(f"\n== D: contexts under {SHORT_CONTEXT} points (MASE relative to FlowState-9.1M, geomean)")
    for label, pick in (("majority short (share >= 0.5)", lambda s: s["ctx_short"] >= 0.5),
                        ("some short (0 < share < 0.5)", lambda s: 0 < s["ctx_short"] < 0.5),
                        ("none short", lambda s: s["ctx_short"] == 0)):
        cs = [c for c in stats if pick(stats[c])]
        print(f"  {label:32s} n {len(cs):3d}  x{geomean(ratios[c][1] for c in cs):.3f}")
    cs = sorted(stats)
    for name, key in (("tail mass", "tail_mass"), ("tail share", "tail_share"), ("mean weight", "weight")):
        x = [stats[c][key] for c in cs]
        print(f"== E: Spearman of {name:11s} with MASE/FlowState {spearman(x, [ratios[c][1] for c in cs]):+.2f}, "
              f"with the decoupling {spearman(x, [ratios[c][2] for c in cs]):+.2f}")
    print("== E: ten heaviest tails (mass of the raw deviation at |z| > 2)")
    for c in sorted(cs, key=lambda c: -stats[c]["tail_mass"])[:10]:
        print(f"  {c:42s} mass {stats[c]['tail_mass']:.3f}  MASE/FlowState x{ratios[c][1]:.3f}  "
              f"decoupling {ratios[c][2]:.2f}")


# ------------------------------------------------------------------ flat
def cmd_flat(args):
    import torch
    from hydra import compose, initialize_config_dir

    sys.path.insert(0, str(TIMEJEPA / "scripts"))
    from evaluate_gift import prepare_context
    from timejepa.evaluation import create_model_from_config, gift, load_checkpoint

    with initialize_config_dir(version_base=None, config_dir=str((HERE / "configs").resolve())):
        cfg = compose(config_name=args.config_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(create_model_from_config(cfg), args.checkpoint, device).to(device).eval()
    gift_root = Path(args.gift_data_dir) if args.gift_data_dir else TIMEJEPA / "data" / "gift_eval"

    for config in [c.strip() for c in args.configs.split(",")]:
        h = gift.prediction_length(config)
        m = gift.seasonality(config.split("/")[1])
        width = m if 2 <= m <= h // 2 else 24
        series = gift.load_series(gift_root, config)
        windows = gift.num_windows(config, min(len(s) for s in series))
        insts = list(gift.iter_test_instances(series, h, windows))
        if len(insts) > args.max_instances:
            insts = [insts[j] for j in np.linspace(0, len(insts) - 1, args.max_instances).astype(int)]
        by_len = {}
        for inst in insts:
            ctx = prepare_context(inst.context, 1024, 1, 1)
            if ctx is not None and len(ctx) >= 4 * width:
                by_len.setdefault(len(ctx), []).append((ctx, inst.target))
        amp_model, amp_truth, change = [], [], []
        for items in by_len.values():
            for i in range(0, len(items), args.batch_size):
                chunk = items[i:i + args.batch_size]
                ctx = np.stack([c for c, _ in chunk])
                batch = torch.from_numpy(ctx).unsqueeze(-1).to(device)
                with torch.no_grad():
                    out = model.forecast(batch, n=h, return_representations=True)
                median = out["forecast_denorm"].squeeze(-1).float().cpu().numpy()        # [B, h]
                ref = amplitude_profile(ctx[:, -4 * width:], width).mean(axis=1, keepdims=True)
                ref = np.where(ref > 0, ref, np.nan)
                amp_model.append(amplitude_profile(median, width) / ref)
                amp_truth.append(amplitude_profile(np.stack([t for _, t in chunk]), width) / ref)
                rep = out["future_representations"].float()
                step = (rep[:, 1:] - rep[:, :-1]).norm(dim=-1) / rep[:, 1:].norm(dim=-1).clamp_min(1e-8)
                change.append(step.cpu().numpy())
        if not amp_model:
            print(f"\n== {config}: no usable instance")
            continue
        am = np.nanmedian(np.concatenate(amp_model), axis=0)
        at = np.nanmedian(np.concatenate(amp_truth), axis=0)
        ch = np.concatenate(change)                                                       # [N, h-1]
        n = sum(len(v) for v in by_len.values())
        print(f"\n== {config}: h {h}, block {width} steps, {n} instances "
              f"(amplitude = std inside a block / mean std of the last 4 context blocks; median over instances)")
        print("  block  steps        model   truth   model/truth   repr change per step")
        blocks = sorted(set(np.unique(np.geomspace(1, len(am), num=min(8, len(am))).astype(int)) - 1))
        for b in blocks:
            lo, hi = b * width, min((b + 1) * width, h) - 1
            print(f"  {b:5d}  {lo:4d}-{hi:<4d}   {am[b]:6.3f}  {at[b]:6.3f}   {am[b] / at[b]:8.3f}      "
                  f"{np.median(ch[:, max(lo - 1, 0):hi]):.4f}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--snapshot", default=str(SNAPSHOT), help="dir of the official per-config CSVs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sn", help="configs at or above the seasonal naive, oracle bound")
    p.add_argument("run")
    p.add_argument("--threshold", type=float, default=0.95)
    p.set_defaults(fn=cmd_sn)
    p = sub.add_parser("data", help="context lengths and target tails per config")
    p.add_argument("--run", default=None, help="run dir to cross with (97 configs)")
    p.add_argument("--configs", default="")
    p.add_argument("--gift-data-dir", default=None)
    p.add_argument("--max-instances", type=int, default=4000)
    p.set_defaults(fn=cmd_data)
    p = sub.add_parser("flat", help="forecast amplitude and representation change along the horizon")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config-name", default="ssm_mini_v3_wide_eval")
    p.add_argument("--configs", default=FLAT_CONFIGS)
    p.add_argument("--gift-data-dir", default=None)
    p.add_argument("--max-instances", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.set_defaults(fn=cmd_flat)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
