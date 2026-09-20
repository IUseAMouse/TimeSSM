"""
Realized batch SIZE of the rationed TemperatureSampler, batch by batch
(2026-09-18). Three 10M processes died at their 111,111th batch whatever
data.batch_size (64, 48) and the seed: with 106 families clamped at >= 1 sample
per batch the nominal batch is 106, not data.batch_size, and the realized size
is set by the fractional quotas (max_samples / num_batches), which fire
together at deterministic batch indices. No data is read: only the size
arithmetic of TemperatureSampler.__iter__ is replayed (it does not depend on
the random generator).

    python scripts/audit_batch_sizes.py --config-name ssm_mid_v3 --world-size 3 --batches 250000
    python scripts/audit_batch_sizes.py --config-name ssm_mid_v3 --world-size 3 --cap 64
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def replay(sizes, samples_per_dataset, max_ratio, num_batches, n, cap=None):
    """Batch sizes of the first n batches, rationed mode. cap: defer (never
    drop) the families beyond `cap` items to the next batch."""
    max_samples = np.array([int(s * max_ratio) for s in sizes], dtype=np.int64)
    quota = max_samples / max(num_batches, 1)
    n_i = np.array(samples_per_dataset, dtype=np.int64)
    allowance = np.zeros(len(sizes))
    drawn = np.zeros(len(sizes), dtype=np.int64)
    out = np.zeros(n, dtype=np.int64)
    for b in range(n):
        allowance += quota
        take = np.minimum(np.minimum(n_i, allowance.astype(np.int64)), max_samples - drawn)
        take = np.maximum(take, 0)
        if cap is not None and take.sum() > cap:
            # keep the families with the largest backlog first, defer the rest
            order = np.argsort(-allowance)
            kept = np.zeros_like(take)
            room = cap
            for i in order:
                if room <= 0:
                    break
                k = min(take[i], room)
                kept[i] = k
                room -= k
            take = kept
        allowance -= take
        drawn += take
        out[b] = take.sum()
    return out, allowance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="ssm_mid_v3")
    ap.add_argument("--world-size", type=int, default=3)
    ap.add_argument("--batches", type=int, default=250000)
    ap.add_argument("--cap", type=int, default=None, help="simulate a cap on the realized batch")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    import os
    from hydra import compose, initialize_config_dir
    from timejepa.data.datamodule import TemperatureSampler
    from train_ssm import build_datamodule
    cfg_dir = os.path.abspath(str(Path(__file__).resolve().parents[1] / "configs"))
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name=args.config_name, overrides=list(args.set))
    dm = build_datamodule(cfg)
    dm.prepare_data()
    dm.setup("fit")
    dm.train_dataloader()
    ref = dm._train_sampler
    s = TemperatureSampler(
        dataset_sizes=list(ref.dataset_sizes), batch_size=int(cfg.data.batch_size),
        temperature=float(cfg.data.sampling_temperature),
        max_oversample_ratio=float(cfg.data.max_oversample_ratio),
        seed=int(cfg.data.seed), rank=0, world_size=args.world_size,
        ration_oversample=bool(cfg.data.get("ration_oversample", False)),
        max_batch_size=cfg.data.get("max_batch_size") if cfg.data.get("fractional_batch") else args.cap,
        fractional_batch=bool(cfg.data.get("fractional_batch", False)))
    print(f"families {s.num_datasets} | data.batch_size {cfg.data.batch_size} | nominal "
          f"sum(samples_per_dataset) = {s.actual_batch_size} | batches per rank {len(s):,}")
    print(f"families clamped at 1 sample/batch: {sum(1 for v in s.samples_per_dataset if v == 1)}")
    if not s.ration_oversample:
        print("ration_oversample is off: the batch is the nominal one until families die out")
        return
    n = min(args.batches, len(s))
    if s.fractional_batch:
        # The fractional path is iterated for real (sizes only are kept).
        out = np.fromiter((len(b) for _, b in zip(range(n), iter(s))), dtype=np.int64, count=n)
        backlog = np.zeros(1)
    else:
        out, backlog = replay(s.dataset_sizes, s.samples_per_dataset, s.max_oversample_ratio,
                              len(s), n, cap=args.cap)
    q = np.percentile(out, [1, 50, 99, 99.99])
    print(f"first {n:,} batches{' (cap ' + str(args.cap) + ')' if args.cap else ''}: mean "
          f"{out.mean():.2f} | p1 {q[0]:.0f} p50 {q[1]:.0f} p99 {q[2]:.0f} p99.99 {q[3]:.0f} | "
          f"max {out.max()} at batch {int(out.argmax()):,}")
    top = np.argsort(-out)[:10]
    print("largest batches:", ", ".join(f"{int(i):,}:{int(out[i])}" for i in sorted(top)))
    lo = max(0, 111_111 - 3)
    if n > 111_115:
        print("around 111,111:", out[lo:111_115].tolist())
    print(f"windows in these batches: {int(out.sum()):,} per rank "
          f"({out.sum() * args.world_size / 1e6:.1f} M on {args.world_size} ranks)")
    per_epoch = out.mean() * len(s) * args.world_size
    print(f"epoch at this composition: {per_epoch / 1e9:.2f} B windows -> "
          f"schedule_fraction for 298M windows = {298e6 / per_epoch:.5f}")
    if args.cap:
        print(f"backlog left after {n:,} capped batches: {backlog.sum():.1f} samples "
              "(bounded = the cap defers, it does not starve)")


if __name__ == "__main__":
    main()
