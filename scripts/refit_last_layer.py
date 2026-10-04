"""
Refit the LAST linear layer of the quantile head with everything else frozen
(arm R1, 2026-10-04): the network becomes a fixed feature map and the final
projection (d_model -> 9 raw quantile outputs) is solved to its optimum on
stored features, with no backpropagation through the body.

    python scripts/refit_last_layer.py --checkpoint <ckpt> --config-name ssm_mini_v3_wide \
        --out checkpoints/timessm_mini_v3_wide_refit/pretrain_False

What is solved. Features phi [N, D] are the inputs of
`decoder.decoder.unpatching.projection` on corpus windows, the target y [N] is
in the model's frame (RobustScale then RevIN, context statistics: the frame of
the training loss). The loss is the head's own pinball on the monotone fan,

    L(W, b) = pinball(monotone(phi W^T + b), y) + lam * ||W - W0||^2,

minimized full-batch by L-BFGS from the trained (W0, b0). The median row is a
convex problem (a least-absolute-deviation regression: the median is the raw
output itself); the width rows go through softplus and a cumulative sum.
`--rows median` moves the median row only. There is no closed form for a
pinball: this is the iterative solution of the last layer's own problem, which
SGD on the whole network only approaches.

Scale. The rows are spread over every visible GPU (`--gb-per-gpu` of
features each, 768 bytes per row at d_model 192) and the loss is accumulated
over slices, so the fit is exact full-batch whatever the number of rows.

Fit on TRAIN windows, report on VAL windows (never seen by the fit); the new
checkpoint is written only if the validation pinball improves. It is the input
checkpoint with two tensors replaced, so every eval script reads it unchanged.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("refit_last_layer")

PROJECTION = "decoder.decoder.unpatching.projection"


SLICE_ROWS = 1_000_000       # rows per forward/backward slice: bounds the transient memory


def head_loss(head, phi, y, weight, bias):
    """Pinball of the head's monotone fan for a projection (weight [Q, D],
    bias [Q]) on features phi [N, D] against y [N]."""
    fan = head._make_monotone(phi @ weight.T + bias)                   # [N, Q]
    return head.loss(fan[None], y[None])


def chunked_loss(head, chunks, weight, bias, backward: bool = False) -> float:
    """Mean pinball over `chunks` = [(phi, y), ...], each on its own device,
    evaluated slice by slice. With `backward`, gradients accumulate into the
    leaves behind (weight, bias): the same gradient as one full-batch pass."""
    total = sum(len(y) for _, y in chunks)
    value = 0.0
    for phi, y in chunks:
        w, b = weight.to(phi.device), bias.to(phi.device)
        for i in range(0, len(y), SLICE_ROWS):
            part = head_loss(head, phi[i:i + SLICE_ROWS], y[i:i + SLICE_ROWS], w, b)
            part = part * (len(y[i:i + SLICE_ROWS]) / total)
            if backward:
                part.backward(retain_graph=True)
            value += float(part.detach())
    return value


def refit(head, chunks, weight0, bias0, lam: float = 0.0, rows: str = "all", max_iter: int = 300):
    """L-BFGS from (weight0, bias0) on `chunks` = [(phi, y), ...] (one per
    device); `rows` = 'all' or 'median' (only the median row moves). Returns
    (weight, bias, loss before, loss after), the losses without the penalty."""
    if rows not in ("all", "median"):
        raise ValueError(f"rows must be 'all' or 'median', got {rows!r}")
    move = torch.zeros(weight0.shape[0], 1, dtype=weight0.dtype, device=weight0.device)
    if rows == "all":
        move += 1.0
    else:
        move[head.median_idx] = 1.0
    dw = torch.zeros_like(weight0, requires_grad=True)
    db = torch.zeros_like(bias0, requires_grad=True)
    opt = torch.optim.LBFGS([dw, db], max_iter=max_iter, history_size=20,
                            tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        value = chunked_loss(head, chunks, weight0 + move * dw, bias0 + move[:, 0] * db, backward=True)
        penalty = lam * (move * dw).pow(2).sum()
        penalty.backward()
        return torch.tensor(value + float(penalty))

    with torch.no_grad():
        before = chunked_loss(head, chunks, weight0, bias0)
    opt.step(closure)
    with torch.no_grad():
        weight, bias = weight0 + move * dw, bias0 + move[:, 0] * db
        after = chunked_loss(head, chunks, weight, bias)
    return weight.detach(), bias.detach(), before, after


def spread(phi, y, devices, rows_per_device: int, rng: np.random.Generator):
    """Split (phi [N, D], y [N]) evenly over `devices`, at most
    rows_per_device each; a random subset is kept when N exceeds the room."""
    room = rows_per_device * len(devices)
    if len(y) > room:
        keep = torch.as_tensor(np.sort(rng.choice(len(y), size=room, replace=False)))
        phi, y = phi[keep], y[keep]
    bounds = np.linspace(0, len(y), len(devices) + 1).astype(int)
    return [(phi[lo:hi].to(dev), y[lo:hi].to(dev)) for dev, lo, hi in zip(devices, bounds[:-1], bounds[1:])]


@torch.no_grad()
def collect(model, dataset, sizes, per_dataset: int, steps: int, batch_size: int, device,
            context_lengths, p_context: float, rng: np.random.Generator):
    """(phi [N, D], y [N]) on `per_dataset` regularly spaced windows of each
    dataset, `steps` horizon positions drawn per batch. The context is cropped
    to a random training length with probability p_context, as in training."""
    head = model.decoder.decoder
    grabbed = {}
    hook = head.unpatching.projection.register_forward_pre_hook(lambda _m, inp: grabbed.update(phi=inp[0]))
    bounds = np.cumsum([0] + list(sizes))
    idx = np.concatenate([np.linspace(lo, hi - 1, min(per_dataset, hi - lo)).astype(int)
                          for lo, hi in zip(bounds[:-1], bounds[1:]) if hi - lo >= 8])
    rng.shuffle(idx)
    feats, targets = [], []
    try:
        for i0 in range(0, len(idx), batch_size):
            items = [dataset[int(j)] for j in idx[i0:i0 + batch_size]]
            ctx = torch.stack([torch.as_tensor(it["context"], dtype=torch.float32).reshape(-1)
                               for it in items]).unsqueeze(-1).to(device)                 # [B, L, 1]
            tgt = torch.stack([torch.as_tensor(it["target"], dtype=torch.float32).reshape(-1)
                               for it in items]).unsqueeze(-1).to(device)                 # [B, h, 1]
            eligible = [n for n in context_lengths if n <= ctx.shape[1]]
            if eligible and rng.random() < p_context:
                ctx = ctx[:, -int(rng.choice(eligible)):]
            model.forecast(ctx, n=tgt.shape[1])
            y = model.robust_scaler.transform(tgt) if model.robust_scaler is not None else tgt
            y = ((y - model.revin.mean) / model.revin.std).squeeze(-1)                    # [B, h]
            pos = torch.as_tensor(rng.choice(tgt.shape[1], size=min(steps, tgt.shape[1]), replace=False),
                                  device=device)
            phi, y = grabbed["phi"][:, pos].float(), y[:, pos]
            ok = torch.isfinite(y)
            feats.append(phi[ok].cpu())
            targets.append(y[ok].cpu())
    finally:
        hook.remove()
    return torch.cat(feats), torch.cat(targets)


def main():
    from hydra import compose, initialize_config_dir
    from timejepa.data.datamodule import MultiDatasetMonashDataModule
    from timejepa.evaluation import create_model_from_config, load_checkpoint

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config-name", default="ssm_mini_v3_wide")
    ap.add_argument("--out", required=True, help="directory of the refit checkpoint (<stem>-refit.ckpt)")
    ap.add_argument("--rows", choices=["all", "median"], default="all")
    ap.add_argument("--lam", type=float, default=0.0, help="ridge penalty toward the trained weights")
    ap.add_argument("--per-dataset", type=int, default=1024, help="train windows per dataset")
    ap.add_argument("--val-per-dataset", type=int, default=256)
    ap.add_argument("--steps", type=int, default=16, help="horizon positions kept per batch")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--gb-per-gpu", type=float, default=15.0, help="feature memory per device")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    with initialize_config_dir(version_base=None, config_dir=str((HERE / "configs").resolve())):
        cfg = compose(config_name=args.config_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(create_model_from_config(cfg), args.checkpoint, device).to(device).eval()
    dm = MultiDatasetMonashDataModule(
        data_dir=cfg.data.data_dir, context_length=cfg.model.seq_length,
        prediction_length=cfg.model.prediction_length, datasets=cfg.data.get("datasets_finetune"),
        dataset_pattern=cfg.data.get("dataset_pattern", "*.npy"),
        combine_mode=cfg.data.get("combine_mode", "concatenate"),
        balanced_sampling=cfg.data.balanced_sampling, sampling_temperature=cfg.data.sampling_temperature,
        max_oversample_ratio=cfg.data.max_oversample_ratio, batch_size=args.batch_size,
        stride=cfg.data.stride, normalize_mode=cfg.data.normalize_mode,
        normalizer_type=cfg.data.normalizer_type, clip_outliers=cfg.data.clip_outliers,
        clip_sigma=cfg.data.clip_sigma, train_val_test_split=cfg.data.train_val_test_split,
        seed=cfg.data.seed, num_workers=0, use_mmap=bool(cfg.data.get("use_mmap", False)),
    )
    dm.prepare_data()
    dm.setup("fit")

    rng = np.random.default_rng(args.seed)
    lengths = list(cfg.training.get("context_lengths") or [])
    p_ctx = float(cfg.training.get("p_random_context_finetune", 0.0))
    common = dict(steps=args.steps, batch_size=args.batch_size, device=device,
                  context_lengths=lengths, p_context=p_ctx, rng=rng)
    phi, y = collect(model, dm.train_dataset, dm.train_dataset_sizes, args.per_dataset, **common)
    phi_val, y_val = collect(model, dm.val_dataset, dm.val_dataset_sizes, args.val_per_dataset, **common)
    logger.info(f"features: train {tuple(phi.shape)}, val {tuple(phi_val.shape)}")

    head = model.decoder.decoder
    proj = head.unpatching.projection
    w0, b0 = proj.weight.detach().float(), proj.bias.detach().float()
    devices = ([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
               if device.type == "cuda" else [device])
    per_device = int(args.gb_per_gpu * 1e9 / (phi.shape[1] * 4))
    chunks = spread(phi, y, devices, per_device, rng)
    logger.info(f"fit on {sum(len(c[1]) for c in chunks)} rows over {len(devices)} device(s), "
                f"{phi.shape[1] * w0.shape[0] + w0.shape[0]} parameters")
    del phi, y
    phi_val, y_val = phi_val.to(device), y_val.to(device)
    weight, bias, before, after = refit(head, chunks, w0, b0, lam=args.lam, rows=args.rows)
    mid = head.median_idx
    with torch.no_grad():
        val_before = float(head_loss(head, phi_val, y_val, w0, b0))
        val_after = float(head_loss(head, phi_val, y_val, weight, bias))
        mae_before = float((phi_val @ w0[mid] + b0[mid] - y_val).abs().mean())
        mae_after = float((phi_val @ weight[mid] + bias[mid] - y_val).abs().mean())
    logger.info(f"train pinball {before:.5f} -> {after:.5f} ({100 * (after / before - 1):+.2f}%)")
    logger.info(f"val   pinball {val_before:.5f} -> {val_after:.5f} ({100 * (val_after / val_before - 1):+.2f}%)")
    logger.info(f"val   median MAE {mae_before:.5f} -> {mae_after:.5f} ({100 * (mae_after / mae_before - 1):+.2f}%)")
    logger.info(f"relative weight change {float((weight - w0).norm() / w0.norm()):.4f}")
    if val_after >= val_before:
        logger.info("the validation pinball does not improve: no checkpoint written")
        return

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    keys = {s: [k for k in sd if k.endswith(f"{PROJECTION}.{s}")] for s in ("weight", "bias")}
    if any(len(v) != 1 for v in keys.values()):
        raise RuntimeError(f"expected one {PROJECTION} weight and bias in the checkpoint, found {keys}")
    sd[keys["weight"][0]] = weight.cpu().to(sd[keys["weight"][0]].dtype)
    sd[keys["bias"][0]] = bias.cpu().to(sd[keys["bias"][0]].dtype)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / (Path(args.checkpoint).stem + "-refit.ckpt")
    torch.save(ckpt, path)
    logger.info(f"written: {path}")


if __name__ == "__main__":
    main()
