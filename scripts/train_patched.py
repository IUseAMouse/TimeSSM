"""
Train PatchedSSM on one GPU.

    python scripts/train_patched.py --config configs/patched/base.yaml run.name=base

Writes <run.out>/<run.name>/{config.yaml, log.jsonl, <run.name>.ckpt}. The held-out
subsets are scored at each run.eval_every, GIFT-Eval never is.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from timessm.corpus import DEV_SUBSETS, WindowBatches, train_subsets
from timessm.patched import build_from_config

from dev_eval import dev_score


def pinball(q: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, levels: torch.Tensor):
    e = (y[..., None] - q) * mask[..., None]
    return (torch.maximum(levels * e, (levels - 1) * e).sum() / mask.sum().clamp_min(1)
            / len(levels))


def schedule(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * t)) * (1 - 1e-3) + 1e-3


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(a.config), OmegaConf.from_dotlist(a.overrides))

    torch.manual_seed(cfg.run.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    run = Path(cfg.run.out) / cfg.run.name
    run.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run / "config.yaml")
    device = torch.device("cuda")
    model = build_from_config(cfg).to(device)
    print(f"{cfg.run.name}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    t = cfg.training
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": t.weight_decay},
                             {"params": other, "weight_decay": 0.0}], lr=t.lr, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: schedule(s, t.warmup, t.steps))
    start = 0
    if cfg.run.resume:
        state = torch.load(cfg.run.resume, map_location=device, weights_only=False)
        model.load_state_dict({k[len("model."):]: v for k, v in state["state_dict"].items()})
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start = state["step"]
    names = train_subsets(cfg.data.root) + (list(DEV_SUBSETS) if cfg.data.include_dev else [])
    data = WindowBatches(cfg.data.root, t.batch, cfg.model.seq_length, max_horizon=t.max_horizon,
                         patch=cfg.model.patch, p_decimate=cfg.data.p_decimate,
                         p_flip=cfg.data.p_flip, p_synth=cfg.data.p_synth,
                         temperature=cfg.data.temperature, seed=cfg.run.seed + start, names=names)
    loader = DataLoader(data, batch_size=None, num_workers=cfg.data.workers, prefetch_factor=4,
                        pin_memory=True, persistent_workers=True)
    levels = torch.tensor(model.quantile_levels, device=device)
    log = (run / "log.jsonl").open("a")
    t0, running, seen = time.time(), 0.0, 0
    model.train()
    for step, (x, y, w) in enumerate(loader, start):
        if step >= t.steps:
            break
        x, y, w = (v.to(device, non_blocking=True) for v in (x, y, w))
        x = model._pad(x)
        mask, ymask = torch.isfinite(x), torch.isfinite(y)
        z, mean, std = model.normalize(x, mask)
        target = torch.asinh((torch.where(ymask, y, torch.zeros_like(y)).double() - mean) / std)
        target = target.float().nan_to_num()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q = model.forward_normalized(z, mask, y.shape[1], w)
            loss = pinball(q.float(), target, ymask, levels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running += loss.item()
        seen += 1
        if (step + 1) % 200 == 0:
            row = {"step": step + 1, "loss": running / seen, "grad": float(grad),
                   "s_per_step": (time.time() - t0) / seen, "lr": sched.get_last_lr()[0]}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            t0, running, seen = time.time(), 0.0, 0
        if (step + 1) % cfg.run.eval_every == 0 or step + 1 == t.steps:
            torch.save({"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
                        "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step + 1},
                       run / f"{cfg.run.name}.ckpt")
            model.eval()
            score = dev_score(model, cfg.data.root)
            model.train()
            row = {"step": step + 1, "dev": score["all"], "per_subset": score}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()


if __name__ == "__main__":
    main()
