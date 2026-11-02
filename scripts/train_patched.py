"""
Train PatchedSSM on corpus v4, one GPU.

    python scripts/train_patched.py --name p16 --steps 60000 --root /workspace/data/v4

Writes runs/<name>/{config.json, log.jsonl, last.pt}. dev_eval.py scores the
checkpoint on the held-out subsets at each --eval-every.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from timessm.corpus import WindowBatches
from timessm.patched import PatchedSSM

from dev_eval import dev_score


def pinball(q: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, levels: torch.Tensor):
    e = (y[..., None] - q) * mask[..., None]
    return (torch.maximum(levels * e, (levels - 1) * e).sum() / mask.sum().clamp_min(1)
            / len(levels))


def build(a) -> PatchedSSM:
    return PatchedSSM(input_length=a.context, patch=a.patch, d_model=a.d_model,
                      n_layers=a.layers, d_state=a.d_state, expand=a.expand, n_heads=a.heads,
                      attention=not a.no_attention, dropout=a.dropout, name=a.name)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--root", default="/workspace/data/v4")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--d-state", type=int, default=32)
    ap.add_argument("--expand", type=int, default=2)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--no-attention", action="store_true")
    ap.add_argument("--p-decimate", type=float, default=0.5)
    ap.add_argument("--p-flip", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=10000)
    ap.add_argument("--resume", default=None)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    run = Path(a.out) / a.name
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(vars(a), indent=1))
    device = torch.device("cuda")
    model = build(a).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{a.name}: {n_params / 1e6:.2f}M parameters", flush=True)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                             {"params": other, "weight_decay": 0.0}], lr=a.lr, fused=True)

    def schedule(step):
        if step < a.warmup:
            return (step + 1) / a.warmup
        t = (step - a.warmup) / max(1, a.steps - a.warmup)
        return 0.5 * (1 + math.cos(math.pi * t)) * (1 - 1e-3) + 1e-3

    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    start = 0
    if a.resume:
        state = torch.load(a.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start = state["step"]
    data = WindowBatches(a.root, a.batch, a.context, patch=a.patch, p_decimate=a.p_decimate,
                         p_flip=a.p_flip, seed=a.seed + start, temperature=a.temperature)
    loader = DataLoader(data, batch_size=None, num_workers=a.workers, prefetch_factor=4,
                        pin_memory=True, persistent_workers=True)
    levels = torch.tensor(model.quantile_levels, device=device)
    log = (run / "log.jsonl").open("a")
    t0, running, seen = time.time(), 0.0, 0
    model.train()
    for step, (x, y, w) in enumerate(loader, start):
        if step >= a.steps:
            break
        x, y, w = (t.to(device, non_blocking=True) for t in (x, y, w))
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
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step + 1, "config": vars(a)},
                       run / "last.pt")
            model.eval()
            score = dev_score(model, a.root)
            model.train()
            row = {"step": step + 1, "dev": score["all"], "per_subset": score}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()


if __name__ == "__main__":
    main()
