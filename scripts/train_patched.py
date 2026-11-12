"""
Train PatchedSSM on one GPU.

    python scripts/train_patched.py --config configs/patched/base.yaml run.name=base

Writes <run.out>/<run.name>/{config.yaml, log.jsonl, <run.name>.ckpt}. The held-out
subsets are scored at each run.eval_every, GIFT-Eval never is.
"""

import argparse
import json
import math
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from timessm.corpus import DEV_SUBSETS, WindowBatches, train_subsets
from timessm.patched import TARGET, build_from_config, next_patch_targets

from dev_eval import dev_multi_score, dev_score


def pinball(q: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, levels: torch.Tensor):
    e = (y[..., None] - q) * mask[..., None]
    return (torch.maximum(levels * e, (levels - 1) * e).sum() / mask.sum().clamp_min(1)
            / len(levels))


def naive_loss(x: torch.Tensor, y: torch.Tensor, ymask: torch.Tensor, w: torch.Tensor,
               levels: torch.Tensor):
    """Pinball of the seasonal naive per item, raw units, its forecast copied at every level."""
    L, H = x.shape[1], y.shape[1]
    season = (24.0 / w).round().clamp(1, L).long()
    idx = L - season[:, None] + (torch.arange(H, device=x.device)[None] % season[:, None])
    last = torch.gather(torch.nan_to_num(x), 1, idx).double()
    e = (y.double().nan_to_num() - last) * ymask
    per = torch.maximum(e, 0 * e) * levels.mean() + torch.maximum(-e, 0 * e) * (1 - levels.mean())
    return per.sum(1) / ymask.sum(1).clamp_min(1)


def score_loss(q, center, scale, y, ymask, x, w, levels, kind):
    """Pinball in raw units per item, over the mean level (`ratio`) or as the log of its ratio
    to the seasonal naive (`logratio`), which targets a geometric mean of ratios."""
    raw = torch.sinh(q.double().clamp(-15, 15)) * scale[:, None] + center[:, None]
    e = (y.double().nan_to_num()[..., None] - raw) * ymask[..., None]
    per = torch.maximum(levels * e, (levels - 1) * e).mean(-1).sum(1)
    per = per / ymask.sum(1).clamp_min(1)
    if kind == "ratio":
        size = (y.double().nan_to_num().abs() * ymask).sum(1) / ymask.sum(1).clamp_min(1)
        return (per / (size + scale[:, 0] * 1e-3)).mean().float()
    base = naive_loss(x, y, ymask, w, levels)
    floor = 1e-3 * scale[:, 0]
    return torch.log((per + floor) / (base + floor)).mean().float()


def schedule(step: int, warmup: int, total: int, kind: str = "cosine", decay: float = 0.2) -> float:
    """Cosine to 1e-3, or warmup-stable-decay: flat, then linear to 1e-3 over the last
    `decay` share of the steps (a checkpoint of the flat part can be decayed on its own)."""
    if step < warmup:
        return (step + 1) / warmup
    if kind == "wsd":
        start = total - int(decay * total)
        return 1.0 if step < start else 1 - (1 - 1e-3) * (step - start) / max(1, total - start)
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * t)) * (1 - 1e-3) + 1e-3


def step_loss(model, x, y, w, levels, t, patch: int, passes=None, roles=None,
              valid=None) -> torch.Tensor:
    """One training loss: x [B, L] context, y [B, H] target, raw units, w [B] Delta scales.
    The forecast pinball in the asinh frame of the last context token (or `score_loss`), plus
    `t.context_loss` times the next-patch pinball of the context tokens.
    Multivariate batch (`roles` given): x [B, V, L], y [B, V, H], roles and valid [B, V]."""
    if roles is not None:
        return _step_loss_variates(model, x, y, w, levels, t, patch, passes, roles, valid)
    x = model._pad(x, patch)
    mask, ymask = torch.isfinite(x), torch.isfinite(y)
    if model.causal:
        z, center, scale = model.normalize_causal(x, mask, patch)
        mean, std = center[:, -1:], scale[:, -1:]
    else:
        z, mean, std = model.normalize(x, mask)
    target = torch.asinh((torch.where(ymask, y, torch.zeros_like(y)).double() - mean) / std)
    target = target.float().nan_to_num()
    stats = model.context_stats(x, mask, mean, std, z) if model.stats is not None else None
    lam = t.get("context_loss", 0.0)
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        out = model.forward_normalized(z, mask, y.shape[1], w, stats, patch,
                                       return_context=lam > 0, passes=passes)
    q, qc = out if lam > 0 else (out, None)
    if t.loss == "asinh":
        loss = pinball(q.float(), target, ymask, levels)
    else:
        loss = score_loss(q, mean, std, y, ymask, x, w, levels, t.loss)
    if qc is not None:
        ct, cm = next_patch_targets(x, mask, center, scale, patch, t.get("context_skip", 2))
        loss = loss + lam * pinball(qc[:, :-1].float().flatten(1, 2), ct, cm, levels)
    return loss


def _step_loss_variates(model, x, y, w, levels, t, patch, passes, roles, valid):
    """Rows normalized one by one (each in its own causal frame); the forecast pinball on
    the target rows only, the next-patch pinball on every valid row (covariates are series
    to model too); a future covariate's y is its known horizon, read by the model."""
    if t.loss != "asinh" or patch != model.patch:
        raise ValueError("multivariate batches: loss asinh and the model's patch size")
    B, V, H = y.shape
    xr, mask, z, center, scale, mean, std, wr, layout, fut = model.prepare_variates(
        x, w, roles, y, valid)
    yr = y.reshape(B * V, H)
    ymask = torch.isfinite(yr) & ((roles == TARGET) & layout.valid).reshape(-1, 1)
    target = torch.asinh((torch.where(ymask, yr, torch.zeros_like(yr)).double() - mean) / std)
    target = target.float().nan_to_num()
    stats = model.context_stats(xr, mask, mean, std, z) if model.stats is not None else None
    lam = t.get("context_loss", 0.0)
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        out = model.forward_normalized(z, mask, H, wr, stats, patch, return_context=lam > 0,
                                       passes=passes, layout=layout, future=fut)
    q, qc = out if lam > 0 else (out, None)
    loss = pinball(q.float(), target, ymask, levels)
    if qc is not None:
        ct, cm = next_patch_targets(xr, mask, center, scale, patch, t.get("context_skip", 2))
        cm = cm & layout.valid.reshape(-1, 1)
        loss = loss + lam * pinball(qc[:, :-1].float().flatten(1, 2), ct, cm, levels)
    return loss


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
    if cfg.training.get("context_loss", 0.0) > 0 and not model.causal:
        raise SystemExit("training.context_loss needs model.causal: the global frame and the "
                         "attention would show each context token its own target")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{cfg.run.name}: {n_params / 1e6:.2f}M ({n_params} parameters)", flush=True)
    if cfg.model.get("max_params") and n_params > cfg.model.max_params:
        raise SystemExit(f"{n_params} parameters, above model.max_params {cfg.model.max_params}")
    p_multi = cfg.data.get("p_multi", 0.0)
    if p_multi > 0 and not model.variate_layers:
        raise SystemExit("data.p_multi needs a model with variate_layers")
    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    t = cfg.training
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": t.weight_decay},
                             {"params": other, "weight_decay": 0.0}], lr=t.lr, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: schedule(s, t.warmup, t.steps, t.get("schedule", "cosine"), t.get("decay", 0.2)))
    start = 0
    if cfg.run.resume:
        state = torch.load(cfg.run.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["live"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start = state["step"]
    names = (train_subsets(cfg.data.root, multivariate=p_multi > 0)
             + (list(DEV_SUBSETS) if cfg.data.include_dev else []))
    # Windows at the longest context the model reads; each step cuts them to the
    # `max_tokens` tokens of the patch size drawn for it.
    data = WindowBatches(cfg.data.root, t.batch, model.input_length, max_horizon=t.max_horizon,
                         patch=model.patches[0], p_decimate=cfg.data.p_decimate,
                         p_flip=cfg.data.p_flip, p_synth=cfg.data.p_synth,
                         temperature=cfg.data.temperature, groups=cfg.data.groups,
                         seed=cfg.run.seed + start, names=names,
                         p_full=cfg.data.get("p_full", 0.0),
                         p_mix=cfg.data.get("p_mix", 0.0),
                         series_frac=cfg.data.get("series_frac", 1.0),
                         p_multi=p_multi, max_variates=cfg.data.get("max_variates", 16),
                         p_cov_synth=cfg.data.get("p_cov_synth", 0.5),
                         balance_rates=cfg.data.get("balance_rates", 0.0))
    loader = DataLoader(data, batch_size=None, num_workers=cfg.data.workers, prefetch_factor=4,
                        pin_memory=True, persistent_workers=True)
    levels = torch.tensor(model.levels, device=device)
    ema = None
    if t.ema > 0:
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(t.ema))
    if ema is not None and cfg.run.resume:
        ema.module.load_state_dict({k[len("model."):]: v for k, v in state["state_dict"].items()})
    log = (run / "log.jsonl").open("a")
    wb = None
    if cfg.get("wandb") and cfg.wandb.get("project"):
        import wandb
        # One W&B run per training run, resumed under the same id after a restart.
        id_file = run / "wandb_id.txt"
        run_id = id_file.read_text().strip() if id_file.exists() else uuid.uuid4().hex[:12]
        id_file.write_text(run_id)
        wb = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.get("entity"),
                        name=cfg.run.name, id=run_id, resume="allow",
                        tags=list(cfg.wandb.get("tags", [])),
                        config=OmegaConf.to_container(cfg, resolve=True))
    t0, running, seen = time.time(), 0.0, 0
    draw = np.random.default_rng(cfg.run.seed + 7919 * start)
    model.train()
    for step, batch in enumerate(loader, start):
        if step >= t.steps:
            break
        x, y, w, *variates = (v.to(device, non_blocking=True) for v in batch)
        patch = model.patches[draw.integers(len(model.patches))]
        # Random depth: each step trains the forecast after 1..loops passes, so that every
        # pass can be read at forecast time (model.halt).
        passes = int(draw.integers(1, model.loops + 1)) if t.get("random_passes") else None
        loss = step_loss(model, x, y, w, levels, t, patch, passes, *variates)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if ema is not None:
            ema.update_parameters(model)
        running += loss.item()
        seen += 1
        if (step + 1) % 200 == 0:
            row = {"step": step + 1, "loss": running / seen, "grad": float(grad),
                   "s_per_step": (time.time() - t0) / seen, "lr": sched.get_last_lr()[0]}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            if wb is not None:
                wb.log({f"train/{k}": v for k, v in row.items() if k != "step"}, step=step + 1)
            t0, running, seen = time.time(), 0.0, 0
        if (step + 1) % cfg.run.eval_every == 0 or step + 1 == t.steps:
            final = ema.module if ema is not None else model
            torch.save({"state_dict": {f"model.{k}": v for k, v in final.state_dict().items()},
                        "live": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step + 1},
                       run / f"{cfg.run.name}.ckpt")
            keep = cfg.run.get("keep_every")
            if keep and (step + 1) % keep == 0:
                # Plateau checkpoints of a WSD run, each one can be decayed on its own later.
                shutil.copyfile(run / f"{cfg.run.name}.ckpt", run / f"{cfg.run.name}_{step + 1}.ckpt")
            model.eval()
            final.eval()
            score = dev_score(final, cfg.data.root)
            row = {"step": step + 1, "dev": score["all"], "per_subset": score}
            if ema is not None:
                row["dev_live"] = dev_score(model, cfg.data.root)["all"]
            if final.variate_layers:
                multi = dev_multi_score(final, cfg.data.root)
                row["dev_multi"], row["dev_multi_single"] = (multi["joint"]["all"],
                                                             multi["single"]["all"])
                row["per_family_multi"] = multi
            if final.halt:
                # The same weights at the full number of passes, against the halted score.
                final.halt = False
                row["dev_full"] = dev_score(final, cfg.data.root)["all"]
                final.halt = True
            model.train()
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            if wb is not None:
                wb.log({"dev/all": row["dev"], **({"dev/live": row["dev_live"]} if "dev_live" in row
                                                  else {}),
                        **({"dev/full": row["dev_full"]} if "dev_full" in row else {}),
                        **({"dev/multi": row["dev_multi"],
                            "dev/multi_single": row["dev_multi_single"]}
                           if "dev_multi" in row else {}),
                        **{f"dev/{k}": v for k, v in score.items() if k != "all"}}, step=step + 1)
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
