"""
Training module: TimeJEPA's FinetuneModule (pinball on the fan, AdamW,
cosine at the real budget, context-length randomization, target masks) plus
ONE thing the SSM needs - multi-rate training on the Delta knob.

Per training batch, with probability `p_delta_scale`, one factor is drawn
from `delta_scales` and passed as `w` to the forecast: the model sees the
same (context, target) pair at Delta * s. The target does not change - Delta
is a latent of rhythm, not a resampling (SSM_cours 6.3): the point is to make
every scale the eval will ask for (w = 1 / k, k up to 48) in-distribution,
the way TimeJEPA's multi-resolution decimation made decimated contexts
in-distribution for RateIN. Validation always runs at s = 1.

Witnesses `aug/delta_scale` and `aug/delta_neq1_frac` (the arm is not
proven active until a log shows it - the B5 lesson).

Random horizon INSIDE the fixed window (2026-09-26, plan B1). The datamodule
yields contiguous windows [context 1024 | target 256]; with probability
`p_random_horizon` the module re-splits the SAME 1280 steps as
[first 1280 - h | last h] for h drawn in `horizon_lengths`, then the usual
context crop applies. No corpus window is lost (the h512 arm of TimeJEPA
lost because it EXTENDED the window to 1536 and dropped the short chunks),
the autonomous rollout is trained up to 512 steps (42 of the 97 GIFT
configs ask for 480-900), and the head learns to calibrate (h512: exact
0.800 coverage). Validation stays at the native 256. Witnesses
`geometry/horizon_len`, `aug/horizon_neq_native_frac`.
"""

import random
from typing import Optional, Sequence

import torch

from timejepa.training.finetune_module import FinetuneModule


class SSMFinetuneModule(FinetuneModule):
    def __init__(self, model, *, delta_scales: Optional[Sequence[float]] = None,
                 p_delta_scale: float = 0.0,
                 horizon_lengths: Optional[Sequence[int]] = None,
                 p_random_horizon: float = 0.0, horizon_min_context: int = 256,
                 **kwargs):
        super().__init__(model, **kwargs)
        self.delta_scales = [float(s) for s in (delta_scales or [])]
        self.p_delta_scale = float(p_delta_scale)
        if self.p_delta_scale > 0 and not self.delta_scales:
            raise ValueError("p_delta_scale > 0 needs a non-empty delta_scales list")
        if any(s <= 0 for s in self.delta_scales):
            raise ValueError(f"delta_scales must be positive, got {self.delta_scales}")
        self._last_delta_scale = 1.0
        self.horizon_lengths = [int(h) for h in (horizon_lengths or [])]
        self.p_random_horizon = float(p_random_horizon)
        self.horizon_min_context = int(horizon_min_context)
        if self.p_random_horizon > 0 and not self.horizon_lengths:
            raise ValueError("p_random_horizon > 0 needs a non-empty horizon_lengths list")
        if any(h <= 0 for h in self.horizon_lengths):
            raise ValueError(f"horizon_lengths must be positive, got {self.horizon_lengths}")
        self._last_horizon = None

    # ------------------------------------------------------------ horizon
    def _maybe_resplit_horizon(self, batch: dict) -> dict:
        """Train only, once per batch: move the context/target split inside
        the SAME window. Returns a new dict (the caller's batch is untouched).
        A window whose padded target steps (target_mask False) would enter
        the context is left alone: padding must never be read as history."""
        context, target = batch["context"], batch["target"]
        native = target.shape[1]
        self._last_horizon = native
        if not self.training or self.p_random_horizon <= 0 or not self.horizon_lengths:
            return batch
        if random.random() >= self.p_random_horizon:
            return batch
        full = torch.cat([context, target], dim=1)
        T = full.shape[1]
        eligible = [h for h in self.horizon_lengths if h <= T - self.horizon_min_context]
        if not eligible:
            return batch
        h = random.choice(eligible)
        if h == native:
            return batch
        mask = batch.get("target_mask")
        new = dict(batch)
        if mask is not None:
            if h < native and not bool(mask[:, :native - h].all()):
                return batch
            if h >= native:
                ones = torch.ones(mask.shape[0], h - native, dtype=mask.dtype, device=mask.device)
                new["target_mask"] = torch.cat([ones, mask], dim=1)
            else:
                new["target_mask"] = mask[:, native - h:]
        new["context"], new["target"] = full[:, :T - h], full[:, T - h:]
        self._last_horizon = h
        return new

    def _draw_delta_scale(self, batch_size: int, device) -> Optional[torch.Tensor]:
        if not self.training or self.p_delta_scale <= 0:
            return None
        if random.random() >= self.p_delta_scale:
            return None
        s = random.choice(self.delta_scales)
        return torch.full((batch_size,), s, device=device)

    def _forward_and_loss(self, context, target, w=None, target_mask=None):
        """FinetuneModule._forward_and_loss for a horizon-free model: the
        parent calls forecast WITHOUT n (the SSM would fall back to its
        native 256 and the pinball would mismatch an h-step target). Same
        normalization chain as the parent (robust_scaler, then RevIN with the
        CONTEXT statistics set by forecast), pinball on the fan with the
        target mask; the JEPA-side terms are refused at construction."""
        if w is None:
            w = self._draw_delta_scale(context.shape[0], context.device)
        self._last_delta_scale = float(w.float().mean()) if w is not None else 1.0
        results = self.model.forecast(context, n=target.shape[1], w=w,
                                      return_representations=False)
        if getattr(self.model, "robust_scaler", None) is not None:
            target = self.model.robust_scaler.transform(target)
        if self.model.revin is not None:
            target = (target - self.model.revin.mean) / self.model.revin.std
        head = self.model.decoder.decoder
        loss = head.loss(results["quantiles"], target, mask=target_mask)
        self._last_anchor = None
        self._last_joint = None
        self._last_sigreg = None
        self._critic_stats = {}
        self._score_stats = {}
        return loss, results, target

    def on_load_checkpoint(self, checkpoint) -> None:
        """RevIN keeps the LAST batch's statistics in its `mean` / `std` buffers
        ([B, 1, 1] at save time, [1] in a fresh module), so a strict resume
        fails on a shape mismatch (10M run, 2026-09-16). They are per-forward
        quantities, recomputed from every context: the GIFT loader drops them
        (loading.py) and the resume replaces them with the fresh buffers."""
        sd = checkpoint.get("state_dict", {})
        own = self.state_dict()
        for key in list(sd):
            if key in own and key.endswith((".mean", ".std")) and ".revin." in key \
                    and sd[key].shape != own[key].shape:
                sd[key] = own[key].clone()

    def training_step(self, batch, batch_idx):
        batch = self._maybe_resplit_horizon(batch)
        loss = super().training_step(batch, batch_idx)
        native = int(self.model.prediction_length)
        self.log("geometry/horizon_len", float(self._last_horizon or native), on_step=True,
                 on_epoch=False, logger=True)
        self.log("aug/horizon_neq_native_frac", float((self._last_horizon or native) != native),
                 on_step=True, on_epoch=True, logger=True)
        self.log("aug/delta_scale", self._last_delta_scale, on_step=True,
                 on_epoch=False, logger=True)
        self.log("aug/delta_neq1_frac", float(self._last_delta_scale != 1.0),
                 on_step=True, on_epoch=True, logger=True)
        # Memory witness (2026-09-18): three processes died at their 111,111th
        # batch whatever the batch size and the seed. Peak since the last
        # reading, so a climb and a single-batch jump look different in W&B.
        if torch.cuda.is_available() and batch_idx % 200 == 0:
            dev = self.device
            self.log("mem/peak_allocated_gib", torch.cuda.max_memory_allocated(dev) / 2**30,
                     on_step=True, on_epoch=False, logger=True)
            self.log("mem/reserved_gib", torch.cuda.memory_reserved(dev) / 2**30,
                     on_step=True, on_epoch=False, logger=True)
            torch.cuda.reset_peak_memory_stats(dev)
        return loss
