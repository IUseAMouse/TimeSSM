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
"""

import random
from typing import Optional, Sequence

import torch

from timejepa.training.finetune_module import FinetuneModule


class SSMFinetuneModule(FinetuneModule):
    def __init__(self, model, *, delta_scales: Optional[Sequence[float]] = None,
                 p_delta_scale: float = 0.0, **kwargs):
        super().__init__(model, **kwargs)
        self.delta_scales = [float(s) for s in (delta_scales or [])]
        self.p_delta_scale = float(p_delta_scale)
        if self.p_delta_scale > 0 and not self.delta_scales:
            raise ValueError("p_delta_scale > 0 needs a non-empty delta_scales list")
        if any(s <= 0 for s in self.delta_scales):
            raise ValueError(f"delta_scales must be positive, got {self.delta_scales}")
        self._last_delta_scale = 1.0

    def _draw_delta_scale(self, batch_size: int, device) -> Optional[torch.Tensor]:
        if not self.training or self.p_delta_scale <= 0:
            return None
        if random.random() >= self.p_delta_scale:
            return None
        s = random.choice(self.delta_scales)
        return torch.full((batch_size,), s, device=device)

    def _forward_and_loss(self, context, target, w=None, target_mask=None):
        if w is None:
            w = self._draw_delta_scale(context.shape[0], context.device)
        self._last_delta_scale = float(w.float().mean()) if w is not None else 1.0
        return super()._forward_and_loss(context, target, w=w, target_mask=target_mask)

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
        loss = super().training_step(batch, batch_idx)
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
