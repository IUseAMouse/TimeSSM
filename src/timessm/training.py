"""
Training module: TimeJEPA's FinetuneModule (pinball on the fan, AdamW,
cosine schedule, context-length randomization, target masks) plus the
augmentations the SSM needs. Validation always runs native, at w = 1.

Multi-rate training. Per batch, with probability `p_delta_scale`, one factor
is drawn from `delta_scales` and passed as `w`: the model sees the same
(context, target) pair at Delta * s. The target does not change, since Delta
is a latent of rhythm rather than a resampling; the point is to make every
scale the evaluation asks for (w = 1 / k) in-distribution. Logged as
`aug/delta_scale` and `aug/delta_neq1_frac`.

Random horizon inside the window. With probability `p_random_horizon` the
contiguous window [context | target] is re-split as [first T - h | last h]
for h drawn in `horizon_lengths`, before the usual context crop. No window
is lost. Logged as `geometry/horizon_len`, `aug/horizon_neq_native_frac`.

Decimated window. With probability `p_decimation` a factor k is drawn from
`decimation_factors` and the same window is read decimated: the target is
the last P*k native steps averaged in blocks of k, the context the rest,
block-averaged and right-aligned like `timejepa.evaluation.ratein.decimate`.
Drawn per batch because the default collate needs one context length per
batch. Logged as `aug/decimation_k`, `aug/decimation_neq1_frac`,
`geometry/horizon_native`.

Frequency-tied Delta. With `delta_from_frequency`, w = BASE_SEASON / season
per item from the `season` batch key (see `_tie_delta`).
"""

import random
from typing import Optional, Sequence

import torch

from timejepa.training.finetune_module import FinetuneModule

from .frequency import BASE_SEASON


class SSMFinetuneModule(FinetuneModule):
    def __init__(self, model, *, delta_scales: Optional[Sequence[float]] = None,
                 p_delta_scale: float = 0.0,
                 horizon_lengths: Optional[Sequence[int]] = None,
                 p_random_horizon: float = 0.0, horizon_min_context: int = 256,
                 decimation_factors: Optional[Sequence[int]] = None,
                 p_decimation: float = 0.0, decimation_min_context: int = 128,
                 delta_from_frequency: bool = False,
                 **kwargs):
        super().__init__(model, **kwargs)
        self.delta_from_frequency = bool(delta_from_frequency)
        self._freq_stats = None
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
        self.decimation_factors = [int(k) for k in (decimation_factors or [])]
        self.p_decimation = float(p_decimation)
        self.decimation_min_context = int(decimation_min_context)
        if self.p_decimation > 0 and not self.decimation_factors:
            raise ValueError("p_decimation > 0 needs a non-empty decimation_factors list")
        if any(k < 1 for k in self.decimation_factors):
            raise ValueError(f"decimation_factors must be >= 1, got {self.decimation_factors}")
        self._last_k = 1

    # ------------------------------------------------------------ decimation
    @staticmethod
    def _block_mean(x: torch.Tensor, k: int) -> torch.Tensor:
        """[B, T, C] -> [B, T // k, C], mean over blocks of k, right-aligned
        (the left remainder is dropped): the tensor twin of `ratein.decimate`."""
        if k == 1:
            return x
        # The datamodule yields [B, T]; the parent's training_step adds the
        # channel axis later. Accept both.
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(-1)
        B, T, C = x.shape
        n = (T // k) * k
        out = x[:, T - n:].reshape(B, n // k, k, C).mean(dim=2)
        return out.squeeze(-1) if squeeze else out

    def _maybe_decimate(self, batch: dict) -> dict:
        """Train only, once per batch: read the window decimated by a drawn k.
        The target becomes the last P*k native steps in blocks of k (P
        decimated steps), the context the rest. A k whose decimated context
        would be shorter than `decimation_min_context` is not eligible.
        Returns a new dict; k = 1 leaves the batch untouched."""
        self._last_k = 1
        if not self.training or self.p_decimation <= 0 or not self.decimation_factors:
            return batch
        if random.random() >= self.p_decimation:
            return batch
        context, target = batch["context"], batch["target"]
        P = target.shape[1]
        T = context.shape[1] + P
        eligible = [k for k in self.decimation_factors
                    if (T - P * k) // k >= self.decimation_min_context]
        if not eligible:
            return batch
        k = random.choice(eligible)
        if k == 1:
            return batch
        full = torch.cat([context, target], dim=1)
        origin = T - P * k
        new = dict(batch)
        new["context"] = self._block_mean(full[:, :origin], k)
        new["target"] = self._block_mean(full[:, origin:], k)
        mask = batch.get("target_mask")
        if mask is not None:
            # The native mask covers the last P steps; the (k-1)*P steps taken
            # from the context are real. A decimated step is real only if
            # every native step of its block is.
            real = torch.ones(mask.shape[0], P * (k - 1), dtype=mask.dtype, device=mask.device)
            native = torch.cat([real, mask], dim=1)                   # [B, P*k]
            new["target_mask"] = native.reshape(mask.shape[0], P, k).all(dim=2)
        self._last_k = k
        return new

    # ------------------------------------------------------------ horizon
    def _maybe_resplit_horizon(self, batch: dict) -> dict:
        """Train only, once per batch: move the context/target split inside
        the same window. Returns a new dict (the caller's batch is untouched).
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
        """FinetuneModule._forward_and_loss with the horizon passed explicitly
        (the parent calls forecast without n, which would mismatch an h-step
        target). Same normalization chain as the parent (robust_scaler, then
        RevIN with the context statistics set by forecast), pinball on the fan
        with the target mask."""
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
        """RevIN keeps the last batch's statistics in its `mean` / `std` buffers
        ([B, 1, 1] at save time, [1] in a fresh module), so a strict resume
        would fail on a shape mismatch. They are recomputed from every
        context, so the resume replaces them with the fresh buffers."""
        sd = checkpoint.get("state_dict", {})
        own = self.state_dict()
        for key in list(sd):
            if key in own and key.endswith((".mean", ".std")) and ".revin." in key \
                    and sd[key].shape != own[key].shape:
                sd[key] = own[key].clone()

    # ------------------------------------------------------------ frequency
    def _tie_delta(self, batch: dict, k: int = 1) -> dict:
        """Delta tied to the declared frequency: w = BASE_SEASON / season per
        item, clamped to the model's `delta_range`. `season` is the batch key
        the datamodule emits with a frequency table (steps of the reference
        cycle; a window decimated by k has a cycle k times shorter). Items
        without a frequency (season 0) share one random draw in train and
        get w = 1 in eval. Off (default): the batch is returned unchanged
        and no random number is drawn."""
        self._freq_stats = None
        if not self.delta_from_frequency:
            return batch
        season = batch.get("season")
        if season is None:
            raise KeyError("delta_from_frequency needs the 'season' batch key: set "
                           "data.frequency_table (timejepa.data.frequency)")
        season = season.float() / k
        known = season > 0
        lo, hi = self.model.delta_range
        w = (BASE_SEASON / season.clamp_min(1e-6)).clamp(lo, hi)
        fallback = 1.0
        if self.training and not bool(known.all()):
            drawn = self._draw_delta_scale(1, season.device)
            fallback = 1.0 if drawn is None else float(drawn[0])
        w = torch.where(known, w, torch.full_like(w, fallback))
        self._freq_stats = (float(known.float().mean()), float(w.min()), float(w.max()),
                            float(torch.unique(w).numel()))
        new = dict(batch)
        new["w"] = w
        return new

    def validation_step(self, batch, batch_idx):
        return super().validation_step(self._tie_delta(batch), batch_idx)

    def test_step(self, batch, batch_idx):
        return super().test_step(self._tie_delta(batch), batch_idx)

    def training_step(self, batch, batch_idx):
        batch = self._maybe_resplit_horizon(batch)
        batch = self._maybe_decimate(batch)
        batch = self._tie_delta(batch, k=self._last_k)
        loss = super().training_step(batch, batch_idx)
        if self._freq_stats is not None:
            for name, value in zip(("labelled_frac", "w_min", "w_max", "n_unique"), self._freq_stats):
                self.log(f"freq/{name}", value, on_step=True, on_epoch=False, logger=True)
        native = int(self.model.prediction_length)
        self.log("aug/decimation_k", float(self._last_k), on_step=True, on_epoch=False, logger=True)
        self.log("aug/decimation_neq1_frac", float(self._last_k != 1), on_step=True,
                 on_epoch=True, logger=True)
        self.log("geometry/horizon_native", float(batch["target"].shape[1] * self._last_k),
                 on_step=True, on_epoch=False, logger=True)
        self.log("geometry/horizon_len", float(self._last_horizon or native), on_step=True,
                 on_epoch=False, logger=True)
        self.log("aug/horizon_neq_native_frac", float((self._last_horizon or native) != native),
                 on_step=True, on_epoch=True, logger=True)
        self.log("aug/delta_scale", self._last_delta_scale, on_step=True,
                 on_epoch=False, logger=True)
        self.log("aug/delta_neq1_frac", float(self._last_delta_scale != 1.0),
                 on_step=True, on_epoch=True, logger=True)
        # Memory log: peak since the last reading, so a slow climb and a
        # single-batch jump look different. Uses the module's own device
        # (a CPU module on a GPU machine made reset_peak_memory_stats raise).
        if self.device.type == "cuda" and batch_idx % 200 == 0:
            dev = self.device
            self.log("mem/peak_allocated_gib", torch.cuda.max_memory_allocated(dev) / 2**30,
                     on_step=True, on_epoch=False, logger=True)
            self.log("mem/reserved_gib", torch.cuda.memory_reserved(dev) / 2**30,
                     on_step=True, on_epoch=False, logger=True)
            torch.cuda.reset_peak_memory_stats(dev)
        return loss
