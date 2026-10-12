"""
Robust arcsinh scaling, composed around RevIN:

    input:  x' = arcsinh((x - median(ctx)) / MAD_scaled(ctx))
    output: x  = sinh(x') * MAD_scaled + median

RevIN, the rollout and the pinball loss operate in the compressed space; only
the `*_denorm` outputs go back to raw space through the inverse.

Why: a near-constant context gives the z-score a tiny scale and projects the
target to thousands of sigma, and a large spike inflates the std and crushes
the rest of the signal. The MAD ignores the spike and arcsinh is logarithmic
in the tail (Toto uses the same "robust arcsinh scaler"). arcsinh is strictly
monotonic, so quantiles stay ordered through the pointwise inverse and the
median commutes with it; MAD * 1.4826 is a consistent sigma estimator for a
Gaussian and arcsinh(z) ~ z for |z| < 1, so well-behaved windows are close to
the z-score.

No learned parameter; the statistics are runtime attributes. A marker buffer
is registered so that loading a checkpoint into a model with a different
scaling fails instead of giving wrong numbers.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# MAD -> equivalent standard deviation for a Gaussian.
MAD_TO_SIGMA = 1.4826


class RobustScale(nn.Module):
    """
    Per-instance, per-channel robust transform with context statistics.

    Same lifecycle as RevIN: `fit(context)` computes and stores (median,
    scale); `transform` / `inverse` reuse them, so the target and every output
    are handled in the context's frame (the target's own frame would leak the
    future).
    """

    # Scale fallback and floor. On a "flat + spikes" window the MAD is exactly
    # zero; a tiny scale offsets the compressed frame by ln(1 / scale) and the
    # sinh inverse re-amplifies that offset exponentially (float32 overflows
    # near sinh(89)). A too large scale is benign (arcsinh becomes near linear
    # and RevIN renormalizes), so the fallback errs on the large side, but it
    # must be conditional: an unconditional max(MAD, 0.1 * std) would let a
    # lone spike reinflate the scale on a healthy window.
    #   scale = MAD * 1.4826          if MAD * 1.4826 > 0.01 * std
    #         = max(0.1 * std, eps)   otherwise (MAD collapsed)
    # For a strictly constant context (std = 0) eps = 1e-3 bounds the frame
    # at arcsinh(X / 1e-3) ~ ln(2000 X).
    STD_FALLBACK = 0.1
    MAD_COLLAPSE_GATE = 0.01

    # Forecast envelope relative to the context. A half-trained quantile head
    # can emit a tail quantile |z| ~ 15 in compressed space, which sinh
    # re-amplifies into about 10^6 * scale; the floor cannot stop that, so the
    # guard sits on the inverse. A forecast does not leave
    #     [min(ctx) - K * w, max(ctx) + K * w],  w = max(range(ctx), scale)
    # with K = 10: far above any plausible future, far below sinh accidents
    # (Chronos bounds its vocabulary at +-15 sigma the same way). The clamp is
    # monotonic, so quantile order survives, and inactive on any reasonable
    # forecast. `w` falls back to the scale for a degenerate context (range 0).
    FORECAST_ENVELOPE = 10.0

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
        # Checkpoint marker (see the module docstring); no influence on the
        # computation.
        self.register_buffer("is_robust", torch.ones(1))
        self.median: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None
        self.ctx_min: torch.Tensor | None = None
        self.ctx_max: torch.Tensor | None = None

    def fit(self, context: torch.Tensor) -> None:
        """context: [B, L, C]; statistics over L, per instance and per channel."""
        med = context.median(dim=1, keepdim=True).values                # [B,1,C]
        mad = (context - med).abs().median(dim=1, keepdim=True).values  # [B,1,C]
        # A one-point context (a short history decimated by a large k) has no
        # unbiased std: torch returns NaN. Zero is the defined value; the eps
        # floor below then takes over.
        if context.shape[1] > 1:
            std = context.std(dim=1, keepdim=True)                      # [B,1,C]
        else:
            std = torch.zeros_like(med)
        mad_sigma = mad * MAD_TO_SIGMA
        fallback = (self.STD_FALLBACK * std).clamp_min(self.eps)
        self.median = med.detach()
        self.scale = torch.where(
            mad_sigma > self.MAD_COLLAPSE_GATE * std, mad_sigma, fallback
        ).clamp_min(self.eps).detach()
        # Context bounds for the forecast envelope (see FORECAST_ENVELOPE).
        self.ctx_min = context.amin(dim=1, keepdim=True).detach()
        self.ctx_max = context.amax(dim=1, keepdim=True).detach()

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.median is not None, "call fit(context) first"
        return torch.asinh((x - self.median) / self.scale)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pointwise inverse, bounded by the context envelope (see
        FORECAST_ENVELOPE). For a quantile tensor [B, n, Q] the monotonicity
        of sinh and of the clamp preserves the level order. Any value whose
        exact inverse falls inside the envelope is restored exactly.
        """
        assert self.median is not None, "call fit(context) first"
        med, scale = self.median, self.scale
        if x.dim() == med.dim() and x.shape[-1] != med.shape[-1]:
            # Quantile outputs [B, n, Q] against stats [B, 1, 1]: the
            # broadcast is already correct.
            pass
        raw = torch.sinh(x) * scale + med
        half = self.FORECAST_ENVELOPE * torch.maximum(self.ctx_max - self.ctx_min, scale)
        return raw.clamp(self.ctx_min - half, self.ctx_max + half)
