"""
Non-parametric probabilistic forecasting head.

Quantiles rather than a parametric distribution: GIFT-Eval ranks on CRPS,
approximated by the weighted quantile loss, which is an average of pinball
losses over a quantile grid. Predicting the grid directly optimizes the
ranking metric, imposes no distributional shape, and gives the exact
conditional median at q = 0.5 (the MAE-optimal point for MASE).

The head cross-attends to the context embeddings: two contexts with the same
conditional mean but different volatility would otherwise collapse to the
same predicted latent.

Quantile crossing is prevented by construction: the median is regressed
directly and the other levels are strictly positive widths accumulated
outward from it.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .patching import UnPatching


DEFAULT_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def pinball_loss(
    quantiles: torch.Tensor,
    target: torch.Tensor,
    levels: Sequence[float],
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Averaged pinball (quantile) loss.

        QL_q(y, y_hat) = 2 * max( q*(y - y_hat), (q-1)*(y - y_hat) )

    The factor 2 matches the GluonTS / GIFT-Eval convention, so a training
    loss and a reported WQL are on the same scale.

    Args:
        quantiles: [B, L, Q], sorted along Q (guaranteed by QuantileHead)
        target:    [B, L, 1] or [B, L]
        levels:    the Q quantile levels
        mask:      [B, L], True on real target positions
    """
    if target.ndim == quantiles.ndim - 1:
        target = target.unsqueeze(-1)
    elif target.shape[-1] == 1:
        pass
    else:
        target = target.unsqueeze(-1)

    q = torch.as_tensor(levels, device=quantiles.device, dtype=quantiles.dtype)
    diff = target - quantiles                      # [B, L, Q]
    loss = 2.0 * torch.maximum(q * diff, (q - 1.0) * diff)
    if mask is None:
        return loss.mean()
    # Positions past the real end of a short series are padded; average over
    # real ones.
    m = mask.to(loss.dtype).unsqueeze(-1)          # [B, L, 1] over the Q axis
    denom = torch.clamp(m.sum() * loss.shape[-1], min=1.0)
    return (loss * m).sum() / denom


class QuantileHead(nn.Module):
    """
    Predicted latents (+ optionally context embeddings) -> a quantile fan.

    Architecture:
        z_pred [B, N_tgt, D]
          -> cross-attention over z_ctx [B, N_ctx, D]   (if use_context)
          -> MLP
          -> UnPatching to [B, L, Q]                    (Q plays the channel role)
          -> monotone reparameterization
        quantiles [B, L, Q], sorted

    Args:
        d_model: model dimension
        patch_size / stride: must match the encoder's patching grid
        prediction_length: horizon in timesteps
        quantile_levels: the grid, defaults to GIFT-Eval's 9 levels
        use_context: cross-attend to the context embeddings
        num_heads: heads for the cross-attention
    """

    def __init__(
        self,
        d_model: int = 128,
        patch_size: int = 16,
        stride: int = 8,
        prediction_length: int = 128,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
        use_context: bool = True,
        num_heads: int = 4,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        # Optional modulation of the fan's spread by a residual signal z.
        # Off: the z_gate attribute does not exist and the state_dict is unchanged.
        use_error_signal: bool = False,
        z_dim: int = 4,
    ):
        super().__init__()

        self.quantile_levels = tuple(float(q) for q in quantile_levels)
        self.num_quantiles = len(self.quantile_levels)
        self.prediction_length = prediction_length
        self.use_context = use_context
        self.use_error_signal = use_error_signal
        self.stride = stride

        if not all(a < b for a, b in zip(self.quantile_levels, self.quantile_levels[1:])):
            raise ValueError(f"quantile_levels must be strictly increasing, got {self.quantile_levels}")

        # Index of the median. With an odd grid this is exact; with an even one
        # the anchor sits just below 0.5, which only shifts which level is
        # regressed directly and never breaks monotonicity.
        self.median_idx = min(
            range(self.num_quantiles),
            key=lambda i: abs(self.quantile_levels[i] - 0.5),
        )

        if use_context:
            self.context_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads,
                dropout=dropout, batch_first=True,
            )
            self.context_norm = nn.LayerNorm(d_model)

        if use_error_signal:
            # Multiplicative gate on _make_monotone's widths:
            # z [B, N, z_dim] -> (g_low, g_up) per patch -> widths * exp(g).
            # Zero init gives exp(0) = 1, an exact identity at start; the
            # median is untouched by construction.
            self.z_gate = nn.Linear(z_dim, 2)
            nn.init.zeros_(self.z_gate.weight)
            nn.init.zeros_(self.z_gate.bias)

        hidden_dim = hidden_dim or d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.mlp_norm = nn.LayerNorm(d_model)

        # Q takes the channel slot: UnPatching maps [B, N, D] -> [B, L, Q].
        self.unpatching = UnPatching(
            patch_size=patch_size,
            stride=stride,
            d_model=d_model,
            num_features=self.num_quantiles,
        )

    def forward(
        self,
        predicted_latents: torch.Tensor,
        context_embeddings: Optional[torch.Tensor] = None,
        target_length: Optional[int] = None,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns:
            quantiles [B, L, Q], strictly increasing along Q.

        `z` (optional): predicted residual stats [B, N, z_dim], modulates the
        spread through z_gate. Built with use_error_signal but no z, or z
        given without the gate, raises ValueError rather than degrading silently.
        """
        if self.use_error_signal and z is None:
            raise ValueError(
                "QuantileHead was built with use_error_signal=True but no z "
                "was passed. JEPATST.forward_finetune forwards it; a caller "
                "invoking the head directly must too."
            )
        if z is not None and not self.use_error_signal:
            raise ValueError(
                "z received but the head was built without use_error_signal "
                "- the gate does not exist, the modulation would be silently "
                "lost."
            )
        target_length = target_length or self.prediction_length
        h = predicted_latents

        if self.use_context:
            if context_embeddings is None:
                raise ValueError(
                    "QuantileHead was built with use_context=True but no "
                    "context_embeddings were passed. JEPATST.forward_finetune "
                    "forwards them; a caller invoking the head directly must too."
                )
            attended, _ = self.context_attn(
                query=h, key=context_embeddings, value=context_embeddings
            )
            h = self.context_norm(h + attended)

        h = self.mlp_norm(h + self.mlp(h))

        raw = self.unpatching(h, target_len=target_length)   # [B, L, Q]

        gates = None
        if z is not None:
            # z is per patch [B, N, z_dim]; the widths are per timestep
            # [B, L, .]. The mapping t -> min(t // stride, N - 1) is simpler
            # than reproducing UnPatching's overlap averaging and good enough
            # for a scale.
            g = self.z_gate(z)                               # [B, N, 2]
            idx = torch.div(
                torch.arange(raw.shape[1], device=raw.device),
                self.stride, rounding_mode='floor',
            ).clamp_max(g.shape[1] - 1)                      # [L]
            gates = g[:, idx, :]                             # [B, L, 2]

        return self._make_monotone(raw, gates)

    def _make_monotone(
        self, raw: torch.Tensor, gates: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Turn unconstrained outputs into a sorted quantile fan.

        The median is regressed directly; every other level is the median plus
        or minus a cumulative sum of softplus widths. Sorting is then a property
        of the parameterization rather than something to penalize or post-process.

        `gates` ([B, L, 2]): log-multiplicative factors (g_low, g_up) applied
        to the widths; exp(g) * softplus stays > 0, so monotonicity and the
        median are preserved.
        """
        mid = self.median_idx
        median = raw[..., mid:mid + 1]                       # [B, L, 1]

        parts = []

        if mid > 0:
            lower_w = F.softplus(raw[..., :mid])             # [B, L, mid]
            if gates is not None:
                lower_w = lower_w * torch.exp(gates[..., 0:1])
            # Accumulate outward from the median, then restore ascending order.
            lower = median - torch.cumsum(lower_w.flip(-1), dim=-1).flip(-1)
            parts.append(lower)

        parts.append(median)

        if mid < raw.shape[-1] - 1:
            upper_w = F.softplus(raw[..., mid + 1:])
            if gates is not None:
                upper_w = upper_w * torch.exp(gates[..., 1:2])
            upper = median + torch.cumsum(upper_w, dim=-1)
            parts.append(upper)

        return torch.cat(parts, dim=-1)

    def loss(self, quantiles: torch.Tensor, target: torch.Tensor,
             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return pinball_loss(quantiles, target, self.quantile_levels, mask=mask)

    def median(self, quantiles: torch.Tensor) -> torch.Tensor:
        """The MAE-optimal point forecast, for MASE and the plots."""
        return quantiles[..., self.median_idx:self.median_idx + 1]
