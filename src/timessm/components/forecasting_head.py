"""
The forecasting head: representations [B, N, d_model] -> a quantile fan
[B, L_pred, Q], with the denormalization through RevIN.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn

from .quantile_head import DEFAULT_QUANTILES, QuantileHead


class ForecastingHead(nn.Module):
    """
    Holds the quantile head as `.decoder` (the attribute name of TimeJEPA's
    ForecastingHead, so a checkpoint trained with it loads as it is) and
    denormalizes its output through RevIN. Only `decoder_type='quantile'`
    is shipped.

    Args:
        revin: RevIN layer used for denormalization (None: no denormalization)
        quantile_hidden_dim: MLP width of the quantile head (None: d_model)
    """

    def __init__(
        self,
        d_model: int = 512,
        patch_size: int = 16,
        stride: int = 8,
        prediction_length: int = 96,
        num_features: int = 1,
        decoder_type: str = 'quantile',
        revin: Optional[nn.Module] = None,
        quantile_levels: Optional[Sequence[float]] = None,
        quantile_use_context: bool = True,
        quantile_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        if decoder_type != 'quantile':
            raise ValueError(f"Unknown decoder_type: {decoder_type} (only 'quantile' is shipped)")

        self.d_model = d_model
        self.prediction_length = prediction_length
        self.decoder_type = decoder_type
        self.revin = revin
        # Sorted quantile fan, optionally cross-attending to the context embeddings.
        self.decoder = QuantileHead(
            d_model=d_model,
            patch_size=patch_size,
            stride=stride,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels or DEFAULT_QUANTILES,
            use_context=quantile_use_context,
            hidden_dim=quantile_hidden_dim,
        )
        # Unused in forward; kept so existing checkpoints load unchanged.
        self.output_norm = nn.LayerNorm(num_features)

    @property
    def is_probabilistic(self) -> bool:
        return True

    def forward(
        self,
        x: torch.Tensor,
        skip_revin: bool = False,
        context_embeddings: Optional[torch.Tensor] = None,
    ):
        """x [B, N, d_model] -> (quantiles [B, L, Q], quantiles_denorm [B, L, Q]);
        the caller extracts the median. `skip_revin` returns the normalized
        fan unchanged as the denormalized one."""
        predictions = self.decoder(x, context_embeddings=context_embeddings)
        if skip_revin or self.revin is None:
            return predictions, predictions
        # `denormalize_target_space` rather than `mode='denorm'`: the head is
        # trained against a plain z-scored target (no RevIN affine), so the
        # affine inverse in `_denormalize` would introduce a scale/offset error.
        return predictions, self.revin.denormalize_target_space(predictions)
