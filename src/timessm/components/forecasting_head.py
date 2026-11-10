"""
The quantile head behind the attribute layout of TimeJEPA's ForecastingHead, so a
checkpoint trained with the TimeJEPA classes loads as it is (keys `decoder.decoder.*`,
`decoder.output_norm.*`). SSMForecaster calls `decoder.decoder` directly.
"""

from typing import Optional, Sequence

import torch.nn as nn

from .quantile_head import DEFAULT_QUANTILES, QuantileHead


class ForecastingHead(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        patch_size: int = 16,
        stride: int = 8,
        prediction_length: int = 96,
        num_features: int = 1,
        revin: Optional[nn.Module] = None,
        quantile_levels: Optional[Sequence[float]] = None,
        quantile_use_context: bool = True,
        quantile_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.prediction_length = prediction_length
        self.revin = revin
        self.decoder = QuantileHead(
            d_model=d_model,
            patch_size=patch_size,
            stride=stride,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels or DEFAULT_QUANTILES,
            use_context=quantile_use_context,
            hidden_dim=quantile_hidden_dim,
        )
        # Unused at inference, kept because published checkpoints carry it.
        self.output_norm = nn.LayerNorm(num_features)
