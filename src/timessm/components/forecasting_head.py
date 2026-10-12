"""
Decoder heads: representations [B, N, d_model] -> values [B, L_pred, C].
"""

import torch
import torch.nn as nn
from typing import Optional, Sequence, Tuple

from .patching import UnPatching


class LinearDecoder(nn.Module):
    """
    Simple linear decoder for forecasting.
    
    Projects patch-level representations back to time series values.
    Used during finetuning for generative tasks.
    
    Architecture:
        Patch representations [B, num_patches, d_model]
        -> Linear projection [B, num_patches, patch_size * C]
        -> Reshape to [B, L_pred, C]
    
    Args:
        d_model: Model dimension (512)
        patch_size: Size of each patch (16)
        stride: Gap between patches (8)
        prediction_length: Length of prediction horizon
        num_features: Number of output features/channels (1 for univariate)
        use_unpatching: Whether to use UnPatching layer
    """
    
    def __init__(
        self,
        d_model: int = 512,
        patch_size: int = 16,
        stride: int = 8,
        prediction_length: int = 96,
        num_features: int = 1,
        use_unpatching: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.patch_size = patch_size
        self.stride = stride
        self.prediction_length = prediction_length
        self.num_features = num_features
        self.use_unpatching = use_unpatching
        
        if use_unpatching:
            self.unpatching = UnPatching(
                patch_size=patch_size,
                stride=stride,
                d_model=d_model,
                num_features=num_features
            )
        else:
            self.projection = nn.Linear(d_model, patch_size * num_features)
    
    def forward(
        self,
        x: torch.Tensor,
        target_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Decode representations to time series values.
        
        Args:
            x: Patch representations [B, num_patches, d_model]
            target_length: Target sequence length (default: prediction_length)
            
        Returns:
            Time series predictions [B, L_pred, C]
        """
        if target_length is None:
            target_length = self.prediction_length
        
        if self.use_unpatching:
            output = self.unpatching(x, target_len=target_length)
        else:
            batch_size, num_patches, _ = x.shape
            x = self.projection(x)  # [B, num_patches, patch_size * C]
            x = x.reshape(batch_size, num_patches * self.patch_size, self.num_features)
            output = x[:, :target_length, :]
        
        return output


class MLPDecoder(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        patch_size: int = 16,
        stride: int = 8,
        prediction_length: int = 96,
        num_features: int = 1,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.patch_size = patch_size
        self.stride = stride
        self.prediction_length = prediction_length
        self.num_features = num_features
        
        hidden_dim = hidden_dim or d_model
        
        # The MLP keeps d_model as output; UnPatching does the final projection.
        layers = []
        for i in range(num_layers):
            in_dim = d_model if i == 0 else hidden_dim
            out_dim = d_model if i == num_layers - 1 else hidden_dim
            
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        
        self.mlp = nn.Sequential(*layers)
        self.unpatching = UnPatching(
            patch_size=patch_size,
            stride=stride,
            d_model=d_model,
            num_features=num_features
        )
    
    def forward(
        self,
        x: torch.Tensor,
        target_length: Optional[int] = None
    ) -> torch.Tensor:
        if target_length is None:
            target_length = self.prediction_length
        
        x = self.mlp(x)  # [B, num_patches, d_model]
        output = self.unpatching(x, target_len=target_length)
        
        return output


class AttentiveDecoder(nn.Module):
    """
    Cross-attention between learnable query embeddings (one per output step)
    and patch representations, as in Perceiver.

    Args:
        d_model: Model dimension
        prediction_length: Prediction horizon
        num_features: Number of output features
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 512,
        prediction_length: int = 96,
        num_features: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.prediction_length = prediction_length
        self.num_features = num_features
        
        self.query_embeddings = nn.Parameter(
            torch.randn(1, prediction_length, d_model) * 0.02
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, num_features)
    
    def forward(
        self,
        x: torch.Tensor,
        target_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Decode with cross-attention.
        
        Args:
            x: Patch representations [B, num_patches, d_model] (keys/values)
            target_length: Target length
            
        Returns:
            Predictions [B, L_pred, C]
        """
        if target_length is None:
            target_length = self.prediction_length
        
        batch_size = x.shape[0]
        queries = self.query_embeddings[:, :target_length, :].expand(batch_size, -1, -1)
        attended, _ = self.cross_attention(
            query=queries,
            key=x,
            value=x
        )
        attended = self.norm(attended)
        output = self.output_proj(attended)  # [B, L_pred, C]
        
        return output


class ForecastingHead(nn.Module):
    """
    Wraps the decoder types behind one interface and handles denormalization
    through RevIN.

    Args:
        d_model: Model dimension
        patch_size: Patch size
        prediction_length: Prediction horizon
        num_features: Number of features
        decoder_type: Type of decoder ('linear', 'mlp', 'attentive', 'quantile')
        revin: Optional RevIN layer for denormalization
        error_signal: Forwarded to QuantileHead (spread gate on a residual signal)
        quantile_hidden_dim: MLP width of the quantile head (None: d_model)
    """
    
    def __init__(
        self,
        d_model: int = 512,
        patch_size: int = 16,
        stride: int = 8,
        prediction_length: int = 96,
        num_features: int = 1,
        decoder_type: str = 'linear',
        revin: Optional[nn.Module] = None,
        quantile_levels: Optional[Sequence[float]] = None,
        quantile_use_context: bool = True,
        error_signal: bool = False,
        quantile_hidden_dim: 'Optional[int]' = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.prediction_length = prediction_length
        self.decoder_type = decoder_type
        self.revin = revin

        if decoder_type == 'linear':
            self.decoder = LinearDecoder(
                d_model=d_model,
                patch_size=patch_size,
                stride=stride,
                prediction_length=prediction_length,
                num_features=num_features
            )
        elif decoder_type == 'mlp':
            self.decoder = MLPDecoder(
                d_model=d_model,
                patch_size=patch_size,
                stride=stride,
                prediction_length=prediction_length,
                num_features=num_features
            )
        elif decoder_type == 'attentive':
            self.decoder = AttentiveDecoder(
                d_model=d_model,
                prediction_length=prediction_length,
                num_features=num_features
            )
        elif decoder_type == 'quantile':
            # Emits a sorted quantile fan and optionally cross-attends to the
            # context embeddings.
            from .quantile_head import QuantileHead, DEFAULT_QUANTILES
            self.decoder = QuantileHead(
                d_model=d_model,
                patch_size=patch_size,
                stride=stride,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels or DEFAULT_QUANTILES,
                use_context=quantile_use_context,
                use_error_signal=error_signal,
                hidden_dim=quantile_hidden_dim,
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

        if error_signal and decoder_type != 'quantile':
            # A point decoder would silently ignore z.
            raise ValueError(
                f"error_signal=True requires decoder_type='quantile' (the z "
                f"path modulates the fan's spread) - got '{decoder_type}'."
            )

        self.output_norm = nn.LayerNorm(num_features)

    @property
    def is_probabilistic(self) -> bool:
        return self.decoder_type == 'quantile'

    def forward(
        self,
        x: torch.Tensor,
        skip_revin: bool = False,
        context_embeddings: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate forecasts.

        Args:
            x: Representations [B, num_patches, d_model]
            skip_revin: Return the normalized forecast unchanged as the "denorm" one
            context_embeddings: Encoder output [B, N_ctx, d_model]. Only the
                quantile head consumes it; the point decoders ignore it, so the
                caller can always pass it.
            z: Predicted residual stats [B, N, z_dim]. Unlike
                context_embeddings, a z given to a point decoder is refused,
                since losing it would be silent.

        Returns:
            For point decoders: (forecast [B, L, C], forecast_denorm [B, L, C])
            For the quantile head: (quantiles [B, L, Q], quantiles_denorm [B, L, Q]);
            the caller extracts the median.
        """
        if self.decoder_type == 'quantile':
            predictions = self.decoder(x, context_embeddings=context_embeddings, z=z)
        else:
            if z is not None:
                raise ValueError(
                    "z (ESJEPA) received by a point decoder - the z path "
                    "modulates a quantile fan, decoder_type='quantile' "
                    "required."
                )
            predictions = self.decoder(x)

        if skip_revin or self.revin is None:
            predictions_denorm = predictions
        else:
            # `denormalize_target_space` rather than `mode='denorm'`: the
            # decoder is trained against a plain z-scored target (no RevIN
            # affine), so the affine inverse in `_denormalize` would introduce
            # a scale/offset error.
            predictions_denorm = self.revin.denormalize_target_space(predictions)

        return predictions, predictions_denorm