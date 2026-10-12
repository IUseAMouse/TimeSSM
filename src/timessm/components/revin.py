"""
Reversible Instance Normalization (RevIN): normalize each instance with its
own statistics, denormalize at the output.

Paper: "Reversible Instance Normalization for Accurate Time-Series Forecasting
        against Distribution Shift" (ICLR 2022)
"""

import torch
import torch.nn as nn
from typing import Optional


class RevIN(nn.Module):
    """
    Reversible Instance Normalization layer.
    
    Normalizes each time series instance independently using its own
    mean and standard deviation, then stores these statistics to 
    reverse the normalization at the output.
    
    Args:
        num_features: Number of features (channels) in the input
        eps: Small constant for numerical stability
        affine: If True, learns affine parameters (scale and shift)
        subtract_last: If True, subtract the last value instead of mean
                      (useful for some forecasting tasks)
    """
    
    def __init__(
        self,
        num_features: int = 1,
        eps: float = 1e-5,
        affine: bool = False,
        subtract_last: bool = False
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        
        # Statistics of the last normalized batch, reused for denormalization.
        self.register_buffer('mean', torch.zeros(1))
        self.register_buffer('std', torch.ones(1))

        # When frozen, `_normalize` reuses the stored mean/std instead of
        # recomputing them, so every roll of a rolling forecast is
        # denormalized with the statistics of the original context.
        self._frozen = False

    def freeze(self):
        """Pin the current (mean, std) so subsequent `_normalize` calls reuse them."""
        self._frozen = True
        return self

    def unfreeze(self):
        """Resume recomputing (mean, std) on every `_normalize` call."""
        self._frozen = False
        return self

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def forward(
        self,
        x: torch.Tensor,
        mode: str = 'norm'
    ) -> torch.Tensor:
        """
        Forward pass: normalize or denormalize
        
        Args:
            x: Input tensor of shape [B, L, C] or [B, L]
            mode: 'norm' for normalization, 'denorm' for denormalization
            
        Returns:
            Normalized or denormalized tensor
        """
        if mode == 'norm':
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise ValueError(f"mode should be 'norm' or 'denorm', got {mode}")
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input.
        
        Args:
            x: Input tensor [B, L] or [B, L, C]
            
        Returns:
            Normalized tensor with same shape
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B, L, 1]

        if self._frozen:
            x = (x - self.mean) / self.std
        else:
            # Statistics per instance, over the time dimension.
            if self.subtract_last:
                self.mean = x[:, -1:, :].detach()  # [B, 1, C]
            else:
                self.mean = x.mean(dim=1, keepdim=True).detach()  # [B, 1, C]

            x = x - self.mean
            self.std = torch.sqrt(
                x.var(dim=1, keepdim=True, unbiased=False) + self.eps
            ).detach()  # [B, 1, C]

            x = x / self.std

        if self.affine:
            x = x * self.affine_weight + self.affine_bias

        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize the output using stored statistics.

        Args:
            x: Normalized tensor [B, L] or [B, L, C]

        Returns:
            Denormalized tensor with same shape
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)

        x = x * self.std + self.mean

        return x

    def denormalize_target_space(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize a forecast that lives in the target space `(y - mean) / std`,
        without the learnable affine.

        The training losses normalize the target as a plain z-score while
        `_normalize` also applies `* w + b` to the context, so the decoder emits
        plain z-scores and `_denormalize` would undo an affine that was never
        applied. `_denormalize` is kept for callers that round-trip through
        `_normalize` (context reconstruction).
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        return x * self.std + self.mean

    def to_input_frame(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map a tensor from the target space `(y - mean) / std` into the encoder
        input space `((y - mean) / std) * w + b`, so that decoder outputs can be
        appended to a context buffer that went through `_normalize`.
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if self.affine:
            return x * self.affine_weight + self.affine_bias
        return x


class RevINMultivariate(nn.Module):
    """
    RevIN for multivariate time series.
    
    Applies RevIN independently to each feature/channel.
    
    Args:
        num_features: Number of features (channels)
        eps: Small constant for numerical stability
        affine: If True, learns separate affine params for each feature
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))
        
        self.register_buffer('mean', torch.zeros(1, 1, num_features))
        self.register_buffer('std', torch.ones(1, 1, num_features))
    
    def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, L, C]
            mode: 'norm' or 'denorm'
            
        Returns:
            Processed tensor [B, L, C]
        """
        if mode == 'norm':
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise ValueError(f"mode should be 'norm' or 'denorm', got {mode}")
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize each feature independently. x: [B, L, C]."""
        self.mean = x.mean(dim=1, keepdim=True).detach()  # [B, 1, C]
        self.std = torch.sqrt(
            x.var(dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()  # [B, 1, C]
        
        x = (x - self.mean) / self.std
        
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        
        return x
    
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize using stored statistics."""
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        
        x = x * self.std + self.mean
        return x