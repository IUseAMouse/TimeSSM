"""
Reversible Instance Normalization (RevIN): normalize each instance with its
own statistics, denormalize at the output.

Paper: "Reversible Instance Normalization for Accurate Time-Series Forecasting
        against Distribution Shift" (ICLR 2022)
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Per-instance z-score over time, statistics kept for the way back.

    Args:
        num_features: number of channels
        eps: added to the variance
        affine: learnable scale and shift applied to the normalized context
    """

    def __init__(self, num_features: int = 1, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        # Statistics of the last normalized batch, reused for denormalization.
        self.register_buffer('mean', torch.zeros(1))
        self.register_buffer('std', torch.ones(1))

    def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        """x [B, L] or [B, L, C] -> normalized [B, L, C]; stores (mean, std)."""
        if mode != 'norm':
            raise ValueError(f"only mode='norm' is supported, got {mode!r}")
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        self.mean = x.mean(dim=1, keepdim=True).detach()                    # [B, 1, C]
        x = x - self.mean
        self.std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        x = x / self.std
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def denormalize_target_space(self, x: torch.Tensor) -> torch.Tensor:
        """
        Back to raw units from the target space `(y - mean) / std`, without the
        affine: the losses normalize the target as a plain z-score, so the decoder
        emits plain z-scores even when the context went through the affine.
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        return x * self.std + self.mean
