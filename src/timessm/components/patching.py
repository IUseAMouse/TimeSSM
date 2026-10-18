"""
Patching for 1D sequences, as in PatchTST: [B, L] -> [B, num_patches, d_model],
and its inverse. TimeSSM uses patch 1 / stride 1: one token per step.
"""

from typing import Optional

import torch
import torch.nn as nn


class Patching(nn.Module):
    """
    Cut a series into patches of `patch_size` steps every `stride` steps
    (non-overlapping when stride is None) and project each patch to d_model.
    `padding` repeats the last value so that the last patch is complete.
    """

    def __init__(
        self,
        patch_size: int,
        d_model: int,
        num_features: int = 1,
        stride: Optional[int] = None,
        padding: bool = True
    ):
        super().__init__()

        self.patch_size = patch_size
        self.d_model = d_model
        self.num_features = num_features
        self.stride = stride if stride is not None else patch_size
        self.padding = padding

        self.projection = nn.Linear(patch_size * num_features, d_model)
        # Unused in forward; kept so existing checkpoints load unchanged.
        self.value_embedding = nn.Linear(num_features, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, L] or [B, L, C] -> [B, num_patches, d_model]."""
        batch_size, seq_len = x.shape[0], x.shape[1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B, L, 1]

        if self.padding:
            remainder = (seq_len - self.patch_size) % self.stride
            if remainder != 0:
                pad_len = self.stride - remainder
                # Pad at the end by repeating the last value.
                x = torch.cat([x, x[:, -1:, :].repeat(1, pad_len, 1)], dim=1)
                seq_len = x.shape[1]

        num_patches = (seq_len - self.patch_size) // self.stride + 1
        patches = x.unfold(dimension=1, size=self.patch_size, step=self.stride)  # [B, num_patches, C, patch_size]
        patches = patches.transpose(2, 3)  # [B, num_patches, patch_size, C]
        patches = patches.reshape(batch_size, num_patches, -1)  # [B, num_patches, patch_size * C]
        patches = self.projection(patches)  # [B, num_patches, d_model]

        return patches

    def get_num_patches(self, seq_len: int) -> int:
        """Number of patches for a series of length seq_len."""
        if self.padding:
            remainder = (seq_len - self.patch_size) % self.stride
            if remainder != 0:
                seq_len += (self.stride - remainder)

        return (seq_len - self.patch_size) // self.stride + 1


class UnPatching(nn.Module):
    """
    Project patches back to a series: [B, num_patches, d_model] ->
    [B, L, num_features], overlapping patches averaged.
    """

    def __init__(
        self,
        patch_size: int,
        d_model: int,
        num_features: int = 1,
        stride: Optional[int] = None
    ):
        super().__init__()

        self.patch_size = patch_size
        self.d_model = d_model
        self.num_features = num_features
        self.stride = stride if stride is not None else patch_size
        self.projection = nn.Linear(d_model, patch_size * num_features)

    def forward(
        self,
        x: torch.Tensor,
        target_len: Optional[int] = None
    ) -> torch.Tensor:
        """x [B, num_patches, d_model] -> [B, L, C], cut to target_len if given."""
        batch_size, num_patches, _ = x.shape
        x = self.projection(x)  # [B, num_patches, patch_size * C]
        x = x.reshape(batch_size, num_patches, self.patch_size, self.num_features)

        if self.stride == self.patch_size:
            x = x.reshape(batch_size, -1, self.num_features)  # [B, num_patches * patch_size, C]
        else:
            # Overlapping patches: average the overlapping regions.
            seq_len = (num_patches - 1) * self.stride + self.patch_size
            output = torch.zeros(
                batch_size, seq_len, self.num_features,
                device=x.device, dtype=x.dtype
            )
            counts = torch.zeros(seq_len, device=x.device, dtype=x.dtype)

            for i in range(num_patches):
                start = i * self.stride
                end = start + self.patch_size
                output[:, start:end, :] += x[:, i, :, :]
                counts[start:end] += 1

            x = output / counts.unsqueeze(0).unsqueeze(-1)

        if target_len is not None:
            x = x[:, :target_len, :]

        return x
