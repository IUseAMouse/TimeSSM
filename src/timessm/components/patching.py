"""
Patching for 1D sequences, as in PatchTST: [B, L] -> [B, num_patches, d_model].
"""

import torch
import torch.nn as nn
from typing import Optional


class Patching(nn.Module):
    """
    Convert time series into patches and embed them.
    
    Takes a time series of length L and converts it into num_patches patches,
    each of length patch_size. Then projects each patch to d_model dimensions.
    
    Args:
        patch_size: Number of timesteps in each patch
        d_model: Embedding dimension
        num_features: Number of input features/channels (default: 1 for univariate)
        stride: Stride for patching (if None, uses patch_size for non-overlapping)
        padding: Whether to pad the sequence to make it divisible by patch_size
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
        """
        Convert time series to patches.
        
        Args:
            x: Input tensor [B, L] or [B, L, C]
            
        Returns:
            Patched tensor [B, num_patches, d_model]
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B, L, 1]
        
        num_features = x.shape[-1]
        
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
        """
        Calculate number of patches for a given sequence length.
        
        Args:
            seq_len: Length of input sequence
            
        Returns:
            Number of patches
        """
        if self.padding:
            remainder = (seq_len - self.patch_size) % self.stride
            if remainder != 0:
                seq_len += (self.stride - remainder)
        
        return (seq_len - self.patch_size) // self.stride + 1


class PatchEmbedding(nn.Module):
    """
    Patching plus optional learned positional embeddings.

    Args:
        patch_size: Size of each patch
        d_model: Embedding dimension
        num_features: Number of input features
        max_patches: Maximum number of patches (for learned pos encoding)
        dropout: Dropout rate
        use_pos_encoding: Whether to add learned positional encoding
    """
    
    def __init__(
        self,
        patch_size: int,
        d_model: int,
        num_features: int = 1,
        max_patches: int = 512,
        dropout: float = 0.1,
        use_pos_encoding: bool = False
    ):
        super().__init__()
        
        self.patching = Patching(patch_size, d_model, num_features)
        self.dropout = nn.Dropout(dropout)
        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(1, max_patches, d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed patches.
        
        Args:
            x: Input [B, L] or [B, L, C]
            
        Returns:
            Embedded patches [B, num_patches, d_model]
        """
        x = self.patching(x)  # [B, num_patches, d_model]
        if self.use_pos_encoding:
            num_patches = x.shape[1]
            x = x + self.pos_encoding[:, :num_patches, :]
        x = self.dropout(x)
        
        return x


class UnPatching(nn.Module):
    """
    Convert patches back to a continuous time series.

    Args:
        patch_size: Size of each patch
        d_model: Embedding dimension
        num_features: Number of output features
        stride: Stride used in patching
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
        """
        Convert patches back to continuous sequence.
        
        Args:
            x: Patched tensor [B, num_patches, d_model]
            target_len: Target sequence length (if None, inferred from patches)
            
        Returns:
            Continuous sequence [B, L, C]
        """
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