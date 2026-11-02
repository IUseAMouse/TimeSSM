"""
PatchedSSM: TimeSSM on patches, S4D blocks alternating with self-attention.

    x [B, L] (NaN = missing) -> mean/std of the observed points, then asinh
    -> tokens of `patch` steps (values and observed mask) -> [context ; future]
    -> blocks -> one patch of quantiles per token.

The rate knob survives patching. `w` scales Delta in the S4D blocks and the
positions of the rotary embedding, so one reference cycle spans the same model
time at any sampling rate. A missing token is read as zeros plus a mask, never
imputed.
"""

import math
from types import SimpleNamespace
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import GatedSSMBlock

Scale = Union[float, torch.Tensor]
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def rotate(x: torch.Tensor, pos: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """x [B, heads, N, dh], pos [B, N] (real-valued)."""
    angle = pos[:, None, :, None] * inv_freq
    cos, sin = angle.cos(), angle.sin()
    a, b = x.chunk(2, dim=-1)
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)


class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, expand: int = 4, dropout: float = 0.0):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, expand * d_model), nn.GELU(),
                                nn.Linear(expand * d_model, d_model))
        self.dropout = dropout
        inv_freq = 1000.0 ** (-torch.arange(0, self.d_head // 2).float() / (self.d_head // 2))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, N, 3, self.n_heads, self.d_head) \
            .permute(2, 0, 3, 1, 4).unbind(0)
        q, k = rotate(q, pos, self.inv_freq), rotate(k, pos, self.inv_freq)
        drop = self.dropout if self.training else 0.0
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :],
                                           dropout_p=drop)
        x = x + self.out(o.transpose(1, 2).reshape(B, N, D))
        return x + self.ff(self.ff_norm(x))


class PatchedSSM(nn.Module):
    rate_knob = "delta"
    handles_nan = True

    def __init__(
        self,
        input_length: int = 4096,
        prediction_length: int = 256,
        patch: int = 16,
        d_model: int = 256,
        n_layers: int = 3,
        d_state: int = 32,
        expand: int = 2,
        n_heads: int = 8,
        attention: bool = True,
        dropout: float = 0.0,
        quantile_levels: Sequence[float] = QUANTILES,
        delta_range: Sequence[float] = (1.0 / 64.0, 8.0),
        name: str = "timessm_patched",
    ):
        super().__init__()
        self.name = name
        self.input_length, self.prediction_length = int(input_length), int(prediction_length)
        self.patch = int(patch)
        self.delta_range = tuple(float(x) for x in delta_range)
        self.expects_frequency = True
        self.predictor = SimpleNamespace(w_film=None)
        self.quantile_levels = tuple(quantile_levels)
        self.embed = nn.Linear(2 * patch, d_model)
        self.embed_mix = nn.Sequential(nn.GELU(), nn.Linear(d_model, d_model))
        self.future_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.ssm = nn.ModuleList([
            GatedSSMBlock(d_model, d_state=d_state, expand=expand, dropout=dropout,
                          dt_min=1e-3 * patch, dt_max=1e-1 * patch)
            for _ in range(n_layers)])
        self.att = nn.ModuleList([
            AttentionBlock(d_model, n_heads, dropout=dropout) if attention else nn.Identity()
            for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                  nn.Linear(2 * d_model, patch * len(quantile_levels)))

    def _fan(self, raw: torch.Tensor) -> torch.Tensor:
        """Sorted quantiles: the median plus cumulated positive widths on each side."""
        mid = len(self.quantile_levels) // 2
        median = raw[..., mid:mid + 1]
        lower = median - F.softplus(raw[..., :mid]).flip(-1).cumsum(-1).flip(-1)
        upper = median + F.softplus(raw[..., mid + 1:]).cumsum(-1)
        return torch.cat([lower, median, upper], dim=-1)

    def forward_normalized(self, x: torch.Tensor, mask: torch.Tensor, n: int,
                           w: Scale = 1.0) -> torch.Tensor:
        """x, mask [B, L], L a multiple of the patch -> quantiles [B, n, Q], asinh frame."""
        B, L = x.shape
        p = self.patch
        n_past, n_future = L // p, -(-n // p)
        values = (x * mask).reshape(B, n_past, p)
        observed = mask.reshape(B, n_past, p)
        h = self.embed(torch.cat([values, observed], dim=-1))
        h = h + self.embed_mix(h)
        h = torch.cat([h, self.future_token.expand(B, n_future, -1)], dim=1)
        N = n_past + n_future
        valid = torch.cat([observed.any(-1), torch.ones(B, n_future, dtype=torch.bool,
                                                        device=x.device)], dim=1)
        scale = torch.as_tensor(w, dtype=h.dtype, device=h.device).reshape(-1, 1)
        pos = torch.arange(N, device=h.device, dtype=h.dtype)[None] * scale
        for ssm, att in zip(self.ssm, self.att):
            h = ssm(h, delta_scale=w)
            h = att(h, pos, valid) if isinstance(att, AttentionBlock) else h
        raw = self.head(self.final_norm(h[:, n_past:]))
        return self._fan(raw.reshape(B, n_future * p, -1)[:, :n])

    @staticmethod
    def normalize(x: torch.Tensor, mask: torch.Tensor):
        """x [B, L] with NaN, mask [B, L] -> (asinh frame, mean [B, 1], std [B, 1])."""
        x0 = torch.where(mask, x, torch.zeros_like(x)).double()
        count = mask.sum(1, keepdim=True).clamp_min(1)
        mean = x0.sum(1, keepdim=True) / count
        var = (mask * (x0 - mean) ** 2).sum(1, keepdim=True) / count
        std = var.sqrt() + 1e-5 + 1e-6 * mean.abs()
        return (torch.asinh((x0 - mean) / std) * mask).float(), mean, std

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, -self.input_length:]
        pad = -x.shape[1] % self.patch
        return F.pad(x, (pad, 0), value=float("nan"))

    def forecast(self, context: torch.Tensor, n: Optional[int] = None, w: Optional[Scale] = None,
                 **_) -> Dict[str, torch.Tensor]:
        """context [B, L] or [B, L, 1], raw scale."""
        n = int(n or self.prediction_length)
        x = self._pad(context.squeeze(-1) if context.dim() == 3 else context)
        mask = torch.isfinite(x)
        z, mean, std = self.normalize(x, mask)
        q = self.forward_normalized(z, mask, n, 1.0 if w is None else w)
        raw = (torch.sinh(q.clamp(-15.0, 15.0).double()) * std[:, None] + mean[:, None]).float()
        mid = len(self.quantile_levels) // 2
        return {"quantiles": q, "quantiles_denorm": raw, "quantile_levels": self.quantile_levels,
                "forecast": q[..., mid], "forecast_denorm": raw[..., mid]}

    def forward(self, context: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        return self.forecast(context, **kwargs)
