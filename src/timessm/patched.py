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
N_STATS = 5


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
    core_prefixes = ("ssm.", "att.", "embed", "head.", "future_token", "final_norm.")

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
        robust_norm: bool = False,
        stats: bool = False,
        patches: Optional[Sequence[int]] = None,
        name: str = "timessm_patched",
    ):
        super().__init__()
        self.name = name
        self.input_length, self.prediction_length = int(input_length), int(prediction_length)
        self.patch = int(patch)
        # Patch sizes the model can read, `patch` among them. A context is cut to
        # `max_tokens` tokens whatever the size, so a size trades resolution for history.
        self.patches = tuple(sorted({self.patch, *(int(q) for q in (patches or ()))}))
        self.max_tokens = self.input_length // self.patch
        self.input_length = self.max_tokens * self.patches[-1]
        self.robust_norm = bool(robust_norm)
        self.delta_range = tuple(float(x) for x in delta_range)
        self.expects_frequency = True
        self.predictor = SimpleNamespace(w_film=None)
        # The harness sees raw steps: the model pads to whole patches itself.
        self.patching = SimpleNamespace(patch_size=1, stride=1)
        self.quantile_levels = tuple(quantile_levels)
        self.embed = nn.Linear(2 * patch, d_model)
        self.embed_mix = nn.Sequential(nn.GELU(), nn.Linear(d_model, d_model))
        self.future_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.ssm = nn.ModuleList([
            GatedSSMBlock(d_model, d_state=d_state, expand=expand, dropout=dropout,
                          dt_min=1e-3 * patch, dt_max=1e-1 * patch)
            for _ in range(n_layers)])
        for block in self.ssm:
            block.ssm.batched_kernels = True
        self.att = nn.ModuleList([
            AttentionBlock(d_model, n_heads, dropout=dropout) if attention else nn.Identity()
            for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                  nn.Linear(2 * d_model, patch * len(quantile_levels)))
        self.stats = None
        if stats:
            # Added to every token; the last layer starts at zero, so the model starts as
            # the one without statistics.
            self.stats = nn.Sequential(nn.Linear(N_STATS, d_model), nn.GELU(),
                                       nn.Linear(d_model, d_model))
            nn.init.zeros_(self.stats[-1].weight)
            nn.init.zeros_(self.stats[-1].bias)
            self.core_prefixes = self.core_prefixes + ("stats.",)
        # One embedding and one last head layer per extra size; `patch` keeps embed/head.
        extra = [q for q in self.patches if q != self.patch]
        self.embed_size = nn.ModuleDict({str(q): nn.Linear(2 * q, d_model) for q in extra})
        self.head_size = nn.ModuleDict({str(q): nn.Linear(2 * d_model, q * len(quantile_levels))
                                        for q in extra})
        if extra:
            self.core_prefixes = self.core_prefixes + ("embed_size.", "head_size.")

    def _fan(self, raw: torch.Tensor) -> torch.Tensor:
        """Sorted quantiles: the median plus cumulated positive widths on each side."""
        mid = len(self.quantile_levels) // 2
        median = raw[..., mid:mid + 1]
        lower = median - F.softplus(raw[..., :mid]).flip(-1).cumsum(-1).flip(-1)
        upper = median + F.softplus(raw[..., mid + 1:]).cumsum(-1)
        return torch.cat([lower, median, upper], dim=-1)

    def context_stats(self, x: torch.Tensor, mask: torch.Tensor, center: torch.Tensor,
                      scale: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """What the normalization removes, unchanged by a change of units: share of exact
        zeros and of missing points, asinh(center / scale), last normalized value, log of
        the observed length. [B, N_STATS]."""
        count = mask.sum(1, keepdim=True).clamp_min(1)
        zeros = ((x == 0) & mask).sum(1, keepdim=True) / count
        missing = 1 - mask.float().mean(1, keepdim=True)
        level = torch.asinh(center / scale)
        last = z[:, -1:]
        length = torch.log(count.float()) / math.log(self.input_length)
        return torch.cat([zeros.float(), missing, level.float(), last.float(), length], dim=1)

    def choose_patch(self, mask: torch.Tensor) -> torch.Tensor:
        """Per item, the smallest size whose `max_tokens` tokens cover the observed history."""
        L = mask.shape[1]
        first = torch.where(mask.any(1), mask.float().argmax(1), torch.full_like(mask[:, 0], L,
                                                                              dtype=torch.long))
        span = L - first
        sizes = torch.tensor(self.patches, device=mask.device)
        fits = span[:, None] <= self.max_tokens * sizes[None]
        return torch.where(fits.any(1), sizes[fits.float().argmax(1)], sizes[-1])

    def forward_normalized(self, x: torch.Tensor, mask: torch.Tensor, n: int,
                           w: Scale = 1.0, stats: Optional[torch.Tensor] = None,
                           patch: Optional[int] = None) -> torch.Tensor:
        """x, mask [B, L], L a multiple of the patch -> quantiles [B, n, Q], asinh frame.
        `stats` [B, N_STATS] from context_stats, required when the model was built with them.
        A token spans `patch` steps: Delta and the rotary positions scale with patch / self.patch,
        the same knob as w."""
        B, L = x.shape
        p = int(patch or self.patch)
        n_past, n_future = L // p, -(-n // p)
        values = (x * mask).reshape(B, n_past, p)
        observed = mask.reshape(B, n_past, p)
        embed = self.embed if p == self.patch else self.embed_size[str(p)]
        h = embed(torch.cat([values, observed], dim=-1))
        h = h + self.embed_mix(h)
        h = torch.cat([h, self.future_token.expand(B, n_future, -1)], dim=1)
        if self.stats is not None:
            h = h + self.stats(stats.to(h.dtype))[:, None]
        N = n_past + n_future
        valid = torch.cat([observed.any(-1), torch.ones(B, n_future, dtype=torch.bool,
                                                        device=x.device)], dim=1)
        if p != self.patch:
            w = w * (p / self.patch)
        scale = torch.as_tensor(w, dtype=h.dtype, device=h.device).reshape(-1, 1)
        pos = torch.arange(N, device=h.device, dtype=h.dtype)[None] * scale
        for ssm, att in zip(self.ssm, self.att):
            h = ssm(h, delta_scale=w)
            h = att(h, pos, valid) if isinstance(att, AttentionBlock) else h
        h = self.final_norm(h[:, n_past:])
        if p == self.patch:
            raw = self.head(h)
        else:
            raw = self.head_size[str(p)](self.head[:-1](h))
        return self._fan(raw.reshape(B, n_future * p, -1)[:, :n])

    def normalize(self, x: torch.Tensor, mask: torch.Tensor):
        """x [B, L] with NaN, mask [B, L] -> (asinh frame, center [B, 1], scale [B, 1])."""
        x0 = torch.where(mask, x, torch.zeros_like(x)).double()
        count = mask.sum(1, keepdim=True).clamp_min(1)
        mean = x0.sum(1, keepdim=True) / count
        var = (mask * (x0 - mean) ** 2).sum(1, keepdim=True) / count
        center, scale = mean, var.sqrt()
        if self.robust_norm:
            nan = torch.where(mask, x, torch.full_like(x, float("nan"))).double()
            center = nan.nanquantile(0.5, dim=1, keepdim=True).nan_to_num()
            mad = (nan - center).abs().nanquantile(0.5, dim=1, keepdim=True).nan_to_num()
            scale = torch.where(mad > 0, 1.4826 * mad, scale)
        scale = scale + 1e-5 + 1e-6 * center.abs()
        return (torch.asinh((x0 - center) / scale) * mask).float(), center, scale

    def _pad(self, x: torch.Tensor, patch: Optional[int] = None) -> torch.Tensor:
        p = int(patch or self.patch)
        x = x[:, -self.max_tokens * p:]
        pad = -x.shape[1] % p
        return F.pad(x, (pad, 0), value=float("nan"))

    def _forecast_by_size(self, context, n, w, sizes) -> Dict[str, torch.Tensor]:
        """Items of one batch read at different patch sizes: one pass per size."""
        out = None
        for p in sorted(set(sizes.tolist())):
            idx = torch.nonzero(sizes == p).squeeze(1)
            wp = w[idx] if isinstance(w, torch.Tensor) and w.dim() > 0 else w
            part = self.forecast(context[idx], n, wp, patch=int(p))
            if out is None:
                out = {k: (torch.empty((len(context),) + v.shape[1:], dtype=v.dtype,
                                       device=v.device) if isinstance(v, torch.Tensor) else v)
                       for k, v in part.items()}
            for k, v in part.items():
                if isinstance(v, torch.Tensor):
                    out[k][idx] = v
        return out

    def forecast(self, context: torch.Tensor, n: Optional[int] = None, w: Optional[Scale] = None,
                 patch: Optional[int] = None, **_) -> Dict[str, torch.Tensor]:
        """context [B, L] or [B, L, 1], raw scale. `patch` forces a size, by default each item
        gets the one `choose_patch` picks."""
        n = int(n or self.prediction_length)
        context = context.squeeze(-1) if context.dim() == 3 else context
        w = 1.0 if w is None else w
        if patch is None and len(self.patches) > 1:
            sizes = self.choose_patch(torch.isfinite(context[:, -self.input_length:]))
            if len(set(sizes.tolist())) > 1:
                return self._forecast_by_size(context, n, w, sizes)
            patch = int(sizes[0])
        x = self._pad(context, patch)
        mask = torch.isfinite(x)
        z, mean, std = self.normalize(x, mask)
        stats = self.context_stats(x, mask, mean, std, z) if self.stats is not None else None
        q = self.forward_normalized(z, mask, n, w, stats, patch)
        raw = (torch.sinh(q.clamp(-15.0, 15.0).double()) * std[:, None] + mean[:, None]).float()
        mid = len(self.quantile_levels) // 2
        return {"quantiles": q, "quantiles_denorm": raw, "quantile_levels": self.quantile_levels,
                "forecast": q[..., mid], "forecast_denorm": raw[..., mid]}

    def forward(self, context: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        return self.forecast(context, **kwargs)


def build_from_config(cfg) -> PatchedSSM:
    """Config -> model; read by scripts/train_patched.py and, through `model.builder`,
    by the evaluation harness."""
    m = cfg.model
    return PatchedSSM(input_length=m.seq_length, prediction_length=m.prediction_length,
                      patch=m.patch, d_model=m.d_model, n_layers=m.n_layers, d_state=m.d_state,
                      expand=m.expand, n_heads=m.n_heads, attention=m.attention,
                      dropout=m.dropout, robust_norm=m.get("robust_norm", False),
                      stats=m.get("stats", False), patches=m.get("patches", None), name=m.name)
