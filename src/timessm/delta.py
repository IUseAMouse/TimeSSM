"""
Delta-rule token mixer (Schlag et al. 2021; Yang et al. 2024, arXiv 2406.06484).

Each head keeps a matrix memory S [d_k, d_v]. Token t writes its value under its key,
replacing what the memory returned for that key, and reads with its query:

    S_t = S_{t-1} + beta_t k_t (v_t - S_{t-1}^T k_t)^T,    o_t = S_t^T q_t

Unlike a linear time-invariant SSM, what is stored depends on the content, so a past cycle
can be recalled exactly. Computed by chunks with the WY form of the products of the
Householder-like updates: one triangular solve per chunk, a recurrence over chunks only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def delta_rule(q, k, v, beta, chunk: int = 64):
    """q, k [B, H, L, dk] (k unit norm), v [B, H, L, dv], beta [B, H, L] -> o [B, H, L, dv]."""
    B, H, L, dk = k.shape
    dv = v.shape[-1]
    pad = -L % chunk
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        beta = F.pad(beta, (0, pad))
    n = (L + pad) // chunk
    q, k, v = (t.reshape(B, H, n, chunk, -1) for t in (q, k, v))
    beta = beta.reshape(B, H, n, chunk, 1)
    kb, vb = k * beta, v * beta
    eye = torch.eye(chunk, dtype=k.dtype, device=k.device)
    # T = (I + tril(diag(beta) K K^T, -1))^-1 ; W = T diag(beta) K, U = T diag(beta) V.
    m = eye + (kb @ k.transpose(-1, -2)).tril(-1)
    w = torch.linalg.solve_triangular(m, kb, upper=False, unitriangular=True)
    u = torch.linalg.solve_triangular(m, vb, upper=False, unitriangular=True)
    causal = torch.ones(chunk, chunk, dtype=torch.bool, device=k.device).tril()
    s = q.new_zeros(B, H, dk, dv)
    out = []
    for i in range(n):
        qi, ki = q[:, :, i], k[:, :, i]
        ui = u[:, :, i] - w[:, :, i] @ s
        att = (qi @ ki.transpose(-1, -2)).masked_fill(~causal, 0)
        out.append(qi @ s + att @ ui)
        s = s + ki.transpose(-1, -2) @ ui
    return torch.stack(out, 2).reshape(B, H, n * chunk, dv)[:, :, :L]


class DeltaBlock(nn.Module):
    """Pre-norm residual block around the delta rule: short causal convolutions on q, k, v,
    SiLU, unit-norm keys and queries, a write strength per head in (0, 1), an output gate.
    Takes `delta_scale` for the interface of the SSM block and ignores it: the memory is
    indexed by content, not by time."""

    def __init__(self, d_model: int, n_heads: int = 4, d_conv: int = 4, dropout: float = 0.0,
                 chunk: int = 64):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.chunk = n_heads, chunk
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.conv = nn.Conv1d(3 * d_model, 3 * d_model, d_conv, groups=3 * d_model,
                              padding=d_conv - 1)
        self.beta = nn.Linear(d_model, n_heads)
        self.gate = nn.Linear(d_model, d_model)
        self.head_norm = nn.LayerNorm(d_model // n_heads)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, delta_scale=1.0) -> torch.Tensor:
        B, L, D = x.shape
        h = self.norm(x)
        qkv = self.conv(self.qkv(h).transpose(1, 2))[..., :L].transpose(1, 2)
        q, k, v = F.silu(qkv).chunk(3, dim=-1)
        q, k, v = (t.reshape(B, L, self.n_heads, -1).transpose(1, 2).float() for t in (q, k, v))
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        beta = torch.sigmoid(self.beta(h)).transpose(1, 2).float()
        with torch.autocast(x.device.type, enabled=False):
            o = delta_rule(q, k, v, beta, self.chunk)
        o = self.head_norm(o.transpose(1, 2)).reshape(B, L, D).to(x.dtype)
        o = o * F.silu(self.gate(h))
        return x + self.dropout(self.out_proj(o))
