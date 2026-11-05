"""
Selective SSM token mixer, in the form of Mamba-2 (Dao & Gu 2024, arXiv 2405.21060).

Each head has a scalar decay a_h < 0; the step, the input matrix and the readout depend on
the token, so what the state keeps depends on the content:

    Delta_t = softplus(W_dt u_t + b_dt) * w
    h_t = exp(Delta_t a_h) h_{t-1} + (exp(Delta_t a_h) - 1) / a_h * B_t x_t^T
    y_t = h_t C_t + D x_t

The input weight is the exact zero-order hold of the continuous system, not the
Delta_t B_t of Mamba: on a stretch where the input is constant, Delta_t, B_t and C_t are
constant too, and two steps at Delta equal one step at 2 Delta exactly, so the rate knob
`w` keeps the meaning it has in the S4D layer. No short convolution before the scan, for
the same reason as in the gated S4D block (it filters in steps, not in time).

Computed by chunks (state space duality): inside a chunk, a masked product with the
decays between positions; across chunks, a recurrence on the states only.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _segsum(x: torch.Tensor) -> torch.Tensor:
    """x [..., T] -> [..., T, T], entry (i, j) = x[j+1] + ... + x[i] for j <= i, -inf above."""
    T = x.shape[-1]
    c = x.cumsum(-1)
    s = c[..., :, None] - c[..., None, :]
    keep = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
    return s.masked_fill(~keep, float("-inf"))


def selective_scan(x, log_a, b, c, chunk: int = 64):
    """x [B, L, H, P] (already weighted by the input step), log_a [B, L, H] (Delta_t a_h),
    b, c [B, L, H, N] -> y [B, L, H, P], y_t = sum_s<=t exp(sum log_a(s+1..t)) c_t.b_s x_s."""
    Bsz, L, H, P = x.shape
    pad = -L % chunk
    if pad:
        x, b, c = (F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (x, b, c))
        log_a = F.pad(log_a, (0, 0, 0, pad))
    n = (L + pad) // chunk
    x, b, c = (t.reshape(Bsz, n, chunk, H, -1) for t in (x, b, c))
    la = log_a.reshape(Bsz, n, chunk, H).permute(0, 3, 1, 2)            # [B, H, n, l]
    cum = la.cumsum(-1)
    # within a chunk
    decay = torch.exp(_segsum(la))                                       # [B, H, n, l, l]
    scores = torch.einsum("bclhn,bcshn->bhcls", c, b) * decay
    y = torch.einsum("bhcls,bcshp->bclhp", scores, x)
    # state at the end of each chunk, from its own inputs
    to_end = torch.exp(cum[..., -1:] - cum)                              # [B, H, n, l]
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", b, to_end, x)
    # recurrence over chunks: state entering chunk i
    total = F.pad(cum[..., -1], (1, 0))                                  # [B, H, n+1]
    across = torch.exp(_segsum(total))                                   # [B, H, n+1, n+1]
    states = torch.cat([torch.zeros_like(states[:, :1]), states], 1)
    entering = torch.einsum("bhzc,bchpn->bzhpn", across, states)[:, :-1]
    y = y + torch.einsum("bclhn,bchpn,bhcl->bclhp", c, entering, torch.exp(cum))
    return y.reshape(Bsz, n * chunk, H, P)[:, :L]


class SelectiveSSMBlock(nn.Module):
    """Pre-norm residual block: in_proj to (x, z, B, C, Delta), the selective scan per head,
    D skip, gated RMS norm, out_proj. Takes `delta_scale` (float or [B]) as the S4D block
    does: it multiplies every Delta_t."""

    def __init__(self, d_model: int, d_state: int = 32, expand: int = 2, head_dim: int = 64,
                 dropout: float = 0.0, dt_min: float = 1e-3, dt_max: float = 1e-1,
                 a_min: float = 1.0, a_max: float = 16.0, chunk: int = 64):
        super().__init__()
        d_inner = expand * d_model
        assert d_inner % head_dim == 0
        self.d_inner, self.d_state, self.chunk = d_inner, d_state, chunk
        self.heads, self.head_dim = d_inner // head_dim, head_dim
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner + 2 * d_state + self.heads)
        # Mamba-2 initialization: Delta log-uniform in [dt_min, dt_max] through the bias,
        # decays uniform in [a_min, a_max].
        dt = torch.exp(torch.rand(self.heads) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))   # softplus^-1(dt)
        self.a_log = nn.Parameter(torch.log(torch.empty(self.heads).uniform_(a_min, a_max)))
        self.D = nn.Parameter(torch.ones(self.heads))
        self.out_norm = nn.RMSNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, delta_scale=1.0) -> torch.Tensor:
        """x [B, L, D] -> [B, L, D]."""
        Bsz, L, _ = x.shape
        u, z, b, c, dt = self.in_proj(self.norm(x)).split(
            [self.d_inner, self.d_inner, self.d_state, self.d_state, self.heads], dim=-1)
        u = F.silu(u)
        f = torch.promote_types(x.dtype, torch.float32)        # the scan never runs in half
        with torch.autocast(x.device.type, enabled=False):
            w = torch.as_tensor(delta_scale, dtype=f, device=x.device)
            w = w.reshape(-1, 1, 1) if w.numel() > 1 else w.reshape(1, 1, 1)
            delta = F.softplus(dt.to(f) + self.dt_bias.to(f)) * w                  # [B, L, H]
            a = -torch.exp(self.a_log.to(f))                                 # [H]
            log_a = delta * a
            gain = torch.expm1(log_a) / a                                      # exact ZOH
            xs = u.to(f).reshape(Bsz, L, self.heads, self.head_dim)
            bh = b.to(f)[:, :, None].expand(Bsz, L, self.heads, self.d_state)
            ch = c.to(f)[:, :, None].expand(Bsz, L, self.heads, self.d_state)
            y = selective_scan(xs * gain[..., None], log_a, bh, ch, self.chunk)
            y = y + self.D.to(f)[:, None] * xs
        y = y.reshape(Bsz, L, self.d_inner).to(x.dtype)
        y = self.out_norm(y * F.silu(z))
        return x + self.dropout(self.out_proj(y))
