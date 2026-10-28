"""EXP-118: translation in the SPACE-HOP formulation (Dutta et al., CVPRW 2026, §3.3, eqs 7-13), kept identical, and a
translation block over the Tesseract tree's encoded tokens.

u = (U_x, U_y, U_z):  U_x = (f_x X/Z + c_x - b_x) / b_w,  U_y = (f_y Y/Z + c_y - b_y) / b_h,
                      U_z = 0.5 (1/s_x + 1/s_y) Z,  s_x = W_full / (alpha b_w),  s_y = H_full / b_h,  alpha = 1.6.
Box (b_x, b_y, b_w, b_h) = (crop_cx, crop_cy, crop_side, crop_side)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

ALPHA = 1.6
W_FULL, H_FULL = 1920, 1200          # SPEED+ image size


def _zscale(bw, bh, W=W_FULL, H=H_FULL):
    return 0.5 * (ALPHA * bw / W + bh / H)                    # 0.5 (1/s_x + 1/s_y)


def to_u(t: np.ndarray, K: np.ndarray, bx, by, bw, bh, W=W_FULL, H=H_FULL) -> np.ndarray:
    """(N,3) camera-frame translation -> (N,3) SPACE-HOP target u. K is (3,3) or per row (N,3,3); (W, H) = full image size
    (SPEED+ by default; EXP-126 passes each dataset's own)."""
    X, Y, Z = t[:, 0], t[:, 1], t[:, 2]
    ux = (K[..., 0, 0] * X / Z + K[..., 0, 2] - bx) / bw
    uy = (K[..., 1, 1] * Y / Z + K[..., 1, 2] - by) / bh
    return np.stack([ux, uy, _zscale(bw, bh, W, H) * Z], 1)


def from_u(u: np.ndarray, K: np.ndarray, bx, by, bw, bh, W=W_FULL, H=H_FULL) -> np.ndarray:
    """Inverse perspective mapping u -> t."""
    Z = u[:, 2] / _zscale(bw, bh, W, H)
    X = (u[:, 0] * bw + bx - K[..., 0, 2]) * Z / K[..., 0, 0]
    Y = (u[:, 1] * bh + by - K[..., 1, 2]) * Z / K[..., 1, 1]
    return np.stack([X, Y, Z], 1)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v (N,3) by unit quaternions q (N,4, wxyz)."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


class TransHead(nn.Module):
    """One learned query cross-attends to the encoded tokens, then an MLP predicts u (3)."""

    def __init__(self, d: int = 256, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(1, 1, d)); nn.init.normal_(self.q, std=0.02)
        self.norm = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 3))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        k = self.norm(h)
        o, _ = self.att(self.q.expand(h.shape[0], -1, -1), k, k, need_weights=False)
        return self.mlp(o[:, 0])
