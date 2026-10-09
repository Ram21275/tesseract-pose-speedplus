"""Phase-4 continuous heads on top of the frozen Phase-3 Tesseract Transformer (EXP-040–042).

Rotations are scalar-first unit quaternions (q ~ -q). Body-frame (right) increments throughout:
    R_tau = R_0 Exp(tau a),  a = Log(R_0^-1 R_1)  =>  dR/dtau = R_tau [a]_x, constant body velocity a.
The network input never sees the raw quaternion (sign-ambiguous); it sees the 3x3 rotation matrix.

- ResidualHead040: one-step tangent residual at a given cell centre, q = q_c Exp(delta).
- FlowHead:        velocity field omega(R_tau, tau | image memory) trained with flow matching.
Both cross-attend to the frozen encoder memory of an EXP-036 HierTransformer.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .phase3 import qconj, qmul, quat_to_rotvec, rotvec_to_quat
from .predictor import fourier

CUT_LOCUS = np.pi - 1e-3          # pairs whose geodesic angle exceeds this have an ill-defined Log; masked out


def qcanon(q):
    return torch.where(q[..., :1] < 0, -q, q)


def quat_to_mat(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)


def relative_rotvec(q0, q1):
    """a = Log(q0^-1 q1) in the body frame of q0 (radians, |a| <= pi)."""
    return quat_to_rotvec(qmul(qconj(q0), q1))


def geo_deg(q0, q1):
    """Geodesic angle in degrees via atan2 of the relative quaternion (well-conditioned near 0 and 180)."""
    r = qmul(qconj(q0), q1)
    return torch.rad2deg(2 * torch.atan2(r[..., 1:].norm(dim=-1), r[..., 0].abs()))


def uniform_quats(n: int, device, generator=None, dtype=torch.float32):
    g = torch.randn(n, 4, device=device, generator=generator, dtype=dtype)
    return qcanon(g / g.norm(dim=-1, keepdim=True))


def local_quats(centre, sigma_rad: float, generator=None):
    """R_c Exp(n), n ~ N(0, sigma^2 I_3) in the body frame of each centre (B,4)."""
    n = torch.randn(centre.shape[0], 3, device=centre.device, generator=generator) * sigma_rad
    return qcanon(qmul(centre, rotvec_to_quat(n)))


def fm_pairs(q0, q1, tau):
    """Geodesic interpolant and its body-frame velocity target; mask = Log well defined."""
    a = relative_rotvec(q0, q1)
    qt = qcanon(qmul(q0, rotvec_to_quat(tau[:, None] * a)))
    return qt, a, a.norm(dim=-1) < CUT_LOCUS


def rot_features(q, n_freq: int = 4):
    return fourier(quat_to_mat(q), n_freq)          # 9 * (1 + 2 n_freq)


class _QueryAttend(nn.Module):
    """Query token(s) cross-attending to frozen image memory."""

    def __init__(self, d_q: int, d: int = 256, layers: int = 2, heads: int = 4, ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_q, d), nn.GELU(), nn.Linear(d, d))
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True), layers)

    def forward(self, qfeat, mem):
        """qfeat (B,S,d_q), mem (B,T,d) -> (B,S,d). S query tokens per image attend jointly (self-attn mixes them)."""
        return self.dec(self.inp(qfeat), mem)


class ResidualHead040(nn.Module):
    def __init__(self, d: int = 256, n_freq: int = 4, **kw):
        super().__init__()
        self.n_freq = n_freq
        self.att = _QueryAttend(9 * (1 + 2 * n_freq), d, **kw)
        self.out = nn.Linear(d, 3)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, mem, q_c):
        """mem (B,T,d), q_c (B,4) -> delta (B,3) radians."""
        return self.out(self.att(rot_features(q_c, self.n_freq)[:, None], mem)[:, 0])


class FlowHead(nn.Module):
    """omega(R_tau, tau | mem). Each query is processed independently (S samples are folded into the batch)."""

    def __init__(self, d: int = 256, n_freq: int = 4, t_freq: int = 8, cell: bool = False, **kw):
        super().__init__()
        self.n_freq, self.t_freq, self.cell = n_freq, t_freq, cell
        self.att = _QueryAttend(9 * (1 + 2 * n_freq) * (2 if cell else 1) + 2 * t_freq, d, **kw)
        self.out = nn.Linear(d, 3)

    def tfeat(self, tau):
        f = 2.0 ** torch.arange(self.t_freq, device=tau.device, dtype=tau.dtype) * np.pi
        a = tau[:, None] * f
        return torch.cat([a.sin(), a.cos()], -1)

    def forward(self, mem, q_t, tau, q_c=None):
        """mem (B,T,d) already expanded to the query batch; q_t (B,4); tau (B,); q_c (B,4) cell centre if cell=True."""
        parts = [rot_features(q_t, self.n_freq), self.tfeat(tau)]
        if self.cell:
            parts.append(rot_features(q_c, self.n_freq))
        z = torch.cat(parts, -1)[:, None]
        return self.out(self.att(z, mem)[:, 0])


@torch.no_grad()
def integrate(vel_fn, q0, nfe: int):
    """Euler on the group: q <- q Exp(dt * omega). vel_fn(q, tau) -> (B,3)."""
    q = q0; dt = 1.0 / nfe
    for i in range(nfe):
        tau = torch.full((q.shape[0],), i * dt, device=q.device, dtype=q.dtype)
        q = qcanon(qmul(q, rotvec_to_quat(dt * vel_fn(q, tau))))
        q = q / q.norm(dim=-1, keepdim=True)
    return q


def kde_mode(samples, bw_deg: float = 5.0):
    """Point estimate from samples (B,M,4): the sample with the highest Gaussian-kernel density among its set."""
    d = torch.rad2deg(2 * torch.arccos(torch.einsum("bmd,bnd->bmn", samples, samples).abs().clamp(max=1.0)))
    score = torch.exp(-0.5 * (d / bw_deg) ** 2).sum(-1)
    return samples.gather(1, score.argmax(1)[:, None, None].expand(-1, 1, 4))[:, 0]
