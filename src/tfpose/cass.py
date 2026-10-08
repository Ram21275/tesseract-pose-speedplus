"""CASS-style spectral attention injection (EXP-018/020), training-free.

Re-implements the spectral strategy of CASS (Kim et al., CVPR 2025; official code
MICV-yonsei/CASS @ 228ecdd, clip/model.py::_apply_spectral_strategy) for a
last-block attention of either DINOv3 (timm Eva) or MoGe-2's DINOv2.

Exact shortcut (not an approximation): a key graph A = K K^T * s has rank <= head_dim,
so its eigenvalues and its top-k low-rank form are obtained exactly from the thin
SVD of K (N x 64) instead of an N x N eigendecomposition / randomized SVD.

Recorded deviations from CASS (pre-registered in EXP-018):
- the block's residual + MLP are kept; prefix (CLS/register) tokens use the original attention;
- DINOv3 keys are taken before RoPE;
- head pairing follows the paper (source head i <-> target head j from the assignment);
  the official code indexes the two adjacency stacks with swapped indices.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

K_EIGEN = 20
ENERGY = 0.95
EPSILON = 1.5
SCALE_FACTOR = 10.0
GAUSS_STD = 5.0


def _gram_eig(K: torch.Tensor):
    """Exact eigenpairs of K K^T via the small Gram matrix K^T K (d x d), descending.

    Returns (ev (H,d) float64 eigenvalues of K K^T, U (H,N,d) float32 orthonormal eigenvectors).
    """
    Kd = K.double()
    G = (Kd.transpose(1, 2) @ Kd).cpu()                     # tiny d x d: CPU LAPACK is far faster than GPU fp64
    ev, V = torch.linalg.eigh(G)                            # ascending, (H,d), (H,d,d)
    ev, V = ev.flip(-1).clamp_min(0).to(K.device), V.flip(-1).to(K.device)
    U = (Kd @ V) / torch.sqrt(ev.clamp_min(1e-30))[:, None, :]
    return ev, U.float()


def _spectrum(K: torch.Tensor, scale: float) -> torch.Tensor:
    """(H,N,d) keys -> (H,K_EIGEN) top eigenvalues of K K^T * scale (descending), float64."""
    ev = _gram_eig(K)[0] * scale
    out = torch.zeros(K.shape[0], K_EIGEN, dtype=torch.float64, device=K.device)
    n = min(K_EIGEN, ev.shape[1])
    out[:, :n] = ev[:, :n]
    return out


def _wasserstein_cost(spec_s: torch.Tensor, spec_t: torch.Tensor) -> np.ndarray:
    """CASS cost[i,j] = 1 - W1(normalized spectra of source head i, target head j) (vectorized)."""
    pa = torch.sort(F.normalize(F.normalize(spec_s, p=2, dim=1), p=1, dim=1), dim=1)[0]
    pb = torch.sort(F.normalize(F.normalize(spec_t, p=2, dim=1), p=1, dim=1), dim=1)[0]
    return (1.0 - (pa[:, None, :] - pb[None, :, :]).abs().sum(-1)).cpu().numpy()


def _scale_sv(s: torch.Tensor) -> torch.Tensor:
    """CASS Eq. 13 with epsilon = 1.5."""
    smin, smax = s.min(), s.max()
    if float(smax - smin) < 1e-12:
        return s
    rng = EPSILON * smax - (2 - EPSILON) * smin
    return (s - smin) / (smax - smin) * rng + (2 - EPSILON) * smin


def _range_to(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    smin, smax = src.min(), src.max()
    return (src - smin) / (smax - smin + 1e-12) * (tgt.max() - tgt.min()) + tgt.min()


_PRIOR = {}


def _gauss_prior(g: int, device) -> torch.Tensor:
    key = (g, str(device))
    if key not in _PRIOR:
        yy, xx = torch.meshgrid(torch.arange(g, device=device), torch.arange(g, device=device), indexing="ij")
        p = torch.stack([yy.reshape(-1), xx.reshape(-1)], 1).float()
        _PRIOR[key] = torch.exp(-torch.cdist(p, p) ** 2 / (2 * GAUSS_STD ** 2))
    return _PRIOR[key]


def fuse_attention(Kt: torch.Tensor, Ks: torch.Tensor, gt: int, gs: int, scale: float):
    """CASS spectral fusion for one image (float32 for N x N work, float64 for spectra).

    Kt: (H, gt*gt, d) target patch keys; Ks: (H, gs*gs, d) source patch keys.
    Returns (attn (H,N,N) softmax-normalized, info dict with spectra and matching).
    """
    H, N, d = Kt.shape
    ks = Ks.permute(0, 2, 1).reshape(H, d, gs, gs)
    ks = F.interpolate(ks.float(), size=(gt, gt), mode="bilinear", align_corners=False).reshape(H, d, N).permute(0, 2, 1)
    kt = Kt.float()
    ks = _range_to(ks, kt)
    spec_t, spec_s = _spectrum(kt, scale), _spectrum(ks, scale)
    cost = _wasserstein_cost(spec_s, spec_t)
    rows, cols = linear_sum_assignment(cost)                 # minimizes 1 - W  -> complementary pairs
    prior = _gauss_prior(gt, Kt.device)
    rows_t = torch.as_tensor(rows, device=Kt.device); cols_t = torch.as_tensor(cols, device=Kt.device)
    w = torch.as_tensor(1.0 - cost[rows, cols], dtype=torch.float32, device=Kt.device)[:, None, None]
    A_t = torch.bmm(kt[cols_t], kt[cols_t].transpose(1, 2)) * scale      # (H,N,N), target head j = cols
    ev, U = _gram_eig(ks[rows_t])                                        # source head i = rows
    ev = (ev * scale).float()                                            # (H,d) eigenvalues of A_s, descending
    cum = torch.cumsum(ev.double(), 1) / ev.double().sum(1, keepdim=True)
    k = ((cum < ENERGY).sum(1) + 1).clamp_max(ev.shape[1])               # per-head rank (0.95 energy)
    keep = torch.arange(ev.shape[1], device=Kt.device)[None, :] < k[:, None]
    big = torch.finfo(ev.dtype).max
    smin = torch.where(keep, ev, torch.full_like(ev, big)).amin(1, keepdim=True)
    smax = torch.where(keep, ev, torch.full_like(ev, -big)).amax(1, keepdim=True)
    rng = EPSILON * smax - (2 - EPSILON) * smin
    scaled = torch.where((smax - smin) > 1e-12, (ev - smin) / (smax - smin).clamp_min(1e-12) * rng + (2 - EPSILON) * smin, ev)
    scaled = scaled * keep                                               # Eq. 13 on the top-k, zero beyond
    A_s = torch.bmm(U * scaled[:, None, :], U.transpose(1, 2))
    A_s.diagonal(dim1=1, dim2=2).zero_()
    def rng_to(src, tgt):
        smn, smx = src.amin((1, 2), keepdim=True), src.amax((1, 2), keepdim=True)
        return (src - smn) / (smx - smn + 1e-12) * (tgt.amax((1, 2), keepdim=True) - tgt.amin((1, 2), keepdim=True)) + tgt.amin((1, 2), keepdim=True)
    A_s = rng_to(A_s, A_t)
    comb = (w * SCALE_FACTOR * A_s + A_t) / (w * SCALE_FACTOR + 1)
    out = torch.empty(H, N, N, dtype=torch.float32, device=Kt.device)
    out[cols_t] = torch.softmax(comb + prior, dim=-1)
    ranks = k.cpu().numpy().tolist()
    info = {"spec_target": spec_t.float().cpu().numpy(), "spec_source": spec_s.float().cpu().numpy(),
            "cost": cost.astype(np.float32), "pairs": np.stack([rows, cols], 1).astype(np.int8),
            "ranks": np.array(ranks, np.int16)}
    return out, info


class LastBlockAttention:
    """Wraps a last-block attention module. mode 'capture' keeps the output unchanged and stores
    pre-RoPE patch keys; mode 'inject' replaces patch->patch attention with the CASS fusion
    using ``self.source`` = (keys (B,H,Ns,d), source grid)."""

    def __init__(self, attn, kind: str, n_prefix: int, grid: int):
        assert kind in ("eva", "dinov2")
        self.attn, self.kind, self.n_prefix, self.grid = attn, kind, n_prefix, grid
        self.mode, self.source, self.keys, self.infos = "capture", None, None, []
        self._orig = attn.forward
        attn.forward = self.forward

    def _qkv(self, x):
        a = self.attn
        B, N, C = x.shape
        if self.kind == "eva":
            if a.q_bias is None:
                qkv = a.qkv(x)
            else:
                qkv = F.linear(x, weight=a.qkv.weight, bias=torch.cat((a.q_bias, a.k_bias, a.v_bias)))
            q, k, v = qkv.reshape(B, N, 3, a.num_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
            q, k = a.q_norm(q), a.k_norm(k)
        else:
            q, k, v = a.qkv(x).reshape(B, N, 3, a.num_heads, C // a.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        return q, k, v

    def forward(self, x, *args, **kwargs):
        q, k, v = self._qkv(x)
        self.keys = k[:, :, self.n_prefix:, :].detach()
        out = self._orig(x, *args, **kwargs)                   # original attention for all tokens
        if self.mode == "capture":
            return out
        a = self.attn
        B, N, C = x.shape
        Ks, gs = self.source
        H, d = a.num_heads, C // a.num_heads
        patch_out = []
        self.infos = []
        for b in range(B):
            attn, info = fuse_attention(self.keys[b], Ks[b], self.grid, gs, d ** -0.5)
            self.infos.append(info)
            patch_out.append((attn.to(v.dtype) @ v[b, :, self.n_prefix:, :]))   # (H,Np,d)
        po = torch.stack(patch_out).permute(0, 2, 1, 3).reshape(B, -1, C)       # merged heads, patches only
        if self.kind == "eva":
            po = a.proj(a.norm(po))
        else:
            po = a.proj(po)
        out = out.clone()
        out[:, self.n_prefix:, :] = po.to(out.dtype)
        return out
