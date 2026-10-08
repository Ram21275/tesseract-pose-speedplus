"""Training-free spectral graph tools (EXP-017/019; inspired by CASS, arXiv 2411.17150).

All graphs are over the 16x16 patch grid of the GT crop. Constants are fixed a
priori (pre-registered in outputs/reports/EXP-017_spectral_foreground.md).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

SIGMA_NORMAL = 0.2   # A_geo normal term: exp(-(1 - n_i.n_j) / 0.2)
SIGMA_LOGD = 0.1     # A_geo depth term:  exp(-|log d_i - log d_j| / 0.1)


def pool16(m: np.ndarray) -> np.ndarray:
    """(74,74,C) -> (16,16,C) area pooling."""
    t = torch.from_numpy(np.ascontiguousarray(m, dtype=np.float32)).permute(2, 0, 1)[None]
    return F.adaptive_avg_pool2d(t, 16)[0].permute(1, 2, 0).numpy()


def dino_affinity(tokens: np.ndarray) -> np.ndarray:
    """(16,16,C) tokens -> (256,256) cosine affinity clipped at 0."""
    X = tokens.reshape(256, -1).astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return np.clip(X @ X.T, 0.0, None)


def geo_affinity(normal74: np.ndarray, depth74: np.ndarray) -> np.ndarray:
    """MoGe-2 normals (74,74,3) and depth (74,74,1) -> (256,256) geometric affinity (mask not used)."""
    n = pool16(normal74).reshape(256, 3).astype(np.float64)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    ld = np.log(np.clip(pool16(depth74).reshape(256).astype(np.float64), 1e-6, None))
    a_n = np.exp(-(1.0 - n @ n.T) / SIGMA_NORMAL)
    a_d = np.exp(-np.abs(ld[:, None] - ld[None, :]) / SIGMA_LOGD)
    return a_n * a_d


def laplacian_eig(A: np.ndarray):
    """Symmetric normalized Laplacian eigenpairs, ascending eigenvalues."""
    A = 0.5 * (A + A.T)
    d = A.sum(1)
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    L = np.eye(len(A)) - dinv[:, None] * A * dinv[None, :]
    return np.linalg.eigh(L)


def border_ring(g: int = 16) -> np.ndarray:
    r = np.zeros((g, g), bool)
    r[0, :] = r[-1, :] = r[:, 0] = r[:, -1] = True
    return r.reshape(-1)


def fiedler_score(A: np.ndarray, res: int = 74):
    """Foreground score from the Fiedler vector; border = background. Returns (score_res, eigvecs)."""
    w, V = laplacian_eig(A)
    f = V[:, 1].copy()
    ring = border_ring()
    if f[ring].mean() > f[~ring].mean():
        f = -f
    t = torch.from_numpy(f.reshape(16, 16).astype(np.float32))[None, None]
    up = F.interpolate(t, size=(res, res), mode="bilinear", align_corners=False)[0, 0].numpy()
    return up, V
