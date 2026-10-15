"""EXP-108: geometry-conditioned iterative path denoiser (renamed after the 2026-10-15 review; NOT a discrete flow-matching
model: no time conditioning, no defined probability path, argmax re-decoding). A bidirectional reviser re-predicts every
Tesseract decision (root + children) of a current path, conditioned on the frozen tree-encoder memory (image) and, unless
geo=False (matched control), on the geometry of the hypothesis (its leaf rotation and the three 180-degree body-axis flips).
Trained to denoise paths corrupted token-wise towards geometry-aware SOURCE paths (flip / nearby / uniform); token mixing
is not geometry-preserving (see scripts/exp108_corruption_geometry.py)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import phase3 as P
from . import tesseract as T
from .fusion import combine
from .predictor import decode_torch, fourier
from .so3flow import rot_features

FLIPS = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float64)     # 180 deg about body x, y, z


class PathReviser(nn.Module):
    def __init__(self, depth: int = 5, d: int = 256, layers: int = 2, heads: int = 4, ff: int = 512, dropout: float = 0.1,
                 n_freq: int = 6, r_freq: int = 4, geo: bool = True):
        super().__init__()
        self.geo = geo                                   # False: same modules, geometry features zeroed (matched control)
        self.depth, self.n_freq, self.r_freq = depth, n_freq, r_freq
        self.root_emb = nn.Embedding(T.N_CHARTS, d)
        self.child_emb = nn.Embedding(T.N_CHILD, d)
        self.level_emb = nn.Embedding(depth + 2, d)
        self.cell_proj = nn.Linear(4 + 8 * n_freq, d)
        self.geo_proj = nn.Sequential(nn.Linear(4 * 9 * (1 + 2 * r_freq), d), nn.GELU(), nn.Linear(d, d))
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True), layers)
        self.root_head = nn.Linear(d, T.N_CHARTS)
        self.child_head = nn.Linear(d, T.N_CHILD)

    def forward(self, mem, chart, path):
        """mem (B,T,d) frozen encoder memory; chart (B,), path (B,L) current decisions -> root (B,4), child (B,L,8) logits."""
        B, L = path.shape
        dev = path.device
        toks = [self.root_emb(chart) + self.level_emb.weight[0] + self.cell_proj(fourier(decode_torch(chart, path[:, :0]), self.n_freq))]
        for l in range(L):
            toks.append(self.child_emb(path[:, l]) + self.level_emb.weight[l + 1]
                        + self.cell_proj(fourier(decode_torch(chart, path[:, :l + 1]), self.n_freq)))
        qleaf = decode_torch(chart, path)                                              # (B,4)
        fl = torch.as_tensor(FLIPS, dtype=qleaf.dtype, device=dev)
        hyp = torch.cat([qleaf[:, None], P.qmul(qleaf[:, None].expand(B, 3, 4).reshape(-1, 4), fl.repeat(B, 1)).view(B, 3, 4)], 1)
        gf = rot_features(hyp.reshape(-1, 4), self.r_freq).view(B, -1)
        geo = self.geo_proj(gf if self.geo else torch.zeros_like(gf)) + self.level_emb.weight[L + 1]
        x = torch.stack(toks + [geo], 1)                                               # (B, L+2, d), no causal mask
        h = self.dec(x, mem)
        return self.root_head(h[:, 0]), self.child_head(h[:, 1:L + 1])


def corrupt(q: np.ndarray, chart: np.ndarray, path: np.ndarray, t: np.ndarray, rng, p_flip=0.35, p_near=0.25, return_info=False):
    """Discrete flow-matching corruption: keep each GT decision with prob t, else take it from a source path
    (180-deg body flip of GT | GT perturbed 3-15 deg | uniform). Any combination is a valid path."""
    N, L = path.shape
    u = rng.random(N)
    src_q = q.copy()
    fl = u < p_flip
    if fl.any():
        f = FLIPS[rng.integers(0, 3, fl.sum())]
        a = src_q[fl]
        src_q[fl] = np.stack([a[:, 0] * f[:, 0] - (a[:, 1:] * f[:, 1:]).sum(1),
                              *(a[:, 0:1] * f[:, 1:] + f[:, 0:1] * a[:, 1:] + np.cross(a[:, 1:], f[:, 1:])).T], 1)
    nr = (u >= p_flip) & (u < p_flip + p_near)
    if nr.any():
        ax = rng.normal(size=(nr.sum(), 3)); ax /= np.linalg.norm(ax, axis=1, keepdims=True)
        ang = np.radians(rng.uniform(3, 15, nr.sum()))
        e = np.concatenate([np.cos(ang / 2)[:, None], np.sin(ang / 2)[:, None] * ax], 1); a = src_q[nr]
        src_q[nr] = np.stack([a[:, 0] * e[:, 0] - (a[:, 1:] * e[:, 1:]).sum(1),
                              *(a[:, 0:1] * e[:, 1:] + e[:, 0:1] * a[:, 1:] + np.cross(a[:, 1:], e[:, 1:])).T], 1)
    sc, sp = T.encode(src_q, L)
    un = ~(fl | nr)
    sc[un] = rng.integers(0, T.N_CHARTS, un.sum()); sp[un] = rng.integers(0, T.N_CHILD, (un.sum(), L))
    keep = rng.random((N, L + 1)) < t[:, None]
    c = np.where(keep[:, 0], chart, sc); p = np.where(keep[:, 1:], path, sp)
    if return_info:
        return c, p, {"source": np.where(fl, "flip", np.where(nr, "near", "uniform")), "src_chart": sc, "src_path": sp}
    return c, p


@torch.no_grad()
def revise(revisers, mems, chart, path, K: int, rule: str = "poe"):
    """K steps of top-down re-decoding with per-level fused reviser distributions. Returns chart, path, n_changed_steps."""
    chart, path = chart.clone(), path.clone()
    L = path.shape[1]
    steps = torch.zeros(len(chart), dtype=torch.long, device=chart.device)
    for _ in range(K):
        c0, p0 = chart.clone(), path.clone()
        outs = [r(m, chart, path) for r, m in zip(revisers, mems)]
        chart = combine([F.log_softmax(o[0].float(), -1) for o in outs], rule).argmax(-1)
        for l in range(L):
            outs = [r(m, chart, path) for r, m in zip(revisers, mems)]
            path[:, l] = combine([F.log_softmax(o[1][:, l].float(), -1) for o in outs], rule).argmax(-1)
        ch = (chart != c0) | (path != p0).any(1)
        steps += ch.long()
        if not ch.any():
            break
    return chart, path, steps
