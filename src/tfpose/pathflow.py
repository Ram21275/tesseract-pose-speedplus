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


# ======================= EXP-108 (DEC-015): discrete flow matching on the Tesseract path =======================
# Tokens x = (root, c_0..c_4). Coupling (image-conditional, independent of x1 given the image):
#   x0 ~ p_tree(path | image)  (ancestral sample of the frozen EXP-036 per-level PoE tree, temperature 1),  x1 = GT path.
# Probability path (factorized mixture, Gat et al. 2024): x_t,i = x1_i with prob kappa(t), else x0_i; kappa(t) = t.
# Model: posterior p_theta(x1_i | x_t, t, image) for every token (reviser + Fourier(t) embedding). Loss: token CE vs x1.
# Sampler: CTMC Euler with the mixture-path velocity u_t(x_i) = kappa'/(1-kappa) [p_{1|t}(x_i) - delta_{x_t,i}(x_i)]:
#   for step h, each token jumps with probability min(1, h*kappa'/(1-kappa)) to a draw from p_{1|t}; the last step
#   (t+h = 1) jumps with probability 1. Branches are combined per token by a normalized product (heuristic, as in the
#   tree PoE; not an exact product of the two CTMCs).

class PathDFM(PathReviser):
    def __init__(self, *a, t_freq: int = 8, **k):
        super().__init__(*a, **k)
        d = self.root_head.in_features
        self.t_freq = t_freq
        self.t_proj = nn.Sequential(nn.Linear(2 * t_freq, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, mem, chart, path, t):
        f = 2.0 ** torch.arange(self.t_freq, device=t.device, dtype=torch.float32) * np.pi
        te = self.t_proj(torch.cat([(t[:, None] * f).sin(), (t[:, None] * f).cos()], -1))      # (B,d)
        B, L = path.shape
        dev = path.device
        toks = [self.root_emb(chart) + self.level_emb.weight[0] + self.cell_proj(fourier(decode_torch(chart, path[:, :0]), self.n_freq))]
        for l in range(L):
            toks.append(self.child_emb(path[:, l]) + self.level_emb.weight[l + 1]
                        + self.cell_proj(fourier(decode_torch(chart, path[:, :l + 1]), self.n_freq)))
        qleaf = decode_torch(chart, path)
        fl = torch.as_tensor(FLIPS, dtype=qleaf.dtype, device=dev)
        hyp = torch.cat([qleaf[:, None], P.qmul(qleaf[:, None].expand(B, 3, 4).reshape(-1, 4), fl.repeat(B, 1)).view(B, 3, 4)], 1)
        geo = self.geo_proj(rot_features(hyp.reshape(-1, 4), self.r_freq).view(B, -1)) + self.level_emb.weight[L + 1]
        x = torch.stack(toks + [geo], 1) + te[:, None]
        h = self.dec(x, mem)
        return self.root_head(h[:, 0]), self.child_head(h[:, 1:L + 1])


@torch.no_grad()
def sample_tree_paths(trees, mems, temperature: float = 1.0, generator=None):
    """Ancestral sample of the per-level PoE tree (EXP-036 decoding rule, sampled instead of argmax). mems: per-tree
    encoder memories (B,T,d). Returns chart (B,), path (B,L)."""
    B = mems[0].shape[0]; dev = mems[0].device; L = trees[0].depth
    lp = F.log_softmax(sum(F.log_softmax(m.root(h).float(), -1) for m, h in zip(trees, mems)), -1)
    chart = torch.multinomial((lp / temperature).softmax(-1), 1, generator=generator)[:, 0]
    path = torch.zeros(B, 0, dtype=torch.long, device=dev)
    for _ in range(L):
        lp = F.log_softmax(sum(F.log_softmax(m.child_logits_prefix(h, chart, path).float(), -1) for m, h in zip(trees, mems)), -1)
        path = torch.cat([path, torch.multinomial((lp / temperature).softmax(-1), 1, generator=generator)], 1)
    return chart, path


def mix_tokens(c0, p0, c1, p1, t, generator=None):
    """Factorized mixture path: each token from x1 with probability kappa(t) = t, else from x0."""
    keep = torch.rand((len(t), p0.shape[1] + 1), device=t.device, generator=generator) < t[:, None]
    return torch.where(keep[:, 0], c1, c0), torch.where(keep[:, 1:], p1, p0)


@torch.no_grad()
def dfm_sample(models, mems, c, p, nfe: int, rule: str = "poe", generator=None):
    """CTMC Euler sampler for the mixture path, kappa(t) = t, from x0 = (c, p) at t = 0 to t = 1 in nfe steps."""
    c, p = c.clone(), p.clone(); B, L = p.shape; dev = p.device; h = 1.0 / nfe
    for s in range(nfe):
        t = s * h
        tt = torch.full((B,), t, device=dev)
        outs = [m(mem, c, p, tt) for m, mem in zip(models, mems)]
        lr = combine([F.log_softmax(o[0].float(), -1) for o in outs], rule)
        lc = combine([F.log_softmax(o[1].float(), -1) for o in outs], rule)                 # (B,L,8)
        lam = 1.0 if s == nfe - 1 else min(1.0, h / (1.0 - t))                              # h * kappa'/(1-kappa)
        jump = torch.rand((B, L + 1), device=dev, generator=generator) < lam
        nr = torch.multinomial(lr.softmax(-1), 1, generator=generator)[:, 0]
        nc = torch.multinomial(lc.softmax(-1).reshape(-1, 8), 1, generator=generator).view(B, L)
        c = torch.where(jump[:, 0], nr, c); p = torch.where(jump[:, 1:], nc, p)
    return c, p
