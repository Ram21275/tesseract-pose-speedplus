"""Phase-3 predictor variants (EXP-031/033/035), sharing the EXP-030 trunk and tree.

- geodesic soft targets  (EXP-033): per level, target over the 4 charts / 8 children of the GT parent
  = softmax(-d^2 / tau_l^2), d = SO(3) distance from the GT rotation to each candidate cell centre,
  tau_l = mean own-cell quantization error at that level (EXP-002; fixed a priori).
- GRU path decoder       (EXP-031): child logits from a GRU over previously chosen children.
- tangent residual       (EXP-035): q = q_leaf (x) Exp(delta), delta in R^3 predicted from [h, leaf centre].
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import tesseract as T
from .predictor import decode_torch, fourier

# EXP-002 mean own-cell error per level (deg); level 0 = chart choice (L1 cell size used)
TAU_DEG = {0: 39.27, 1: 39.27, 2: 19.95, 3: 10.01, 4: 5.01, 5: 2.51}


# ---------------- quaternion helpers (scalar-first, torch) ----------------
def qmul(a, b):
    aw, ax, ay, az = a.unbind(-1); bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def qconj(q):
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device, dtype=q.dtype)


def rotvec_to_quat(v):
    th = v.norm(dim=-1, keepdim=True)
    k = torch.where(th > 1e-8, torch.sin(th / 2) / th.clamp_min(1e-8), 0.5 - th ** 2 / 48)
    return torch.cat([torch.cos(th / 2), v * k], -1)


def quat_to_rotvec(q):
    q = torch.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]; s = v.norm(dim=-1, keepdim=True)
    th = 2 * torch.atan2(s, q[..., :1])
    k = torch.where(s > 1e-8, th / s.clamp_min(1e-8), 2.0 + s ** 2 / 3)
    return v * k


# ---------------- soft targets ----------------
def soft_targets(q: np.ndarray, chart: np.ndarray, path: np.ndarray):
    """Returns root (N,4) and child (N,L,8) target distributions (float32)."""
    N, L = path.shape
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)

    def dist_deg(centres):  # centres (N,K,4)
        d = np.abs(np.einsum("nd,nkd->nk", qn, centres))
        return np.degrees(2 * np.arccos(np.clip(d, 0, 1)))

    def sm(d, tau):
        z = -(d / tau) ** 2
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return (e / e.sum(1, keepdims=True)).astype(np.float32)
    roots = np.stack([T.decode(np.full(N, c), np.zeros((N, 0), np.int64)) for c in range(4)], 1)
    root = sm(dist_deg(roots), TAU_DEG[0])
    child = np.empty((N, L, 8), np.float32)
    for l in range(L):
        cands = np.stack([T.decode(chart, np.concatenate([path[:, :l], np.full((N, 1), c)], 1)) for c in range(8)], 1)
        child[:, l] = sm(dist_deg(cands), TAU_DEG[l + 1])
    return root, child


def leaf_rotvec_targets(q: np.ndarray, chart: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Tangent residual target delta* = Log(q_leaf^-1 (x) q_gt), (N,3) radians."""
    qc = torch.as_tensor(T.decode(chart, path), dtype=torch.float64)
    qg = torch.as_tensor(q / np.linalg.norm(q, axis=1, keepdims=True), dtype=torch.float64)
    return quat_to_rotvec(qmul(qconj(qc), qg)).float().numpy()


def soft_ce(logits, target):
    return -(target * F.log_softmax(logits, -1)).sum(-1).mean()


# ---------------- residual head ----------------
class ResidualHead(nn.Module):
    def __init__(self, hidden: int = 512, n_freq: int = 6):
        super().__init__()
        self.n_freq = n_freq
        self.net = nn.Sequential(nn.Linear(hidden + 4 + 8 * n_freq, hidden), nn.GELU(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, h, q_leaf):
        return self.net(torch.cat([h, fourier(q_leaf, self.n_freq)], -1))


# ---------------- GRU path decoder ----------------
class HierGRU(nn.Module):
    """Same trunk/root as HierMLP; children from a GRU over (previous child, level, parent centre)."""

    def __init__(self, d_in: int, depth: int, hidden: int = 512, emb: int = 128, gru: int = 256, dropout: float = 0.1, n_freq: int = 6):
        super().__init__()
        self.depth, self.n_freq = depth, n_freq
        self.trunk = nn.Sequential(nn.LayerNorm(d_in), nn.Dropout(dropout), nn.Linear(d_in, hidden), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU())
        self.root = nn.Linear(hidden, T.N_CHARTS)
        self.h0 = nn.Linear(hidden, gru)
        self.prev_emb = nn.Embedding(9, emb)          # 0..7 = previous child, 8 = start (chart chosen)
        self.chart_emb = nn.Embedding(4, emb)
        self.level_emb = nn.Embedding(depth, emb)
        self.parent = nn.Linear(4 + 8 * n_freq, emb)
        self.cell = nn.GRUCell(3 * emb, gru)
        self.out = nn.Sequential(nn.Linear(gru + hidden, hidden), nn.GELU(), nn.Linear(hidden, T.N_CHILD))

    def encode(self, x):
        return self.trunk(x)

    def init_state(self, h, chart):
        return torch.tanh(self.h0(h))

    def step(self, h, state, chart, prev, level, parent_q):
        inp = torch.cat([self.prev_emb(prev) + self.chart_emb(chart), self.level_emb(level),
                         self.parent(fourier(parent_q, self.n_freq))], -1)
        state = self.cell(inp, state)
        return self.out(torch.cat([state, h], -1)), state

    def child_logits_tf(self, h, chart, path, parent_q):
        """Teacher-forced child logits (B,L,8)."""
        B, L = path.shape
        st = self.init_state(h, chart); outs = []
        prev = torch.full((B,), 8, dtype=torch.long, device=h.device)
        for l in range(L):
            lg, st = self.step(h, st, chart, prev, torch.full((B,), l, device=h.device, dtype=torch.long), parent_q[:, l])
            outs.append(lg); prev = path[:, l]
        return torch.stack(outs, 1)

    @torch.no_grad()
    def beam_search(self, x, beam: int = 1):
        h = self.encode(x); B = h.shape[0]; dev = x.device
        lp = F.log_softmax(self.root(h).float(), -1)
        k0 = min(beam, 4)
        score, chart = lp.topk(k0, -1)
        path = torch.zeros(B, k0, 0, dtype=torch.long, device=dev)
        st = self.init_state(h, None)[:, None].expand(B, k0, -1).reshape(B * k0, -1)
        prev = torch.full((B * k0,), 8, dtype=torch.long, device=dev)
        for l in range(self.depth):
            K = chart.shape[1]
            pq = decode_torch(chart.reshape(-1), path.reshape(B * K, l))
            hk = h[:, None].expand(B, K, h.shape[-1]).reshape(B * K, -1)
            lg, st = self.step(hk, st, chart.reshape(-1), prev, torch.full((B * K,), l, device=dev, dtype=torch.long), pq)
            tot = (score[:, :, None] + F.log_softmax(lg.float(), -1).view(B, K, 8)).view(B, K * 8)
            score, idx = tot.topk(min(beam, K * 8), -1)
            src, child = idx // 8, idx % 8
            chart = chart.gather(1, src)
            path = torch.cat([path.gather(1, src[:, :, None].expand(-1, -1, l)), child[:, :, None]], -1)
            st = st.view(B, K, -1).gather(1, src[:, :, None].expand(-1, -1, st.shape[-1])).reshape(B * src.shape[1], -1)
            prev = child.reshape(-1)
        return chart, path, score
