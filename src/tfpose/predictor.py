"""Hierarchical Tesseract predictor (common MLP head for Phase-1 controls).

trunk(feature) -> h
root:  logits over 4 charts from h
child: logits over 8 children from [h, level_emb(l), parent_emb(centre of parent cell)]

The parent cell is represented by the Fourier-encoded quaternion of its
centre, so the same head generalises across the 4*8^(l-1) parents of a level.
Training uses teacher forcing (GT parent); inference never sees the GT path.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import tesseract as T


def parent_centres(chart: np.ndarray, path: np.ndarray) -> np.ndarray:
    """(N, depth, 4): centre of the parent cell for each child level (level 0 parent = chart root)."""
    n, depth = path.shape
    out = np.empty((n, depth, 4))
    for l in range(depth):
        out[:, l] = T.decode(chart, path[:, :l])
    return out


def fourier(q: torch.Tensor, n_freq: int = 6) -> torch.Tensor:
    f = 2.0 ** torch.arange(n_freq, device=q.device, dtype=q.dtype) * np.pi
    a = q[..., None] * f
    return torch.cat([q, a.sin().flatten(-2), a.cos().flatten(-2)], -1)


class HierMLP(nn.Module):
    def __init__(self, d_in: int, depth: int, hidden: int = 512, emb: int = 128, dropout: float = 0.1, n_freq: int = 6):
        super().__init__()
        self.depth, self.n_freq = depth, n_freq
        self.trunk = nn.Sequential(nn.LayerNorm(d_in), nn.Dropout(dropout), nn.Linear(d_in, hidden), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU())
        self.root = nn.Linear(hidden, T.N_CHARTS)
        self.level_emb = nn.Embedding(depth, emb)
        self.parent = nn.Linear(4 + 8 * n_freq, emb)
        self.child = nn.Sequential(nn.Linear(hidden + emb, hidden), nn.GELU(), nn.Linear(hidden, T.N_CHILD))

    def encode(self, x):
        return self.trunk(x)

    def child_logits(self, h, level: torch.Tensor, parent_q: torch.Tensor):
        e = self.level_emb(level) + self.parent(fourier(parent_q, self.n_freq))
        return self.child(torch.cat([h, e], -1))

    def loss(self, x, chart, path, parent_q):
        """Teacher-forced CE. chart (B,), path (B,L), parent_q (B,L,4). Returns total and per-level losses."""
        h = self.encode(x)
        losses = [F.cross_entropy(self.root(h), chart)]
        B, L = path.shape
        hl = h[:, None].expand(B, L, h.shape[-1]).reshape(B * L, -1)
        lev = torch.arange(L, device=x.device).repeat(B)
        logits = self.child_logits(hl, lev, parent_q.reshape(B * L, 4)).view(B, L, 8)
        for l in range(L):
            losses.append(F.cross_entropy(logits[:, l], path[:, l]))
        return sum(losses), losses

    @torch.no_grad()
    def beam_search(self, x, beam: int = 1, depth: int | None = None):
        """Returns chart (B,K), path (B,K,depth), logp (B,K) sorted best-first, nodes_scored (int per image)."""
        depth = depth or self.depth
        h = self.encode(x)
        B = h.shape[0]
        lp = F.log_softmax(self.root(h).float(), -1)
        k0 = min(beam, T.N_CHARTS)
        score, chart = lp.topk(k0, -1)
        path = torch.zeros(B, k0, 0, dtype=torch.long, device=x.device)
        nodes = T.N_CHARTS
        for l in range(depth):
            K = chart.shape[1]
            pq = T.decode(chart.reshape(-1).cpu().numpy(), path.reshape(B * K, l).cpu().numpy())
            pq = torch.as_tensor(pq, dtype=h.dtype, device=x.device)
            hk = h[:, None].expand(B, K, h.shape[-1]).reshape(B * K, -1)
            lev = torch.full((B * K,), l, device=x.device, dtype=torch.long)
            clp = F.log_softmax(self.child_logits(hk, lev, pq).float(), -1).view(B, K, 8)
            nodes += K * T.N_CHILD
            tot = (score[:, :, None] + clp).view(B, K * 8)
            score, idx = tot.topk(min(beam, K * 8), -1)
            src, child = idx // 8, idx % 8
            chart = chart.gather(1, src)
            path = torch.cat([path.gather(1, src[:, :, None].expand(-1, -1, l)), child[:, :, None]], -1)
        return chart, path, score, nodes


_FREE_T = {}


def decode_torch(chart: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
    """GPU Tesseract cell centres, identical to tesseract.decode. chart (...,), path (..., l) -> (..., 4).

    After l levels the free coordinate is lo + 2^-l with lo = -1 + sum_t b_t * 2^(1-t) (t = 1..l).
    """
    dev = path.device
    if dev not in _FREE_T:
        _FREE_T[dev] = torch.as_tensor(T.FREE, device=dev)
    l = path.shape[-1]
    if l:
        w = 2.0 ** (1 - torch.arange(1, l + 1, device=dev, dtype=torch.float32))      # (l,)
        bits = torch.stack([(path >> k) & 1 for k in range(3)], -1).float()          # (..., l, 3)
        u = -1.0 + (bits * w[:, None]).sum(-2) + 2.0 ** (-l)                          # (..., 3)
    else:
        u = torch.zeros(path.shape[:-1] + (3,), device=dev)
    x = torch.ones(path.shape[:-1] + (4,), device=dev)
    x.scatter_(-1, _FREE_T[dev][chart], u)
    return F.normalize(x, dim=-1)


@torch.no_grad()
def beam_search_fast(model: "HierMLP", x, beam: int = 1, depth: int | None = None):
    """Same search as HierMLP.beam_search but fully on the device (no NumPy decode, no host syncs)."""
    depth = depth or model.depth
    h = model.encode(x)
    B = h.shape[0]
    lp = F.log_softmax(model.root(h).float(), -1)
    k0 = min(beam, T.N_CHARTS)
    score, chart = lp.topk(k0, -1)
    path = torch.zeros(B, k0, 0, dtype=torch.long, device=x.device)
    for l in range(depth):
        K = chart.shape[1]
        pq = decode_torch(chart.reshape(-1), path.reshape(B * K, l)).to(h.dtype)
        hk = h[:, None].expand(B, K, h.shape[-1]).reshape(B * K, -1)
        lev = torch.full((B * K,), l, device=x.device, dtype=torch.long)
        clp = F.log_softmax(model.child_logits(hk, lev, pq).float(), -1).view(B, K, 8)
        tot = (score[:, :, None] + clp).view(B, K * 8)
        score, idx = tot.topk(min(beam, K * 8), -1)
        src, child = idx // 8, idx % 8
        chart = chart.gather(1, src)
        path = torch.cat([path.gather(1, src[:, :, None].expand(-1, -1, l)), child[:, :, None]], -1)
    return chart, path, score


class FlatAnchorMLP(nn.Module):
    """SPACE-HOP-style flat classifier over K anchor rotations, on the SAME trunk as HierMLP.

    Prediction = argmax over K logits -> anchor rotation (no continuous offset; EXP-101 compares
    discrete prediction only, like the Tesseract head without residual).
    """

    def __init__(self, d_in: int, n_anchors: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(d_in), nn.Dropout(dropout), nn.Linear(d_in, hidden), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU())
        self.cls = nn.Linear(hidden, n_anchors)

    def forward(self, x):
        return self.cls(self.trunk(x))
