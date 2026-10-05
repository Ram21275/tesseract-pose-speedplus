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
