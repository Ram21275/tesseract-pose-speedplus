"""Arm K (EXP-105 / EXP-106): the original Kurz, Pfaff & Hanebeck (MFI 2017) corner formulation as a tree.

Differences from `tesseract` (centre representatives, 4 positive facets, unique paths):
- all 8 facets (axis a = f % 4, sign + for f < 4, - for f >= 4), so q and -q are both represented;
  a facet-f cell with free coordinates u is the 4D point x_a = sign, x[FREE[a]] = u (no sign flip on u);
- rotations are represented by the CORNERS of the depth-L cubes (radially projected); a corner is shared by up to
  8 cubes (across facets at facet boundaries), so one rotation has many root-to-leaf paths;
- training target: the nearest corner of q and of -q; uniform mass over every leaf cube that has either corner
  (<= 8 per copy, <= 16 in total);
- decoding: top-k leaves vote their probability onto their 8 corners; votes for the same rotation (corner z or -z)
  are summed; prediction = the corner with the largest total.
Corners are handled as integer grids k in {0..n}^4 (n = 2^L, z = -1 + 2k/n), which are shared exactly across facets.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import tesseract as T
from .fusion import combine
from .phase3 import HierTransformer
from .predictor import fourier

N_FACETS = 8
KMAX = 16
_FREE_T = {}


def _free(dev):
    if dev not in _FREE_T:
        _FREE_T[dev] = torch.as_tensor(T.FREE, device=dev)
    return _FREE_T[dev]


def decode8_torch(chart: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
    """Signed cell-centre quaternions of 8-facet cells. chart (...,) in [0, 8), path (..., l) -> (..., 4)."""
    dev = path.device
    l = path.shape[-1]
    if l:
        w = 2.0 ** (1 - torch.arange(1, l + 1, device=dev, dtype=torch.float32))
        bits = torch.stack([(path >> k) & 1 for k in range(3)], -1).float()
        u = -1.0 + (bits * w[:, None]).sum(-2) + 2.0 ** (-l)
    else:
        u = torch.zeros(path.shape[:-1] + (3,), device=dev)
    sign = 1.0 - 2.0 * (chart >= 4).float()
    x = sign[..., None].expand(path.shape[:-1] + (4,)).clone()
    x.scatter_(-1, _free(dev)[chart % 4], u)
    return F.normalize(x, dim=-1)


class HierTransformer8(HierTransformer):
    """EXP-036 Transformer tree with an 8-way facet root and signed cell embeddings."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.root_head = nn.Linear(self.root_head.in_features, N_FACETS)

    def _seq(self, chart, path):
        B, l = path.shape
        toks = [self.root_tok.expand(B, -1, -1)]
        for j in range(l + 1):
            qc = decode8_torch(chart, path[:, :j])
            lev = torch.full((B,), j, device=path.device, dtype=torch.long)
            toks.append((self.level_emb(lev) + self.cell_proj(fourier(qc, self.n_freq)))[:, None])
        return torch.cat(toks, 1)


def corner_targets(q: np.ndarray, depth: int):
    """Multi-path targets. q (N, 4) -> chart (N, KMAX), path (N, KMAX, depth), valid (N, KMAX) bool."""
    n = 2 ** depth
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    y = q / np.abs(q).max(1, keepdims=True)
    k = np.clip(np.rint((y + 1) * n / 2).astype(np.int64), 0, n)
    N = len(q)
    charts, paths, valid = [], [], []
    for kc in (k, n - k):                                  # the corner of q and of -q
        for a in range(4):
            on = (kc[:, a] == 0) | (kc[:, a] == n)
            f = a + 4 * (kc[:, a] == 0)
            kk = kc[:, T.FREE[a]]                          # (N, 3) free corner ints
            for o in range(8):
                off = np.array([(o >> j) & 1 for j in range(3)])
                i = kk - 1 + off                           # cube index per free axis
                ok = on & np.all((i >= 0) & (i <= n - 1), 1)
                i = np.clip(i, 0, n - 1)
                p = np.zeros((N, depth), np.int64)
                for l in range(depth):
                    b = (i >> (depth - 1 - l)) & 1
                    p[:, l] = b[:, 0] + 2 * b[:, 1] + 4 * b[:, 2]
                charts.append(f); paths.append(p); valid.append(ok)
    C, P, V = np.stack(charts, 1), np.stack(paths, 1), np.stack(valid, 1)          # (N, 64, ...)
    cnt = V.sum(1)
    assert cnt.min() >= 1 and cnt.max() <= KMAX, (cnt.min(), cnt.max())
    order = np.argsort(~V, axis=1, kind="stable")[:, :KMAX]
    take = lambda A: np.take_along_axis(A, order if A.ndim == 2 else order[..., None], 1)
    return take(C), take(P), take(V)


@torch.no_grad()
def beam8(models, xs, rule: str, beam: int = 8, return_h: bool = False):
    """Per-level fused beam search over the 8-facet tree. Returns chart (B,K), path (B,K,L), score (B,K) best first."""
    dev = xs[0].device
    hs = [m.encode(x) for m, x in zip(models, xs)]
    B = hs[0].shape[0]
    lp = combine([F.log_softmax(m.root(h).float(), -1) for m, h in zip(models, hs)], rule)
    score, chart = lp.topk(min(beam, N_FACETS), -1)
    path = torch.zeros(B, chart.shape[1], 0, dtype=torch.long, device=dev)
    for l in range(models[0].depth):
        K = chart.shape[1]
        clps = []
        for m, h in zip(models, hs):
            hk = h[:, None].expand(B, K, *h.shape[1:]).reshape(B * K, *h.shape[1:])
            clps.append(F.log_softmax(m.child_logits_prefix(hk, chart.reshape(-1), path.reshape(B * K, l)).float(), -1))
        tot = (score[:, :, None] + combine(clps, rule).view(B, K, 8)).view(B, K * 8)
        score, idx = tot.topk(min(beam, K * 8), -1)
        src, child = idx // 8, idx % 8
        chart = chart.gather(1, src)
        path = torch.cat([path.gather(1, src[:, :, None].expand(-1, -1, l)), child[:, :, None]], -1)
    return (chart, path, score, hs) if return_h else (chart, path, score)


def corner_vote(chart: torch.Tensor, path: torch.Tensor, score: torch.Tensor):
    """Corner-vote decoding. chart (B,K), path (B,K,L), score (B,K) joint log-probs -> (B, 4) quaternions."""
    B, K, L = path.shape
    n = 2 ** L
    dev = path.device
    w = 2 ** (L - 1 - torch.arange(L, device=dev))
    bits = torch.stack([(path >> j) & 1 for j in range(3)], -1)                     # (B,K,L,3)
    i = (bits * w[:, None]).sum(-2)                                                 # (B,K,3) cube index per free axis
    off = torch.tensor([[(o >> j) & 1 for j in range(3)] for o in range(8)], device=dev)
    kk = i[:, :, None] + off                                                        # (B,K,8,3)
    a = chart % 4
    k4 = torch.where(chart >= 4, 0, n)[:, :, None, None].expand(B, K, 8, 4).clone()
    k4.scatter_(-1, _free(dev)[a][:, :, None].expand(B, K, 8, 3), kk)
    k4 = k4.reshape(B, K * 8, 4)
    base = (n + 1) ** torch.arange(4, device=dev)
    key = torch.minimum((k4 * base).sum(-1), ((n - k4) * base).sum(-1))           # z and -z share a key
    p = score.float().exp()[:, :, None].expand(B, K, 8).reshape(B, K * 8)
    tot = ((key[:, :, None] == key[:, None, :]).float() * p[:, None, :]).sum(-1)
    best = tot.argmax(1)
    z = -1.0 + 2.0 * k4[torch.arange(B, device=dev), best].float() / n
    return F.normalize(z, dim=-1)
