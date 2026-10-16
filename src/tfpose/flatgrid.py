"""EXP-113: matched flat Hopf-anchor classifier. It is the Tesseract tree's own network (same input layer, positional
encoding, encoder and 2-layer decoder) with only the [ROOT] query token and a K-way head over a count-matched
SPACE-HOP Hopf grid (10,923 directions x 12 rolls = 131,076 anchors vs 131,072 Tesseract leaves). The flat head is
necessarily larger (d x K parameters), which is inherent to flat classification and reported."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .grids import spacehop_hopf_grid
from .phase3 import HierTransformer

HOPF_POINTS, HOPF_ROLLS = 10923, 12
TAU_LEAF_DEG = 2.51           # = phase3.TAU_DEG[5], the tree's leaf-level soft-target temperature
TOPK_TARGET = 32


def matched_hopf_grid() -> np.ndarray:
    return spacehop_hopf_grid(HOPF_POINTS, HOPF_ROLLS)


class FlatTransformer(HierTransformer):
    def __init__(self, token_shape, n_anchors: int, **k):
        super().__init__(token_shape, 5, **k)
        self.flat_head = nn.Linear(self.root_head.in_features, n_anchors)

    def logits(self, mem):
        return self.flat_head(self._run(mem, self.root_tok.expand(mem.shape[0], -1, -1))[:, 0])


def soft_anchor_targets(q: torch.Tensor, G: torch.Tensor, tau_deg: float = TAU_LEAF_DEG, k: int = TOPK_TARGET):
    """Geodesic soft targets over the k nearest anchors (as the tree's leaf level): indices (B,k), weights (B,k)."""
    d, idx = (q @ G.T).abs().topk(k, -1)
    ang = torch.rad2deg(2 * torch.arccos(d.clamp(max=1.0)))
    w = torch.softmax(-(ang / tau_deg) ** 2, -1)
    return idx, w


def soft_ce(logits, idx, w):
    return -(w * F.log_softmax(logits.float(), -1).gather(1, idx)).sum(-1).mean()


def nearest_anchor(q: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    return (q @ G.T).abs().argmax(-1)
