"""Training-free posterior fusion of several Tesseract probes inside beam search (EXP-025/026).

Rules combine the per-level conditional log-distributions (root: 4 charts, children: 8):
- ``single``: first model only
- ``poe``:    renormalized product  log p = sum_i log p_i
- ``avg``:    equal mixture         log p = log mean_i p_i
- ``ent``:    reliability-weighted product, weights w_i ∝ 1/(H_i + eps), normalized to sum to n
- ``margin``: reliability-weighted product, weights w_i ∝ (top1_i - top2_i) + eps, normalized to sum to n
All rules are fixed (no fitted parameters); equal weights reduce ``ent``/``margin`` to ``poe``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .predictor import decode_torch

EPS = 1e-3


def combine(lps: list[torch.Tensor], rule: str) -> torch.Tensor:
    if rule == "single" or len(lps) == 1:
        return lps[0]
    if rule == "poe":
        return F.log_softmax(sum(lps), -1)
    if rule == "avg":
        return torch.logsumexp(torch.stack(lps), 0) - np.log(len(lps))
    if rule in ("ent", "margin"):
        if rule == "ent":
            rel = [1.0 / (-(lp.exp() * lp).sum(-1, keepdim=True) + EPS) for lp in lps]
        else:
            rel = []
            for lp in lps:
                top = lp.exp().topk(2, -1).values
                rel.append((top[..., :1] - top[..., 1:2]) + EPS)
        tot = sum(rel)
        w = [len(lps) * r / tot for r in rel]
        return F.log_softmax(sum(wi * lp for wi, lp in zip(w, lps)), -1)
    raise ValueError(rule)


@torch.no_grad()
def fused_beam(models, xs, rule: str, beam: int = 1, return_h: bool = False, return_all: bool = False):
    """Beam search over the Tesseract tree with per-level fused distributions. Returns best (chart, path).

    Works for HierMLP (stateless children), phase3.HierGRU (stateful path decoder) and
    phase3.HierTransformer (prefix decoder over visual tokens) members.
    """
    dev = xs[0].device
    hs = [m.encode(x) for m, x in zip(models, xs)]
    B = hs[0].shape[0]
    gru = [isinstance(getattr(m, "cell", None), torch.nn.GRUCell) for m in models]
    lp = combine([F.log_softmax(m.root(h).float(), -1) for m, h in zip(models, hs)], rule)
    k0 = min(beam, 4)
    score, chart = lp.topk(k0, -1)
    path = torch.zeros(B, k0, 0, dtype=torch.long, device=dev)
    states = [m.init_state(h, None)[:, None].expand(B, k0, -1).reshape(B * k0, -1) if g else None
              for m, h, g in zip(models, hs, gru)]
    prev = torch.full((B * k0,), 8, dtype=torch.long, device=dev)
    for l in range(models[0].depth):
        K = chart.shape[1]
        pq = decode_torch(chart.reshape(-1), path.reshape(B * K, l))
        lev = torch.full((B * K,), l, device=dev, dtype=torch.long)
        clps, new_states = [], []
        for m, h, g, st in zip(models, hs, gru, states):
            hk = h[:, None].expand(B, K, *h.shape[1:]).reshape(B * K, *h.shape[1:])
            if getattr(m, "prefix_decoder", False):
                lg = m.child_logits_prefix(hk, chart.reshape(-1), path.reshape(B * K, l))
            elif g:
                lg, st = m.step(hk, st, chart.reshape(-1), prev, lev, pq)
            else:
                lg = m.child_logits(hk, lev, pq)
            clps.append(F.log_softmax(lg.float(), -1)); new_states.append(st)
        tot = (score[:, :, None] + combine(clps, rule).view(B, K, 8)).view(B, K * 8)
        score, idx = tot.topk(min(beam, K * 8), -1)
        src, child = idx // 8, idx % 8
        chart = chart.gather(1, src)
        path = torch.cat([path.gather(1, src[:, :, None].expand(-1, -1, l)), child[:, :, None]], -1)
        states = [st.view(B, K, -1).gather(1, src[:, :, None].expand(-1, -1, st.shape[-1])).reshape(B * src.shape[1], -1)
                  if st is not None else None for st in new_states]
        prev = child.reshape(-1)
    if return_all:   # every beam, best first: chart (B,K), path (B,K,L), joint log-prob score (B,K)
        return chart, path, score, hs
    if return_h:
        return chart[:, 0], path[:, 0], hs
    return chart[:, 0], path[:, 0]
