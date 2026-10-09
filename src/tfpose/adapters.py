"""LoRA adapters for frozen backbones (EXP-102: backbone adaptation, a separately named experiment).

Each targeted nn.Linear W (frozen) becomes  y = W x + (alpha / r) * B(A(x)),  with A ~ Kaiming, B = 0,
so an adapted model starts exactly at the frozen backbone. Only A and B are trainable.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def inject_lora(model: nn.Module, names=("qkv", "proj"), r: int = 8, alpha: float = 16.0, blocks_attr: str = "blocks") -> int:
    """Wrap attention Linear layers named in ``names`` inside every block's ``attn``. Returns #trainable params."""
    for p in model.parameters():
        p.requires_grad_(False)
    blocks = getattr(model, blocks_attr)
    n = 0
    for blk in blocks:
        attn = blk.attn
        for nm in names:
            lin = getattr(attn, nm, None)
            if isinstance(lin, nn.Linear):
                setattr(attn, nm, LoRALinear(lin, r, alpha))
                n += r * (lin.in_features + lin.out_features)
    return n


def lora_parameters(model: nn.Module):
    return [p for m in model.modules() if isinstance(m, LoRALinear) for p in (m.A, m.B)]


def lora_state_dict(model: nn.Module) -> dict:
    return {k: v for k, v in model.state_dict().items() if k.endswith(".A") or k.endswith(".B")}
