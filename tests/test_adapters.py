import torch
import torch.nn as nn

from tfpose.adapters import LoRALinear, inject_lora, lora_parameters


class _Attn(nn.Module):
    def __init__(self):
        super().__init__(); self.qkv = nn.Linear(8, 24); self.proj = nn.Linear(8, 8)


class _Blk(nn.Module):
    def __init__(self):
        super().__init__(); self.attn = _Attn(); self.mlp = nn.Linear(8, 8)


class _Net(nn.Module):
    def __init__(self):
        super().__init__(); self.blocks = nn.ModuleList([_Blk(), _Blk()])

    def forward(self, x):
        for b in self.blocks:
            x = b.mlp(b.attn.proj(b.attn.qkv(x)[:, :8]))
        return x


def test_lora_starts_at_frozen_model_and_only_adapters_train():
    torch.manual_seed(0)
    net = _Net(); x = torch.randn(5, 8); y0 = net(x).detach()
    n = inject_lora(net, r=2)
    assert torch.allclose(net(x), y0, atol=1e-6)                 # B = 0 -> identical at init
    trainable = [p for p in net.parameters() if p.requires_grad]
    assert len(trainable) == len(lora_parameters(net)) == 8 and sum(p.numel() for p in trainable) == n
    net(x).sum().backward()
    assert all(p.grad is None for m in net.modules() if isinstance(m, nn.Linear) for p in m.parameters())
