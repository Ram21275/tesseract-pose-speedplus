"""EXP-128: inference precision of the downstream heads (the frozen backbones are untouched).
  fp32 : no autocast (reference)
  bf16 : bf16 autocast (the precision used for every result before EXP-128)
  fp16 : fp16 autocast
  int8w: weight-only int8, simulated: every matmul weight (nn.Linear weights and nn.MultiheadAttention in-projection /
         out-projection weights) is rounded to symmetric per-output-channel int8 and de-quantized in place; activations,
         biases, LayerNorms, embeddings and tokens stay fp32 and the forward runs in fp32 (accuracy of an int8-weight
         deployment, not its speed; speed of real int8 kernels is measured separately)."""
from contextlib import nullcontext

import torch
import torch.nn as nn

PRECISIONS = ("fp32", "bf16", "fp16", "int8w")


def ctx(precision: str):
    if precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    assert precision in ("fp32", "int8w"), precision
    return nullcontext()


@torch.no_grad()
def _q8(w: torch.Tensor) -> torch.Tensor:
    s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
    return (w / s).round().clamp(-127, 127) * s


@torch.no_grad()
def fake_quant_int8_(model: nn.Module) -> dict:
    """In-place weight-only int8 round trip of all matmul weights. Returns counts for the report."""
    n_t = n_q = 0
    for mod in model.modules():
        ws = []
        if isinstance(mod, nn.Linear):
            ws.append(mod.weight)
        elif isinstance(mod, nn.MultiheadAttention):
            if mod.in_proj_weight is not None:
                ws.append(mod.in_proj_weight)
            else:
                ws += [mod.q_proj_weight, mod.k_proj_weight, mod.v_proj_weight]
        for w in ws:
            w.copy_(_q8(w.float()).to(w.dtype)); n_t += 1; n_q += w.numel()
    total = sum(p.numel() for p in model.parameters())
    return {"int8_tensors": n_t, "int8_params": n_q, "total_params": total, "int8_param_frac": n_q / total}


def prepare(model: nn.Module, precision: str) -> nn.Module:
    if precision == "int8w":
        model.int8_info = fake_quant_int8_(model)
    return model
