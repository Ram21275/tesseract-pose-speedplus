"""Frozen-backbone feature extraction with an immutable HDF5 cache.

Both backbones are frozen: eval(), requires_grad_(False), torch.no_grad, and
``assert_frozen`` checks that no parameter has a gradient.

Preprocessing (identical crop for both backbones, only resolution differs):
grayscale SPEED+ image -> GT square crop (bilinear, zero padding) -> resize
to the backbone resolution -> replicate to 3 channels -> [0, 1].
- DINOv3 (timm Eva port of Meta's LVD-1689M weights): 256x256, patch 16 ->
  16x16 tokens + 1 CLS + 4 registers; ImageNet mean/std normalisation.
- VGGT-1B: 518x518, patch 14 -> 37x37 tokens + 1 camera + 4 registers;
  VGGT normalises internally. Single frame (S=1), so the camera head carries
  no relative-pose signal and is not used as a feature.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from . import data as D

DINO_LAYERS = {"vit_base_patch16_dinov3.lvd1689m": [3, 7, 11],
               "vit_large_patch16_dinov3.lvd1689m": [5, 11, 17, 23]}
VGGT_LAYERS = [4, 11, 17, 23]
VGGT_MAP_RES = 74  # cached depth/point map resolution (518 / 7)


def assert_frozen(model: torch.nn.Module):
    assert not model.training, "backbone must be in eval mode"
    bad = [n for n, p in model.named_parameters() if p.requires_grad or p.grad is not None]
    assert not bad, f"backbone parameters not frozen: {bad[:3]}"


def freeze(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    model.requires_grad_(False)
    assert_frozen(model)
    return model


def preprocess_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


def load_crops(root: Path, rows, size: int) -> torch.Tensor:
    ims = []
    for r in rows:
        img = D.read_gray(root, r.image_relpath)
        ims.append(D.crop_square(img, r.crop_cx, r.crop_cy, r.crop_side, size))
    x = torch.from_numpy(np.stack(ims)).float().div_(255.0)
    return x[:, None].expand(-1, 3, -1, -1).contiguous()


class DinoV3:
    def __init__(self, name: str, device):
        import timm
        self.name = name
        self.model = freeze(timm.create_model(name, pretrained=True).to(device))
        cfg = self.model.pretrained_cfg
        self.size = cfg["input_size"][-1]
        self.mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
        self.layers = DINO_LAYERS[name]
        self.device = device
        self.spec = {"model": name, "hf_hub_id": cfg.get("hf_hub_id"), "size": self.size, "layers": self.layers,
                     "norm": "imagenet", "intermediate_norm": True, "crop": "gt_square", "gray_to_rgb": "replicate"}

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> dict:
        x = ((x.to(self.device) - self.mean) / self.std)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, inter = self.model.forward_intermediates(x, indices=self.layers, return_prefix_tokens=True,
                                                        norm=True, output_fmt="NLC")
        assert_frozen(self.model)
        out = {}
        for li, (patch, prefix) in zip(self.layers, inter):
            out[f"l{li}_cls"] = prefix[:, 0].float()
            out[f"l{li}_mean"] = patch.float().mean(1)
        g = int(round(inter[-1][0].shape[1] ** 0.5))
        out["last_grid"] = inter[-1][0].float().reshape(-1, g, g, inter[-1][0].shape[-1])  # (B,16,16,C)
        return out


class VGGTExtractor:
    def __init__(self, device, repo: str = "facebook/VGGT-1B"):
        from vggt.models.vggt import VGGT
        self.model = freeze(VGGT.from_pretrained(repo).to(device))
        self.size = 518
        self.device = device
        self._tokens = None
        self.model.aggregator.register_forward_hook(lambda m, i, o: setattr(self, "_tokens", o))
        self.spec = {"model": repo, "size": self.size, "layers": VGGT_LAYERS, "frames": 1, "map_res": VGGT_MAP_RES,
                     "autocast": "bf16", "crop": "gt_square", "gray_to_rgb": "replicate"}

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> dict:
        x = x.to(self.device)[:, None]  # (B, S=1, 3, H, W)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = self.model(x)
        assert_frozen(self.model)
        tokens, ps = self._tokens
        out = {}
        for li in VGGT_LAYERS:
            t = tokens[li][:, 0].float()  # (B, P, 2048)
            out[f"l{li}_cam"] = t[:, 0]
            out[f"l{li}_mean"] = t[:, ps:].mean(1)
        last = tokens[VGGT_LAYERS[-1]][:, 0, ps:].float()
        g = int(round(last.shape[1] ** 0.5))
        grid = last.reshape(-1, g, g, last.shape[-1]).permute(0, 3, 1, 2)
        out["last_grid8"] = F.adaptive_avg_pool2d(grid, 8).permute(0, 2, 3, 1)  # (B,8,8,2048)

        def pool(m):  # (B,H,W,C) -> (B,R,R,C)
            return F.adaptive_avg_pool2d(m.permute(0, 3, 1, 2).float(), VGGT_MAP_RES).permute(0, 2, 3, 1)
        out["depth"] = pool(pred["depth"][:, 0])
        out["depth_conf"] = pool(pred["depth_conf"][:, 0, ..., None])
        out["points"] = pool(pred["world_points"][:, 0])
        out["points_conf"] = pool(pred["world_points_conf"][:, 0, ..., None])
        return out


def cache_path(cache_root: Path, tag: str, spec: dict, manifest_hash: str) -> Path:
    return Path(cache_root) / f"{tag}__{preprocess_hash(spec)}__{manifest_hash}"


def extract_to_cache(extractor, root: Path, df, out_dir: Path, batch: int, code_version: str,
                     manifest_hash: str, log=print):
    """Writes features.h5 (fp16) + manifest.json. Refuses to overwrite."""
    out_dir = Path(out_dir)
    if (out_dir / "manifest.json").exists():
        log(f"cache exists, reusing: {out_dir}")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "features.h5.partial"
    rows = list(df.itertuples())
    n = len(rows)
    import time
    t0 = time.time()
    with h5py.File(tmp, "w") as f:
        f.create_dataset("image_relpath", data=np.array(df.image_relpath.tolist(), dtype="S"))
        dsets = {}
        for s in range(0, n, batch):
            x = load_crops(root, rows[s:s + batch], extractor.size)
            out = extractor(x)
            for k, v in out.items():
                v = v.cpu().numpy().astype(np.float16)
                if k not in dsets:
                    dsets[k] = f.create_dataset(k, shape=(n,) + v.shape[1:], dtype=np.float16,
                                                chunks=(min(64, n),) + v.shape[1:])
                dsets[k][s:s + len(v)] = v
            if (s // batch) % 20 == 0:
                log(f"  {s + len(x)}/{n}  {(time.time() - t0) / (s + len(x)) * 1000:.1f} ms/img")
    tmp.rename(out_dir / "features.h5")
    man = {"spec": extractor.spec, "preprocess_hash": preprocess_hash(extractor.spec), "manifest_hash": manifest_hash,
           "n": n, "code_version": code_version, "seconds": round(time.time() - t0, 1),
           "ms_per_image": round((time.time() - t0) / n * 1000, 2),
           "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else None}
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=2))
    return out_dir
