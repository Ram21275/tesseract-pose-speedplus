"""Named, deterministic feature vectors built from cached backbone outputs.

Spec string: ``<backbone>:<featset>``, e.g. ``dinov3_vitb16:l11_cls+mean``.
featsets:
- ``l<i>_<a>[+<b>]``   concatenation of cached per-layer vectors (cls/mean/cam)
- ``grid4`` / ``grid2`` last-layer patch grid, 4x4 / 2x2 average pooled, flattened
- ``tokens8``          DINOv3 last-layer tokens pooled 16x16 -> 8x8, flattened (64*C), fp16 (EXP-032/036)
- ``parts4``           DINOv3 tokens pooled by 4 soft spectral parts (EXP-019) + part centroid and second moments
- ``depthconf16``      depth / per-image median and log confidence (VGGT) or masked median + mask (MoGe-2), 16x16 pooled
- ``points16``         point map centred (conf/mask-weighted) and RMS-scaled, plus conf (VGGT: log) or mask (MoGe-2), 16x16 pooled
- ``normals16``        MoGe-2 surface normals (masked) plus foreground mask, 16x16 pooled
- suffix ``~and``     use the EXP-015 consensus mask: ``normals16~and``, ``grid4~and`` (mask-weighted 4x4 pooling)
No statistics are fitted here; standardisation happens in the predictor pipeline
with train-split statistics only.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]


def find_cache(backbone: str, subset: str) -> Path:
    hits = sorted((REPO / "outputs/shared_cache").glob(f"{backbone}__subset_{subset}__*"))
    hits = [h for h in hits if (h / "manifest.json").exists()]
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one complete cache for {backbone}/{subset}, found {hits}")
    return hits[0]


def _pool(a, r: int, chunk: int = 256) -> np.ndarray:
    """(N,H,W,C) array or HDF5 dataset -> (N,r,r,C) float32, processed in chunks to bound memory."""
    out = []
    for s in range(0, a.shape[0], chunk):
        t = torch.from_numpy(np.asarray(a[s:s + chunk], dtype=np.float32)).permute(0, 3, 1, 2)
        out.append(F.adaptive_avg_pool2d(t, r).permute(0, 2, 3, 1).numpy())
    return np.concatenate(out)


def consensus_mask(subset: str, names: list[str]) -> np.ndarray:
    """EXP-015 training-free AND mask (MoGe-2 mask AND DINOv3 PCA foreground), (N,74,74,1) float32."""
    p = REPO / f"outputs/shared_cache/consensus_masks__subset_{subset}"
    if not (p / "manifest.json").exists():
        raise FileNotFoundError(p)
    with h5py.File(p / "masks.h5", "r") as f:
        if [s.decode() for s in f["image_relpath"][:]] != names:
            raise ValueError("consensus mask order does not match feature cache")
        return f["and_mask"][:].astype(np.float32)[..., None]


def load(spec: str, subset: str):
    backbone, fs = spec.split(":")
    fs, _, mask_src = fs.partition("~")      # e.g. normals16~and : use the EXP-015 consensus mask
    if mask_src not in ("", "and"):
        raise ValueError(f"unknown mask source {mask_src}")
    path = find_cache(backbone, subset)
    with h5py.File(path / "features.h5", "r") as f:
        names = [s.decode() for s in f["image_relpath"][:]]
        if fs == "grid4" and mask_src == "and":  # mask-weighted 4x4 pooling of the 16x16 token grid (EXP-016)
            m = _pool(consensus_mask(subset, names), 16)[..., 0]           # (N,16,16) area fraction
            out = []
            for s0 in range(0, len(names), 256):
                G = np.asarray(f["last_grid"][s0:s0 + 256], dtype=np.float32)  # (n,16,16,C)
                w = m[s0:s0 + 256]
                n, H, W, C = G.shape
                Gc = G.reshape(n, 4, 4, 4, 4, C)                       # (n, cy, py, cx, px, C)
                wc = w.reshape(n, 4, 4, 4, 4)[..., None]
                num = (Gc * wc).sum((2, 4)); den = wc.sum((2, 4))
                unweighted = Gc.mean((2, 4))
                out.append(np.where(den > 1e-6, num / np.maximum(den, 1e-6), unweighted))  # empty cell -> plain mean
            X = np.concatenate(out).reshape(len(names), -1)
        elif fs == "tokens8":  # EXP-032/036: 16x16 token grid pooled to 8x8, kept fp16 (N, 64*C)
            ds = f["last_grid"]; N, C = ds.shape[0], ds.shape[-1]
            X = np.empty((N, 64 * C), np.float16)                      # filled chunk-wise: no full float32 copy
            for s0 in range(0, N, 1024):
                X[s0:s0 + 1024] = _pool(ds[s0:s0 + 1024], 8).astype(np.float16).reshape(-1, 64 * C)
        elif fs in ("grid4", "grid2"):
            key = "last_grid" if "last_grid" in f else "last_grid8"
            X = _pool(f[key], int(fs[-1])).reshape(len(names), -1)
        elif fs == "parts4":  # EXP-019: spectral part pooling (fused-graph eigvecs 2-5, weights v^2)
            pp = REPO / f"outputs/shared_cache/spectral_parts__subset_{subset}"
            with h5py.File(pp / "parts.h5", "r") as fp:
                if [s.decode() for s in fp["image_relpath"][:]] != names:
                    raise ValueError("parts cache order does not match feature cache")
                V = fp["eigvecs_2_5"][:].astype(np.float32).reshape(len(names), 256, 4)
            W = V ** 2
            W = W / (W.sum(1, keepdims=True) + 1e-12)                       # (N,256,4) soft part weights, sign-invariant
            yy, xx = np.meshgrid((np.arange(16) + 0.5) / 16, (np.arange(16) + 0.5) / 16, indexing="ij")
            P = np.stack([xx.reshape(-1), yy.reshape(-1)], 1).astype(np.float32)   # (256,2)
            out = []
            for s0 in range(0, len(names), 256):
                G = np.asarray(f["last_grid"][s0:s0 + 256], dtype=np.float32).reshape(-1, 256, f["last_grid"].shape[-1])
                w = W[s0:s0 + 256]
                feat = np.einsum("npk,npc->nkc", w, G)                     # (n,4,C) weighted mean tokens
                cen = np.einsum("npk,pd->nkd", w, P)                        # (n,4,2) centroids
                dx = P[None, :, None, :] - cen[:, None, :, :]               # (n,256,4,2)
                mom = np.stack([(w * dx[..., 0] ** 2).sum(1), (w * dx[..., 0] * dx[..., 1]).sum(1), (w * dx[..., 1] ** 2).sum(1)], -1)
                out.append(np.concatenate([feat, cen, mom], -1).reshape(len(G), -1))   # (n, 4*(C+5))
            X = np.concatenate(out)
        elif fs == "depthconf16":
            d = f["depth"][:].astype(np.float32)
            if "depth_conf" in f:  # VGGT: confidence map, log-scaled
                c = f["depth_conf"][:].astype(np.float32)
                d = d / np.median(d.reshape(len(d), -1), 1)[:, None, None, None]
                X = np.concatenate([_pool(d, 16), _pool(np.log(c), 16)], -1)
            else:  # MoGe-2: foreground mask in [0,1]; depth / masked median, background zeroed
                m = f["mask"][:].astype(np.float32)
                med = np.array([np.median(di[mi > 0.5]) if (mi > 0.5).any() else np.median(di)
                                for di, mi in zip(d, m)], dtype=np.float32)
                d = d / med[:, None, None, None] * m
                X = np.concatenate([_pool(d, 16), _pool(m, 16)], -1)
            X = X.reshape(len(names), -1)
        elif fs == "points16":
            p = f["points"][:].astype(np.float32)
            c = f["points_conf"][:].astype(np.float32) if "points_conf" in f else f["mask"][:].astype(np.float32) + 1e-6
            w = c / c.sum((1, 2, 3), keepdims=True)
            mu = (p * w).sum((1, 2), keepdims=True)
            p = p - mu
            p = p / np.sqrt((p ** 2 * w).sum((1, 2, 3), keepdims=True))
            cfeat = np.log(c) if "points_conf" in f else c
            X = np.concatenate([_pool(p, 16), _pool(cfeat, 16)], -1).reshape(len(names), -1)
        elif fs == "normals16":  # MoGe-2 surface normals (camera frame) + foreground mask
            nrm = f["normal"][:].astype(np.float32)
            m = consensus_mask(subset, names) if mask_src == "and" else f["mask"][:].astype(np.float32)
            X = np.concatenate([_pool(nrm * m, 16), _pool(m, 16)], -1).reshape(len(names), -1)
        else:
            layer, parts = fs.split("_", 1)
            X = np.concatenate([f[f"{layer}_{p}"][:].astype(np.float32) for p in parts.split("+")], 1)
    man = json.loads((path / "manifest.json").read_text())
    X = X if X.dtype == np.float16 else X.astype(np.float32)
    return names, X, {"cache": str(path.relative_to(REPO)), "cache_manifest": man}


def standardize_to_tensor(X: np.ndarray, train_mask: np.ndarray, device, dtype, pin: bool = False, chunk: int = 4096):
    """Train-split standardization, chunked (float64 statistics), written into a torch tensor.

    Avoids materializing a full float32 copy of very large feature matrices.
    """
    tr = np.flatnonzero(train_mask)
    s1 = np.zeros(X.shape[1]); s2 = np.zeros(X.shape[1])
    for i in range(0, len(tr), chunk):
        c = X[tr[i:i + chunk]].astype(np.float64); s1 += c.sum(0); s2 += (c ** 2).sum(0)
    mu = s1 / len(tr); sd = np.sqrt(np.maximum(s2 / len(tr) - mu ** 2, 0)) + 1e-6
    out = torch.empty(X.shape, dtype=dtype, device=device, pin_memory=pin and str(device) == "cpu")
    for i in range(0, len(X), chunk):
        out[i:i + chunk] = torch.from_numpy(((X[i:i + chunk].astype(np.float32) - mu) / sd).astype(np.float32)).to(dtype)
    return out
