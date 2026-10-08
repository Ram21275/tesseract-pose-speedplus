"""Named, deterministic feature vectors built from cached backbone outputs.

Spec string: ``<backbone>:<featset>``, e.g. ``dinov3_vitb16:l11_cls+mean``.
featsets:
- ``l<i>_<a>[+<b>]``   concatenation of cached per-layer vectors (cls/mean/cam)
- ``grid4``            last-layer patch grid, 4x4 average pooled, flattened
- ``depthconf16``      VGGT depth / per-image median depth and log depth-confidence, 16x16 pooled
- ``points16``         VGGT point map centred (conf-weighted) and RMS-scaled, plus log conf, 16x16 pooled
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


def load(spec: str, subset: str):
    backbone, fs = spec.split(":")
    path = find_cache(backbone, subset)
    with h5py.File(path / "features.h5", "r") as f:
        names = [s.decode() for s in f["image_relpath"][:]]
        if fs == "grid4":
            key = "last_grid" if "last_grid" in f else "last_grid8"
            X = _pool(f[key], 4).reshape(len(names), -1)
        elif fs == "depthconf16":
            d = f["depth"][:].astype(np.float32)
            c = f["depth_conf"][:].astype(np.float32)
            d = d / np.median(d.reshape(len(d), -1), 1)[:, None, None, None]
            X = np.concatenate([_pool(d, 16), _pool(np.log(c), 16)], -1).reshape(len(names), -1)
        elif fs == "points16":
            p = f["points"][:].astype(np.float32)
            c = f["points_conf"][:].astype(np.float32)
            w = c / c.sum((1, 2, 3), keepdims=True)
            mu = (p * w).sum((1, 2), keepdims=True)
            p = p - mu
            p = p / np.sqrt((p ** 2 * w).sum((1, 2, 3), keepdims=True))
            X = np.concatenate([_pool(p, 16), _pool(np.log(c), 16)], -1).reshape(len(names), -1)
        else:
            layer, parts = fs.split("_", 1)
            X = np.concatenate([f[f"{layer}_{p}"][:].astype(np.float32) for p in parts.split("+")], 1)
    man = json.loads((path / "manifest.json").read_text())
    return names, X.astype(np.float32), {"cache": str(path.relative_to(REPO)), "cache_manifest": man}
