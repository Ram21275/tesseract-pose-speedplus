"""Training-free foreground masks and their evaluation (EXP-015).

All masks live on the 74x74 grid that covers the GT square crop (the MoGe-2
map grid). Ground truth is used for evaluation only.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import data as D

RES = 74          # evaluation grid
RASTER = 518      # GT rasterization resolution (same as MoGe-2 input), then area-pooled to RES


def _box_corners(xs, ys, zs):
    return np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)


def silhouette_parts(pts: np.ndarray, model: str) -> list[np.ndarray]:
    """3D point sets whose projected convex hulls are unioned into the silhouette.

    pts: (11,3) Tango points; 0-3 top quad (z=0.3215), 4-7 bottom quad (z=0), 8-10 antenna tips.
    model 'hull8'  : one convex hull of the 8 body points.
    model 'box+plate': bus box (bottom-quad footprint extruded to the top height) U top plate quad.
    """
    body = pts[:8]
    if model == "hull8":
        return [body]
    if model == "two-quads":   # union of the top quad and the bottom quad as flat plates
        return [pts[0:4], pts[4:8]]
    if model == "top-quad":
        return [pts[0:4]]
    if model == "box+plate":
        bot = pts[4:8]
        ztop = pts[0:4, 2].max()
        box = _box_corners([bot[:, 0].min(), bot[:, 0].max()], [bot[:, 1].min(), bot[:, 1].max()], [bot[:, 2].min(), ztop])
        return [box, pts[0:4]]
    raise ValueError(model)


def to_crop(uv: np.ndarray, cx: float, cy: float, side: float, out: int) -> np.ndarray:
    """Full-image pixel coords -> coords in an out x out crop (same transform as data.crop_square)."""
    s = out / side
    return np.stack([(uv[:, 0] - cx) * s + out / 2, (uv[:, 1] - cy) * s + out / 2], 1)


def gt_silhouette(pts, q, t, cam, cx, cy, side, model: str) -> np.ndarray:
    """Soft GT body silhouette on the RES grid (fraction of each cell covered)."""
    canvas = np.zeros((RASTER, RASTER), np.uint8)
    shift = 4  # sub-pixel rasterization
    for part in silhouette_parts(pts, model):
        uv = to_crop(D.project_points(part, q, t, cam), cx, cy, side, RASTER)
        hull = cv2.convexHull((uv * (1 << shift)).round().astype(np.int32))
        cv2.fillConvexPoly(canvas, hull, 1, lineType=cv2.LINE_8, shift=shift)
    return cv2.resize(canvas.astype(np.float32), (RES, RES), interpolation=cv2.INTER_AREA)


def tip_points(pts, q, t, cam, cx, cy, side) -> np.ndarray:
    """Antenna-tip keypoints (8-10) in RES-grid coordinates."""
    return to_crop(D.project_points(pts[8:11], q, t, cam), cx, cy, side, RES)


def dino_pca_score(tokens: np.ndarray) -> np.ndarray:
    """(16,16,C) patch tokens -> (RES,RES) foreground score via per-image PCA (first component).

    Sign: border ring (outer patches) is background, i.e. border mean < interior mean.
    """
    g = tokens.shape[0]
    X = tokens.reshape(g * g, -1).astype(np.float64)
    X = X - X.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    s = (X @ vt[0]).reshape(g, g)
    ring = np.zeros((g, g), bool)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    if s[ring].mean() > s[~ring].mean():
        s = -s
    t = torch.from_numpy(s.astype(np.float32))[None, None]
    return F.interpolate(t, size=(RES, RES), mode="bilinear", align_corners=False)[0, 0].numpy()


def otsu(score: np.ndarray, bins: int = 256) -> float:
    """Otsu threshold on a real-valued map (deterministic, per image)."""
    lo, hi = float(score.min()), float(score.max())
    if hi - lo < 1e-12:
        return hi
    h, edges = np.histogram(score, bins=bins, range=(lo, hi))
    p = h / h.sum()
    w0 = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + edges[1:]) / 2)
    mt = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mt * w0 - mu) ** 2 / (w0 * (1 - w0))
    sb[~np.isfinite(sb)] = -1
    return float(edges[int(np.argmax(sb)) + 1])


def boundary(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT)).astype(bool)


def mask_metrics(pred: np.ndarray, gt: np.ndarray, tips: np.ndarray, tol: int = 2) -> dict:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    out = {"iou": inter / union if union else 1.0,
           "precision": inter / pred.sum() if pred.sum() else 0.0,
           "recall": inter / gt.sum() if gt.sum() else 1.0}
    bp, bg = boundary(pred), boundary(gt)
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    dp = cv2.dilate(bp.astype(np.uint8), k).astype(bool)
    dg = cv2.dilate(bg.astype(np.uint8), k).astype(bool)
    bprec = (bp & dg).sum() / bp.sum() if bp.sum() else 0.0
    brec = (bg & dp).sum() / bg.sum() if bg.sum() else 0.0
    out["boundary_f"] = 2 * bprec * brec / (bprec + brec) if (bprec + brec) else 0.0
    ti = np.round(tips).astype(int)
    inside = [(0 <= x < RES and 0 <= y < RES and pred[y, x]) for x, y in ti]
    valid = [(0 <= x < RES and 0 <= y < RES) for x, y in ti]
    out["tip_coverage"] = float(np.sum(inside) / max(1, np.sum(valid)))
    return out


def envelope(pts, q, t, cam, cx, cy, side, dilate_cells: int = 3) -> np.ndarray:
    """Generous object envelope on the RES grid: hull of all 11 keypoints, any coverage, dilated.

    Pixels outside it are background beyond doubt (EXP-015 amendment A1).
    """
    canvas = np.zeros((RASTER, RASTER), np.uint8)
    uv = to_crop(D.project_points(pts, q, t, cam), cx, cy, side, RASTER)
    hull = cv2.convexHull((uv * 16).round().astype(np.int32))
    cv2.fillConvexPoly(canvas, hull, 1, lineType=cv2.LINE_8, shift=4)
    env = cv2.resize(canvas.astype(np.float32), (RES, RES), interpolation=cv2.INTER_AREA) > 0
    k = np.ones((2 * dilate_cells + 1, 2 * dilate_cells + 1), np.uint8)
    return cv2.dilate(env.astype(np.uint8), k).astype(bool)


def leakage(pred: np.ndarray, env: np.ndarray) -> float:
    """Fraction of predicted foreground outside the envelope (0 if the mask is empty)."""
    n = pred.sum()
    return float((pred & ~env).sum() / n) if n else 0.0
