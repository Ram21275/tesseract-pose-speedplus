"""Rotation metrics (ground rules sec. 6). Predicted quaternions are validated before scoring."""
from __future__ import annotations

import numpy as np

from . import rotations as rot

THRESH = (1, 3, 5, 10, 20)


def validate_quats(q: np.ndarray):
    if not np.all(np.isfinite(q)):
        raise ValueError("non-finite predicted quaternion")
    n = np.linalg.norm(q, axis=1)
    if np.any(n < 1e-8):
        raise ValueError("zero-norm predicted quaternion")
    return q / n[:, None]


def rotation_metrics(q_pred: np.ndarray, q_gt: np.ndarray, prefix: str = "") -> dict:
    e = rot.geodesic_distance_deg(validate_quats(q_pred), q_gt)
    m = {f"{prefix}n": int(len(e)), f"{prefix}mean_deg": float(e.mean()), f"{prefix}median_deg": float(np.median(e)),
         f"{prefix}p95_deg": float(np.percentile(e, 95))}
    for t in THRESH:
        m[f"{prefix}acc@{t}"] = float(np.mean(e < t))
    return m


def path_metrics(chart_pred, path_pred, chart_gt, path_gt, topk_leaf_hit=None, prefix: str = "") -> dict:
    """chart_pred (N,), path_pred (N,L) best beam; topk_leaf_hit dict k -> (N,) bool."""
    m = {f"{prefix}root_acc": float(np.mean(chart_pred == chart_gt))}
    ok = chart_pred == chart_gt
    for l in range(path_gt.shape[1]):
        ok = ok & (path_pred[:, l] == path_gt[:, l])
        m[f"{prefix}path_acc_L{l + 1}"] = float(np.mean(ok))
    m[f"{prefix}full_path_acc"] = float(np.mean(ok))
    for k, hit in (topk_leaf_hit or {}).items():
        m[f"{prefix}top{k}_leaf_recall"] = float(np.mean(hit))
    return m
