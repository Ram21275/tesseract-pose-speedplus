"""EXP-109: in-plane rotation of a GT square crop (sampled from the original image, no black corners), the matching
label rotation about the camera ray through the crop centre, and photometric augmentation.

Label conventions (the EXP-109 gate picks one by checking frozen-tree predictions on rotated crops):
  pre+ : q' = Exp(+theta * d) (x) q      pre- : q' = Exp(-theta * d) (x) q
  post+: q' = q (x) Exp(+theta * d)      post-: q' = q (x) Exp(-theta * d)
theta = the image rotation angle (cv2 convention: positive = counter-clockwise on screen), d = unit ray to the crop centre.
"""
from __future__ import annotations

import cv2
import numpy as np

MODES = ("pre+", "pre-", "post+", "post-")


def crop_rotated(img: np.ndarray, cx: float, cy: float, side: float, out: int, theta_deg: float) -> np.ndarray:
    s = out / side
    C = np.array([[s, 0, out / 2 - s * cx], [0, s, out / 2 - s * cy]], np.float64)
    R = cv2.getRotationMatrix2D((out / 2, out / 2), float(theta_deg), 1.0)
    M = np.concatenate([R[:, :2] @ C[:, :2], (R[:, :2] @ C[:, 2] + R[:, 2])[:, None]], 1)
    return cv2.warpAffine(img, M, (out, out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def ray(K: np.ndarray, cx, cy) -> np.ndarray:
    d = np.linalg.solve(K, np.stack([np.atleast_1d(cx), np.atleast_1d(cy), np.ones(np.size(cx))], 0)).T
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _qmul(a, b):
    return np.stack([a[:, 0] * b[:, 0] - (a[:, 1:] * b[:, 1:]).sum(1),
                     *(a[:, :1] * b[:, 1:] + b[:, :1] * a[:, 1:] + np.cross(a[:, 1:], b[:, 1:])).T], 1)


def rotate_label(q: np.ndarray, theta_deg, d: np.ndarray, mode: str) -> np.ndarray:
    th = np.radians(np.broadcast_to(theta_deg, (len(q),))) * (1 if mode.endswith("+") else -1)
    e = np.concatenate([np.cos(th / 2)[:, None], np.sin(th / 2)[:, None] * d], 1)
    out = _qmul(e, q) if mode.startswith("pre") else _qmul(q, e)
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out * np.where(out[:, :1] < 0, -1, 1)


def photometric(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """x float32 (H,W,C) in [0,1] -> augmented copy (pre-registered EXP-109 ranges)."""
    H, W = x.shape[:2]
    x = x * rng.uniform(0.5, 1.5)
    m = x.mean(); x = (x - m) * rng.uniform(0.6, 1.4) + m
    x = np.clip(x, 0, 1) ** rng.uniform(0.6, 1.6)
    if rng.random() < 0.5:
        sig = rng.uniform(0.01, 1.5); x = cv2.GaussianBlur(x, (0, 0), sig)
    if rng.random() < 0.3:                                                     # glare: 1-3 saturating Gaussian blobs
        yy, xx = np.mgrid[0:H, 0:W]
        for _ in range(rng.integers(1, 4)):
            cy_, cx_, r = rng.uniform(0, H), rng.uniform(0, W), rng.uniform(0.05, 0.25) * H
            g = rng.uniform(0.5, 1.5) * np.exp(-((yy - cy_) ** 2 + (xx - cx_) ** 2) / (2 * r * r))
            x = x + (g[..., None] if x.ndim == 3 else g)
    x = x + rng.normal(0, rng.uniform(0, 0.06), x.shape)
    x = np.clip(x, 0, 1).astype(np.float32)
    if rng.random() < 0.3:                                                     # random erasing, 5-20 % of the area
        a = rng.uniform(0.05, 0.2) * H * W; ar = rng.uniform(0.3, 3.3)
        h = int(min(H, np.sqrt(a * ar))); w = int(min(W, np.sqrt(a / ar)))
        y0, x0 = rng.integers(0, H - h + 1), rng.integers(0, W - w + 1); x[y0:y0 + h, x0:x0 + w] = rng.random()
    if rng.random() < 0.5:                                                     # JPEG 30-90
        ok, enc = cv2.imencode(".jpg", (x * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 91))])
        x = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        if x.ndim == 2 and x.shape != (H, W):
            x = x.reshape(H, W)
    return x


def ray_undistorted(K: np.ndarray, dist: np.ndarray, cx, cy) -> np.ndarray:
    """Unit camera ray through (distorted) pixel(s) (cx, cy): undistort, then lift (z = 1)."""
    p = np.stack([np.atleast_1d(cx), np.atleast_1d(cy)], 1).astype(np.float64).reshape(-1, 1, 2)
    n = cv2.undistortPoints(p, K, dist).reshape(-1, 2)
    d = np.concatenate([n, np.ones((len(n), 1))], 1)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def crop_rolled_homography(img: np.ndarray, cx: float, cy: float, side: float, out: int, theta_deg: float, K: np.ndarray) -> np.ndarray:
    """EXP-064: exact camera roll by Rot(d, -theta) about the ray d through the crop centre (pinhole, no distortion):
    x' ~ K Rot(d, -theta) K^-1 x, then the same square crop. The crop centre is a fixed point, so labels follow exactly."""
    d = np.linalg.solve(K, np.array([cx, cy, 1.0])); d /= np.linalg.norm(d)
    th = -np.radians(theta_deg)
    Kx = np.array([[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]])
    Rs = np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx
    s = out / side
    C = np.array([[s, 0, out / 2 - s * cx], [0, s, out / 2 - s * cy], [0, 0, 1.0]])
    M = C @ K @ Rs @ np.linalg.inv(K)
    return cv2.warpPerspective(img, M, (out, out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
