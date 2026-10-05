"""Rotation utilities.

Conventions (used at every interface in this package):
- Quaternions are scalar-first ``[w, x, y, z]``, Hamilton product, unit norm.
- ``quat_to_matrix(q)`` returns the active rotation matrix R such that ``v' = R v``.
- Rotation distance: d(q1, q2) = 2 * arccos(|q1 . q2|), in radians (handles q ~ -q).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.all(np.isfinite(q)):
        raise ValueError("non-finite quaternion")
    if np.any(n < EPS):
        raise ValueError("zero-norm quaternion")
    return q / n


def geodesic_distance(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """SO(3) geodesic distance in radians; inputs need not be normalised."""
    q1 = normalize(q1)
    q2 = normalize(q2)
    d = np.abs(np.sum(q1 * q2, axis=-1))
    return 2.0 * np.arccos(np.clip(d, 0.0, 1.0))


def geodesic_distance_deg(q1, q2):
    return np.degrees(geodesic_distance(q1, q2))


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], axis=-1)
    return R.reshape(q.shape[:-1] + (3, 3))


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Shepperd's method; returns scalar-first unit quaternion with w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    shp = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    q = np.empty((R.shape[0], 4))
    tr = np.trace(R, axis1=1, axis2=2)
    diag = np.stack([tr, R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]], axis=1)
    k = np.argmax(diag, axis=1)
    for i in range(R.shape[0]):
        m = R[i]
        if k[i] == 0:
            s = 2 * np.sqrt(1 + tr[i])
            q[i] = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif k[i] == 1:
            s = 2 * np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
            q[i] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif k[i] == 2:
            s = 2 * np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
            q[i] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = 2 * np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
            q[i] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = normalize(q)
    q = np.where(q[:, :1] < 0, -q, q)
    return q.reshape(shp + (4,))


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def axis_angle_to_quat(axis, angle) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    angle = np.broadcast_to(np.asarray(angle, dtype=np.float64), axis.shape[:-1])[..., None]
    return np.concatenate([np.cos(angle / 2), np.sin(angle / 2) * axis], axis=-1)


def random_quats(n: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-uniform rotations via normalised 4D Gaussians."""
    return normalize(rng.standard_normal((n, 4)))
