"""Allocentric-rotation helpers shared by the cross-dataset manifest builders (EXP-064 / EXP-106)."""
import numpy as np

from .rotations import normalize


def mat_to_quat(R):
    """(N,3,3) -> (N,4) scalar-first, w >= 0."""
    R = np.asarray(R, np.float64); N = len(R); q = np.empty((N, 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    for n in range(N):
        m = R[n]
        if tr[n] > 0:
            s = 2 * np.sqrt(tr[n] + 1); q[n] = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2 * np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]); q[n] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = 2 * np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]); q[n] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = 2 * np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]); q[n] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = normalize(q); return q * np.where(q[:, :1] < 0, -1, 1)


def ray_rot(t):
    """(N,3) translations -> (N,3,3) minimal rotations taking e_z to t/|t| (Rodrigues)."""
    d = t / np.linalg.norm(t, axis=1, keepdims=True); z = np.array([0, 0, 1.0])
    v = np.cross(z, d); s = np.linalg.norm(v, axis=1); c = d[:, 2]
    K = np.zeros((len(t), 3, 3)); K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -v[:, 2], v[:, 1], -v[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = v[:, 2], -v[:, 1], v[:, 0]
    f = np.where(s > 1e-12, (1 - c) / np.maximum(s, 1e-12) ** 2, 0.5)
    return np.eye(3)[None] + K + f[:, None, None] * K @ K
