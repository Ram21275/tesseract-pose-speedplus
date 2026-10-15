"""Implicit-PDF / LiePose-style SO(3) visualisation (re-implemented in matplotlib; style follows
liepose/utils/vis.py:visualize_so3_probabilities): Mollweide projection of the rotation's first column
(lon = atan2(x0, -x1), lat = asin(x2)), hsv colour = tilt (first 'xyz' Euler angle), marker area = probability,
ground truth as an open ring in its tilt colour."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402


def _q2R(q):
    q = np.asarray(q, np.float64).reshape(-1, 4)
    return Rotation.from_quat(np.roll(q, -1, axis=1)).as_matrix()       # wxyz -> scipy xyzw


def _coords(R):
    x = R[:, :, 0]
    lon = np.arctan2(x[:, 0], -x[:, 1]); lat = np.arcsin(np.clip(x[:, 2], -1, 1))
    tilt = Rotation.from_matrix(R).as_euler("xyz")[:, 0]
    return lon, lat, tilt


def so3_axes(fig, rect=(0.0, 0.0, 1.0, 1.0)):
    ax = fig.add_axes(rect, projection="mollweide")
    ax.grid(True, linewidth=0.4, alpha=0.5); ax.set_xticklabels([]); ax.set_yticklabels([])
    return ax


def plot_so3(ax, q, prob=None, q_gt=None, size_scale=4e3, const_size=6.0, gt_size=900, alpha=0.9, cmap=plt.cm.hsv):
    """q: (N,4) wxyz rotations; prob: (N,) or None (constant marker); q_gt: (K,4) (one rotation or a symmetry set)."""
    lon, lat, tilt = _coords(_q2R(q))
    s = size_scale * np.asarray(prob) if prob is not None else np.full(len(lon), const_size)
    order = np.argsort(s)[::-1]
    ax.scatter(lon[order], lat[order], s=s[order], c=cmap(0.5 + tilt[order] / (2 * np.pi)), alpha=alpha, linewidths=0)
    if q_gt is not None:
        gl, gb, gt = _coords(_q2R(q_gt))
        ax.scatter(gl, gb, s=gt_size, facecolors="none", edgecolors=cmap(0.5 + gt / (2 * np.pi)), linewidths=3)
        ax.scatter(gl, gb, s=gt_size * 0.6, facecolors="none", edgecolors="k", linewidths=0.6)


def tilt_wheel(fig, rect=(0.86, 0.06, 0.1, 0.1), cmap=plt.cm.hsv):
    ax = fig.add_axes(rect, projection="polar")
    th = np.linspace(-np.pi, np.pi, 256); r = np.linspace(0.6, 1.0, 2)
    T, Rr = np.meshgrid(th, r)
    ax.pcolormesh(T, Rr, 0.5 + T / (2 * np.pi), cmap=cmap, shading="auto", vmin=0, vmax=1)
    ax.set_yticklabels([]); ax.set_xticklabels([]); ax.spines["polar"].set_visible(False); ax.set_title("tilt", fontsize=8)
    return ax
