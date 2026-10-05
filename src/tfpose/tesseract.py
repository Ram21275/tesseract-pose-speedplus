"""Hierarchical Tesseract discretisation of SO(3).

Builds on the recursive tesseract subdivision of Kurz, Pfaff & Hanebeck,
"Discretization of SO(3) using recursive tesseract subdivision", MFI 2017
(reference code: libDirectional lib/util/tesseractsubdivision.m). Shared with
that work: the cubic facets of the 4D hypercube, recursive 8-way halving of
each cube, and radial projection q = x / ||x||. Differences (ours):
- representatives are cell CENTRES, not cube corners, so every rotation
  belongs to exactly one cell and has a unique root-to-leaf path (a corner is
  shared by up to 16 cubes and has no unique parent);
- only the 4 positive facets are kept (antipodal canonicalisation), so
  q and -q map to the same path, giving 4 * 8**L cells instead of
  (2**m+1)**4 - (2**m-1)**4 corner points with antipodal duplicates;
- an explicit encoder/decoder, chart rule and tie rule (below), which the
  flat point-set construction does not need.

Encoding (ground rules sec. 5):
1. normalise q (scalar-first ``[w, x, y, z]``);
2. chart c = first index of max |q_i|; multiply q by sign(q_c) so q_c > 0.
   This makes path(q) == path(-q) exactly (antipodal canonicalisation);
3. project x = q / q_c  (= q / ||q||_inf); the three free coordinates
   u = x[j], j != c (in increasing index order) lie in [-1, 1];
4. at each level split every free-coordinate interval at its midpoint.
   Boundary convention: u >= mid goes to the UPPER half (bit = 1);
5. child id at a level = b0 + 2*b1 + 4*b2 for free coords (u0, u1, u2).

Four root charts x eight children per level => 4 * 8**L leaves at depth L.
Leaf id = c * 8**L + sum_l child_l * 8**(L-1-l).
Cell centre = centre of the cube cell in projected coordinates, lifted to the
sphere (x_c = 1) and normalised.
"""
from __future__ import annotations

import numpy as np

from .rotations import normalize

N_CHARTS = 4
N_CHILD = 8
FREE = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])


def canonicalize(q: np.ndarray):
    q = normalize(np.atleast_2d(q))
    chart = np.argmax(np.abs(q), axis=1)  # argmax returns the first maximum
    sgn = np.sign(q[np.arange(len(q)), chart])
    qc = q * sgn[:, None]
    return qc, chart


def project(q: np.ndarray):
    """Return (chart, u) with u the (N, 3) free coordinates in [-1, 1]."""
    qc, chart = canonicalize(q)
    x = qc / qc[np.arange(len(qc)), chart][:, None]
    u = np.take_along_axis(x, FREE[chart], axis=1)
    return chart, np.clip(u, -1.0, 1.0)


def encode(q: np.ndarray, depth: int):
    """Return (chart (N,), path (N, depth) child ids in [0, 8))."""
    chart, u = project(q)
    lo = -np.ones_like(u)
    hi = np.ones_like(u)
    path = np.zeros((len(u), depth), dtype=np.int64)
    for l in range(depth):
        mid = 0.5 * (lo + hi)
        bits = (u >= mid)
        path[:, l] = bits[:, 0] + 2 * bits[:, 1] + 4 * bits[:, 2]
        lo = np.where(bits, mid, lo)
        hi = np.where(bits, hi, mid)
    return chart, path


def _cell_bounds(chart, path):
    path = np.atleast_2d(path)
    n, depth = path.shape
    lo = -np.ones((n, 3))
    hi = np.ones((n, 3))
    for l in range(depth):
        c = path[:, l]
        bits = np.stack([(c >> 0) & 1, (c >> 1) & 1, (c >> 2) & 1], axis=1).astype(bool)
        mid = 0.5 * (lo + hi)
        lo = np.where(bits, mid, lo)
        hi = np.where(bits, hi, mid)
    return lo, hi


def _lift(chart, u):
    chart = np.asarray(chart)
    x = np.ones((len(u), 4))
    np.put_along_axis(x, FREE[chart], u, axis=1)
    return normalize(x)


def decode(chart, path) -> np.ndarray:
    """Cell-centre quaternions (canonical: chart coordinate positive)."""
    chart = np.atleast_1d(chart)
    lo, hi = _cell_bounds(chart, path)
    return _lift(chart, 0.5 * (lo + hi))


def cell_corners(chart, path) -> np.ndarray:
    """(N, 8, 4) normalised corner quaternions of each cell."""
    chart = np.atleast_1d(chart)
    lo, hi = _cell_bounds(chart, path)
    out = []
    for k in range(8):
        bits = np.array([(k >> 0) & 1, (k >> 1) & 1, (k >> 2) & 1], dtype=bool)
        out.append(_lift(chart, np.where(bits, hi, lo)))
    return np.stack(out, axis=1)


def leaf_id(chart, path) -> np.ndarray:
    path = np.atleast_2d(path)
    depth = path.shape[1]
    idx = np.asarray(chart, dtype=np.int64) * (8 ** depth)
    for l in range(depth):
        idx = idx + path[:, l] * 8 ** (depth - 1 - l)
    return idx


def leaf_to_path(leaf, depth: int):
    leaf = np.asarray(leaf, dtype=np.int64)
    chart = leaf // (8 ** depth)
    rem = leaf % (8 ** depth)
    path = np.zeros((len(leaf), depth), dtype=np.int64)
    for l in range(depth - 1, -1, -1):
        path[:, l] = rem % 8
        rem //= 8
    return chart, path


def all_cells(depth: int):
    leaves = np.arange(N_CHARTS * 8 ** depth)
    return leaf_to_path(leaves, depth)


def codebook(depth: int) -> np.ndarray:
    """All leaf-centre quaternions at a depth, ordered by leaf id."""
    chart, path = all_cells(depth)
    return decode(chart, path)
