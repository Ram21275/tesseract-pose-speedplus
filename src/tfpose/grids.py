"""Reference SO(3) grids and grid-quality measurements.

- Hopf grid (Yershova et al., IJRR 2010): HEALPix on S^2 x uniform S^1,
  72 * 8**r points at resolution r. HEALPix ring-scheme pix2ang is implemented
  here (Gorski et al. 2005) to avoid a healpy dependency.
- Super-Fibonacci spirals (Alexa, CVPR 2022): near-uniform, any N. Used as the
  "another near-uniform SO(3) grid" substitute for cubochoric (recorded in
  the EXP-003 report).
"""
from __future__ import annotations

import numpy as np
import torch

from .rotations import normalize


def healpix_pix2ang_ring(nside: int) -> tuple[np.ndarray, np.ndarray]:
    npix = 12 * nside * nside
    p = np.arange(npix, dtype=np.int64)
    ncap = 2 * nside * (nside - 1)
    theta = np.empty(npix)
    phi = np.empty(npix)
    # north cap
    m = p < ncap
    ph = p[m] + 1
    i = np.floor(np.sqrt(ph / 2 - np.sqrt(np.floor(ph / 2)))).astype(np.int64) + 1
    j = ph - 2 * i * (i - 1)
    theta[m] = np.arccos(1 - i ** 2 / (3.0 * nside ** 2))
    phi[m] = np.pi / (2 * i) * (j - 0.5)
    # equatorial belt
    m = (p >= ncap) & (p < npix - ncap)
    pp = p[m] - ncap
    i = pp // (4 * nside) + nside
    j = pp % (4 * nside) + 1
    s = (i - nside + 1) % 2
    theta[m] = np.arccos(4.0 / 3.0 - 2.0 * i / (3.0 * nside))
    phi[m] = np.pi / (2 * nside) * (j - s / 2.0)
    # south cap
    m = p >= npix - ncap
    ph = npix - p[m]
    i = np.floor(np.sqrt(ph / 2 - np.sqrt(np.floor(ph / 2)))).astype(np.int64) + 1
    j = 4 * i + 1 - (ph - 2 * i * (i - 1))
    theta[m] = np.arccos(-1 + i ** 2 / (3.0 * nside ** 2))
    phi[m] = np.pi / (2 * i) * (j - 0.5)
    return theta, phi


def hopf_grid(r: int) -> np.ndarray:
    """Yershova Hopf grid at resolution r: 72 * 8**r unit quaternions [w,x,y,z]."""
    nside = 2 ** r
    theta, phi = healpix_pix2ang_ring(nside)
    n_psi = 6 * 2 ** r
    psi = (np.arange(n_psi) + 0.5) * 2 * np.pi / n_psi
    th = np.repeat(theta, n_psi)
    ph = np.repeat(phi, n_psi)
    ps = np.tile(psi, len(theta))
    q = np.stack([np.cos(th / 2) * np.cos(ps / 2), np.cos(th / 2) * np.sin(ps / 2),
                  np.sin(th / 2) * np.cos(ph + ps / 2), np.sin(th / 2) * np.sin(ph + ps / 2)], axis=1)
    return normalize(q)


def super_fibonacci(n: int) -> np.ndarray:
    phi = np.sqrt(2.0)
    psi = 1.533751168755204288118041
    s = np.arange(n) + 0.5
    r = np.sqrt(s / n)
    R = np.sqrt(1.0 - s / n)
    a = 2 * np.pi * s / phi
    b = 2 * np.pi * s / psi
    q = np.stack([r * np.sin(a), r * np.cos(a), R * np.sin(b), R * np.cos(b)], axis=1)
    return normalize(q)


def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def nearest(queries: np.ndarray, grid: np.ndarray, chunk: int = 2048, exclude_self: bool = False):
    """Nearest grid point under |q.g| (SO(3) metric). Returns (index, distance_rad, float64)."""
    dev = _dev()
    G = torch.as_tensor(grid, dtype=torch.float32, device=dev)
    idx = np.empty(len(queries), dtype=np.int64)
    for s in range(0, len(queries), chunk):
        Q = torch.as_tensor(queries[s:s + chunk], dtype=torch.float32, device=dev)
        dots = (Q @ G.T).abs_()
        if exclude_self:
            rows = torch.arange(len(Q), device=dev)
            dots[rows, rows + s] = -1.0
        idx[s:s + chunk] = dots.argmax(1).cpu().numpy()
    d = np.abs(np.sum(queries * grid[idx], axis=1))
    return idx, 2 * np.arccos(np.clip(d, 0, 1))


def error_summary(err_rad: np.ndarray, prefix: str = "") -> dict:
    e = np.degrees(err_rad)
    return {f"{prefix}mean_deg": float(e.mean()), f"{prefix}median_deg": float(np.median(e)),
            f"{prefix}p95_deg": float(np.percentile(e, 95)), f"{prefix}max_deg": float(e.max())}


def spacehop_hopf_grid(num_points: int = 256, num_rolls: int = 12) -> np.ndarray:
    """SPACE-HOP anchor grid (Team-M3OW/SPACE-HOP src/hopf_grid.py::generate_hopf_so3_grid), as quaternions.

    Fibonacci-lattice directions on S^2 -> frame with z along the direction (up=[0,0,1], [1,0,0] near the
    poles) -> num_rolls in-plane rotations about z. Reproduced verbatim in NumPy (float64).
    """
    i = np.arange(num_points, dtype=np.float64)
    phi = np.arccos(1 - 2 * (i + 0.5) / num_points)
    theta = np.pi * (1 + 5 ** 0.5) * i
    z = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], -1)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    up = np.tile([0.0, 0.0, 1.0], (num_points, 1))
    up[np.abs(z[:, 2]) > 0.999] = [1.0, 0.0, 0.0]
    x = np.cross(up, z); x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.cross(z, x); y /= np.linalg.norm(y, axis=1, keepdims=True)
    Rb = np.stack([x, y, z], -1)                                      # columns [X, Y, Z]
    a = np.linspace(0, 2 * np.pi, num_rolls + 1)[:-1]
    Rr = np.zeros((num_rolls, 3, 3)); Rr[:, 0, 0] = np.cos(a); Rr[:, 0, 1] = -np.sin(a)
    Rr[:, 1, 0] = np.sin(a); Rr[:, 1, 1] = np.cos(a); Rr[:, 2, 2] = 1.0
    R = (Rb[:, None] @ Rr[None]).reshape(-1, 3, 3)
    from .rotations import matrix_to_quat
    return matrix_to_quat(R)
