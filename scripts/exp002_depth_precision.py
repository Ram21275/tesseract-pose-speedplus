"""EXP-002: Tesseract fixed-depth precision, L = 1..5.

Measures, per depth:
- own-cell quantisation error (what the decoder returns) on Haar-uniform
  rotations and on the SPEED+ GT distribution;
- EXACT worst-case own-cell error: max over cells of centre->corner distance
  (angle to a fixed direction is quasiconvex on a convex projected cell, so
  the maximum is attained at a corner);
- nearest-centre covering error (sampled), which lower-bounds the true
  covering radius;
- cell-volume variation: exact density of the L_inf chart map on S^3 is
  (1+|u|^2)^-2, integrated per cell with Gauss-Legendre quadrature;
- empirical occupancy under Haar sampling and centre nearest-neighbour spacing;
- codebook memory and search cost (greedy tree vs flat).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import rotations as rot  # noqa: E402
from tfpose import tesseract as T  # noqa: E402
from tfpose.grids import error_summary, nearest  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n-haar", type=int, default=2_000_000)
ap.add_argument("--max-depth", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
run = Run("EXP-002", "depth_precision", vars(args), seed=args.seed)
try:
    rng = np.random.default_rng(args.seed)
    qh = rot.random_quats(args.n_haar, rng)
    man = pd.read_csv(REPO / "outputs/data_manifests/speedplus_tesseract_L5.csv")
    qgt = man[[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    gl_x, gl_w = np.polynomial.legendre.leggauss(6)
    rows = []
    for L in range(1, args.max_depth + 1):
        chart, path = T.all_cells(L)
        cb = T.decode(chart, path)
        n = len(cb)
        row = {"depth": L, "n_cells": n}
        # own-cell error
        c, p = T.encode(qh, L)
        row.update(error_summary(rot.geodesic_distance(qh, T.decode(c, p)), "haar_own_"))
        c2, p2 = T.encode(qgt, L)
        row.update(error_summary(rot.geodesic_distance(qgt, T.decode(c2, p2)), "gt_own_"))
        # exact worst case own-cell error via corners
        corners = T.cell_corners(chart, path)
        cell_r = np.max(2 * np.arccos(np.clip(np.abs(np.einsum("nd,nkd->nk", cb, corners)), 0, 1)), axis=1)
        row["exact_own_cell_max_deg"] = float(np.degrees(cell_r.max()))
        row["cell_radius_min_deg"] = float(np.degrees(cell_r.min()))
        # nearest-centre covering (sampled lower bound of true covering radius)
        nh = min(len(qh), 1_000_000)
        _, dn = nearest(qh[:nh], cb)
        row.update(error_summary(dn, "haar_nearest_"))
        row["nearest_differs_from_own_frac"] = float(np.mean(T.leaf_id(c[:nh], p[:nh]) != nearest(qh[:nh], cb)[0]))
        # exact cell volume via quadrature of (1+|u|^2)^-2 over each cube cell
        lo = -np.ones((n, 3)); hi = np.ones((n, 3))
        for l in range(L):
            b = np.stack([(path[:, l] >> k) & 1 for k in range(3)], 1).astype(bool)
            mid = 0.5 * (lo + hi)
            lo = np.where(b, mid, lo); hi = np.where(b, hi, mid)
        vol = np.zeros(n)
        h = 0.5 * (hi - lo)
        m = 0.5 * (hi + lo)
        for i, wi in zip(gl_x, gl_w):
            for j, wj in zip(gl_x, gl_w):
                for k, wk in zip(gl_x, gl_w):
                    u = m + h * np.array([i, j, k])
                    vol += wi * wj * wk / (1 + np.sum(u * u, 1)) ** 2
        vol *= np.prod(h, 1)
        row["volume_max_over_min"] = float(vol.max() / vol.min())
        row["volume_cv"] = float(vol.std() / vol.mean())
        occ = np.bincount(T.leaf_id(c, p), minlength=n)
        row["haar_occupancy_cv"] = float(occ.std() / occ.mean())
        row["haar_occ_vs_volume_corr"] = float(np.corrcoef(occ, vol)[0, 1])
        # centre nearest-neighbour spacing
        _, nn = nearest(cb, cb, exclude_self=True)
        nn = np.degrees(nn)
        row.update({"nn_mean_deg": float(nn.mean()), "nn_min_deg": float(nn.min()), "nn_max_deg": float(nn.max()),
                    "nn_cv": float(nn.std() / nn.mean())})
        row["codebook_MB_fp32"] = n * 4 * 4 / 2**20
        row["greedy_nodes_scored"] = 4 + 8 * L
        row["flat_nodes_scored"] = n
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    run.write_metrics(rows, "metrics")
    (run.sub("tables") / "depth_table.md").write_text(md_table(rows))
    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    for col, lab in [("haar_own_mean_deg", "own-cell mean"), ("haar_own_p95_deg", "own-cell p95"),
                     ("exact_own_cell_max_deg", "own-cell max (exact)"), ("haar_nearest_max_deg", "nearest-centre max (sampled)")]:
        ax.semilogy(df["depth"], df[col], "o-", label=lab)
    ax.set_xlabel("Tesseract depth L"); ax.set_ylabel("error (deg)"); ax.grid(True, which="both", alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(run.sub("figures") / "error_vs_depth.png", dpi=130)
    run.done()
    print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc())
    raise
