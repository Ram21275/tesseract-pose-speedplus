"""Qualitative pose figures: best / worst rotation predictions per domain, drawn on the GT crop.

For each image the Tango body-frame bounding box (from the extents of the 11 tangoPoints keypoints) and the
body axes are projected with the GT translation and (green) the GT rotation, (red) the predicted rotation.
Translation is GT in both overlays, so the figure isolates rotation error (the pipeline is rotation-only so far).

Usage:
  scripts/make_pose_figures.py --pred <predictions csv> [--out <figures dir>] [--n 6] [--title ...]
The csv needs image_relpath, a group column (``domain`` or ``group``), q_pred_w..z and err_deg.
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose.data import crop_square, load_camera, load_tango_points, project_points, read_gray  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True)
ap.add_argument("--out", default=None, help="default: <run dir>/figures")
ap.add_argument("--n", type=int, default=6, help="best and worst images per domain")
ap.add_argument("--size", type=int, default=320)
ap.add_argument("--title", default="")
args = ap.parse_args()

pred_path = Path(args.pred)
out = Path(args.out) if args.out else pred_path.parent.parent / "figures"
out.mkdir(parents=True, exist_ok=True)
paths = yaml.safe_load(open(REPO / "configs/paths.yaml"))
root = Path(paths["speedplus_root"])
cam = load_camera(root)
kp = load_tango_points(paths["tango_points"])
lo, hi = kp.min(0), kp.max(0)
corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
L = 0.6 * np.abs(hi - lo).max()
axes_pts = np.array([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L]], float)

man = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv").set_index("image_relpath")
pr = pd.read_csv(pred_path)
gcol = "group" if "group" in pr.columns else "domain"
pr = pr[pr[gcol] != "synthetic_train_sample"]
if {"lightbox_val", "lightbox_test"} & set(pr[gcol]):          # DEC-006: show lightbox as one test domain
    pr = pr.copy(); pr.loc[pr[gcol].isin(["lightbox_val", "lightbox_test"]), gcol] = "lightbox"


def to_crop(uv, cx, cy, side):
    s = args.size / side
    return np.stack([s * (uv[:, 0] - cx) + args.size / 2, s * (uv[:, 1] - cy) + args.size / 2], 1)


def draw(ax, row):
    m = man.loc[row.image_relpath]
    img = crop_square(read_gray(root, row.image_relpath), m.crop_cx, m.crop_cy, m.crop_side, args.size)
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    r = np.array([m.t_x, m.t_y, m.t_z])
    for q, col, lw in [(np.array([m.q_src_w, m.q_src_x, m.q_src_y, m.q_src_z]), "lime", 1.6),
                       (np.array([row.q_pred_w, row.q_pred_x, row.q_pred_y, row.q_pred_z]), "red", 1.2)]:
        c = to_crop(project_points(corners, q, r, cam), m.crop_cx, m.crop_cy, m.crop_side)
        for a, b in EDGES:
            ax.plot(*c[[a, b]].T, color=col, lw=lw, alpha=0.9)
        p = to_crop(project_points(axes_pts, q, r, cam), m.crop_cx, m.crop_cy, m.crop_side)
        for k in range(1, 4):
            ax.plot(*p[[0, k]].T, color=col, lw=lw + 0.6, ls="-" if col == "lime" else "--")
        ax.text(*p[1], "x", color=col, fontsize=7)
    ax.set_xlim(0, args.size); ax.set_ylim(args.size, 0); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{row.err_deg:.1f}°  {Path(row.image_relpath).name}", fontsize=8)


written = []
for dom in ["synthetic_val", "lightbox", "sunlamp"]:
    g = pr[pr[gcol] == dom].sort_values("err_deg")
    if g.empty:
        continue
    n = min(args.n, len(g) // 2)
    fig, axs = plt.subplots(2, n, figsize=(2.6 * n, 5.8))
    axs = np.atleast_2d(axs)
    for j, (_, row) in enumerate(g.head(n).iterrows()):
        draw(axs[0, j], row)
    for j, (_, row) in enumerate(g.tail(n).iloc[::-1].iterrows()):
        draw(axs[1, j], row)
    axs[0, 0].set_ylabel("best", fontsize=10); axs[1, 0].set_ylabel("worst", fontsize=10)
    fig.suptitle(f"{args.title} {dom}: mean {g.err_deg.mean():.1f}° / median {g.err_deg.median():.1f}°  "
                 f"(green = GT, red = predicted; GT translation in both)", fontsize=10)
    fig.tight_layout()
    f = out / f"best_worst_{dom}.png"; fig.savefig(f, dpi=110); plt.close(fig); written.append(f)
print("\n".join(str(w) for w in written))
