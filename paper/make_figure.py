"""Paper Fig. 1 (pipeline). Vector PDF via matplotlib; one real SPEED+ lightbox crop for the orbit views.
Run from the repo root:  .venv/bin/python paper/make_figure.py"""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
C_FROZEN, C_TRAIN, C_MOGE, C_INF, C_FUSE = "#e9edf2", "#dcebf8", "#fbeedd", "#e3f4e8", "#efe6f8"
E_FROZEN, E_TRAIN, E_MOGE, E_INF, E_FUSE = "#6b7685", "#2f6fae", "#c0661c", "#2f8a4c", "#6e48a6"


def box(ax, x, y, w, h, title, lines=(), fc="white", ec="black", ls="-", tsize=6.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc=fc, ec=ec, lw=0.9, ls=ls))
    ax.text(x + w / 2, y + h - 0.022, title, ha="center", va="top", fontsize=tsize, weight="bold")
    for i, l in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.062 - i * 0.043, l, ha="center", va="top", fontsize=5.7)


def arrow(ax, x0, y0, x1, y1, color="#333333", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.9, color=color, ls=ls))


def crop_image():
    root = Path(yaml.safe_load(open(REPO / "configs/paths.yaml"))["speedplus_root"])
    t = pd.read_csv(REPO / "outputs/data_manifests/exp109_test3.csv"); r = t[t.domain == "lightbox"].iloc[7]
    img = cv2.imread(str(root / r.image_relpath), cv2.IMREAD_GRAYSCALE)
    s = r.crop_side; M = np.array([[128 / s, 0, 64 - 128 / s * r.crop_cx], [0, 128 / s, 64 - 128 / s * r.crop_cy]])
    return cv2.warpAffine(img, M, (128, 128), borderMode=cv2.BORDER_REFLECT)


def main():
    fig = plt.figure(figsize=(7.0, 3.35)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    crop = crop_image()
    # ---- training row
    ax.text(0.008, 0.975, "Training (synthetic images only)", fontsize=7.6, weight="bold", va="top")
    ia = fig.add_axes([0.012, 0.64, 0.085, 0.24]); ia.imshow(crop, cmap="gray"); ia.axis("off")
    ax.text(0.055, 0.615, "GT crop, $(q, t)$", ha="center", fontsize=6)
    box(ax, 0.110, 0.585, 0.185, 0.33, "3 views / image", ["clean", "photometric + in-plane roll $\\theta$", "$q'=\\mathrm{Exp}(-\\theta d)\\otimes q$", "$t'=R(d,-\\theta)\\,t$", "$d$: crop-centre ray"])
    box(ax, 0.315, 0.765, 0.135, 0.15, "DINOv3-L (frozen)", ["8$\\times$8$\\times$1024 tokens"], C_FROZEN, E_FROZEN, "--")
    box(ax, 0.315, 0.585, 0.135, 0.15, "MoGe-2 (frozen)", ["normals 16$\\times$16$\\times$4"], C_FROZEN, E_FROZEN, "--")
    box(ax, 0.48, 0.765, 0.20, 0.15, "Tesseract tree Transformer", ["4 charts + 5 levels $\\times$ 8 children", "soft targets, 2.4 M params"], C_TRAIN, E_TRAIN)
    box(ax, 0.48, 0.585, 0.20, 0.15, "Tesseract tree Transformer", ["same architecture, own weights"], C_MOGE, E_MOGE)
    box(ax, 0.71, 0.765, 0.12, 0.15, "Translation block", ["SPACE-HOP target $u$", "(optional, 6-DoF)"], C_TRAIN, E_TRAIN)
    box(ax, 0.71, 0.585, 0.275, 0.15, "Orbit distillation (optional)", ["teacher: orbit over the 3 views", "student loss $L_{GT}+\\beta H(\\mathrm{teacher}\\to v)$"], C_INF, E_INF, "--")
    box(ax, 0.845, 0.765, 0.14, 0.15, "Loss", ["$\\Sigma_\\ell$ CE(level $\\ell$)", "$+\\,\\|u-u^*\\|^2$"])
    arrow(ax, 0.097, 0.76, 0.110, 0.76); arrow(ax, 0.295, 0.84, 0.315, 0.84); arrow(ax, 0.295, 0.66, 0.315, 0.66)
    arrow(ax, 0.45, 0.84, 0.48, 0.84); arrow(ax, 0.45, 0.66, 0.48, 0.66); arrow(ax, 0.68, 0.84, 0.71, 0.84); arrow(ax, 0.83, 0.84, 0.845, 0.84)
    ax.plot([0.005, 0.995], [0.545, 0.545], color="#bbbbbb", lw=0.6)
    # ---- inference row
    ax.text(0.008, 0.525, "Inference: orbit voting over 4 in-plane views", fontsize=7.6, weight="bold", va="top")
    for i, k in enumerate((0, 90, 180, 270)):
        R = cv2.getRotationMatrix2D((64, 64), k, 1.0); v = cv2.warpAffine(crop, R, (128, 128), borderMode=cv2.BORDER_REFLECT)
        a = fig.add_axes([0.012 + (i % 2) * 0.05, 0.27 - (i // 2) * 0.155, 0.046, 0.145]); a.imshow(v, cmap="gray"); a.axis("off")
        a.text(4, 18, f"{k}°", color="white", fontsize=5.5)
    box(ax, 0.125, 0.11, 0.19, 0.36, "Per view $k$", ["frozen backbones + trees", "per-level PoE:", "$\\log p_k \\propto \\log p_{D}+\\log p_{M}$ (per level)", "beam search $\\to$ 8 leaves", "(DINOv3-only: no MoGe-2)"], C_FUSE, E_FUSE)
    box(ax, 0.345, 0.11, 0.19, 0.36, "Map back", ["$c=\\mathrm{Exp}(+\\theta_k d)\\otimes c_k$", "4 views $\\times$ 8 = 32", "candidate rotations", "(exact inverse of the", "training label transform)"], C_INF, E_INF)
    box(ax, 0.565, 0.11, 0.23, 0.36, "Orbit score", ["$s(c)=\\Sigma_k \\log p_k(\\mathrm{Exp}(-\\theta_k d)\\otimes c)$", "tree likelihood of each candidate", "as seen from every view", "$\\hat q=\\arg\\max_c s(c)$", "(alt.: weighted vote, 5° kernel)"], C_INF, E_INF)
    box(ax, 0.825, 0.11, 0.16, 0.36, "Output", ["$\\hat q$: leaf centre", "$\\hat t$: block $u$ per view,", "back-rolled and averaged", "", "frozen: 634 M params", "trained: 2.4 / 4.5 M"])
    arrow(ax, 0.11, 0.29, 0.125, 0.29); arrow(ax, 0.315, 0.29, 0.345, 0.29); arrow(ax, 0.535, 0.29, 0.565, 0.29); arrow(ax, 0.795, 0.29, 0.825, 0.29)
    ax.text(0.5, 0.03, "a look-alike (e.g. 180° flip) rarely wins in all rotated views; the true pose, mapped back, is consistent across them",
            ha="center", fontsize=6.2, style="italic", color="#2f8a4c")
    out = REPO / "paper/figures"; out.mkdir(exist_ok=True)
    fig.savefig(out / "pipeline.pdf"); fig.savefig(out / "pipeline.png", dpi=200)
    print("saved", out / "pipeline.pdf")


if __name__ == "__main__":
    main()
