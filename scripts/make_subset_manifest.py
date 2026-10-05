"""Seeded subset manifest with ground-truth square crops (Phase 1 controls).

Draws from the official splits only. Crop box = square around the 11 Tango
keypoints projected with GT pose and lens distortion, enlarged by --margin per
side. Writes outputs/data_manifests/subset_<name>.csv (+ .tsv, hash).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import data as D  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="v1")
ap.add_argument("--n-train", type=int, default=6000)
ap.add_argument("--n-val", type=int, default=1500)
ap.add_argument("--n-lightbox", type=int, default=1000)
ap.add_argument("--n-sunlamp", type=int, default=1000)
ap.add_argument("--margin", type=float, default=0.15)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

paths = yaml.safe_load(open(REPO / "configs/paths.yaml"))
root = Path(paths["speedplus_root"])
cam = D.load_camera(root)
pts = D.load_tango_points(paths["tango_points"])
man = pd.read_csv(REPO / "outputs/data_manifests/speedplus_tesseract_L5.csv")
rng = np.random.default_rng(args.seed)
parts = []
for (dom, sp), n in [(("synthetic", "train"), args.n_train), (("synthetic", "validation"), args.n_val),
                     (("lightbox", "test"), args.n_lightbox), (("sunlamp", "test"), args.n_sunlamp)]:
    m = man[(man.domain == dom) & (man.split == sp)]
    parts.append(m.iloc[np.sort(rng.choice(len(m), size=min(n, len(m)), replace=False))])
sub = pd.concat(parts).reset_index(drop=True)
boxes = []
for r in sub.itertuples():
    q = np.array([r.q_can_w, r.q_can_x, r.q_can_y, r.q_can_z])
    t = np.array([r.t_x, r.t_y, r.t_z])
    uv = D.project_points(pts, q, t, cam)
    boxes.append(D.gt_square_box(uv, cam, margin=args.margin))
sub[["crop_cx", "crop_cy", "crop_side"]] = np.array(boxes)
sub["crop_margin"] = args.margin
out = REPO / f"outputs/data_manifests/subset_{args.name}.csv"
sub.to_csv(out, index=False, float_format="%.9g")
sub.to_csv(out.with_suffix(".tsv"), index=False, sep="\t", float_format="%.9g")
h = D.manifest_hash(sub["image_relpath"].tolist() + [f"{v:.4f}" for v in sub["crop_side"]])
(out.parent / f"subset_{args.name}.sha256.txt").write_text(f"{h}  rows={len(sub)}  args={vars(args)}\n")
print(sub.groupby(["domain", "split"]).size(), "\nhash", h, "\ncrop side px:", sub.crop_side.describe().round(1).to_dict())
