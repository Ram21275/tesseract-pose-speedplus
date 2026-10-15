"""EXP-064 data stage, YCB-Video original PoseCNN release (CPU only): crop manifest with allocentric rotations.

Objects (declared before any model result; see the EXP-064 pre-registration): 003_cracker_box, 006_mustard_bottle,
011_banana, 025_mug, 035_power_drill, 037_scissors. Mix of box / bottle / elongated / near-symmetric (mug: handle
breaks the symmetry, the closest analogue of SPEED+ flips) / thin objects; the release has no symmetry metadata,
so declared-symmetric YCB objects (bowl, wood block, clamps, foam brick) are left out and raw geodesic error is used.
Splits: train = every 3rd frame of image_sets/train.txt minus held-out sequences (one per object, rng 0) = validation (same stride); test = image_sets/keyframe.txt (2,949 frames, sequences 0048-0059), all instances. Train/val keep instances
with >= 1,000 visible label pixels; <= 12,000 train crops per object (rng 0).
Crop = GT square box of the projected model points (points.xyz), margin 0.15, min 32 px. Allocentric rotation as for
T-LESS. Check: rebuilt egocentric rotation reprojects the points identically; projected points vs the label mask bbox.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import data as D  # noqa: E402

ROOT = Path(yaml.safe_load(open(REPO / "configs/paths.yaml"))["ycbv_root"])
CLASSES = (ROOT / "image_sets/classes.txt").read_text().split()
OBJ = ["003_cracker_box", "006_mustard_bottle", "011_banana", "025_mug", "035_power_drill", "037_scissors"]
CID = {CLASSES.index(o) + 1: o for o in OBJ}
PTS = {o: np.loadtxt(ROOT / "models" / o / "points.xyz") for o in OBJ}
OUT = REPO / "outputs/data_manifests/ycbv_exp064"
STRIDE, MIN_PIX, CAP = 3, 1000, 12000
from tfpose.allo import mat_to_quat, ray_rot  # noqa: E402

train = [l.strip() for l in (ROOT / "image_sets/train.txt").read_text().split()]
key = [l.strip() for l in (ROOT / "image_sets/keyframe.txt").read_text().split()]
seqs = sorted({f.split("/")[0] for f in train})
first = {sq: next(f for f in train if f.startswith(sq + "/")) for sq in seqs}
objs_in = {sq: set(sio.loadmat(ROOT / "data" / f"{first[sq]}-meta.mat")["cls_indexes"].ravel()) for sq in seqs}
vrng = np.random.default_rng(0); val_seqs = set()
for c in CID:                                   # one held-out sequence per object (rng 0), so every object is validated
    cand = [sq for sq in seqs if c in objs_in[sq] and sq not in val_seqs]
    if not any(c in objs_in[v] for v in val_seqs):
        val_seqs.add(vrng.choice(cand))
frames = [(f, "validation" if f.split("/")[0] in val_seqs else "train") for i, f in enumerate(train) if int(f.split("/")[1]) % STRIDE == 0]
frames += [(f, "test") for f in key]
rows, maxerr, ious = [], 0.0, []
for f, split in frames:
    m = sio.loadmat(ROOT / "data" / f"{f}-meta.mat")
    cls = m["cls_indexes"].ravel(); K = m["intrinsic_matrix"]
    lab = cv2.imread(str(ROOT / "data" / f"{f}-label.png"), cv2.IMREAD_UNCHANGED)
    for k, c in enumerate(cls):
        if c not in CID:
            continue
        o = CID[c]; Rt = m["poses"][:, :, k]; R, t = Rt[:, :3], Rt[:, 3]
        npx = int((lab == c).sum())
        if split != "test" and npx < MIN_PIX:
            continue
        X = PTS[o] @ R.T + t; uv = np.stack([K[0, 0] * X[:, 0] / X[:, 2] + K[0, 2], K[1, 1] * X[:, 1] / X[:, 2] + K[1, 2]], 1)
        Rr = ray_rot(t[None])[0]; Ra = Rr.T @ R
        X2 = PTS[o] @ (Rr @ Ra).T + t; uv2 = np.stack([K[0, 0] * X2[:, 0] / X2[:, 2] + K[0, 2], K[1, 1] * X2[:, 1] / X2[:, 2] + K[1, 2]], 1)
        maxerr = max(maxerr, float(np.abs(uv - uv2).max()))
        if npx > 0:
            ys, xs = np.nonzero(lab == c); a = np.r_[uv.min(0), uv.max(0)]; b = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)
            inter = np.clip(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0, None).prod()
            ious.append(inter / ((a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - inter))
        cx, cy, side = D.gt_square_box(uv, None, margin=0.15, min_size=32)
        qe, qa = mat_to_quat(R[None])[0], mat_to_quat(Ra[None])[0]
        rows.append({"image_relpath": f"data/{f}-color.png", "split": split, "domain": "ycbv", "obj": o, "obj_cls": int(c),
                     "seq": f.split("/")[0], "frame": f.split("/")[1], "inst_id": k, "visible_px": npx,
                     **{f"q_ego_{a_}": v for a_, v in zip("wxyz", qe)}, **{f"q_allo_{a_}": v for a_, v in zip("wxyz", qa)},
                     **{f"t_{a_}": v for a_, v in zip("xyz", t)}, "cam_fx": K[0, 0], "cam_fy": K[1, 1], "cam_cx": K[0, 2], "cam_cy": K[1, 2],
                     "crop_cx": cx, "crop_cy": cy, "crop_side": side, "crop_margin": 0.15})
df = pd.DataFrame(rows)
rng = np.random.default_rng(0); drop = []
for o in OBJ:
    ii = np.flatnonzero(((df.split == "train") & (df.obj == o)).to_numpy())
    if len(ii) > CAP:
        drop += list(rng.choice(ii, len(ii) - CAP, replace=False))
df = df.drop(index=drop).reset_index(drop=True)
df.to_csv(f"{OUT}.csv", index=False, float_format="%.9g")
Path(f"{OUT}.sha256.txt").write_text(D.manifest_hash((df.image_relpath + "#" + df.obj).tolist()) + "\n")
print(df.groupby(["obj", "split"]).size().unstack(1).fillna(0).astype(int).to_string())
print("val seqs", sorted(val_seqs), "| reprojection max px", f"{maxerr:.2e}", "| points-bbox vs label-bbox IoU median / p5",
      f"{np.median(ious):.3f} / {np.percentile(ious, 5):.3f}", "| total", len(df))
