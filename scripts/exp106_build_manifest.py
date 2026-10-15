"""EXP-106 data stage (CPU only, reads BOP JSON; no images): T-LESS crop manifest, allocentric rotations, symmetry sets.

Splits (pre-registered, DEC-009 paths):
  train = train_pbr scenes 0-44 + train_primesense (real), visib_fract >= 0.3, <= 12,000 crops per object (rng 0)
  validation = train_pbr scenes 45-49 (whole scenes held out), visib_fract >= 0.3 (same filter as train)
  test = BOP19 targets of the objects in test_primesense, all instances (no visibility filter)
Crop = GT square box of the projected 3-D model bounding box (data.gt_square_box, margin 0.15, min 32 px).
Allocentric rotation R_allo = R_ray^T R_ego, R_ray = minimal rotation taking the optical axis z to the object ray t/|t|.
Checks: (1) R_ego rebuilt from R_allo and t reprojects the bounding-box corners to the same pixels (max error);
        (2) the projected model box agrees with BOP's bbox_obj (median IoU), validating K, units and pose convention.
Symmetries from models_cad/models_info.json in the model frame (equivalents R S): discrete as given, continuous in 36 steps.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import data as D  # noqa: E402
from tfpose.rotations import normalize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--objects", default="1,4,5,11,18,21,27")
ap.add_argument("--min-visib", type=float, default=0.3)
ap.add_argument("--cap", type=int, default=12000)
ap.add_argument("--cont-steps", type=int, default=36)
ap.add_argument("--out", default="outputs/data_manifests/tless_exp106")
args = ap.parse_args()
ROOT = Path(yaml.safe_load(open(REPO / "configs/paths.yaml"))["tless_root"])
OBJ = [int(o) for o in args.objects.split(",")]
info = json.load(open(ROOT / "models_cad/models_info.json"))


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


def box_corners(o):
    m = info[str(o)]; lo = np.array([m["min_x"], m["min_y"], m["min_z"]]); sz = np.array([m["size_x"], m["size_y"], m["size_z"]])
    return np.array([lo + sz * np.array([(k >> 0) & 1, (k >> 1) & 1, (k >> 2) & 1]) for k in range(8)])


def project(R, t, K, P):
    X = np.einsum("nij,kj->nki", R, P) + t[:, None]
    uv = np.einsum("nij,nkj->nki", K, X); return uv[..., :2] / uv[..., 2:]


def sym_set(o):
    m = info[str(o)]; S = [np.eye(3)]
    for s in m.get("symmetries_discrete", []):
        S.append(np.asarray(s, np.float64).reshape(4, 4)[:3, :3])
    for c in m.get("symmetries_continuous", []):
        a = np.asarray(c["axis"], np.float64); a /= np.linalg.norm(a)
        Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        S += [np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx for th in 2 * np.pi * np.arange(1, args.cont_steps) / args.cont_steps]
    return np.stack(S)


def scene_rows(split_dir, scene, domain, split, keep=None, min_visib=None):
    sd = ROOT / split_dir / f"{scene:06d}"
    gt, gi, cam = (json.load(open(sd / f)) for f in ("scene_gt.json", "scene_gt_info.json", "scene_camera.json"))
    ext = "jpg" if split_dir == "train_pbr" else "png"
    rows = []
    for im, insts in gt.items():
        if keep is not None and int(im) not in keep:
            continue
        for k, (g, ii) in enumerate(zip(insts, gi[im])):
            o = g["obj_id"]
            if o not in OBJ or (keep is not None and o not in keep[int(im)]):
                continue
            if min_visib is not None and ii["visib_fract"] < min_visib:
                continue
            rows.append({"image_relpath": f"{split_dir}/{scene:06d}/rgb/{int(im):06d}.{ext}", "domain": domain, "split": split, "obj_id": o,
                         "scene_id": scene, "im_id": int(im), "inst_id": k, "visib_fract": ii["visib_fract"], "bbox_obj": ii["bbox_obj"],
                         "R": np.asarray(g["cam_R_m2c"]).reshape(3, 3), "t": np.asarray(g["cam_t_m2c"], np.float64),
                         "K": np.asarray(cam[im]["cam_K"]).reshape(3, 3)})
    return rows


rows = []
for s in range(50):
    rows += scene_rows("train_pbr", s, "pbr", "train" if s < 45 else "validation", min_visib=args.min_visib)
for o in OBJ:
    rows += scene_rows("train_primesense", o, "real", "train", min_visib=args.min_visib)
tg = json.load(open(ROOT / "test_targets_bop19.json")); keep = {}
for e in tg:
    if e["obj_id"] in OBJ:
        keep.setdefault(e["scene_id"], {}).setdefault(e["im_id"], set()).add(e["obj_id"])
for sc, k in keep.items():
    rows += scene_rows("test_primesense", sc, "test", "test", keep=k)
df = pd.DataFrame(rows)
# cap per object over train (pbr + real), seeded
rng = np.random.default_rng(0); drop = []
for o in OBJ:
    ii = np.flatnonzero(((df.split == "train") & (df.obj_id == o)).to_numpy())
    if len(ii) > args.cap:
        drop += list(rng.choice(ii, len(ii) - args.cap, replace=False))
df = df.drop(index=drop).reset_index(drop=True)

R, t, K = np.stack(df.R), np.stack(df.t), np.stack(df.K)
Rr = ray_rot(t); Ra = np.transpose(Rr, (0, 2, 1)) @ R
qe, qa = mat_to_quat(R), mat_to_quat(Ra)
uv = np.stack([project(R[i:i + 1], t[i:i + 1], K[i:i + 1], box_corners(o))[0] for i, o in enumerate(df.obj_id)])
uv2 = np.stack([project((Rr @ Ra)[i:i + 1], t[i:i + 1], K[i:i + 1], box_corners(o))[0] for i, o in enumerate(df.obj_id)])
reproj = np.abs(uv - uv2).max((1, 2))
box = np.concatenate([uv.min(1), uv.max(1)], 1); bb = np.stack(df.bbox_obj).astype(float); bb[:, 2:] += bb[:, :2]
inter = np.clip(np.minimum(box[:, 2:], bb[:, 2:]) - np.maximum(box[:, :2], bb[:, :2]), 0, None).prod(1)
iou = inter / ((box[:, 2:] - box[:, :2]).prod(1) + (bb[:, 2:] - bb[:, :2]).prod(1) - inter)
crops = np.array([D.gt_square_box(u, None, margin=0.15, min_size=32) for u in uv])
out = df.drop(columns=["R", "t", "K", "bbox_obj"]).copy()
for i, k in enumerate("wxyz"):
    out[f"q_ego_{k}"] = qe[:, i]; out[f"q_allo_{k}"] = qa[:, i]
for i, k in enumerate("xyz"):
    out[f"t_{k}"] = t[:, i]
for i, k in enumerate(["fx", "fy", "cx", "cy"]):
    out[f"cam_{k}"] = K[:, [0, 1, 0, 1][i], [0, 1, 2, 2][i]]
out["crop_cx"], out["crop_cy"], out["crop_side"], out["crop_margin"] = crops[:, 0], crops[:, 1], crops[:, 2], 0.15
out["bbox_iou_bop"] = iou
p = REPO / args.out
out.to_csv(f"{p}.csv", index=False, float_format="%.9g")
(Path(f"{p}.sha256.txt")).write_text(D.manifest_hash((out.image_relpath + "#" + out.obj_id.astype(str) + "#" + out.inst_id.astype(str)).tolist()) + "\n")
np.savez(f"{p}_symmetries.npz", **{f"obj{o}": sym_set(o) for o in OBJ})
summ = out.groupby(["split", "domain", "obj_id"]).size().unstack(0).fillna(0).astype(int)
print(summ.to_string())
print("total", len(out), "| reprojection max px", f"{reproj.max():.2e}",
      "| BOP bbox IoU median / p5 / min", f"{np.median(iou):.4f} / {np.percentile(iou, 5):.4f} / {iou.min():.4f}")
print("symmetry set sizes", {o: len(sym_set(o)) for o in OBJ})
json.dump({"n": len(out), "reprojection_max_px": float(reproj.max()), "bbox_iou_median": float(np.median(iou)),
           "bbox_iou_p5": float(np.percentile(iou, 5)), "counts": {f"{a}|{b}|{c}": int(v) for (a, b, c), v in out.groupby(["split", "domain", "obj_id"]).size().items()}},
          open(f"{p}_checks.json", "w"), indent=2)
