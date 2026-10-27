"""EXP-122 data stage (CPU only): YCB-Video `data_syn` training crops for the six EXP-064 objects, composited onto real
backgrounds, plus the crop manifest in the EXP-064 format.

`data_syn` renders have a black background. As in DenseFusion / FFB6D, each selected render is pasted onto a real YCB-V
frame where label == 0. Background frames are drawn (rng 0) only from the EXP-064 real *train* rows, so neither the
held-out validation sequences nor the test sequences (0048-0059) are ever used. Composites are written to
ycbvsyn_root (hdd2/RAM); the dataset folder stays read-only.
Selection: instances of the six objects with >= 1,000 visible label pixels; at most 12,000 per object (rng 0), the same
cap and visibility rule as the real rows. Crop, allocentric rotation and reprojection check as exp064_ycbv_manifest.py.
Writes outputs/data_manifests/exp064_ycbvsyn_{all,train}.csv (identical; all rows split = train)."""
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import data as D  # noqa: E402
from tfpose.allo import mat_to_quat, ray_rot  # noqa: E402

PATHS = yaml.safe_load(open(REPO / "configs/paths.yaml"))
ROOT = Path(PATHS["ycbv_root"]); OUT_ROOT = Path(PATHS["ycbvsyn_root"])
CLASSES = (ROOT / "image_sets/classes.txt").read_text().split()
OBJ = ["003_cracker_box", "006_mustard_bottle", "011_banana", "025_mug", "035_power_drill", "037_scissors"]
CID = {CLASSES.index(o) + 1: o for o in OBJ}
PTS = {o: np.loadtxt(ROOT / "models" / o / "points.xyz") for o in OBJ}
MIN_PIX, CAP = 1000, 12000
SYN = sorted(int(p.name[:6]) for p in (ROOT / "data_syn").glob("*-meta.mat"))


def scan(i):
    m = sio.loadmat(ROOT / "data_syn" / f"{i:06d}-meta.mat", variable_names=["cls_indexes", "poses", "intrinsic_matrix"])
    cls = m["cls_indexes"].ravel().astype(int)
    if not any(c in CID for c in cls):
        return []
    lab = cv2.imread(str(ROOT / "data_syn" / f"{i:06d}-label.png"), cv2.IMREAD_UNCHANGED); K = m["intrinsic_matrix"]; out = []
    for k, c in enumerate(cls):
        if c not in CID or (npx := int((lab == c).sum())) < MIN_PIX:
            continue
        o = CID[c]; R, t = m["poses"][:, :3, k], m["poses"][:, 3, k]
        X = PTS[o] @ R.T + t; uv = np.stack([K[0, 0] * X[:, 0] / X[:, 2] + K[0, 2], K[1, 1] * X[:, 1] / X[:, 2] + K[1, 2]], 1)
        Rr = ray_rot(t[None])[0]; Ra = Rr.T @ R
        X2 = PTS[o] @ (Rr @ Ra).T + t; uv2 = np.stack([K[0, 0] * X2[:, 0] / X2[:, 2] + K[0, 2], K[1, 1] * X2[:, 1] / X2[:, 2] + K[1, 2]], 1)
        cx, cy, side = D.gt_square_box(uv, None, margin=0.15, min_size=32)
        qe, qa = mat_to_quat(R[None])[0], mat_to_quat(Ra[None])[0]
        out.append({"image_relpath": f"data_syn/{i:06d}-color.jpg", "split": "train", "domain": "ycbv_syn", "obj": o, "obj_cls": int(c),
                    "seq": "syn", "frame": f"{i:06d}", "inst_id": k, "visible_px": npx,
                    **{f"q_ego_{a_}": v for a_, v in zip("wxyz", qe)}, **{f"q_allo_{a_}": v for a_, v in zip("wxyz", qa)},
                    **{f"t_{a_}": v for a_, v in zip("xyz", t)}, "cam_fx": K[0, 0], "cam_fy": K[1, 1], "cam_cx": K[0, 2], "cam_cy": K[1, 2],
                    "crop_cx": cx, "crop_cy": cy, "crop_side": side, "crop_margin": 0.15, "_reproj": float(np.abs(uv - uv2).max())})
    return out


def composite(job):
    i, bg = job
    dst = OUT_ROOT / "data_syn" / f"{i:06d}-color.jpg"
    if dst.exists():
        return
    fg = cv2.imread(str(ROOT / "data_syn" / f"{i:06d}-color.jpg")); lab = cv2.imread(str(ROOT / "data_syn" / f"{i:06d}-label.png"), cv2.IMREAD_UNCHANGED)
    b = cv2.imread(str(ROOT / bg)); assert fg.shape == b.shape, (i, bg)
    out = np.where((lab > 0)[..., None], fg, b)
    cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, 95])


if __name__ == "__main__":
    with Pool(6) as pool:
        rows = [r for rs in pool.imap(scan, SYN, chunksize=64) for r in rs]
    df = pd.DataFrame(rows); rng = np.random.default_rng(0); keep = []
    for o in OBJ:
        ii = np.flatnonzero((df.obj == o).to_numpy())
        keep += list(rng.choice(ii, CAP, replace=False) if len(ii) > CAP else ii)
    df = df.iloc[np.sort(keep)].reset_index(drop=True)
    print("reprojection max px", f"{df._reproj.max():.2e}"); df = df.drop(columns="_reproj")
    real = pd.read_csv(REPO / "outputs/data_manifests/exp064_ycbv_all.csv")
    bgs = np.array(sorted(set(real.loc[real.split == "train", "image_relpath"])))          # real train frames only
    assert not set(real.loc[real.split != "train", "seq"].astype(str).str.zfill(4)) & {p.split("/")[1] for p in bgs}
    frames = sorted(set(df.frame.astype(int))); brng = np.random.default_rng(0); pick = brng.choice(len(bgs), len(frames))
    (OUT_ROOT / "data_syn").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame": frames, "background": bgs[pick]}).to_csv(OUT_ROOT / "backgrounds.csv", index=False)
    with Pool(6) as pool:
        pool.map(composite, list(zip(frames, bgs[pick])), chunksize=64)
    out = REPO / "outputs/data_manifests/exp064_ycbvsyn"
    for s in ("all", "train"):
        df.to_csv(f"{out}_{s}.csv", index=False, float_format="%.9g")
    Path(f"{out}.sha256.txt").write_text(D.manifest_hash((df.image_relpath + "#" + df.obj).tolist()) + "\n")
    print(df.groupby("obj").size().to_string()); print("frames composited", len(frames), "| background pool", len(bgs), "| rows", len(df))
