"""Assets for the architecture figure (fig_arch.py), from the real T-LESS data pipeline (run from the repo root):
    .venv/bin/python video/paper_figures/prep_arch.py
- training example: a fixed-rule T-LESS PBR training crop of object 5 (first row of exp064_tless_train.csv with object 5 and
  visib_fract > 0.95) as the clean view and its two cached augmented copies, re-created with the exact extraction code
  (scripts/extract_compact.py: homography roll with the cached angle, photometric rng([seed, row]));
- test example: the T-LESS teaser image (median-error test image of object 5), its four homography-rolled views, and its
  cached frozen features: DINOv3 8x8x1024 tokens (PCA to RGB, per image), MoGe-2 normals under the consensus mask (74x74).
Requires assets.json from prep_assets.py (for the teaser image)."""
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent / "assets"
sys.path.insert(0, str(REPO / "src"))
from tfpose import data as D, inplane as IP  # noqa: E402

ROOT = Path(yaml.safe_load(open(REPO / "configs/paths.yaml"))["tless_root"]); S = 320; CMP = Path("/media/kavinder/hdd2/RAM/compact")
A = json.loads((OUT / "assets.json").read_text())


def rgb(rel):
    im = cv2.imread(str(ROOT / rel), cv2.IMREAD_COLOR); return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def crop(r, ang=0.0):
    img = rgb(r.image_relpath)
    if ang:
        K = np.array([[r.cam_fx, 0, r.cam_cx], [0, r.cam_fy, r.cam_cy], [0, 0, 1.0]])
        return IP.crop_rolled_homography(img, r.crop_cx, r.crop_cy, r.crop_side, S, ang, K)
    return D.crop_square(img, r.crop_cx, r.crop_cy, r.crop_side, S)


def save(a, name):
    a = np.asarray(a); a = (np.clip(a, 0, 1) * 255).astype(np.uint8) if a.dtype != np.uint8 else a
    cv2.imwrite(str(OUT / f"{name}.png"), cv2.cvtColor(a, cv2.COLOR_RGB2BGR))


def cache(prefix):
    h = [p for p in CMP.glob(prefix + "__*") if (p / "manifest.json").exists()]; assert len(h) == 1, h; return h[0]


# ---- training example: clean + the two cached augmented copies (exact re-creation)
tr = pd.read_csv(REPO / "outputs/data_manifests/exp064_tless_train.csv")
i = int(tr.index[(tr.obj_id == 5) & (tr.visib_fract > 0.95)][0]); r = tr.iloc[i]
key = f"{r.image_relpath}#{r.obj_id}#{r.inst_id}"
save(crop(r), "arch_train_clean"); info = dict(train_key=key, train_row=i, aug={})
for seed in (1, 2):
    d = cache(f"dinov3_vitl16__subset_exp064_tless_aug{seed}"); aug = pd.read_csv(d / "aug.csv")
    assert aug.key.iloc[i] == key, (aug.key.iloc[i], key)
    ang = float(aug.rot_deg.iloc[i])
    c = crop(r, ang).astype(np.float32) / 255.0
    c = IP.photometric(c, np.random.default_rng([seed, i]))
    save(c, f"arch_train_aug{seed}"); info["aug"][seed] = ang

# ---- test example: the teaser image, rolled views and its cached features
al = pd.read_csv(REPO / "outputs/data_manifests/exp064_tless_all.csv")
im = A["teaser"]["tless"]["image"]; j = int(al.index[(al.image_relpath == im) & (al.obj_id == 5)][0]); t = al.iloc[j]
tkey = f"{t.image_relpath}#{t.obj_id}#{t.inst_id}"
for k in range(4):
    save(crop(t, 90.0 * k), f"arch_view{k}")
with h5py.File(cache("dinov3_vitl16__subset_exp064_tless_all") / "features.h5") as f:
    assert f["image_relpath"][j].decode() == tkey
    tok = f["last_grid8"][j].astype(np.float32).reshape(64, -1)
with h5py.File(cache("moge2_vitl__subset_exp064_tless_all") / "features.h5") as f:
    assert f["image_relpath"][j].decode() == tkey
    nrm = f["normal"][j].astype(np.float32)
with h5py.File(CMP / "consensus_masks__subset_exp064_tless_all/masks.h5") as f:
    assert f["image_relpath"][j].decode() == tkey
    msk = f["and_mask"][j].astype(bool)
x = tok - tok.mean(0); _, _, V = np.linalg.svd(x, full_matrices=False); p = x @ V[:3].T
p = (p - p.min(0)) / (p.max(0) - p.min(0) + 1e-9)
save(cv2.resize(p.reshape(8, 8, 3), (S, S), interpolation=cv2.INTER_NEAREST), "arch_dino_pca")
n = np.where(msk[..., None], (nrm + 1) / 2, 0.93)
save(cv2.resize(n, (S, S), interpolation=cv2.INTER_NEAREST), "arch_moge_normals")
info.update(test_key=tkey, test_row=j, mask_frac=float(msk.mean()))
A["arch"] = info
(OUT / "assets.json").write_text(json.dumps(A, indent=1, default=float))
print(json.dumps(info, indent=1))
