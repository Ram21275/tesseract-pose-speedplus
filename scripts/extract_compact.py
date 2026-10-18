"""Compact frozen-feature extraction for the cross-dataset block (EXP-064 / EXP-106) and its SPEED+ equivalence check.

Stores only what the locked recipe reads (about 175 KB per crop instead of about 750 KB):
  dinov3_vitl16 : last_grid8 (8x8x1024, = featsets `tokens8`), dino_fg (74x74 uint8, EXP-015 per-image PCA-1 Otsu)
  moge2_vitl    : normal (74x74x3), mask (74x74x1)   (= the inputs of `normals16`)
`build_consensus` then writes consensus_masks__subset_<tag> (MoGe-2 mask > 0.5 AND dino_fg), as EXP-015 did.
Caches live under --cache-root (compact_cache_root, hdd2) and are symlinked into the featsets cache root (cache_root).
Row key = image_relpath (+ "#obj#inst" when an image has several targets). RGB datasets are read in colour (--rgb).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose.cachepaths import CACHE, COMPACT  # noqa: E402
from tfpose import data as D, features as Fx, inplane as IP, masks as M  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True, help="CSV with image_relpath + crop_cx/cy/side (outputs/data_manifests/...)")
ap.add_argument("--root-key", required=True, help="configs/paths.yaml key of the image root")
ap.add_argument("--tag", required=True, help="subset name used by featsets.load(spec, tag)")
ap.add_argument("--backbone", choices=["dinov3_vitl16", "moge2_vitl", "consensus"], required=True)
ap.add_argument("--rgb", action="store_true")
ap.add_argument("--key-cols", default="", help="extra columns appended to image_relpath to make a unique row key, e.g. obj_id,inst_id")
ap.add_argument("--cache-root", default=None, help="default: configs/paths.yaml compact_cache_root")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--rot-deg", type=float, default=None, help="EXP-109: rotate every crop in-plane by this angle (deg, cv2 sign)")
ap.add_argument("--rot-random-seed", type=int, default=None, help="EXP-109: per-row angle ~ U(-180,180), rng(seed) in row order")
ap.add_argument("--roll-homography", action="store_true", help="EXP-064: exact camera-roll homography (needs cam_fx/fy/cx/cy columns)")
ap.add_argument("--store-grid16", action="store_true", help="EXP-109 B: also store the 16x16 DINOv3 grid (fp16)")
ap.add_argument("--photo-seed", type=int, default=None, help="EXP-109: per-row photometric augmentation, rng([seed, row])")
args = ap.parse_args()
args.cache_root = args.cache_root or str(COMPACT)

paths = yaml.safe_load(open(REPO / "configs/paths.yaml"))
root = Path(paths[args.root_key])
df = pd.read_csv(REPO / args.manifest if not Path(args.manifest).is_absolute() else args.manifest)
if args.limit:
    df = df.iloc[:args.limit]
keys = df.image_relpath.astype(str)
for c in [c for c in args.key_cols.split(",") if c]:
    keys = keys + "#" + df[c].astype(str)
keys = keys.tolist()
assert len(set(keys)) == len(keys), "row keys are not unique: pass --key-cols"
ANG = np.zeros(len(df))
if args.rot_random_seed is not None:
    ANG = np.random.default_rng(args.rot_random_seed).uniform(-180, 180, len(df))
if args.rot_deg is not None:                                   # fixed angle, added to the random one if both are given
    ANG = ANG + args.rot_deg
mhash = D.manifest_hash(keys + [f"rot={args.rot_deg}|{args.rot_random_seed}|photo={args.photo_seed}"])
shared = CACHE


class Crops(torch.utils.data.Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return len(df)

    def __getitem__(self, i):
        r = df.iloc[i]
        if args.rgb:
            img = cv2.imread(str(root / r.image_relpath), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(root / r.image_relpath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = D.read_gray(root, r.image_relpath)
        if ANG[i] != 0 and args.roll_homography:
            Km = np.array([[r.cam_fx, 0, r.cam_cx], [0, r.cam_fy, r.cam_cy], [0, 0, 1.0]])
            c = IP.crop_rolled_homography(img, r.crop_cx, r.crop_cy, r.crop_side, self.size, ANG[i], Km)
        elif ANG[i] != 0:
            c = IP.crop_rotated(img, r.crop_cx, r.crop_cy, r.crop_side, self.size, ANG[i])
        else:
            c = D.crop_square(img, r.crop_cx, r.crop_cy, r.crop_side, self.size)
        c = c.astype(np.float32) / 255.0
        if args.photo_seed is not None:
            c = IP.photometric(c, np.random.default_rng([args.photo_seed, i]))
        t = torch.from_numpy(np.ascontiguousarray(c))
        return t.permute(2, 0, 1).contiguous() if t.ndim == 3 else t[None].expand(3, -1, -1).contiguous()


def link(out):
    ln = shared / out.name
    if not ln.exists():
        ln.symlink_to(out)


if args.backbone == "consensus":
    def one(bb):
        hits = [h for h in shared.glob(f"{bb}__subset_{args.tag}__*") if (h / "manifest.json").exists()]
        assert len(hits) == 1, hits
        return hits[0]
    dp, mp = one("dinov3_vitl16"), one("moge2_vitl")
    out = Path(args.cache_root) / f"consensus_masks__subset_{args.tag}"
    if (out / "manifest.json").exists():
        link(out); print("cache exists", out); sys.exit(0)
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(dp / "features.h5") as fd, h5py.File(mp / "features.h5") as fm, h5py.File(out / "masks.h5", "w") as fo:
        names = fd["image_relpath"][:]
        assert (names == fm["image_relpath"][:]).all() and [s.decode() for s in names] == keys
        fo.create_dataset("image_relpath", data=names)
        ds = fo.create_dataset("and_mask", shape=(len(names), M.RES, M.RES), dtype=np.uint8, chunks=(256, M.RES, M.RES))
        for s0 in range(0, len(names), 4096):
            ds[s0:s0 + 4096] = ((fm["mask"][s0:s0 + 4096][..., 0] > 0.5) & (fd["dino_fg"][s0:s0 + 4096] > 0)).astype(np.uint8)
    (out / "manifest.json").write_text(json.dumps({"rule": "MoGe-2 mask>0.5 AND DINOv3-L PCA-1 Otsu (compact)", "dino_cache": dp.name,
                                                    "moge_cache": mp.name, "n": len(names)}, indent=2))
    link(out); print("done", out); sys.exit(0)

dev = torch.device("cuda")
torch.manual_seed(0)
if args.backbone == "dinov3_vitl16":
    ex = Fx.DinoV3("vit_large_patch16_dinov3.lvd1689m", dev)
else:
    ex = Fx.MoGe2Extractor(dev)
spec = {**ex.spec, "compact": True, "gray_to_rgb": "rgb" if args.rgb else "replicate",
        "rot_deg": args.rot_deg, "rot_random_seed": args.rot_random_seed, "photo_seed": args.photo_seed,
        "roll_homography": args.roll_homography}
out = Path(args.cache_root) / f"{args.backbone}__subset_{args.tag}__{Fx.preprocess_hash(spec)}__{mhash}"
if (out / "manifest.json").exists():
    print("cache exists", out); link(out); sys.exit(0)
out.mkdir(parents=True, exist_ok=True)
dl = torch.utils.data.DataLoader(Crops(ex.size), batch_size=args.batch, num_workers=args.workers, shuffle=False)
code = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
N = len(df); t0 = time.time(); s = 0
with h5py.File(out / "features.h5.partial", "w") as f:
    f.create_dataset("image_relpath", data=np.array(keys, dtype="S"))
    dsets = {}
    for x in dl:
        for attempt in range(6):
            try:
                o = ex(x); break
            except torch.OutOfMemoryError:
                if attempt == 5:
                    raise
                torch.cuda.empty_cache(); print(f"  CUDA OOM at {s}, retry in 60 s", flush=True); time.sleep(60)
        if args.backbone == "dinov3_vitl16":
            g16 = o["last_grid"]                                                          # (B,16,16,C) float32
            g8 = F.adaptive_avg_pool2d(g16.permute(0, 3, 1, 2), 8).permute(0, 2, 3, 1)
            g16h = g16.half().float().cpu().numpy()                                       # as cached fp16 in EXP-015
            fg = np.stack([(lambda sc: (sc > M.otsu(sc)))(M.dino_pca_score(g)) for g in g16h]).astype(np.uint8)
            vals = {"last_grid8": g8.cpu().numpy().astype(np.float16), "dino_fg": fg}
            if args.store_grid16:
                vals["last_grid16"] = g16.cpu().numpy().astype(np.float16)
        else:
            vals = {"normal": o["normal"].cpu().numpy().astype(np.float16), "mask": o["mask"].cpu().numpy().astype(np.float16)}
        for k, v in vals.items():
            if k not in dsets:
                dsets[k] = f.create_dataset(k, shape=(N,) + v.shape[1:], dtype=v.dtype, chunks=(min(64, N),) + v.shape[1:])
            dsets[k][s:s + len(v)] = v
        s += len(x)
        if (s // args.batch) % 50 == 0:
            print(f"  {s}/{N}  {(time.time() - t0) / s * 1000:.1f} ms/img", flush=True)
(out / "features.h5.partial").rename(out / "features.h5")
(out / "manifest.json").write_text(json.dumps({"spec": spec, "preprocess_hash": Fx.preprocess_hash(spec), "manifest_hash": mhash, "n": N,
                                               "code_version": code, "seconds": round(time.time() - t0, 1),
                                               "ms_per_image": round((time.time() - t0) / N * 1000, 2), "source_manifest": args.manifest,
                                               "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)}, indent=2))
pd.DataFrame({"key": keys, "rot_deg": ANG}).to_csv(out / "aug.csv", index=False)
link(out); print("done", out)
