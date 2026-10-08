"""EXP-018 extraction: CASS spectral attention injection in both directions (+ EXP-020 spectra).

Per batch: (1) DINOv3-L plain pass -> last-block keys; (2) MoGe-2 with DINOv3 graph injected
(18a) -> keys + spectra; (3) DINOv3-L with MoGe-2 graph injected (18b).
Writes caches  moge2_vitl_cassdino__subset_<s>__...  dinov3_vitl16_cassmoge__subset_<s>__...
and          cass_spectra__subset_<s>/spectra.h5 (EXP-020).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import cass, features as Fx  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--subset", default="v1")
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--limit", type=int, default=0, help="debug: only the first N images (no cache written)")
args = ap.parse_args()

paths = yaml.safe_load(open(REPO / "configs/paths.yaml"))
root = Path(paths["speedplus_root"])
df = pd.read_csv(REPO / f"outputs/data_manifests/subset_{args.subset}.csv")
if args.limit:
    df = df.iloc[:args.limit]
mhash = (REPO / f"outputs/data_manifests/subset_{args.subset}.sha256.txt").read_text().split()[0]
dev = torch.device("cuda")
dino = Fx.DinoV3("vit_large_patch16_dinov3.lvd1689m", dev)
moge = Fx.MoGe2Extractor(dev)
wd = cass.LastBlockAttention(dino.model.blocks[23].attn, "eva", n_prefix=dino.model.num_prefix_tokens, grid=16)
mb = moge.model.encoder.backbone
wm = cass.LastBlockAttention(mb.blocks[23].attn, "dinov2", n_prefix=1 + mb.num_register_tokens, grid=37)
code = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
cass_spec = {k: getattr(cass, k) for k in ["K_EIGEN", "ENERGY", "EPSILON", "SCALE_FACTOR", "GAUSS_STD"]}
spec_a = dict(moge.spec, cass_injection="DINOv3-L block23 keys -> MoGe-2 block23", **cass_spec)
spec_b = dict(dino.spec, cass_injection="MoGe-2 block23 keys -> DINOv3-L block23", **cass_spec)
outs = {}
if not args.limit:
    for tag, spec in [("moge2_vitl_cassdino", spec_a), ("dinov3_vitl16_cassmoge", spec_b)]:
        p = Fx.cache_path(REPO / "outputs/shared_cache", f"{tag}__subset_{args.subset}", spec, mhash)
        if (p / "manifest.json").exists():
            raise FileExistsError(p)
        p.mkdir(parents=True, exist_ok=True)
        outs[tag] = (p, spec)
    sp_dir = REPO / f"outputs/shared_cache/cass_spectra__subset_{args.subset}"
    if (sp_dir / "manifest.json").exists():
        raise FileExistsError(sp_dir)
    sp_dir.mkdir(parents=True, exist_ok=True)

rows = list(df.itertuples())
N = len(rows)
files, dsets = {}, {}
for tag, (p, _) in outs.items():
    files[tag] = h5py.File(p / "features.h5.partial", "w")
    files[tag].create_dataset("image_relpath", data=np.array(df.image_relpath.tolist(), dtype="S"))
if outs:
    sf = h5py.File(sp_dir / "spectra.h5.partial", "w")
    sf.create_dataset("image_relpath", data=np.array(df.image_relpath.tolist(), dtype="S"))
    for k, shp, dt in [("spec_moge", (16, 20), "f4"), ("spec_dino", (16, 20), "f4"), ("cost", (16, 16), "f4"),
                       ("pairs", (16, 2), "i1"), ("ranks_18a", (16,), "i2"), ("ranks_18b", (16,), "i2")]:
        sf.create_dataset(k, shape=(N,) + shp, dtype=dt)


def write(tag, s, out):
    f = files[tag]
    for k, v in out.items():
        v = v.cpu().numpy().astype(np.float16)
        if (tag, k) not in dsets:
            dsets[(tag, k)] = f.create_dataset(k, shape=(N,) + v.shape[1:], dtype=np.float16, chunks=(min(64, N),) + v.shape[1:])
        dsets[(tag, k)][s:s + len(v)] = v


t0 = time.time()
for s in range(0, N, args.batch):
    chunk = rows[s:s + args.batch]
    x518 = Fx.load_crops(root, chunk, 518)
    x256 = Fx.load_crops(root, chunk, 256)
    wd.mode = "capture"; dino(x256); Kd = wd.keys                                  # (1)
    wm.mode, wm.source = "inject", (Kd, 16); out_a = moge(x518); Km = wm.keys     # (2) 18a
    infos_a = wm.infos
    wd.mode, wd.source = "inject", (Km, 37); out_b = dino(x256)                    # (3) 18b
    infos_b = wd.infos
    for name, o in [("18a", out_a), ("18b", out_b)]:
        for k, v in o.items():
            if not torch.isfinite(v).all():
                raise ValueError(f"non-finite {name} {k}")
    if outs:
        write("moge2_vitl_cassdino", s, out_a)
        write("dinov3_vitl16_cassmoge", s, out_b)
        for j, (ia, ib) in enumerate(zip(infos_a, infos_b)):
            i = s + j
            sf["spec_moge"][i] = ia["spec_target"]; sf["spec_dino"][i] = ia["spec_source"]
            sf["cost"][i] = ia["cost"]; sf["pairs"][i] = ia["pairs"]
            sf["ranks_18a"][i] = ia["ranks"]; sf["ranks_18b"][i] = ib["ranks"]
    if (s // args.batch) % 50 == 0:
        print(f"  {s + len(chunk)}/{N}  {(time.time() - t0) / (s + len(chunk)) * 1000:.0f} ms/img  "
              f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)

for tag, (p, spec) in outs.items():
    files[tag].close()
    (p / "features.h5.partial").rename(p / "features.h5")
    (p / "manifest.json").write_text(json.dumps({"spec": spec, "preprocess_hash": Fx.preprocess_hash(spec), "manifest_hash": mhash,
                                                 "n": N, "code_version": code, "seconds": round(time.time() - t0, 1)}, indent=2))
if outs:
    sf.close()
    (sp_dir / "spectra.h5.partial").rename(sp_dir / "spectra.h5")
    (sp_dir / "manifest.json").write_text(json.dumps({"what": "per-image per-head top-20 eigenvalues of last-block key graphs; "
                                                      "spec_moge = MoGe-2 (target in 18a), spec_dino = DINOv3-L (source in 18a, "
                                                      "interpolated to 37x37); cost = 1 - W1 (CASS); pairs = (dino head, moge head)",
                                                      "code_version": code, **cass_spec}, indent=2))
print("done", round(time.time() - t0, 1), "s")
