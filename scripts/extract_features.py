"""Extract frozen DINOv3 / VGGT features for a subset manifest into outputs/shared_cache/."""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import features as Fx  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backbone", required=True, choices=["dinov3_vitb16", "dinov3_vitl16", "vggt_1b"])
ap.add_argument("--subset", default="v1")
ap.add_argument("--batch", type=int, default=32)
args = ap.parse_args()

paths = yaml.safe_load(open(REPO / "configs/paths.yaml"))
root = Path(paths["speedplus_root"])
df = pd.read_csv(REPO / f"outputs/data_manifests/subset_{args.subset}.csv")
mhash = (REPO / f"outputs/data_manifests/subset_{args.subset}.sha256.txt").read_text().split()[0]
dev = torch.device("cuda")
torch.manual_seed(0)
if args.backbone.startswith("dinov3"):
    name = {"dinov3_vitb16": "vit_base_patch16_dinov3.lvd1689m", "dinov3_vitl16": "vit_large_patch16_dinov3.lvd1689m"}[args.backbone]
    ex = Fx.DinoV3(name, dev)
else:
    ex = Fx.VGGTExtractor(dev)
code = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
out = Fx.cache_path(REPO / "outputs/shared_cache", f"{args.backbone}__subset_{args.subset}", ex.spec, mhash)
print("->", out, flush=True)
Fx.extract_to_cache(ex, root, df, out, args.batch, code, mhash, log=lambda s: print(s, flush=True))
print("done", out)
