"""Adaptation / deployment cost inventory from existing artifacts (no training, CPU only). Writes
outputs/comparisons/cost_inventory.json, used by paper/make_tables.py (Tables B and C).
  - frozen backbones: parameter count and fp32 file size of the cached weights (DINOv3-L/16 timm safetensors, MoGe-2 ViT-L .pt)
  - trainable heads: parameter counts of phase3.HierTransformer (both token shapes) and trans.TransHead; median best.pt size
  - feature extraction: per dataset and backbone, images extracted (all caches of that dataset, all views) and GPU hours
    (n * ms_per_image recorded in each cache manifest; shared GPU, upper bounds)
  - head training: wall-clock per head from status.json of completed runs (median / min / max; shared GPU)
  - detector training (EXP-124) and end-to-end timing rows (EXP-112) by reference."""
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
E = REPO / "outputs/experiments"
out = {"sources": {}}

# frozen backbones
from safetensors import safe_open  # noqa: E402
import torch  # noqa: E402
hub = Path.home() / ".cache/huggingface/hub"
dino = next(hub.glob("models--timm--vit_large_patch16_dinov3.lvd1689m/snapshots/*/model.safetensors"))
with safe_open(str(dino), "pt") as f:
    n_d = sum(int(np.prod(f.get_slice(k).get_shape())) for k in f.keys())
moge = next(hub.glob("models--Ruicheng--moge-2-vitl-normal/snapshots/*/model.pt"))
ck = torch.load(moge, map_location="cpu", weights_only=False)["model"]
n_m = sum(v.numel() for v in ck.values() if hasattr(v, "numel")); del ck
out["backbones"] = {"DINOv3 ViT-L/16": {"params": n_d, "file_MB": os.path.getsize(os.path.realpath(dino)) / 2**20},
                    "MoGe-2 ViT-L": {"params": n_m, "file_MB": os.path.getsize(os.path.realpath(moge)) / 2**20}}
out["sources"]["backbones"] = [str(dino), str(moge)]

# trainable heads
from tfpose import phase3 as P, trans as TR  # noqa: E402
cnt = lambda m: sum(p.numel() for p in m.parameters())
out["heads"] = {"tree_dino": cnt(P.HierTransformer((64, 1024), 5)), "tree_moge": cnt(P.HierTransformer((256, 4), 5)), "translation": cnt(TR.TransHead())}

# feature extraction per dataset / backbone
TAGS = {"SPEED+": ("subset_full", "subset_exp109", "subset_exp124"), "T-LESS": ("subset_exp064_tless",), "SYMSOL-T": ("subset_exp064_symsolt",),
        "YCB-V": ("subset_exp064_ycbv_", "subset_exp064_ycbvsyn")}
ext = defaultdict(lambda: {"images": 0, "gpu_h": 0.0})
for mf in glob.glob(str(REPO / "outputs/shared_cache/*/manifest.json")):
    name = Path(mf).parent.name; bb = name.split("__")[0]
    if bb not in ("dinov3_vitl16", "moge2_vitl"):
        continue
    m = json.load(open(mf))
    if "ms_per_image" not in m or "n" not in m:
        continue
    for ds, pre in TAGS.items():
        if any(name.split("__")[1].startswith(p) for p in pre):
            ext[f"{ds}|{bb}"]["images"] += m["n"]; ext[f"{ds}|{bb}"]["gpu_h"] += m["n"] * m["ms_per_image"] / 3.6e6
out["extraction"] = dict(ext)

# head training wall-clock and checkpoint sizes
tr = defaultdict(list); sz = defaultdict(list)
def add(key, d):
    s = json.load(open(d / "status.json"))
    if s.get("state") != "completed":
        return
    tr[key].append(s["wall_clock_s"] / 60)
    if (d / "checkpoints/best.pt").exists():
        sz[key].append(os.path.getsize(d / "checkpoints/best.pt") / 2**20)
for d in (E / "EXP-109_real_domain_aug_tokens16_orbit").glob("RUN-*"):
    c = yaml.safe_load((d / "config.yaml").read_text())
    if c.get("aug_tags") and c.get("features") in ("dinov3_vitl16:tokens8", "moge2_vitl:normals16~and"):
        add(f"SPEED+|rotation tree|{c['features'].split('_')[0]}", d)
for d in (E / "EXP-117_orbit_distillation").glob("RUN-*"):
    c = yaml.safe_load((d / "config.yaml").read_text())
    if c.get("mode") == "distill":
        add(f"SPEED+|distilled student|{c['features'].split('_')[0]}", d)
for d in (E / "EXP-118_translation_6dof").glob("RUN-*"):
    c = yaml.safe_load((d / "config.yaml").read_text())
    if c.get("features") and c.get("trans") in ("frozen", "joint"):
        add(f"SPEED+|translation ({c['trans']})|{c['features'].split('_')[0]}", d)
for exp, lab in (("EXP-106_tless_multipath_tree", "rotation tree"), ("EXP-122_ycbv_synthetic_training", "rotation tree"), ("EXP-126_translation_xobj", "translation (frozen)")):
    for d in (E / exp).glob("RUN-*"):
        c = yaml.safe_load((d / "config.yaml").read_text())
        if c.get("stage") != "train" or (exp.startswith("EXP-106") and c.get("label") != "multi"):
            continue
        add(f"{ {'tless': 'T-LESS', 'symsolt': 'SYMSOL-T', 'ycbv': 'YCB-V'}[c['dataset']] }|{lab}|{c['features'].split('_')[0]}", d)
out["training_min"] = {k: {"n": len(v), "median": float(np.median(v)), "min": float(min(v)), "max": float(max(v))} for k, v in tr.items()}
out["checkpoint_MB"] = {k: float(np.median(v)) for k, v in sz.items() if v}
det = json.load(open(E / "EXP-124_speedplus_detected_official_score/RUN-20261029-145416-seed0/status.json"))
out["detector_training_h"] = det["wall_clock_s"] / 3600
out["sources"]["detector"] = "EXP-124_speedplus_detected_official_score/RUN-20261029-145416-seed0"
out["sources"]["timing"] = "EXP-112_locked_recipe_eval_timing/RUN-20261016-231146-seed0/metrics/timing.csv"
(REPO / "outputs/comparisons").mkdir(exist_ok=True)
(REPO / "outputs/comparisons/cost_inventory.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
