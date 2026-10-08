"""EXP-018 mask evaluation: CASS-injected MoGe-2 mask and injected-DINO PCA foreground vs originals.

Same metrics/envelope as EXP-015 (amendment A1); originals are read from the EXP-015 per-image file.
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import data as D, featsets, masks as M  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--subset", default="v1")
ap.add_argument("--exp015-run", default="outputs/experiments/EXP-015_mask_consensus/RUN-20261008-234038-seed0")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
run = Run("EXP-018", "cass_attention_injection", dict(vars(args), stage="mask_eval"), seed=args.seed)
try:
    paths = yaml.safe_load(open(REPO / "configs/paths.yaml"))
    cam = D.load_camera(Path(paths["speedplus_root"])); pts = D.load_tango_points(paths["tango_points"])
    df = pd.read_csv(REPO / f"outputs/data_manifests/subset_{args.subset}.csv")
    dom = np.where(df.domain == "synthetic", np.where(df.split == "train", "synthetic_train", "synthetic_val"), df.domain)
    base = pd.read_csv(REPO / args.exp015_run / "predictions/per_image_mask_metrics.csv")
    rows = []
    with h5py.File(featsets.find_cache("moge2_vitl_cassdino", args.subset) / "features.h5") as fm, \
         h5py.File(featsets.find_cache("dinov3_vitl16_cassmoge", args.subset) / "features.h5") as fd:
        for s0 in range(0, len(df), 256):
            MG = fm["mask"][s0:s0 + 256][..., 0]
            G = fd["last_grid"][s0:s0 + 256]
            for j in range(len(MG)):
                i = s0 + j; r = df.iloc[i]
                q = np.array([r.q_can_w, r.q_can_x, r.q_can_y, r.q_can_z]); t = np.array([r.t_x, r.t_y, r.t_z])
                gt = M.gt_silhouette(pts, q, t, cam, r.crop_cx, r.crop_cy, r.crop_side, "hull8") > 0.5
                env = M.envelope(pts, q, t, cam, r.crop_cx, r.crop_cy, r.crop_side)
                tips = M.tip_points(pts, q, t, cam, r.crop_cx, r.crop_cy, r.crop_side)
                sc = M.dino_pca_score(G[j])
                preds = {"moge2_inj": MG[j] > 0.5, "dino_inj": sc > M.otsu(sc)}
                row = {"idx": i, "domain": dom[i]}
                for m, p in preds.items():
                    row.update({f"{m}_{k}": v for k, v in M.mask_metrics(p, gt, tips).items()})
                    row[f"{m}_leakage"] = M.leakage(p, env); row[f"{m}_area"] = float(p.mean())
                rows.append(row)
    per = pd.DataFrame(rows).merge(base.drop(columns=["domain"]), on="idx")
    per.to_csv(run.sub("predictions") / "per_image_mask_metrics.csv", index=False, float_format="%.5g")
    rng = np.random.default_rng(args.seed)
    keys = ["leakage", "recall", "iou", "boundary_f", "tip_coverage", "area"]
    agg, diffs = [], []
    for d in ["synthetic_val", "lightbox", "sunlamp"]:
        g = per[per.domain == d]
        for m in ["moge2", "moge2_inj", "dino", "dino_inj", "and"]:
            agg.append({"domain": d, "method": m, "n": len(g), **{k: float(g[f"{m}_{k}"].mean()) for k in keys}})
        B = rng.integers(0, len(g), size=(args.boot, len(g)))
        for a, b in [("moge2_inj", "moge2"), ("dino_inj", "dino"), ("moge2_inj", "and")]:
            for k in ["leakage", "recall", "iou", "boundary_f"]:
                dl = (g[f"{a}_{k}"] - g[f"{b}_{k}"]).to_numpy(); bs = dl[B].mean(1)
                diffs.append({"domain": d, "comparison": f"{a} - {b}", "metric": k, "mean_diff": float(dl.mean()),
                              "ci_low": float(np.percentile(bs, 2.5)), "ci_high": float(np.percentile(bs, 97.5))})
    run.write_metrics(agg, "mask_metrics"); run.write_metrics(diffs, "mask_paired_differences")
    (run.sub("tables") / "mask_metrics.md").write_text(md_table(agg))
    run.done()
    print(md_table(agg)); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc())
    raise
