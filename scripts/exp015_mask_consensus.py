"""EXP-015: training-free two-way DINOv3 <-> MoGe-2 foreground consensus.

Protocol pre-registered in outputs/reports/EXP-015_mask_consensus.md (incl. amendment A1).
CPU only; DINO tokens streamed in chunks. Writes the AND masks to
outputs/shared_cache/consensus_masks__subset_<s>/ for EXP-016.
"""
import argparse
import json
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
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
run = Run("EXP-015", "mask_consensus", vars(args), seed=args.seed)
try:
    paths = yaml.safe_load(open(REPO / "configs/paths.yaml"))
    root = Path(paths["speedplus_root"])
    cam = D.load_camera(root)
    pts = D.load_tango_points(paths["tango_points"])
    df = pd.read_csv(REPO / f"outputs/data_manifests/subset_{args.subset}.csv")
    dom = np.where(df.domain == "synthetic", np.where(df.split == "train", "synthetic_train", "synthetic_val"), df.domain)
    N = len(df)
    dino_path = featsets.find_cache("dinov3_vitl16", args.subset)
    moge_path = featsets.find_cache("moge2_vitl", args.subset)
    methods = ["moge2", "dino", "and", "or", "avg"]
    rows = []
    and_masks = np.zeros((N, M.RES, M.RES), np.uint8)
    with h5py.File(dino_path / "features.h5") as fd, h5py.File(moge_path / "features.h5") as fm:
        assert [s.decode() for s in fd["image_relpath"][:]] == df.image_relpath.tolist()
        assert [s.decode() for s in fm["image_relpath"][:]] == df.image_relpath.tolist()
        for s0 in range(0, N, 256):
            G = fd["last_grid"][s0:s0 + 256]
            Mg = fm["mask"][s0:s0 + 256][..., 0].astype(np.float32)
            for j in range(len(G)):
                i = s0 + j
                r = df.iloc[i]
                q = np.array([r.q_can_w, r.q_can_x, r.q_can_y, r.q_can_z]); t = np.array([r.t_x, r.t_y, r.t_z])
                score = M.dino_pca_score(G[j])
                dino = score > M.otsu(score)
                moge = Mg[j] > 0.5
                sn = (score - score.min()) / (score.max() - score.min() + 1e-12)
                pred = {"moge2": moge, "dino": dino, "and": moge & dino, "or": moge | dino, "avg": (0.5 * (Mg[j] + sn)) > 0.5}
                and_masks[i] = pred["and"]
                gt = M.gt_silhouette(pts, q, t, cam, r.crop_cx, r.crop_cy, r.crop_side, "hull8") > 0.5
                env = M.envelope(pts, q, t, cam, r.crop_cx, r.crop_cy, r.crop_side)
                tips = M.tip_points(pts, q, t, cam, r.crop_cx, r.crop_cy, r.crop_side)
                row = {"idx": i, "image_relpath": r.image_relpath, "domain": dom[i], "crop_side": r.crop_side}
                for m in methods:
                    mm = M.mask_metrics(pred[m], gt, tips)
                    row.update({f"{m}_{k}": v for k, v in mm.items()})
                    row[f"{m}_leakage"] = M.leakage(pred[m], env)
                    row[f"{m}_area"] = float(pred[m].mean())
                rows.append(row)
            print(f"  {min(s0 + 256, N)}/{N}", flush=True)
    per = pd.DataFrame(rows)
    pred_dir = run.sub("predictions")
    per.to_csv(pred_dir / "per_image_mask_metrics.csv", index=False, float_format="%.5g")

    # derived consensus masks for EXP-016 (immutable cache with manifest)
    out = REPO / f"outputs/shared_cache/consensus_masks__subset_{args.subset}"
    if (out / "manifest.json").exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    with h5py.File(out / "masks.h5", "w") as f:
        f.create_dataset("image_relpath", data=np.array(df.image_relpath.tolist(), dtype="S"))
        f.create_dataset("and_mask", data=and_masks, compression="gzip")
    (out / "manifest.json").write_text(json.dumps({"rule": "MoGe-2 mask>0.5 AND DINOv3-L PCA-1 Otsu", "dino_cache": dino_path.name,
                                                   "moge_cache": moge_path.name, "producer_run": run.rel()}, indent=2))

    # aggregates with paired bootstrap CIs
    rng = np.random.default_rng(args.seed)
    keys = ["leakage", "recall", "iou", "precision", "boundary_f", "tip_coverage", "area"]
    agg, diffs = [], []
    for d in ["synthetic_val", "lightbox", "sunlamp", "synthetic_train"]:
        g = per[per.domain == d]
        for m in methods:
            agg.append({"domain": d, "method": m, "n": len(g), **{k: float(g[f"{m}_{k}"].mean()) for k in keys}})
        B = rng.integers(0, len(g), size=(args.boot, len(g)))
        for a, b in [("and", "moge2"), ("and", "dino"), ("or", "moge2"), ("avg", "moge2"), ("dino", "moge2")]:
            for k in ["leakage", "recall", "iou", "boundary_f"]:
                dlt = (g[f"{a}_{k}"] - g[f"{b}_{k}"]).to_numpy()
                bs = dlt[B].mean(1)
                diffs.append({"domain": d, "comparison": f"{a} - {b}", "metric": k, "mean_diff": float(dlt.mean()),
                              "ci_low": float(np.percentile(bs, 2.5)), "ci_high": float(np.percentile(bs, 97.5))})
    case = []
    for d in ["synthetic_val", "lightbox", "sunlamp"]:
        g = per[per.domain == d]
        mw, dw = g.moge2_leakage > 0.10, g.dino_leakage > 0.10
        for mlab, mm in [("MoGe-2 right", ~mw), ("MoGe-2 wrong", mw)]:
            for dlab, dd in [("DINO right", ~dw), ("DINO wrong", dw)]:
                c = g[mm & dd]
                case.append({"domain": d, "moge2": mlab, "dino": dlab, "frac_images": len(c) / len(g),
                             "and_leakage": float(c.and_leakage.mean()) if len(c) else float("nan"),
                             "and_recall": float(c.and_recall.mean()) if len(c) else float("nan"),
                             "moge2_leakage": float(c.moge2_leakage.mean()) if len(c) else float("nan"),
                             "dino_leakage": float(c.dino_leakage.mean()) if len(c) else float("nan")})
    run.write_metrics(agg, "metrics")
    run.write_metrics(diffs, "paired_differences")
    run.write_metrics(case, "case_table")
    t = run.sub("tables")
    (t / "metrics.md").write_text(md_table(agg))
    (t / "paired_differences.md").write_text(md_table(diffs))
    (t / "case_table.md").write_text(md_table(case))
    run.done(consensus_cache=str(out.relative_to(REPO)))
    print((t / "metrics.md").read_text()); print((t / "case_table.md").read_text())
    print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc())
    raise
