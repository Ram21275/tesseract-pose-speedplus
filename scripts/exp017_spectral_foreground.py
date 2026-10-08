"""EXP-017: spectral foreground from fused DINOv3 x MoGe-2 patch graphs (CASS option 1).

Protocol pre-registered in outputs/reports/EXP-017_spectral_foreground.md.
Baselines (MoGe-2, PCA-DINO, AND) are read from the EXP-015 per-image metrics
(identical metric definitions). Also writes eigenvectors 2-5 of the fused
graph to outputs/shared_cache/spectral_parts__subset_<s>/ for EXP-019.
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
from tfpose import data as D, featsets, masks as M, spectral as S  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--subset", default="v1")
ap.add_argument("--exp015-run", default="outputs/experiments/EXP-015_mask_consensus/RUN-20261008-234038-seed0")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
run = Run("EXP-017", "spectral_foreground", vars(args), seed=args.seed)
try:
    paths = yaml.safe_load(open(REPO / "configs/paths.yaml"))
    cam = D.load_camera(Path(paths["speedplus_root"]))
    pts = D.load_tango_points(paths["tango_points"])
    df = pd.read_csv(REPO / f"outputs/data_manifests/subset_{args.subset}.csv")
    dom = np.where(df.domain == "synthetic", np.where(df.split == "train", "synthetic_train", "synthetic_val"), df.domain)
    N = len(df)
    base = pd.read_csv(REPO / args.exp015_run / "predictions/per_image_mask_metrics.csv")
    assert (base.idx.values == np.arange(N)).all()
    methods = ["spec_dino", "spec_fused", "spec_avg"]
    rows, parts = [], np.zeros((N, 16, 16, 4), np.float16)
    with h5py.File(featsets.find_cache("dinov3_vitl16", args.subset) / "features.h5") as fd, \
         h5py.File(featsets.find_cache("moge2_vitl", args.subset) / "features.h5") as fm:
        for s0 in range(0, N, 256):
            G = fd["last_grid"][s0:s0 + 256]
            NR = fm["normal"][s0:s0 + 256]
            DP = fm["depth"][s0:s0 + 256]
            for j in range(len(G)):
                i = s0 + j
                r = df.iloc[i]
                q = np.array([r.q_can_w, r.q_can_x, r.q_can_y, r.q_can_z]); t = np.array([r.t_x, r.t_y, r.t_z])
                Ad = S.dino_affinity(G[j]); Ag = S.geo_affinity(NR[j], DP[j])
                graphs = {"spec_dino": Ad, "spec_fused": Ad * Ag, "spec_avg": 0.5 * (Ad + Ag)}
                gt = M.gt_silhouette(pts, q, t, cam, r.crop_cx, r.crop_cy, r.crop_side, "hull8") > 0.5
                env = M.envelope(pts, q, t, cam, r.crop_cx, r.crop_cy, r.crop_side)
                tips = M.tip_points(pts, q, t, cam, r.crop_cx, r.crop_cy, r.crop_side)
                row = {"idx": i, "domain": dom[i]}
                for m, A in graphs.items():
                    score, V = S.fiedler_score(A)
                    pred = score > M.otsu(score)
                    if m == "spec_fused":
                        parts[i] = V[:, 1:5].reshape(16, 16, 4)
                    mm = M.mask_metrics(pred, gt, tips)
                    row.update({f"{m}_{k}": v for k, v in mm.items()})
                    row[f"{m}_leakage"] = M.leakage(pred, env)
                    row[f"{m}_area"] = float(pred.mean())
                rows.append(row)
            print(f"  {min(s0 + 256, N)}/{N}", flush=True)
    per = pd.DataFrame(rows).merge(base.drop(columns=["domain"]), on="idx")
    per.to_csv(run.sub("predictions") / "per_image_mask_metrics.csv", index=False, float_format="%.5g")

    out = REPO / f"outputs/shared_cache/spectral_parts__subset_{args.subset}"
    if (out / "manifest.json").exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    with h5py.File(out / "parts.h5", "w") as f:
        f.create_dataset("image_relpath", data=np.array(df.image_relpath.tolist(), dtype="S"))
        f.create_dataset("eigvecs_2_5", data=parts, compression="gzip")
    (out / "manifest.json").write_text(json.dumps({"graph": "A_dino * A_geo (EXP-017 primary)", "eigvecs": "2..5 of sym. normalized Laplacian",
                                                   "producer_run": run.rel()}, indent=2))

    rng = np.random.default_rng(args.seed)
    keys = ["leakage", "recall", "iou", "precision", "boundary_f", "tip_coverage", "area"]
    allm = ["moge2", "dino", "and"] + methods
    agg, diffs = [], []
    for d in ["synthetic_val", "lightbox", "sunlamp"]:
        g = per[per.domain == d]
        for m in allm:
            agg.append({"domain": d, "method": m, "n": len(g), **{k: float(g[f"{m}_{k}"].mean()) for k in keys}})
        B = rng.integers(0, len(g), size=(args.boot, len(g)))
        for a, b in [("spec_fused", "and"), ("spec_fused", "moge2"), ("spec_fused", "dino"), ("spec_dino", "dino"), ("spec_avg", "and")]:
            for k in ["leakage", "recall", "iou", "boundary_f"]:
                dlt = (g[f"{a}_{k}"] - g[f"{b}_{k}"]).to_numpy()
                bs = dlt[B].mean(1)
                diffs.append({"domain": d, "comparison": f"{a} - {b}", "metric": k, "mean_diff": float(dlt.mean()),
                              "ci_low": float(np.percentile(bs, 2.5)), "ci_high": float(np.percentile(bs, 97.5))})
    run.write_metrics(agg, "metrics")
    run.write_metrics(diffs, "paired_differences")
    t = run.sub("tables")
    (t / "metrics.md").write_text(md_table(agg)); (t / "paired_differences.md").write_text(md_table(diffs))
    run.done(parts_cache=str(out.relative_to(REPO)))
    print((t / "metrics.md").read_text()); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc())
    raise
