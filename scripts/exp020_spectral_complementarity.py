"""EXP-020: CASS spectral head-matching distance as a label-free complementarity measure (descriptive).

Uses the per-image spectra saved by scripts/exp018_cass_extract.py and the seed-0 greedy
predictions of the EXP-010 (DINOv3-L grid4) and EXP-013 (MoGe-2 grid4) probes.
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose.runlog import Run, md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--subset", default="v1")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
run = Run("EXP-020", "spectral_head_complementarity", vars(args), seed=args.seed)
try:
    df = pd.read_csv(REPO / f"outputs/data_manifests/subset_{args.subset}.csv")
    with h5py.File(REPO / f"outputs/shared_cache/cass_spectra__subset_{args.subset}/spectra.h5") as f:
        assert [s.decode() for s in f["image_relpath"][:]] == df.image_relpath.tolist()
        cost = f["cost"][:]; pairs = f["pairs"][:]
    W = 1.0 - cost                                                  # Wasserstein distance per (dino head, moge head)
    Dm = np.array([W[n, pairs[n, :, 0], pairs[n, :, 1]].mean() for n in range(len(df))])   # mean matched distance
    Dall = W.reshape(len(df), -1).mean(1)                           # mean over all head pairs

    def preds(exp, spec):
        for d in sorted((REPO / "outputs/experiments" / exp).glob("RUN-*-seed0")):
            if yaml.safe_load((d / "config.yaml").read_text()).get("features") == spec and (d / "predictions/predictions_greedy.csv").exists():
                return pd.read_csv(d / "predictions/predictions_greedy.csv", dtype={"path_gt": str, "path_pred": str})
        raise FileNotFoundError(exp)
    a = preds("EXP-010_dinov3_baseline", "dinov3_vitl16:grid4"); b = preds("EXP-013_moge2_controls", "moge2_vitl:grid4")
    m = a.merge(b, on=["image_relpath", "domain"], suffixes=("_d", "_m"))
    idx = pd.Series(np.arange(len(df)), index=df.image_relpath)
    m["D_matched"] = Dm[idx[m.image_relpath].values]; m["D_all"] = Dall[idx[m.image_relpath].values]
    qd = m[[f"q_pred_{k}_d" for k in "wxyz"]].to_numpy(); qm = m[[f"q_pred_{k}_m" for k in "wxyz"]].to_numpy()
    m["disagree_deg"] = np.degrees(2 * np.arccos(np.clip(np.abs((qd * qm).sum(1)), 0, 1)))
    rows = []
    for d, g in m.groupby("domain"):
        row = {"domain": d, "n": len(g), "D_matched_mean": float(g.D_matched.mean()), "D_matched_std": float(g.D_matched.std())}
        for col in ["D_matched", "D_all"]:
            row[f"spearman_{col}_vs_disagree"] = float(spearmanr(g[col], g.disagree_deg).statistic)
            row[f"spearman_{col}_vs_dino_err"] = float(spearmanr(g[col], g.err_deg_d).statistic)
            row[f"spearman_{col}_vs_moge_err"] = float(spearmanr(g[col], g.err_deg_m).statistic)
            y = (g.err_deg_d > 20).astype(int)
            row[f"auroc_{col}_dino_err_gt20"] = float(roc_auc_score(y, g[col])) if 0 < y.sum() < len(y) else float("nan")
        row["auroc_disagree_dino_err_gt20"] = float(roc_auc_score((g.err_deg_d > 20).astype(int), g.disagree_deg)) if 0 < (g.err_deg_d > 20).sum() < len(g) else float("nan")
        rows.append(row)
    # head-level complementarity map: how often each (dino head, moge head) pair is matched
    pm = np.zeros((16, 16))
    for n in range(len(df)):
        pm[pairs[n, :, 0], pairs[n, :, 1]] += 1
    np.savetxt(run.sub("tables") / "pair_frequency_dino_x_moge.csv", pm / len(df), delimiter=",", fmt="%.4f")
    m[["image_relpath", "domain", "D_matched", "D_all", "disagree_deg", "err_deg_d", "err_deg_m"]].to_csv(
        run.sub("predictions") / "per_image.csv", index=False, float_format="%.5g")
    run.write_metrics(rows, "metrics")
    (run.sub("tables") / "metrics.md").write_text(md_table(rows))
    run.done()
    print(md_table(rows)); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc())
    raise
