"""EXP-012: DINOv3 vs VGGT complementarity and frequency audit.

Uses the saved greedy predictions of the selected single-branch probes (same
subset, same predictor, same seeds) -- no new training.
1. error correlation, solved-by-one-only, oracle-of-two, posterior agreement;
2. error vs object scale (GT crop size tertiles) per branch;
3. feature drift synthetic -> lightbox/sunlamp (label-free statistics);
4. spatial band energy of patch-token maps per domain (low/mid/high).
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import featsets  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dino", required=True, help="feature spec of selected DINO probe (EXP-010)")
ap.add_argument("--vggt", required=True, help="feature spec of the second branch probe (VGGT in EXP-011, MoGe-2 in EXP-013)")
ap.add_argument("--second-exp", default="EXP-011_vggt_controls", help="experiment holding the second-branch probe runs")
ap.add_argument("--second-map", default="depth", help="dense map of the second branch used for the band-energy audit")
ap.add_argument("--run-exp", default="EXP-012_complementarity", help="experiment id_name for this audit run")
ap.add_argument("--subset", default="v1")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--thresh", type=float, default=10.0)
args = ap.parse_args()
run = Run(args.run_exp.split("_", 1)[0], args.run_exp.split("_", 1)[1], vars(args), seed=0)
try:
    seeds = [int(s) for s in args.seeds.split(",")]

    def preds(exp, spec, seed):
        for d in sorted((REPO / "outputs/experiments" / exp).glob(f"RUN-*-seed{seed}")):
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            if cfg["features"] == spec and cfg["epochs"] == 100 and (d / "predictions/predictions_greedy.csv").exists():
                return pd.read_csv(d / "predictions/predictions_greedy.csv", dtype={"path_gt": str, "path_pred": str}), d.name
        raise FileNotFoundError(f"{exp} {spec} seed {seed}")

    df = pd.read_csv(REPO / f"outputs/data_manifests/subset_{args.subset}.csv")
    side = dict(zip(df.image_relpath, df.crop_side))
    rows, scale_rows, used = [], [], []
    for s in seeds:
        a, ra = preds("EXP-010_dinov3_baseline", args.dino, s)
        b, rb = preds(args.second_exp, args.vggt, s)
        used += [ra, rb]
        m = a.merge(b, on=["image_relpath", "domain"], suffixes=("_d", "_v"))
        for dom, g in m.groupby("domain"):
            ed, ev = g.err_deg_d.to_numpy(), g.err_deg_v.to_numpy()
            sd, sv = ed < args.thresh, ev < args.thresh
            same_root = g.chart_pred_d.to_numpy() == g.chart_pred_v.to_numpy()
            same_l2 = same_root & (g.path_pred_d.str[:2] == g.path_pred_v.str[:2]).to_numpy()
            qd = g[[f"q_pred_{k}_d" for k in "wxyz"]].to_numpy()
            qv = g[[f"q_pred_{k}_v" for k in "wxyz"]].to_numpy()
            dist = np.degrees(2 * np.arccos(np.clip(np.abs((qd * qv).sum(1)), 0, 1)))
            agree = dist < 2 * args.thresh
            rows.append({"seed": s, "domain": dom, "n": len(g),
                         "dino_mean_deg": ed.mean(), "vggt_mean_deg": ev.mean(),
                         "spearman_err": spearmanr(ed, ev).statistic,
                         f"both_lt{args.thresh:g}": np.mean(sd & sv), f"dino_only_lt{args.thresh:g}": np.mean(sd & ~sv),
                         f"vggt_only_lt{args.thresh:g}": np.mean(~sd & sv), f"neither_lt{args.thresh:g}": np.mean(~sd & ~sv),
                         f"oracle_of_two_acc@{args.thresh:g}": np.mean(sd | sv),
                         "root_agree": same_root.mean(), "L2_prefix_agree": same_l2.mean(),
                         "pred_pred_dist_median_deg": np.median(dist),
                         f"agree_frac(dist<{2 * args.thresh:g})": agree.mean(),
                         f"dino_acc_when_agree": np.mean(sd[agree]) if agree.any() else np.nan,
                         f"dino_acc_when_disagree": np.mean(sd[~agree]) if (~agree).any() else np.nan,
                         f"vggt_acc_when_disagree": np.mean(sv[~agree]) if (~agree).any() else np.nan})
            sz = np.array([side[p] for p in g.image_relpath])
            t = np.quantile(sz, [1 / 3, 2 / 3])
            for name, mk in [("small", sz <= t[0]), ("medium", (sz > t[0]) & (sz <= t[1])), ("large", sz > t[1])]:
                scale_rows.append({"seed": s, "domain": dom, "object_size": name, "crop_side_px_median": float(np.median(sz[mk])),
                                   "dino_mean_deg": ed[mk].mean(), "vggt_mean_deg": ev[mk].mean()})
    comp = pd.DataFrame(rows)
    comp_mean = comp.drop(columns="seed").groupby("domain").mean().reset_index()
    scale = pd.DataFrame(scale_rows).drop(columns="seed").groupby(["domain", "object_size"]).mean().reset_index()

    # feature drift and band energy (label-free)
    drift_rows, band_rows = [], []
    dom = np.where(df.domain == "synthetic", np.where(df.split == "train", "synthetic_train", "synthetic_val"), df.domain)
    for spec in [args.dino, args.vggt]:
        _, X, _ = featsets.load(spec, args.subset)
        ref = X[dom == "synthetic_train"]
        mu, sd = ref.mean(0), ref.std(0) + 1e-6
        Z = (X - mu) / sd
        spread = np.sqrt((Z[dom == "synthetic_train"] ** 2).sum(1).mean())
        for d in ["synthetic_val", "lightbox", "sunlamp"]:
            drift_rows.append({"features": spec, "domain": d,
                               "mean_shift_over_spread": float(np.linalg.norm(Z[dom == d].mean(0)) / spread)})
    for backbone, key in [(args.dino.split(":")[0], "last_grid"), (args.vggt.split(":")[0], args.second_map)]:
        path = featsets.find_cache(backbone, args.subset)
        # Streamed in chunks (float32 / complex64): the earlier whole-array
        # version peaked at ~100 GB and most likely took the machine down.
        with h5py.File(path / "features.h5") as f:
            ds = f[key]
            N, H, W, C = ds.shape
            fy, fx = np.meshgrid(np.fft.fftfreq(H), np.fft.fftfreq(W), indexing="ij")
            r = np.sqrt(fx ** 2 + fy ** 2) / 0.5  # 1 = Nyquist along an axis
            bands = {"low(<0.25)": r < 0.25, "mid(0.25-0.5)": (r >= 0.25) & (r < 0.5), "high(>=0.5)": r >= 0.5}
            frac = {bn: np.empty(N, dtype=np.float64) for bn in bands}
            for s in range(0, N, 128):
                g = np.asarray(ds[s:s + 128], dtype=np.float32)
                g -= g.mean((1, 2), keepdims=True)
                P = np.abs(np.fft.fft2(g, axes=(1, 2)).astype(np.complex64)) ** 2  # (n,H,W,C)
                Pc = P.sum(-1)  # energy summed over channels, (n,H,W)
                tot = Pc.sum((1, 2))
                for bn, bm in bands.items():
                    frac[bn][s:s + len(g)] = Pc[:, bm].sum(1) / tot
                del g, P, Pc
        for d in ["synthetic_train", "synthetic_val", "lightbox", "sunlamp"]:
            mk = dom == d
            row = {"map": f"{backbone}:{key}", "grid": f"{H}x{W}", "domain": d}
            for bn in bands:
                row[bn] = float(frac[bn][mk].mean())
            band_rows.append(row)
    run.write_metrics(comp.to_dict("records"), "complementarity_per_seed")
    run.write_metrics(comp_mean.to_dict("records"), "metrics")
    run.write_metrics(scale.to_dict("records"), "scale")
    run.write_metrics(drift_rows, "drift")
    run.write_metrics(band_rows, "band_energy")
    (run.dir / "source_runs.txt").write_text("\n".join(used) + "\n")
    t = run.sub("tables")
    (t / "complementarity.md").write_text(md_table(comp_mean.to_dict("records")))
    (t / "scale.md").write_text(md_table(scale.to_dict("records")))
    (t / "drift.md").write_text(md_table(drift_rows))
    (t / "band_energy.md").write_text(md_table(band_rows))
    run.done()
    for p in ["complementarity", "scale", "drift", "band_energy"]:
        print((t / f"{p}.md").read_text(), "\n")
    print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc())
    raise
