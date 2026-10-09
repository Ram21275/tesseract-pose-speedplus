"""EXP-104: audit of the ~180 deg failures of the frozen Phase-3 predictor (no training).

Tests, per candidate cause (see report):
 1 encoder/decoder bug   : teacher-forced GT path -> production decoders (tesseract.decode, predictor.decode_torch)
 2 metric / convention    : quaternion vs matrix geodesic, sign invariance, transpose test on failures
 3 chart seam            : chart margin |q|_(1) - |q|_(2) of GT vs error
 4 hierarchical routing   : first divergent level (0 = root chart) of failures
 5 visual symmetry       : body-frame relative rotation dR = R_g^T R_p (angle, axis), and error after
                            allowing 180 deg about the body x / y / z axis
Inputs: greedy predictions csv(s) of EXP-036 (single branches; seed 0).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import tesseract as T  # noqa: E402
from tfpose.predictor import decode_torch  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--preds", nargs="+", default=[
    "outputs/experiments/EXP-036_transformer_soft_targets/RUN-20261009-111708-seed0/predictions/predictions_greedy.csv",
    "outputs/experiments/EXP-036_transformer_soft_targets/RUN-20261009-112706-seed0/predictions/predictions_greedy.csv"])
ap.add_argument("--names", nargs="+", default=["DINOv3", "MoGe-2"])
args = ap.parse_args()
run = Run("EXP-104", "flip_audit", vars(args), seed=0)
try:
    man = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv")
    qall = man[[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    out = {}

    # ---- 1. teacher-forced decoding with the production decoders, all 69,491 GT rotations ----
    chart, path = T.encode(qall, 5)
    qd_np = T.decode(chart, path)
    qd_t = decode_torch(torch.as_tensor(chart), torch.as_tensor(path)).double().numpy()

    def geo(a, b):
        return np.degrees(2 * np.arccos(np.clip(np.abs((a * b).sum(-1)), 0, 1)))
    e_np, e_t = geo(qd_np, qall), geo(qd_t, qall)
    stored = np.stack([man.root_chart.to_numpy(), man.child_path.astype(str).str.zfill(5).to_numpy()], 1)
    out["tf_decode_max_deg_numpy"] = float(e_np.max()); out["tf_decode_max_deg_torch"] = float(e_t.max())
    out["tf_decode_mean_deg"] = float(e_np.mean())
    out["manifest_path_matches_encoder"] = float(np.mean([(int(c) == ch) and (s == "".join(map(str, p)))
                                                         for (c, s), ch, p in zip(stored, chart, path)]))
    out["encode_pm_q_identical"] = bool(np.array_equal(T.encode(-qall, 5)[1], path))

    # ---- per-prediction audit ----
    man_i = man.set_index("image_relpath")
    rows, allf = [], []
    for name, f in zip(args.names, args.preds):
        p = pd.read_csv(REPO / f)
        m = man_i.loc[p.image_relpath]
        qg = m[[f"q_can_{k}" for k in "wxyz"]].to_numpy(); qp = p[[f"q_pred_{k}" for k in "wxyz"]].to_numpy()
        qp = qp / np.linalg.norm(qp, axis=1, keepdims=True)
        Rg, Rp = Rot.from_quat(qg[:, [1, 2, 3, 0]]), Rot.from_quat(qp[:, [1, 2, 3, 0]])
        dq = geo(qp, qg)
        Mg, Mp = Rg.as_matrix(), Rp.as_matrix()
        dR = np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", Mp, Mg) - 1) / 2, -1, 1)))     # tr(Rp Rg^T)
        dT = np.degrees(np.arccos(np.clip((np.einsum("nji,nij->n", Mp, Mg) - 1) / 2, -1, 1)))     # Rp^T vs Rg
        dneg = geo(-qp, qg)
        rel = (Rg.inv() * Rp)                                    # body frame: R_p = R_g dR
        rv = rel.as_rotvec(); ang = np.degrees(np.linalg.norm(rv, axis=1)); ax = rv / np.maximum(np.linalg.norm(rv, axis=1, keepdims=True), 1e-9)
        sym = {"x": Rot.from_rotvec([np.pi, 0, 0]), "y": Rot.from_rotvec([0, np.pi, 0]), "z": Rot.from_rotvec([0, 0, np.pi])}
        dsym = {k: np.degrees((Rg * S).inv().__mul__(Rp).magnitude()) for k, S in sym.items()}
        dsym_all = np.minimum.reduce([dq] + list(dsym.values()))
        srt = np.sort(np.abs(qg), 1); margin = srt[:, -1] - srt[:, -2]
        gt_c, gt_p = m.root_chart.to_numpy(), m.child_path.astype(str).str.zfill(5).to_numpy()
        pr_c, pr_p = p.chart_pred.to_numpy(), p.path_pred.astype(str).str.zfill(5).to_numpy()
        first = np.array([0 if a != b else next((l + 1 for l in range(5) if s[l] != t[l]), 6) for a, b, s, t in zip(gt_c, pr_c, gt_p, pr_p)])
        allf.append(pd.DataFrame({"branch": name, "domain": p.domain.values, "image_relpath": p.image_relpath.values, "err_quat": dq,
                                  "err_mat": dR, "err_transposed": dT, "err_negq": dneg, "rel_angle": ang, "ax_x": ax[:, 0], "ax_y": ax[:, 1],
                                  "ax_z": ax[:, 2], "err_sym_x": dsym["x"], "err_sym_y": dsym["y"], "err_sym_z": dsym["z"],
                                  "err_sym_best": dsym_all, "chart_margin": margin, "first_div_level": first}))
        for dom in ["synthetic_val", "lightbox", "sunlamp"]:
            k = p.domain.values == dom; f150 = k & (dq > 150)
            r = {"branch": name, "domain": dom, "n": int(k.sum()), "frac_gt150": float(f150.sum() / k.sum()),
                 "max_abs_quat_minus_mat_deg": float(np.abs(dq - dR)[k].max()), "max_abs_negq_diff_deg": float(np.abs(dq - dneg)[k].max()),
                 "fails_fixed_by_transpose(<20)": float(np.mean(dT[f150] < 20)) if f150.any() else np.nan,
                 "mean_err": float(dq[k].mean()), "mean_err_sym_best": float(dsym_all[k].mean()),
                 **{f"fails_within20_after_180{a}": float(np.mean(dsym[a][f150] < 20)) if f150.any() else np.nan for a in "xyz"},
                 "fails_within20_after_any180": float(np.mean(dsym_all[f150] < 20)) if f150.any() else np.nan,
                 "fails_first_div_root": float(np.mean(first[f150] == 0)) if f150.any() else np.nan,
                 "fails_first_div_L1": float(np.mean(first[f150] == 1)) if f150.any() else np.nan,
                 "fails_median_chart_margin": float(np.median(margin[f150])) if f150.any() else np.nan,
                 "all_median_chart_margin": float(np.median(margin[k])),
                 "fails_frac_margin_lt_0.05": float(np.mean(margin[f150] < 0.05)) if f150.any() else np.nan,
                 "all_frac_margin_lt_0.05": float(np.mean(margin[k] < 0.05))}
            rows.append(r)
    A = pd.concat(allf); A.to_csv(run.sub("tables") / "per_image_audit.csv", index=False, float_format="%.5g")
    run.write_metrics(rows, "metrics"); run.write_metrics(out, "decoder_checks")
    (run.sub("tables") / "metrics.md").write_text(md_table(rows))

    # ---- figures ----
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for j, dom in enumerate(["synthetic_val", "lightbox", "sunlamp"]):
        a = A[(A.domain == dom) & (A.branch == args.names[0])]
        axs[0, j].hist(a.err_quat, bins=90, range=(0, 180), alpha=0.6, label="standard"); axs[0, j].hist(a.err_sym_best, bins=90, range=(0, 180), alpha=0.6, label="min over 180° about body x/y/z")
        axs[0, j].set_yscale("log"); axs[0, j].set_title(f"{args.names[0]} {dom}: error"); axs[0, j].legend(fontsize=7)
        f = A[(A.domain == dom) & (A.err_quat > 150)]
        for b, mk in zip(args.names, "o^"):
            ff = f[f.branch == b]; s = np.sign(ff.ax_z + 1e-9)          # fold axis sign (axis and -axis at 180 deg are equivalent)
            axs[1, j].scatter(ff.ax_x * s, ff.ax_y * s, s=6, marker=mk, alpha=0.5, label=f"{b} (n={len(ff)})")
        axs[1, j].set_xlim(-1, 1); axs[1, j].set_ylim(-1, 1); axs[1, j].set_aspect("equal")
        axs[1, j].set_title(f"{dom}: body-frame axis of dR for errors >150° (x,y; z folded ≥0)", fontsize=8); axs[1, j].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(run.sub("figures") / "error_hist_and_flip_axes.png", dpi=110); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    a = A[A.branch == args.names[0]]
    axs[0].scatter(a.chart_margin, a.err_quat, s=2, alpha=0.3); axs[0].set_xlabel("GT chart margin |q|(1)-|q|(2)"); axs[0].set_ylabel("error (deg)")
    for dom in ["synthetic_val", "lightbox", "sunlamp"]:
        d = a[(a.domain == dom) & (a.err_quat > 150)].first_div_level
        axs[1].hist(d, bins=np.arange(-0.5, 7), alpha=0.5, label=dom, density=True)
    axs[1].set_xlabel("first divergent level of >150° failures (0 = root chart)"); axs[1].legend()
    fig.tight_layout(); fig.savefig(run.sub("figures") / "chart_margin_and_divergence.png", dpi=110); plt.close(fig)
    run.done()
    print(out); print(md_table(rows)); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
