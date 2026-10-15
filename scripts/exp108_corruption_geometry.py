"""EXP-108 review point 2 (CPU, no model): geometry of the corruption process.

For synthetic-train rotations and t in 11 bins: decoded angular error of the corrupted path vs GT and vs its source path,
min over the two endpoints (is the mixed path near either?), split by source type (flip / near / uniform), by whether
the root changed, and by seam cases (GT within a few degrees of a chart boundary: second-largest |q_i| / largest > 0.95).
Contrast: a geometry-preserving alternative, prefix mixing (GT decisions above depth j, source below, j = round(t*(L+1)))
-- reported only, not used for training (the pre-registered token-wise kernel is kept).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import pathflow as PF, tesseract as T  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

run = Run("EXP-108", "path_flow_revision", {"stage": "corruption_geometry"}, seed=0)
try:
    L = 5
    df = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv")
    q = df[(df.domain == "synthetic") & (df.split == "train")][[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    rng = np.random.default_rng(0); q = q[rng.choice(len(q), 20000, replace=False)]
    c, p = T.encode(q, L)
    a = np.sort(np.abs(q), 1); seam = a[:, -2] / a[:, -1] > 0.95
    geo = lambda x, y: np.degrees(2 * np.arccos(np.clip(np.abs((x * y).sum(1)), 0, 1)))
    rows = []
    for t in np.linspace(0, 1, 11):
        tt = np.full(len(q), t)
        cc, pp, info = PF.corrupt(q, c, p, tt, np.random.default_rng(int(t * 1000) + 7), return_info=True)
        qc, qs = T.decode(cc, pp), T.decode(info["src_chart"], info["src_path"])
        e_gt, e_src = geo(qc, q), geo(qc, qs)
        j = int(round(t * (L + 1)))                                         # prefix mixing: first j decisions from GT
        dec = np.concatenate([c[:, None], p], 1); sdec = np.concatenate([info["src_chart"][:, None], info["src_path"]], 1)
        mix = np.where(np.arange(L + 1)[None] < j, dec, sdec)
        e_pref = geo(T.decode(mix[:, 0], mix[:, 1:]), q)
        rootchg = cc != c
        for src in ("flip", "near", "uniform", "all"):
            m = (info["source"] == src) if src != "all" else np.ones(len(q), bool)
            for sub, ms in (("all", m), ("root_changed", m & rootchg), ("root_kept", m & ~rootchg), ("seam", m & seam)):
                if ms.sum() == 0:
                    continue
                rows.append({"t": round(t, 1), "source": src, "subset": sub, "n": int(ms.sum()),
                             "err_vs_gt_mean": e_gt[ms].mean(), "err_vs_gt_median": np.median(e_gt[ms]),
                             "err_vs_source_mean": e_src[ms].mean(), "min_endpoint_err_mean": np.minimum(e_gt, e_src)[ms].mean(),
                             "frac_far_from_both_gt30": float((np.minimum(e_gt, e_src)[ms] > 30).mean()),
                             "prefix_mix_err_vs_gt_mean": e_pref[ms].mean()})
    out = pd.DataFrame(rows); run.write_metrics(rows, "corruption_geometry")
    main = out[(out.subset == "all")]
    (run.sub("tables") / "corruption_geometry.md").write_text(md_table(out.to_dict("records")))
    print(md_table(main[main.source.isin(["all", "flip", "near"])].to_dict("records"),
                   ["t", "source", "n", "err_vs_gt_mean", "err_vs_source_mean", "min_endpoint_err_mean", "frac_far_from_both_gt30", "prefix_mix_err_vs_gt_mean"]))
    rc = out[(out.source == "all") & (out.subset.isin(["root_changed", "root_kept", "seam"]))]
    print(md_table(rc.to_dict("records"), ["t", "subset", "n", "err_vs_gt_mean", "min_endpoint_err_mean", "frac_far_from_both_gt30"]))
    run.done(seam_fraction=float(seam.mean())); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
