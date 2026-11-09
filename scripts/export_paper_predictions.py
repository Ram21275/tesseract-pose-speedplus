"""Slim per-image inference results behind the paper's figures and literature tables (git-friendly CSV.gz).

Full prediction npz files (with all 32 orbit candidates and scores) are git-ignored and stay on disk; this keeps one row
per image and system: ground truth, final prediction (1 view and orbit), predicted translation and the errors.
Sources: EXP-130 (T-LESS, YCB-V; 3 seeds) and the FIG-qualitative seed-0 re-runs (SYMSOL-T, SPEED+ GT and detected crops).
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]; E = REPO / "outputs/experiments"; OUT = REPO / "outputs/paper_results"
SRC = {
    "tless": E / "EXP-130_literature_matched_metrics/RUN-20261104-143610-seed0",
    "symsolt": E / "FIG-qualitative_qualitative_figures/RUN-20261103-235636-seed0",
    "speedplus_gt": E / "FIG-qualitative_qualitative_figures/RUN-20261104-000830-seed0",
    "speedplus_det": E / "FIG-qualitative_qualitative_figures/RUN-20261104-001739-seed0",
}
ycb = [d for d in sorted((E / "EXP-130_literature_matched_metrics").glob("RUN-*")) if list((d / "predictions").glob("ycbv_*seed2_DINO.npz"))]
SRC["ycbv"] = ycb[-1]
Q = lambda p, a: {f"{p}_{c}": a[:, i] for i, c in enumerate("wxyz")}
T = lambda p, a: {f"{p}_{c}": a[:, i] for i, c in enumerate("xyz")}


def bop_like(ds, f):
    z = dict(np.load(f)); stem = f.stem.split("_seed"); obj = stem[0][len(ds) + 1:]; seed = int(stem[1].split("_")[0]); var = stem[1].split("_")[1]
    d = dict(dataset=ds, obj=obj, seed=seed, variant=var, image=z["image_relpath"], split=z["split"])
    d |= Q("q_gt", z["q_ego_gt"]) | T("t_gt", z["t_gt"]) | T("t_pred", z["t_used"])
    for m in ("greedy", "orbit_product"):
        d |= Q(f"q_{m}", z[f"q_ego_{m}"]); d[f"err_rot_deg_{m}"] = z[f"err_{m}"]
    d["err_t"] = np.linalg.norm(z["t_used"] - z["t_gt"], axis=1)
    return pd.DataFrame(d)


def speedplus(ds, f):
    z = dict(np.load(f)); d = dict(dataset=ds, obj="tango", seed=0, variant="S1_distilledPoE+DINO_t", image=z["image_relpath"], split=z["domain"])
    d |= Q("q_gt", z["q_gt"]) | T("t_gt", z["t_gt"]) | T("t_pred", z["t_orbit_mean"])
    for m in ("greedy", "orbit_product", "orbit_vote32"):
        d |= Q(f"q_{m}", z[f"q_{m}"]); d[f"err_rot_deg_{m}"] = z[f"err_{m}"]
    d["err_t"] = np.linalg.norm(z["t_orbit_mean"] - z["t_gt"], axis=1)
    return pd.DataFrame(d)


OUT.mkdir(exist_ok=True); rows = []
for ds, run in SRC.items():
    fs = sorted((run / "predictions").glob("*.npz"))
    df = pd.concat([speedplus(ds, f) if ds.startswith("speedplus") else bop_like(ds, f) for f in fs])
    p = OUT / f"{ds}_per_image.csv.gz"; df.to_csv(p, index=False, float_format="%.6g", compression="gzip")
    rows.append(dict(file=p.name, source_run=str(run.relative_to(REPO)), rows=len(df), seeds=sorted(df.seed.unique().tolist()), variants=sorted(df.variant.unique().tolist())))
    print(p.name, len(df), round(p.stat().st_size / 1e6, 2), "MB")
pd.DataFrame(rows).to_csv(OUT / "index.csv", index=False)
