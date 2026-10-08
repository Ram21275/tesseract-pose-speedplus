"""EXP-101 part A: Tesseract vs SPACE-HOP Hopf grid discretization (no learning)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import rotations as rot, tesseract as T  # noqa: E402
from tfpose.grids import error_summary, nearest, spacehop_hopf_grid  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n-haar", type=int, default=1_000_000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
run = Run("EXP-101", "tesseract_vs_spacehop_hopf", dict(vars(args), part="A_discretization"), seed=args.seed)
try:
    rng = np.random.default_rng(args.seed)
    qh = rot.random_quats(args.n_haar, rng)
    man = pd.read_csv(REPO / "outputs/data_manifests/speedplus_tesseract_L5.csv") if (REPO / "outputs/data_manifests/speedplus_tesseract_L5.csv").exists() else None
    if man is None:
        raise FileNotFoundError("run scripts/exp001_gt_tesseract_paths.py first (regenerates the L5 manifest)")
    qgt = man[[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    rows = []

    def spacing_cv(G):
        _, nn = nearest(G, G, exclude_self=True)
        return float(nn.std() / nn.mean()), float(np.degrees(nn.min()))

    for L in [3, 4, 5]:
        cb = T.codebook(L)
        c, p = T.encode(qh, L); cg, pg = T.encode(qgt, L)
        _, dn = nearest(qh, cb)
        cv, nnmin = spacing_cv(cb)
        rows.append({"grid": "tesseract", "config": f"L{L}", "n": len(cb),
                     **error_summary(rot.geodesic_distance(qh, T.decode(c, p)), "haar_decoded_"),
                     **error_summary(dn, "haar_nearest_"),
                     **error_summary(rot.geodesic_distance(qgt, T.decode(cg, pg)), "gt_decoded_"),
                     "spacing_cv": cv, "nn_min_deg": nnmin, "decode": f"tree, {4 + 8 * L} nodes (greedy)"})
        print(rows[-1]["config"], rows[-1]["n"], round(rows[-1]["haar_decoded_mean_deg"], 3), flush=True)
    for npts, nroll, tag in [(256, 12, "paper"), (171, 12, "~L3"), (1365, 12, "~L4"), (10923, 12, "~L5")]:
        G = spacehop_hopf_grid(npts, nroll)
        _, dn = nearest(qh, G); _, dg = nearest(qgt, G)
        cv, nnmin = spacing_cv(G)
        rows.append({"grid": "spacehop_hopf", "config": f"{npts}x{nroll} ({tag})", "n": len(G),
                     **error_summary(dn, "haar_decoded_"), **error_summary(dn, "haar_nearest_"),
                     **error_summary(dg, "gt_decoded_"), "spacing_cv": cv, "nn_min_deg": nnmin,
                     "decode": f"flat argmax over {len(G)}"})
        print(rows[-1]["config"], rows[-1]["n"], round(rows[-1]["haar_decoded_mean_deg"], 3), flush=True)
    run.write_metrics(rows, "discretization")
    keys = ["grid", "config", "n", "haar_decoded_mean_deg", "haar_decoded_p95_deg", "haar_decoded_max_deg",
            "haar_nearest_mean_deg", "gt_decoded_mean_deg", "spacing_cv", "nn_min_deg", "decode"]
    (run.sub("tables") / "discretization.md").write_text(md_table(rows, keys))
    run.done()
    print(md_table(rows, keys)); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc())
    raise
