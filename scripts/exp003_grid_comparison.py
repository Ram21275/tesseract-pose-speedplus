"""EXP-003: Tesseract vs Hopf (Yershova) vs Super-Fibonacci grids.

Cubochoric is replaced by Super-Fibonacci (ground rules sec. 7.10 permit
"another near-uniform SO(3) grid"); the substitution is recorded in the report.

Matching (ground rules sec. 7): hypothesis counts are matched exactly for
Super-Fibonacci; Hopf counts (72*8^r) cannot equal Tesseract counts (4*8^L),
so the nearest Hopf levels are reported together with their counts. Matched
covering radius is reported as the Super-Fibonacci size needed to reach the
Tesseract p95 error (estimated from the N^-1/3 law, then measured).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import rotations as rot  # noqa: E402
from tfpose import tesseract as T  # noqa: E402
from tfpose.grids import error_summary, hopf_grid, nearest, super_fibonacci  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n-haar", type=int, default=1_000_000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
run = Run("EXP-003", "grid_comparison", vars(args), seed=args.seed)
try:
    rng = np.random.default_rng(args.seed)
    qh = rot.random_quats(args.n_haar, rng)

    def measure(name, level, G, own_err=None, greedy=None):
        idx, d = nearest(qh, G)
        occ = np.bincount(idx, minlength=len(G))
        _, nn = nearest(G, G, exclude_self=True)
        row = {"grid": name, "level": level, "n": len(G)}
        row.update(error_summary(d, "nearest_"))
        if own_err is not None:
            row.update(error_summary(own_err, "decoded_"))
        row.update({"voronoi_occupancy_cv": float(occ.std() / occ.mean()),
                    "nn_spacing_cv": float(nn.std() / nn.mean()),
                    "nn_min_deg": float(np.degrees(nn.min())),
                    "search_nodes_greedy_tree": greedy if greedy is not None else "n/a (flat)",
                    "search_nodes_flat": len(G)})
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        return row

    rows = []
    for L in range(1, 6):
        cb = T.codebook(L)
        c, p = T.encode(qh, L)
        rows.append(measure("tesseract", f"L{L}", cb, rot.geodesic_distance(qh, T.decode(c, p)), 4 + 8 * L))
        rows.append(measure("super_fibonacci@tess_count", f"L{L}", super_fibonacci(len(cb))))
    for r in range(0, 5):
        G = hopf_grid(r)
        # Yershova's hierarchy: 72 roots then 8 children per level
        rows.append(measure("hopf", f"r{r}", G, greedy=72 + 8 * r))

    # matched covering: SF size needed to reach the Tesseract decoded-p95 error
    match = []
    for L in range(1, 6):
        t = next(x for x in rows if x["grid"] == "tesseract" and x["level"] == f"L{L}")
        s = next(x for x in rows if x["grid"] == "super_fibonacci@tess_count" and x["level"] == f"L{L}")
        n_est = int(round(s["n"] * (s["nearest_p95_deg"] / t["decoded_p95_deg"]) ** 3))
        _, d = nearest(qh, super_fibonacci(n_est))
        match.append({"tesseract_level": f"L{L}", "tesseract_n": t["n"], "tesseract_decoded_p95_deg": t["decoded_p95_deg"],
                      "sf_n_for_equal_p95": n_est, "sf_measured_p95_deg": float(np.degrees(np.percentile(d, 95))),
                      "sf_n_over_tess_n": n_est / t["n"]})
        print(match[-1], flush=True)
    run.write_metrics(rows, "metrics")
    run.write_metrics(match, "matched_covering")
    keys = ["grid", "level", "n", "nearest_mean_deg", "nearest_p95_deg", "nearest_max_deg", "decoded_mean_deg",
            "decoded_p95_deg", "decoded_max_deg", "voronoi_occupancy_cv", "nn_spacing_cv", "search_nodes_greedy_tree"]
    (run.sub("tables") / "grid_table.md").write_text(md_table(rows, keys) + "\n\n" + md_table(match))
    comp = REPO / "outputs/comparisons"
    comp.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(comp / "rotation_grid_summary.csv", index=False)
    pd.DataFrame(rows).to_csv(comp / "rotation_grid_summary.tsv", index=False, sep="\t")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    for g, mk in [("tesseract", "o-"), ("super_fibonacci@tess_count", "s--"), ("hopf", "^-")]:
        d = df[df.grid == g]
        ax.loglog(d["n"], d["nearest_p95_deg"], mk, label=f"{g} nearest p95")
    d = df[df.grid == "tesseract"]
    ax.loglog(d["n"], d["decoded_p95_deg"], "o:", label="tesseract own-cell (decoded) p95")
    ax.set_xlabel("number of rotation hypotheses"); ax.set_ylabel("p95 error (deg)")
    ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(run.sub("figures") / "p95_vs_n.png", dpi=130)
    run.done()
    print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc())
    raise
