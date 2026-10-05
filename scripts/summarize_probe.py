"""Aggregate completed probe runs of an experiment (mean ± std over seeds).

Only runs with state=completed and the requested epoch budget are included
(e.g. 5-epoch smoke tests are excluded). Writes tables under the experiment's
directory and prints a Markdown table for the report.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose.runlog import md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--beam", type=int, default=1)
ap.add_argument("--out", default=None)
args = ap.parse_args()

rows, runs = [], []
for d in sorted((REPO / "outputs/experiments" / args.exp).glob("RUN-*")):
    st = json.loads((d / "status.json").read_text())
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    runs.append({"run": d.name, "features": cfg.get("features"), "seed": cfg.get("seed"), "epochs": cfg.get("epochs"),
                 "state": st["state"], "wall_clock_s": st.get("wall_clock_s"), "best_epoch": st.get("best_epoch"),
                 "params": st.get("trainable_params"), "feature_dim": st.get("feature_dim")})
    if st["state"] != "completed" or cfg.get("epochs") != args.epochs:
        continue
    m = pd.read_csv(d / "metrics/metrics.csv")
    m["run"] = d.name
    rows.append(m)
df = pd.concat(rows)
keys = ["mean_deg", "median_deg", "p95_deg", "acc@5", "acc@10", "acc@20", "root_acc", "path_acc_L1", "path_acc_L2",
        "path_acc_L3", "full_path_acc", "top5_leaf_recall", "L3_mean_deg", "nodes_scored_per_image"]
keys = [k for k in keys if k in df.columns]
g = df.groupby(["features", "domain", "beam"])[keys]
agg = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std")).join(g.size().rename("n_seeds")).reset_index()
out = Path(args.out) if args.out else REPO / "outputs/experiments" / args.exp
agg.to_csv(out / "summary_by_features.csv", index=False)
agg.to_csv(out / "summary_by_features.tsv", index=False, sep="\t")
pd.DataFrame(runs).to_csv(out / "run_list.csv", index=False)

b = agg[agg.beam == args.beam]
disp = []
for _, r in b.iterrows():
    row = {"features": r["features"], "domain": r["domain"], "seeds": r["n_seeds"]}
    for k in ["mean_deg", "median_deg", "acc@10", "acc@20", "root_acc", "path_acc_L2", "path_acc_L3"]:
        if f"{k}_mean" in b.columns:
            row[k] = f"{r[k + '_mean']:.3f} ± {r[k + '_std']:.3f}"
    disp.append(row)
print(md_table(disp))
