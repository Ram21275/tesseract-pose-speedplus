"""EXP-127 smoke gate: the tree rows of an EXP-127 orbit_eval run (seed 0) must reproduce EXP-113's seed-0 tree rows
(RUN-20261017-042606) for PoE / DINO x greedy / orbit_product on every domain (|diff| <= 0.02 deg). Exit 1 otherwise."""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ref = pd.read_csv(REPO / "outputs/experiments/EXP-113_tesseract_vs_flat_hopf/RUN-20261017-042606-seed0/metrics/per_seed.csv")
new = pd.read_csv(REPO / sys.argv[1] / "metrics/per_seed.csv")
k = ["variant", "mode", "seed", "domain"]
m = ref[(ref.family == "tree") & (ref.seed == 0)].merge(new[new.family == "tree"], on=k, suffixes=("_ref", "_new"))
m["diff"] = (m.mean_deg_ref - m.mean_deg_new).abs()
print(m[k + ["mean_deg_ref", "mean_deg_new", "diff"]].to_string(index=False))
ok = len(m) >= 16 and (m["diff"] <= 0.02).all()
print("GATE", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
