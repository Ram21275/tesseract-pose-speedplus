"""EXP-119 base-arm rule (pre-registered 2026-10-27): per dataset, the EXP-106 arm (A0 / A1 / A2) with the lowest validation
symmetry-aware PoE greedy mean (object-balanced, 3 seeds), read from the newest finished EXP-106 evaluation of each arm.
Prints shell assignments: BASE_ARM, BASE_DIR (train-run folder), BASE_FILTER, BASE_LABEL."""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
ds = sys.argv[1]; best = {}
for d in sorted((REPO / "outputs/experiments/EXP-106_tless_multipath_tree").glob("RUN-*")):
    cfg = yaml.safe_load((d / "config.yaml").read_text()); st = json.loads((d / "status.json").read_text())
    if cfg.get("stage") == "eval" and cfg.get("dataset") == ds and st.get("state") == "completed" and cfg.get("tag"):
        best[cfg["tag"]] = (d, cfg)                                                   # newest wins (sorted by time)
assert {"A0", "A1", "A2"} <= set(best), best.keys()
score = {}
for tag, (d, cfg) in best.items():
    ob = pd.read_csv(d / "metrics/object_balanced.csv")
    score[tag] = float(ob.query("split == 'validation' and variant == 'PoE' and mode == 'greedy'").sym_mean.iloc[0])
arm = min(score, key=score.get); d, cfg = best[arm]
lab = dict(kv.split("=", 1) for kv in cfg.get("filter", "").split(",") if kv).get("label", "raw")
print(f"# EXP-119 base arm for {ds}: " + ", ".join(f"{k} {v:.3f}" for k, v in sorted(score.items())), file=sys.stderr)
print(f"BASE_ARM={arm}; BASE_DIR={cfg['runs_dir']}; BASE_FILTER={cfg.get('filter', '')}; BASE_LABEL={lab}")
