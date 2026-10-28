"""Exit 0 if outputs/experiments/<dir> holds a finished train run (checkpoints/best.pt) whose config matches every k=v
(missing keys take the trainer defaults); stage=eval matches a completed evaluation run. Used by resume queues to skip completed runs."""
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
D = {"label": "raw", "extra_train": "None", "teacher_variant": "None", "stage": "train"}
want = dict(a.split("=", 1) for a in sys.argv[2:]); want.setdefault("stage", "train")
for d in (REPO / "outputs/experiments" / sys.argv[1]).glob("RUN-*"):
    st = d / "status.json"
    if (d / "checkpoints/best.pt").exists() or (want["stage"] == "eval" and st.exists() and '"completed"' in st.read_text()):
        c = yaml.safe_load((d / "config.yaml").read_text())
        if all(str(c.get(k, D.get(k))) == v for k, v in want.items()):
            sys.exit(0)
sys.exit(1)
