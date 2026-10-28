"""Print the newest finished train run (outputs/experiments-relative path) in outputs/experiments/<dir> whose config matches
every k=v (missing keys take the trainer defaults). Exit 1 if none."""
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
D = {"label": "raw", "extra_train": "None", "teacher_variant": "None", "trans": "none", "stage": "train"}
want = dict(a.split("=", 1) for a in sys.argv[2:] if a); want.setdefault("stage", "train"); hit = None
for d in sorted((REPO / "outputs/experiments" / sys.argv[1]).glob("RUN-*")):
    if (d / "checkpoints/best.pt").exists():
        c = yaml.safe_load((d / "config.yaml").read_text())
        if all(str(c.get(k, D.get(k))) == v for k, v in want.items()):
            hit = d
if hit is None:
    sys.exit(1)
print(f"{sys.argv[1]}/{hit.name}")
