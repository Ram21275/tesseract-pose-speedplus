#!/usr/bin/env bash
# EXP-016: consensus mask -> pose, both directions; one guarded probe at a time.
set -u; cd "$(dirname "$0")/.."
for spec in moge2_vitl:normals16~and dinov3_vitl16:grid4~and; do for s in 0 1 2; do
  scripts/guarded.sh 16 -- .venv/bin/python scripts/train_probe.py --exp EXP-016_consensus_pose --features $spec --seed $s 2>&1 | grep -v -i warn | grep -E "seed|Error|Traceback|memguard" | head -3
  echo "EXIT_CODE=${PIPESTATUS[0]} $spec seed $s"
done; done; echo SWEEP_DONE
