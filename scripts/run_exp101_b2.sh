#!/usr/bin/env bash
# EXP-101 amendment B2: flat Hopf heads on the consensus feature; waits for part C timing to finish first.
set -u; cd "$(dirname "$0")/.."
until grep -q -E "EXP101_DONE" outputs/experiments/EXP-101_tesseract_vs_spacehop_hopf/driver.log 2>/dev/null; do sleep 30; done
for cfg in "256 12" "1365 12" "10923 12"; do set -- $cfg; for s in 0 1 2; do
  scripts/guarded.sh 24 -- .venv/bin/python scripts/train_flat_probe.py --features moge2_vitl:normals16~and --hopf-points $1 --hopf-rolls $2 --seed $s 2>&1 | grep -v -i warn | grep -E "flat_hopf|Error|Traceback|memguard" | head -3
  echo "EXIT_CODE=${PIPESTATUS[0]} flat $1x$2 seed $s"
done; done
echo B2_DONE
