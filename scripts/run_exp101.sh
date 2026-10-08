#!/usr/bin/env bash
# EXP-101 parts B then C, one guarded job at a time.
set -u; cd "$(dirname "$0")/.."
G=scripts/guarded.sh; PY=.venv/bin/python
for cfg in "256 12" "1365 12" "10923 12"; do set -- $cfg; for s in 0 1 2; do
  $G 24 -- $PY scripts/train_flat_probe.py --hopf-points $1 --hopf-rolls $2 --seed $s 2>&1 | grep -v -i warn | grep -E "flat_hopf|Error|Traceback|memguard" | head -3
  echo "EXIT_CODE=${PIPESTATUS[0]} flat $1x$2 seed $s"
done; done
echo PARTB_DONE
$G 24 -- $PY scripts/exp101_timing.py 2>&1 | grep -v -i warn | grep -E "RUN|Error|Traceback|memguard" ; echo "EXIT_CODE=${PIPESTATUS[0]} timing"
echo EXP101_DONE
