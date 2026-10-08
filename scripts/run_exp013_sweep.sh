#!/usr/bin/env bash
# EXP-013: MoGe-2 single-image geometry controls on subset v1 (mirrors EXP-011).
# Runs one probe at a time, each inside its own memory-capped cgroup.
set -u
cd "$(dirname "$0")/.."
until ls outputs/shared_cache/moge2_vitl__subset_v1__*/manifest.json >/dev/null 2>&1; do sleep 30; done
for fs in l5_cls+mean l11_cls+mean l17_cls+mean l23_cls+mean l23_cls l23_mean grid4 depthconf16 points16 normals16; do
  for s in 0 1 2; do
    scripts/guarded.sh 16 -- .venv/bin/python scripts/train_probe.py --exp EXP-013_moge2_controls --features moge2_vitl:$fs --seed $s 2>&1 \
      | grep -v -i warn | grep -E "seed|Error|Traceback|memguard" | head -3
    echo "EXIT_CODE=${PIPESTATUS[0]} $fs seed $s"
  done
done
echo SWEEP_DONE
