#!/usr/bin/env bash
# HISTORICAL: the exact Phase-1 sweeps behind EXP-010 (ViT-L part) and EXP-011 (VGGT,
# retired by DEC-001). Kept for reproducibility; use scripts/run_phase1.sh for new runs.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
wait_cache() { until ls outputs/shared_cache/$1__subset_v1__*/manifest.json >/dev/null 2>&1; do sleep 30; done; }
sweep() {  # exp backbone featsets...
  local exp=$1 bb=$2; shift 2
  for fs in "$@"; do for s in 0 1 2; do
    $PY scripts/train_probe.py --exp $exp --features $bb:$fs --seed $s 2>&1 | grep -v -i warn | grep -E "seed|Error|Traceback" | head -3
  done; done
}
wait_cache dinov3_vitl16
sweep EXP-010_dinov3_baseline dinov3_vitl16 l11_cls+mean l17_cls+mean l23_cls+mean grid4
wait_cache vggt_1b
sweep EXP-011_vggt_controls vggt_1b l4_cam+mean l11_cam+mean l17_cam+mean l23_cam+mean l23_cam l23_mean grid4 depthconf16 points16
echo SWEEPS_DONE
