#!/usr/bin/env bash
# Phase-1 controls for the CURRENT branches (configs/branches.yaml):
#   DINOv3-L (primary) + MoGe-2 (geometric; replaced VGGT, DEC-001).
# Every step runs alone inside its own memory-capped cgroup (scripts/guarded.sh).
# Usage: scripts/run_phase1.sh <subset-name> <exp-suffix>
#   e.g. scripts/run_phase1.sh v1 subset      (reproduces the EXP-010/013 setup)
#        scripts/run_phase1.sh full fulldata  (after making the full-data manifest)
set -u
cd "$(dirname "$0")/.."
SUBSET=${1:?subset name}; TAG=${2:?experiment suffix}
PY=.venv/bin/python
G="scripts/guarded.sh"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { echo ">>> $*"; "$@"; rc=$?; echo "EXIT_CODE=$rc $*"; return 0; }

# 1) feature extraction (one backbone at a time; GPU is shared -> small MoGe-2 batch)
run $G 24 -- $PY scripts/extract_features.py --backbone dinov3_vitl16 --subset "$SUBSET" --batch 32
run $G 24 -- $PY scripts/extract_features.py --backbone dinov3_vitb16 --subset "$SUBSET" --batch 32   # non-geometric control
run $G 24 -- $PY scripts/extract_features.py --backbone moge2_vitl --subset "$SUBSET" --batch 4

# 2) probes, 3 seeds each, one run at a time
for fs in l11_cls+mean l17_cls+mean l23_cls+mean grid4; do for s in 0 1 2; do
  run $G 16 -- $PY scripts/train_probe.py --exp "EXP-010_dinov3_${TAG}" --subset "$SUBSET" --features dinov3_vitl16:$fs --seed $s
done; done
for s in 0 1 2; do
  run $G 16 -- $PY scripts/train_probe.py --exp "EXP-010_dinov3_${TAG}" --subset "$SUBSET" --features dinov3_vitb16:grid4 --seed $s
done
for fs in l11_cls+mean l17_cls+mean l23_cls+mean grid4 depthconf16 points16 normals16; do for s in 0 1 2; do
  run $G 16 -- $PY scripts/train_probe.py --exp "EXP-013_moge2_${TAG}" --subset "$SUBSET" --features moge2_vitl:$fs --seed $s
done; done

# 3) complementarity audits: MoGe-2, then the non-geometric control (DINOv3-B)
run $G 24 -- $PY scripts/exp012_complementarity.py --subset "$SUBSET" --primary-exp "EXP-010_dinov3_${TAG}" \
    --second-exp "EXP-013_moge2_${TAG}" --run-exp "EXP-014_moge2_complementarity_${TAG}"
run $G 24 -- $PY scripts/exp012_complementarity.py --subset "$SUBSET" --primary-exp "EXP-010_dinov3_${TAG}" \
    --geo dinov3_vitb16:grid4 --second-exp "EXP-010_dinov3_${TAG}" --second-map last_grid \
    --run-exp "EXP-014_moge2_complementarity_${TAG}"
echo PHASE1_DONE
