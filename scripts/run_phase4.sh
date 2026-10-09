#!/usr/bin/env bash
# Phase 4: EXP-040 (residual), EXP-041 (local flow), EXP-042 (uniform flow) on frozen EXP-036; one guarded job at a time.
set -u; cd "$(dirname "$0")/.."
G=scripts/guarded.sh; PY=.venv/bin/python; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ERR=outputs/experiments/EXP-040_tangent_residual_predicted_leaf/RUN-20261010-015411-seed0/tables/poe_synthval_greedy_err.csv
step() { echo ">>> $(date +%T) $*"; "$@" 2>&1 | grep -v -i warn | grep -E "seed|RUN|Error|Traceback|memguard|OutOfMemory|^fused|^\|" | tail -30; rc=${PIPESTATUS[0]}; echo "EXIT_CODE=$rc"; [ $rc -eq 0 ] || { echo CHAIN_FAILED; exit $rc; }; }
for spec in "040_tangent_residual_predicted_leaf|--mode residual" "041_cell_conditioned_rfm|--mode flow --base local --sigma-b-deg 3" "042_image_conditioned_so3_posterior|--mode flow --base uniform"; do
  IFS='|' read -r name opts <<< "$spec"
  for s in 0 1 2; do
    for f in dinov3_vitl16:tokens8 moge2_vitl:normals16~and; do
      step $G 32 -- $PY scripts/train_phase4.py --exp EXP-$name --features $f $opts --centre mix --err-dist $ERR --seed $s
    done
  done
  extra=""; [ "$name" = "042_image_conditioned_so3_posterior" ] && extra="--save-samples"
  step $G 32 -- $PY scripts/exp04x_fused_eval.py --exp EXP-$name $extra
done
echo CHAIN_DONE
