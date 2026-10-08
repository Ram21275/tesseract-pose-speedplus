#!/usr/bin/env bash
# Phase-3 batch: EXP-034 (beam, eval only) then EXP-031 / 033 / 035 (train both branches x 3 seeds, then fused eval).
set -u; cd "$(dirname "$0")/.."
G=scripts/guarded.sh; PY=.venv/bin/python; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { echo ">>> $(date +%T) $*"; "$@" 2>&1 | grep -v -i warn | grep -E "seed|RUN|Error|Traceback|memguard" | tail -3; rc=${PIPESTATUS[0]}; echo "EXIT_CODE=$rc"; [ $rc -eq 0 ] || { echo CHAIN_FAILED; exit $rc; }; }
step $G 32 -- $PY scripts/exp03x_fused_eval.py --exp EXP-030_full_data_mlp_baseline --beams 1,2,4,8 --run-exp EXP-034_beam_decoding
for spec in "031_gru_path_decoder|--decoder gru" "033_geodesic_soft_targets|--targets soft" "035_tangent_residual|--residual"; do
  IFS='|' read -r name opts <<< "$spec"
  for f in dinov3_vitl16:grid4 moge2_vitl:normals16~and; do for s in 0 1 2; do
    step $G 32 -- $PY scripts/train_probe_p3.py --exp EXP-$name --features $f $opts --seed $s
  done; done
  step $G 32 -- $PY scripts/exp03x_fused_eval.py --exp EXP-$name --beams 1,4 --run-exp EXP-$name
done
echo CHAIN_DONE
