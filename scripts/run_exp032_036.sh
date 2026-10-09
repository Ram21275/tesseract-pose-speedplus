#!/usr/bin/env bash
# EXP-032 (Transformer, hard) then EXP-036 (Transformer, soft); one guarded job at a time.
set -u; cd "$(dirname "$0")/.."
G=scripts/guarded.sh; PY=.venv/bin/python; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { echo ">>> $(date +%T) $*"; "$@" 2>&1 | grep -v -i warn | grep -E "seed|RUN|Error|Traceback|memguard|OutOfMemory" | tail -3; rc=${PIPESTATUS[0]}; echo "EXIT_CODE=$rc"; [ $rc -eq 0 ] || { echo CHAIN_FAILED; exit $rc; }; }
for spec in "032_transformer_decoder|" "036_transformer_soft_targets|--targets soft"; do
  IFS='|' read -r name opts <<< "$spec"
  for s in 0 1 2; do
    step $G 32 -- $PY scripts/train_probe_p3.py --exp EXP-$name --features dinov3_vitl16:tokens8 --decoder transformer --token-shape 64x1024 --feat-device gpu --amp bf16 $opts --seed $s
    step $G 32 -- $PY scripts/train_probe_p3.py --exp EXP-$name --features moge2_vitl:normals16~and --decoder transformer --token-shape 256x4 --feat-device gpu --amp bf16 $opts --seed $s
  done
  step $G 32 -- $PY scripts/exp03x_fused_eval.py --exp EXP-$name --a dinov3_vitl16:tokens8 --b moge2_vitl:normals16~and --beams 1,4 --run-exp EXP-$name
done
echo CHAIN_DONE
