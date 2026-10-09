#!/usr/bin/env bash
# EXP-102 Stage 1 (+ matched controls); waits for the EXP-032/036 chain to finish; one guarded job at a time.
set -u; cd "$(dirname "$0")/.."
G=scripts/guarded.sh; PY=.venv/bin/python; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q -E "CHAIN_DONE|CHAIN_FAILED" outputs/experiments/EXP-032_transformer_decoder/chain.log 2>/dev/null; do sleep 60; done
step() { echo ">>> $(date +%T) $*"; "$@" 2>&1 | grep -v -i warn | grep -E "adapter|seed|RUN|Error|Traceback|memguard|OutOfMemory" | tail -3; rc=${PIPESTATUS[0]}; echo "EXIT_CODE=$rc"; return 0; }
# matched frozen controls (cheap), 3 seeds each
for s in 0 1 2; do
  step $G 32 -- $PY scripts/train_probe_p3.py --exp EXP-102_lora_adapters --features dinov3_vitl16:grid4 --epochs 5 --batch 32 --clip 1.0 --eval-every 1 --beams 1 --seed $s
  step $G 32 -- $PY scripts/train_probe_p3.py --exp EXP-102_lora_adapters --features moge2_vitl:normals16~and --epochs 2 --batch 8 --train-frac 0.5 --clip 1.0 --eval-every 1 --beams 1 --seed $s
done
# adapter Stage 1 (seed 0)
step $G 32 -- $PY scripts/train_adapter.py --branch dino --epochs 5 --batch 32 --seed 0
step $G 32 -- $PY scripts/train_adapter.py --branch moge --epochs 2 --batch 4 --accum 2 --train-frac 0.5 --grad-ckpt --seed 0
step $G 32 -- $PY scripts/exp102_fused_eval.py
echo STAGE1_DONE
