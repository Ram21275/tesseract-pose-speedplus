#!/usr/bin/env bash
# EXP-130 (DEC-029): per-image predictions, 3 seeds, EXP-129 configurations; then the literature metrics (CPU).
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; N="--exp EXP-130 --exp-name literature_matched_metrics --runs-dir EXP-126_translation_xobj --filter trans=frozen --save-preds --modes greedy,orbit_product --pose-t-variant DINO"
G() { GUARD_GPU_GB=${GG:-12} scripts/guarded.sh ${GMEM:-16} -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
R() { local i; for i in $(seq 1 12); do G "$@" && return 0; echo "RETRY $i $(date +%T)"; sleep 900; done; return 1; }
block() { local name=$1; shift; echo "=== $(date +%T) BLOCK $name"; if "$@"; then echo "=== $(date +%T) BLOCK_DONE $name"; else echo "=== $(date +%T) BLOCK_FAILED $name"; return 1; fi; }
block ycbv R $PY scripts/exp064_eval.py $N --dataset ycbv --tag exp130 --pose-t greedy && \
block tless R $PY scripts/exp064_eval.py $N --dataset tless --tag exp130 --pose-t orbit_mean && \
GUARD_GPU_GB=0 GMEM=32 block metrics G $PY scripts/exp130_metrics.py
echo "=== $(date +%T) QUEUE20_DONE"
