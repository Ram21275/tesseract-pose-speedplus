#!/usr/bin/env bash
# Supplementary flow figures: per-example data (one light GPU job; peak RAM of the same loading in EXP-107: 21.8 GB).
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== $(date +%T) BLOCK flow"
if GUARD_GPU_GB=4 scripts/guarded.sh 32 -- .venv/bin/python scripts/fig_flow_supp.py 2>&1 | grep -v -i warn | grep -v nn.Transformer; then echo "=== $(date +%T) BLOCK_DONE flow"; else echo "=== $(date +%T) BLOCK_FAILED flow"; fi
echo "=== $(date +%T) QUEUE19_DONE"
