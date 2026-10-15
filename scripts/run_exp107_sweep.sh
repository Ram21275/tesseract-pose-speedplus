#!/usr/bin/env bash
# EXP-107 speed-accuracy sweep: fused EXP-042 flow, seed 0, all domains; NFE sweep at M=32, M sweep at NFE=10.
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
G() { scripts/guarded.sh 32 -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
run() { echo ">>> $(date +%T) nfe=$1 m=$2"; G $PY scripts/exp04x_fused_eval.py --exp EXP-042_image_conditioned_so3_posterior \
  --run-exp EXP-107_paper_visuals_metrics_efficiency --seeds 0 --fused-only --feat-cpu --queries 1024 --nfe $1 --m $2 || echo "FAILED nfe=$1 m=$2"; }
for n in 1 2 5 10 20 50; do run $n 32; done
for m in 1 4 8 16 64; do run 10 $m; done
echo SWEEP_DONE
