#!/usr/bin/env bash
# Paper figures: seed-0 re-runs of the finalised configurations with per-image predictions saved (--save-preds).
# Metrics must reproduce the seed-0 rows of EXP-129 (T-LESS, YCB-V), EXP-126 (SYMSOL-T) and EXP-124 (SPEED+ S1).
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; N="--exp FIG-qualitative --exp-name qualitative_figures"
G() { GUARD_GPU_GB=${GG:-12} scripts/guarded.sh ${GMEM:-56} -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
R() { local i; for i in $(seq 1 12); do G "$@" && return 0; echo "RETRY $i $(date +%T)"; sleep 900; done; return 1; }
block() { local name=$1; shift; echo "=== $(date +%T) BLOCK $name"; if "$@"; then echo "=== $(date +%T) BLOCK_DONE $name"; else echo "=== $(date +%T) BLOCK_FAILED $name"; fi; }
M() { local g=$1; shift; GMEM=$g R "$@"; }   # per-job memory cap (GB), from the recorded peaks (EXP-129 5.4 GB, EXP-124 24.9 GB)
E64="scripts/exp064_eval.py $N --runs-dir EXP-126_translation_xobj --filter trans=frozen --seeds 0 --save-preds --modes greedy,orbit_product --pose-t-variant DINO"
block tless M 16 $PY $E64 --dataset tless --tag fig_tless --pose-t orbit_mean
block ycbv M 16 $PY $E64 --dataset ycbv --tag fig_ycbv --pose-t greedy
block symsolt M 16 $PY $E64 --dataset symsolt --tag fig_symsolt --pose-t gt --variants DINO
block spp_gt M 32 $PY scripts/exp124_eval.py $N --crops gt --seeds 0 --systems S1 --save-preds
block spp_det M 32 $PY scripts/exp124_eval.py $N --crops det --seeds 0 --systems S1 --save-preds
echo "=== $(date +%T) QUEUE18_DONE"
