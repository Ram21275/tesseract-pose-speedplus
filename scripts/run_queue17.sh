#!/usr/bin/env bash
# DEC-028 / EXP-129: T-LESS and YCB-V rotation + joint pose with predicted instead of GT translation (evaluation only).
# Starts after QUEUE16_DONE; one GPU job at a time through guarded.sh; does not touch QUEUE_PAUSE.
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
G() { GUARD_GPU_GB=${GG:-8} scripts/guarded.sh ${GMEM:-40} -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
R() { local i; for i in 1 2 3; do G "$@" && return 0; echo "RETRY $i $(date +%T)"; sleep 600; done; return 1; }
block() { local name=$1; shift; echo "=== $(date +%T) BLOCK $name"; if "$@"; then echo "=== $(date +%T) BLOCK_DONE $name"; else echo "=== $(date +%T) BLOCK_FAILED $name"; fi; }
until grep -q QUEUE16_DONE outputs/experiments/queue16.log 2>/dev/null; do sleep 120; done
echo "=== $(date +%T) queue17 starts"
exp129() { local ds=$1 t
  for t in gt greedy orbit_mean; do echo ">>> $(date +%T) EXP-129 $ds pose-t $t"
    R $PY scripts/exp064_eval.py --dataset $ds --exp EXP-129 --exp-name rotation_with_predicted_translation --runs-dir EXP-126_translation_xobj \
      --filter trans=frozen --tag pose_t_$t --pose-t $t --pose-t-variant DINO || return 1; done; }
block EXP-129-tless exp129 tless
block EXP-129-ycbv exp129 ycbv
echo "=== $(date +%T) QUEUE17_DONE"
