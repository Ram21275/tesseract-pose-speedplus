#!/usr/bin/env bash
# Queue part 2 (DEC-017): after the running EXP-109 eval -> EXP-111 -> EXP-108 -> EXP-110 -> EXP-107. Non-blocking blocks.
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
G() { scripts/guarded.sh ${GMEM:-56} -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
block() { local name=$1; shift; echo "=== $(date +%T) BLOCK $name"; if "$@"; then echo "=== $(date +%T) BLOCK_DONE $name"; else echo "=== $(date +%T) BLOCK_FAILED $name"; fi; }
while pgrep -f 'scripts/[e]xp109_eval.py' >/dev/null; do sleep 60; done
b111() { local CONV; CONV=$(cat outputs/experiments/exp109_convention.txt) || return 1
  for s in 0 1 2; do echo ">>> $(date +%T) EXP-111 flow seed=$s"
    G $PY scripts/train_phase4.py --exp EXP-111_dino_aug_flow --features dinov3_vitl16:tokens8 --src-exp EXP-109_real_domain_aug_tokens16_orbit \
      --mode flow --base uniform --feat-cpu --aug-tags exp109_train_aug1,exp109_train_aug2 --aug-convention $CONV --seed $s || return 1; done
  G $PY scripts/exp111_eval.py || echo "(summary/timing failed; continuing)"
  echo ">>> $(date +%T) EXP-111 verifier arm (EXP-045 design)"; G $PY scripts/exp111_verifier.py; }
b108() { echo ">>> $(date +%T) EXP-108 DFM smoke"; G $PY scripts/train_pathdfm.py --exp EXP-099_p3_smoke --epochs 1 --eval-every 1 || return 1
  for s in 0 1 2; do echo ">>> $(date +%T) EXP-108 DFM seed=$s"; G $PY scripts/train_pathdfm.py --seed $s || return 1; done
  G $PY scripts/exp108_eval.py; }
b110() { G $PY scripts/exp110_weighted_poe.py; }
b107() { G $PY scripts/exp04x_fused_eval.py --exp EXP-042_image_conditioned_so3_posterior --run-exp EXP-107_paper_visuals_metrics_efficiency \
           --seeds 0 --fused-only --feat-cpu --queries 1024 --nfe 10 --m 8 || echo "(sweep point failed; continuing)"
         G $PY scripts/exp107_figures.py || echo "(figures failed; continuing)"; }
block EXP-111 b111
block EXP-108 b108
block EXP-110 b110
block EXP-107 b107
echo "=== $(date +%T) QUEUE2_DONE"
