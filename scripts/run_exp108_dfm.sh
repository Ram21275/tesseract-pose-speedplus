#!/usr/bin/env bash
# DEC-015: after the EXP-109 part of the master chain, EXP-108 discrete flow matching (smoke, 3 seeds, eval), then the
# EXP-107 jobs (sweep point M=8, figures, timing). The old master chain stops at its superseded EXP-108 step.
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
G() { scripts/guarded.sh 40 -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
until grep -qE "EXP109_DONE|_FAILED" outputs/experiments/master_chain.log 2>/dev/null && ! pgrep -f '[r]un_exp109_master' >/dev/null; do sleep 120; done
grep -q EXP109_DONE outputs/experiments/master_chain.log || { echo "EXP-109 did not finish; not starting EXP-108"; exit 1; }
echo ">>> $(date +%T) EXP-108 DFM smoke"
G $PY scripts/train_pathdfm.py --exp EXP-099_p3_smoke --epochs 1 --eval-every 1 || { echo SMOKE_FAILED; exit 1; }
for s in 0 1 2; do echo ">>> $(date +%T) EXP-108 DFM seed=$s"; G $PY scripts/train_pathdfm.py --seed $s || { echo TRAIN_FAILED; exit 1; }; done
echo ">>> $(date +%T) EXP-108 eval"; G $PY scripts/exp108_eval.py || echo EVAL108_FAILED
echo CHAIN108_DONE
echo ">>> $(date +%T) EXP-110 weighted PoE"; G $PY scripts/exp110_weighted_poe.py || echo EXP110_FAILED
echo ">>> $(date +%T) EXP-107 sweep rerun nfe=10 m=8"
G $PY scripts/exp04x_fused_eval.py --exp EXP-042_image_conditioned_so3_posterior --run-exp EXP-107_paper_visuals_metrics_efficiency --seeds 0 --fused-only --feat-cpu --queries 1024 --nfe 10 --m 8 || echo SWEEP8_FAILED
echo ">>> $(date +%T) EXP-107 figures"; G $PY scripts/exp107_figures.py || echo FIGURES_FAILED
echo ">>> $(date +%T) EXP-107 timing"; G $PY scripts/exp107_timing.py || echo TIMING_FAILED
echo DFM_CHAIN_DONE
