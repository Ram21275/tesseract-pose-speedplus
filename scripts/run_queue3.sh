#!/usr/bin/env bash
# Retry of the EXP-111 block (killed by systemd-oomd at 02:55 on 2026-10-16); protocol unchanged.
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
G() { scripts/guarded.sh ${GMEM:-48} -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
block() { local name=$1; shift; echo "=== $(date +%T) BLOCK $name"; if "$@"; then echo "=== $(date +%T) BLOCK_DONE $name"; else echo "=== $(date +%T) BLOCK_FAILED $name"; fi; }
b111() { local CONV; CONV=$(cat outputs/experiments/exp109_convention.txt) || return 1
  for s in 0 1 2; do echo ">>> $(date +%T) EXP-111 flow seed=$s"
    G $PY scripts/train_phase4.py --exp EXP-111_dino_aug_flow --features dinov3_vitl16:tokens8 --src-exp EXP-109_real_domain_aug_tokens16_orbit \
      --mode flow --base uniform --feat-cpu --aug-tags exp109_train_aug1,exp109_train_aug2 --aug-convention $CONV --seed $s || return 1; done
  G $PY scripts/exp111_eval.py || echo "(summary/timing failed; continuing)"
  echo ">>> $(date +%T) EXP-111 verifier arm"; G $PY scripts/exp111_verifier.py; }
block EXP-111 b111
echo "=== $(date +%T) QUEUE3_DONE"
