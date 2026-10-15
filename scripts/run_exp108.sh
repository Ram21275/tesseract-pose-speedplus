#!/usr/bin/env bash
# EXP-108 chain: waits for the EXP-107 chain, then smoke (1 epoch, EXP-099), 3 seeds x 2 branches, PoE revision eval.
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
G() { scripts/guarded.sh 40 -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
until grep -q CHAIN107_DONE outputs/experiments/exp107_after050.log 2>/dev/null; do sleep 120; done
echo ">>> $(date +%T) smoke"
G $PY scripts/train_pathflow.py --exp EXP-099_p3_smoke --features moge2_vitl:normals16~and --epochs 1 --eval-every 1 || { echo SMOKE_FAILED; exit 1; }
for s in 0 1 2; do for f in dinov3_vitl16:tokens8 moge2_vitl:normals16~and; do
  echo ">>> $(date +%T) train $f seed=$s"; G $PY scripts/train_pathflow.py --features $f --seed $s || { echo TRAIN_FAILED; exit 1; }
done; done
echo ">>> $(date +%T) eval"; G $PY scripts/exp108_eval.py
echo CHAIN108_DONE
