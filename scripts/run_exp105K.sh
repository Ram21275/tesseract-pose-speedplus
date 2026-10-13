#!/usr/bin/env bash
# EXP-105 arm K chain: waits for the C-arm scoring run, then smoke-tests and trains the 8-facet Kurz trees
# (both branches, seeds 0-2, EXP-036 budget) and evaluates their PoE corner vote vs the EXP-036 PoE.
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
E=EXP-105_candidate_expansion_best_fit; PY=.venv/bin/python
G() { scripts/guarded.sh 40 -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
until grep -q EXP105_DONE outputs/experiments/exp105_main.log 2>/dev/null; do sleep 120; done
echo ">>> $(date +%T) smoke (1 epoch, EXP-099)"
G $PY scripts/train_kurz8.py --exp EXP-099_p3_smoke --features moge2_vitl:normals16~and --token-shape 256x4 --epochs 1 --eval-every 1 \
  || { echo SMOKE_FAILED; exit 1; }
for s in 0 1 2; do
  for f in "dinov3_vitl16:tokens8 64x1024" "moge2_vitl:normals16~and 256x4"; do
    set -- $f; echo ">>> $(date +%T) train K $1 seed=$s"
    G $PY scripts/train_kurz8.py --exp $E --features $1 --token-shape $2 --seed $s || { echo TRAIN_FAILED; exit 1; }
  done
done
echo ">>> $(date +%T) arm K eval"
G $PY scripts/exp105_armK_eval.py
echo CHAIN_DONE
