#!/usr/bin/env bash
# After EXP-050: EXP-107 figures + timing, then the compact-extraction equivalence check on 512 SPEED+ crops.
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
G() { scripts/guarded.sh 32 -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
until grep -q EXP050_DONE outputs/experiments/exp050.log 2>/dev/null; do sleep 60; done
echo ">>> $(date +%T) figures"; G $PY scripts/exp107_figures.py || echo FIGURES_FAILED
echo ">>> $(date +%T) timing"; G $PY scripts/exp107_timing.py || echo TIMING_FAILED
for bb in dinov3_vitl16 moge2_vitl; do
  echo ">>> $(date +%T) compact check $bb"
  G $PY scripts/extract_compact.py --manifest outputs/data_manifests/speedcheck_512.csv --root-key speedplus_root --tag speedcheck512 --backbone $bb || echo CHECK_FAILED
done
G $PY scripts/extract_compact.py --manifest outputs/data_manifests/speedcheck_512.csv --root-key speedplus_root --tag speedcheck512 --backbone consensus || echo CHECK_FAILED
echo CHAIN107_DONE
