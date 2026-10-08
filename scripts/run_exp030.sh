#!/usr/bin/env bash
# EXP-030 full-data chain; strictly one guarded job at a time.
set -u; cd "$(dirname "$0")/.."
G=scripts/guarded.sh; PY=.venv/bin/python; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { echo ">>> $(date +%T) $*"; "$@"; rc=$?; echo "EXIT_CODE=$rc $*"; [ $rc -eq 0 ] || { echo CHAIN_FAILED; exit $rc; }; }
step $G 8 -- $PY scripts/make_subset_manifest.py --name full --n-train 47966 --n-val 11994 --n-lightbox 6740 --n-sunlamp 2791
step $G 24 -- $PY scripts/extract_features.py --backbone dinov3_vitl16 --subset full --batch 32
step $G 24 -- $PY scripts/extract_features.py --backbone moge2_vitl --subset full --batch 4
step $G 16 -- $PY scripts/exp015_mask_consensus.py --subset full
for spec in dinov3_vitl16:grid4 moge2_vitl:normals16~and; do for s in 0 1 2; do
  step $G 32 -- $PY scripts/train_probe.py --exp EXP-030_full_data_mlp_baseline --subset full --features $spec --gpu-dtype fp16 --beams 1 --seed $s
done; done
step $G 32 -- $PY scripts/exp025_posterior_fusion.py --subset full --a EXP-030_full_data_mlp_baseline:dinov3_vitl16:grid4 --b EXP-030_full_data_mlp_baseline:moge2_vitl:normals16~and --run-exp EXP-030_full_data_mlp_baseline
echo CHAIN_DONE
