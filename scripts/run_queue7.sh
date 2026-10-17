#!/usr/bin/env bash
# EXP-115 retry: training block was stopped at 00:46 on 2026-10-17 by memguard (system available RAM < 16 GB floor,
# other users' load); its evaluation then ran without the new arms (archived to EXP-099). Protocol unchanged.
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; CONV=$(cat outputs/experiments/exp109_convention.txt)
G() { scripts/guarded.sh ${GMEM:-56} -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
block() { local name=$1; shift; echo "=== $(date +%T) BLOCK $name"; if "$@"; then echo "=== $(date +%T) BLOCK_DONE $name"; else echo "=== $(date +%T) BLOCK_FAILED $name"; fi; }
A109=tree:EXP-109_real_domain_aug_tokens16_orbit:exp109_train_aug1+exp109_train_aug2
b115_train() { for arm in photo rot; do for s in 0 1 2; do for f in "dinov3_vitl16:tokens8 64x1024" "moge2_vitl:normals16~and 256x4"; do set -- $f
    echo ">>> $(date +%T) EXP-115 $arm $1 seed=$s"
    G $PY scripts/train_probe_p3.py --exp EXP-115_augmentation_ablation --features $1 --decoder transformer --token-shape $2 --feat-device cpu \
      --amp bf16 --targets soft --aug-tags exp115_train_${arm}1,exp115_train_${arm}2 --aug-convention $CONV --seed $s || return 1
  done; done; done; }
b115_eval() { G $PY scripts/orbit_eval.py --exp EXP-115_augmentation_ablation --modes greedy,orbit_product,orbit_vote32 \
  --families tree:EXP-036_transformer_soft_targets:,$A109,tree:EXP-115_augmentation_ablation:exp115_train_photo1+exp115_train_photo2,tree:EXP-115_augmentation_ablation:exp115_train_rot1+exp115_train_rot2; }
block EXP-115-train b115_train
block EXP-115-eval b115_eval
echo "=== $(date +%T) QUEUE7_DONE"
