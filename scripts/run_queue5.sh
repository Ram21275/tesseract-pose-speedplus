#!/usr/bin/env bash
# DEC-019: after queue3 (EXP-111) and queue4 (EXP-112): EXP-114 -> EXP-113 trainings -> EXP-115 features + trainings
# -> EXP-113 / EXP-115 evaluations. Non-blocking blocks; one GPU job at a time; all through guarded.sh.
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; CONV=$(cat outputs/experiments/exp109_convention.txt)
G() { scripts/guarded.sh ${GMEM:-56} -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
block() { local name=$1; shift; echo "=== $(date +%T) BLOCK $name"; if "$@"; then echo "=== $(date +%T) BLOCK_DONE $name"; else echo "=== $(date +%T) BLOCK_FAILED $name"; fi; }
X() { local man=$1 tag=$2; shift 2
  for bb in dinov3_vitl16 moge2_vitl; do echo ">>> $(date +%T) extract $tag $bb"
    G $PY scripts/extract_compact.py --manifest $man --root-key speedplus_root --tag $tag --backbone $bb "$@" || return 1; done
  G $PY scripts/extract_compact.py --manifest $man --root-key speedplus_root --tag $tag --backbone consensus || return 1; }
while pgrep -f '^bash scripts/run_queue[34].sh' >/dev/null; do sleep 60; done
A109=tree:EXP-109_real_domain_aug_tokens16_orbit:exp109_train_aug1+exp109_train_aug2
b114() { G $PY scripts/orbit_eval.py --exp EXP-114_orbit_scoring_rules --families $A109; }
b113_train() { for s in 0 1 2; do for f in "dinov3_vitl16:tokens8 64x1024" "moge2_vitl:normals16~and 256x4"; do set -- $f
    echo ">>> $(date +%T) EXP-113 flat $1 seed=$s"; G $PY scripts/train_flat_tf.py --features $1 --token-shape $2 --aug-convention $CONV --seed $s || return 1
  done; done; }
b115_feat() { local M=outputs/data_manifests/exp109_syn_train.csv
  X $M exp115_train_photo1 --photo-seed 1 && X $M exp115_train_photo2 --photo-seed 2 && \
  X $M exp115_train_rot1 --rot-random-seed 1 && X $M exp115_train_rot2 --rot-random-seed 2; }
b115_train() { for arm in photo rot; do for s in 0 1 2; do for f in "dinov3_vitl16:tokens8 64x1024" "moge2_vitl:normals16~and 256x4"; do set -- $f
    echo ">>> $(date +%T) EXP-115 $arm $1 seed=$s"
    G $PY scripts/train_probe_p3.py --exp EXP-115_augmentation_ablation --features $1 --decoder transformer --token-shape $2 --feat-device cpu \
      --amp bf16 --targets soft --aug-tags exp115_train_${arm}1,exp115_train_${arm}2 --aug-convention $CONV --seed $s || return 1
  done; done; done; }
b113_eval() { G $PY scripts/orbit_eval.py --exp EXP-113_tesseract_vs_flat_hopf --modes greedy,orbit_product \
  --families $A109,flat:EXP-113_tesseract_vs_flat_hopf:exp109_train_aug1+exp109_train_aug2; }
b115_eval() { G $PY scripts/orbit_eval.py --exp EXP-115_augmentation_ablation --modes greedy,orbit_product \
  --families tree:EXP-036_transformer_soft_targets:,$A109,tree:EXP-115_augmentation_ablation:exp115_train_photo1+exp115_train_photo2,tree:EXP-115_augmentation_ablation:exp115_train_rot1+exp115_train_rot2; }
block EXP-114 b114
block EXP-113-train b113_train
block EXP-115-features b115_feat
block EXP-115-train b115_train
block EXP-113-eval b113_eval
block EXP-115-eval b115_eval
echo "=== $(date +%T) QUEUE5_DONE"
