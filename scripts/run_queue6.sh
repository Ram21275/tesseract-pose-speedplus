#!/usr/bin/env bash
# Re-run of the blocks that failed on 2026-10-16 ~14:00 because featsets.load called relative_to(REPO) on the moved
# caches (DEC-020; fixed). Order (DEC-019): finish EXP-115 features -> EXP-111 seed 2 + summary + verifier -> EXP-112
# -> EXP-114 -> EXP-113 trainings -> EXP-115 trainings -> EXP-113 eval -> EXP-115 eval. Non-blocking blocks.
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; CONV=$(cat outputs/experiments/exp109_convention.txt)
G() { scripts/guarded.sh ${GMEM:-56} -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
block() { local name=$1; shift; echo "=== $(date +%T) BLOCK $name"; if "$@"; then echo "=== $(date +%T) BLOCK_DONE $name"; else echo "=== $(date +%T) BLOCK_FAILED $name"; fi; }
X() { local man=$1 tag=$2; shift 2
  for bb in dinov3_vitl16 moge2_vitl; do echo ">>> $(date +%T) extract $tag $bb"
    G $PY scripts/extract_compact.py --manifest $man --root-key speedplus_root --tag $tag --backbone $bb "$@" || return 1; done
  G $PY scripts/extract_compact.py --manifest $man --root-key speedplus_root --tag $tag --backbone consensus || return 1; }
while pgrep -f '^/home/kavinder/RAM/pose-estimation/\.venv/bin/python scripts/extract_compact.py' >/dev/null; do sleep 60; done
A109=tree:EXP-109_real_domain_aug_tokens16_orbit:exp109_train_aug1+exp109_train_aug2
b115_feat() { local M=outputs/data_manifests/exp109_syn_train.csv
  X $M exp115_train_photo1 --photo-seed 1 && X $M exp115_train_photo2 --photo-seed 2 && \
  X $M exp115_train_rot1 --rot-random-seed 1 && X $M exp115_train_rot2 --rot-random-seed 2; }
b111() { echo ">>> $(date +%T) EXP-111 flow seed=2"
  GMEM=48 G $PY scripts/train_phase4.py --exp EXP-111_dino_aug_flow --features dinov3_vitl16:tokens8 --src-exp EXP-109_real_domain_aug_tokens16_orbit \
    --mode flow --base uniform --centre gt --feat-cpu --aug-tags exp109_train_aug1,exp109_train_aug2 --aug-convention $CONV --seed 2 || return 1
  G $PY scripts/exp111_eval.py || echo "(summary/timing failed; continuing)"
  echo ">>> $(date +%T) EXP-111 verifier arm"; G $PY scripts/exp111_verifier.py; }
b112() { G $PY scripts/exp112_eval.py || echo "(EXP-112 accuracy failed; continuing)"; G $PY scripts/exp112_timing.py; }
b114() { G $PY scripts/orbit_eval.py --exp EXP-114_orbit_scoring_rules --families $A109; }
b113_train() { for s in 0 1 2; do for f in "dinov3_vitl16:tokens8 64x1024" "moge2_vitl:normals16~and 256x4"; do set -- $f
    echo ">>> $(date +%T) EXP-113 flat $1 seed=$s"; G $PY scripts/train_flat_tf.py --features $1 --token-shape $2 --aug-convention $CONV --seed $s || return 1
  done; done; }
b115_train() { for arm in photo rot; do for s in 0 1 2; do for f in "dinov3_vitl16:tokens8 64x1024" "moge2_vitl:normals16~and 256x4"; do set -- $f
    echo ">>> $(date +%T) EXP-115 $arm $1 seed=$s"
    G $PY scripts/train_probe_p3.py --exp EXP-115_augmentation_ablation --features $1 --decoder transformer --token-shape $2 --feat-device cpu \
      --amp bf16 --targets soft --aug-tags exp115_train_${arm}1,exp115_train_${arm}2 --aug-convention $CONV --seed $s || return 1
  done; done; done; }
b113_eval() { G $PY scripts/orbit_eval.py --exp EXP-113_tesseract_vs_flat_hopf --modes greedy,orbit_product \
  --families $A109,flat:EXP-113_tesseract_vs_flat_hopf:exp109_train_aug1+exp109_train_aug2; }
b115_eval() { G $PY scripts/orbit_eval.py --exp EXP-115_augmentation_ablation --modes greedy,orbit_product \
  --families tree:EXP-036_transformer_soft_targets:,$A109,tree:EXP-115_augmentation_ablation:exp115_train_photo1+exp115_train_photo2,tree:EXP-115_augmentation_ablation:exp115_train_rot1+exp115_train_rot2; }
block EXP-115-features b115_feat
block EXP-111 b111
block EXP-112 b112
block EXP-114 b114
block EXP-113-train b113_train
block EXP-115-train b115_train
block EXP-113-eval b113_eval
block EXP-115-eval b115_eval
echo "=== $(date +%T) QUEUE6_DONE"
