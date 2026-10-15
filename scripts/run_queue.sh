#!/usr/bin/env bash
# Single non-blocking queue (user, 2026-10-15: "even if the runs fails we should start further experiments").
# Each block runs its steps in order; a failed step ends only that block (logged as BLOCK_FAILED <name>), and the queue
# moves on. One GPU job at a time, all through guarded.sh (waits for RAM / GPU headroom, never kills anything).
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
G() { scripts/guarded.sh ${GMEM:-40} -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
X() { local man=$1 tag=$2; shift 2
  for bb in dinov3_vitl16 moge2_vitl; do echo ">>> $(date +%T) extract $tag $bb"
    G $PY scripts/extract_compact.py --manifest $man --root-key speedplus_root --tag $tag --backbone $bb "$@" || return 1; done
  G $PY scripts/extract_compact.py --manifest $man --root-key speedplus_root --tag $tag --backbone consensus || return 1; }
block() { local name=$1; shift; echo "=== $(date +%T) BLOCK $name"; if "$@"; then echo "=== $(date +%T) BLOCK_DONE $name"; else echo "=== $(date +%T) BLOCK_FAILED $name"; fi; }

b109_features() {
  [ -f outputs/shared_cache/tokens16_full_stats.npz ] || GUARD_GPU_GB=0 GMEM=16 G $PY scripts/build_tokens16_memmap.py --stats-only || return 1
  local GM=outputs/data_manifests/exp109_gate512.csv
  X $GM gate512_r0 && X $GM gate512_r90 --rot-deg 90 && X $GM gate512_r37 --rot-deg 37 || return 1
  GUARD_GPU_GB=0 G $PY scripts/exp109_convention_geometry.py || return 1
  [ "$(cat outputs/experiments/exp109_convention.txt)" != NONE ] || return 1
  G $PY scripts/exp109_gate.py || echo "(descriptive predictor check failed; not blocking)"
  X outputs/data_manifests/exp109_syn_train.csv exp109_train_aug1 --rot-random-seed 1 --photo-seed 1 || return 1
  X outputs/data_manifests/exp109_syn_train.csv exp109_train_aug2 --rot-random-seed 2 --photo-seed 2 || return 1
  X outputs/data_manifests/exp109_syn_val.csv exp109_valaug --rot-random-seed 1000 --photo-seed 1000 --store-grid16 || return 1
  for k in 90 180 270; do
    X outputs/data_manifests/exp109_syn_val.csv exp109_valaug_r$k --rot-random-seed 1000 --photo-seed 1000 --rot-deg $k || return 1
    X outputs/data_manifests/exp109_test3.csv exp109_test3_r$k --rot-deg $k || return 1
  done; }
b109_train() { local CONV; CONV=$(cat outputs/experiments/exp109_convention.txt) || return 1
  for s in 0 1 2; do
    for f in "dinov3_vitl16:tokens8 64x1024" "moge2_vitl:normals16~and 256x4"; do set -- $f
      echo ">>> $(date +%T) EXP-109 A $1 seed=$s"
      G $PY scripts/train_probe_p3.py --exp EXP-109_real_domain_aug_tokens16_orbit --features $1 --decoder transformer --token-shape $2 \
        --feat-device cpu --amp bf16 --targets soft --aug-tags exp109_train_aug1,exp109_train_aug2 --aug-convention $CONV --seed $s || return 1; done
    echo ">>> $(date +%T) EXP-109 B tokens16 seed=$s"
    G $PY scripts/train_probe_p3.py --exp EXP-109_real_domain_aug_tokens16_orbit --features dinov3_vitl16:tokens16 --decoder transformer \
      --token-shape 256x1024 --feat-device cpu --amp bf16 --targets soft --memmap outputs/shared_cache/tokens16_full_std.npy --seed $s || return 1
  done; }
b109_eval() { G $PY scripts/exp109_eval.py; }
b108() { echo ">>> $(date +%T) EXP-108 DFM smoke"; G $PY scripts/train_pathdfm.py --exp EXP-099_p3_smoke --epochs 1 --eval-every 1 || return 1
  for s in 0 1 2; do echo ">>> $(date +%T) EXP-108 DFM seed=$s"; G $PY scripts/train_pathdfm.py --seed $s || return 1; done
  G $PY scripts/exp108_eval.py; }
b110() { G $PY scripts/exp110_weighted_poe.py; }                       # needs the exp109_valaug features
b107() { G $PY scripts/exp04x_fused_eval.py --exp EXP-042_image_conditioned_so3_posterior --run-exp EXP-107_paper_visuals_metrics_efficiency \
           --seeds 0 --fused-only --feat-cpu --queries 1024 --nfe 10 --m 8 || echo "(sweep point failed; continuing)"
         G $PY scripts/exp107_figures.py || echo "(figures failed; continuing)"; G $PY scripts/exp107_timing.py; }

block EXP-109-features b109_features
block EXP-109-train b109_train
block EXP-109-eval b109_eval
block EXP-108 b108
block EXP-110 b110
block EXP-107 b107
echo "=== $(date +%T) QUEUE_DONE"
