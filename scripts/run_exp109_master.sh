#!/usr/bin/env bash
# DEC-014 order: EXP-109 (gate -> augmented / orbit extraction -> A and B trainings -> eval) -> EXP-108 -> EXP-107 jobs.
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; E=EXP-109_real_domain_aug_tokens16_orbit
G() { scripts/guarded.sh ${GMEM:-40} -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
X() {  # manifest tag extra-args...: compact DINO + MoGe + consensus
  local man=$1 tag=$2; shift 2
  for bb in dinov3_vitl16 moge2_vitl; do
    echo ">>> $(date +%T) extract $tag $bb"; G $PY scripts/extract_compact.py --manifest $man --root-key speedplus_root --tag $tag --backbone $bb "$@" || { echo EXTRACT_FAILED $tag; exit 1; }
  done
  G $PY scripts/extract_compact.py --manifest $man --root-key speedplus_root --tag $tag --backbone consensus || { echo EXTRACT_FAILED $tag; exit 1; }
}
until [ -f outputs/shared_cache/tokens16_full_std.json ]; do sleep 60; done
[ -f outputs/shared_cache/tokens16_full_stats.npz ] || GUARD_GPU_GB=0 GMEM=16 G $PY scripts/build_tokens16_memmap.py --stats-only
# ---- convention gate
GM=outputs/data_manifests/exp109_gate512.csv
X $GM gate512_r0; X $GM gate512_r90 --rot-deg 90; X $GM gate512_r37 --rot-deg 37
GUARD_GPU_GB=0 G $PY scripts/exp109_convention_geometry.py || { echo GATE_FAILED; exit 1; }      # derived + verified (no model)
CONV=$(cat outputs/experiments/exp109_convention.txt); echo "CONVENTION=$CONV"
G $PY scripts/exp109_gate.py || echo PREDICTOR_CHECK_FAILED                                       # descriptive only
if [ "$CONV" = "NONE" ]; then echo "GATE_NONE: stop for a decision (pre-registered: drop rotation aug and C)"; exit 1; fi
# ---- augmented / orbit features
X outputs/data_manifests/exp109_syn_train.csv exp109_train_aug1 --rot-random-seed 1 --photo-seed 1
X outputs/data_manifests/exp109_syn_train.csv exp109_train_aug2 --rot-random-seed 2 --photo-seed 2
X outputs/data_manifests/exp109_syn_val.csv exp109_valaug --rot-random-seed 1000 --photo-seed 1000 --store-grid16
for k in 90 180 270; do
  X outputs/data_manifests/exp109_syn_val.csv exp109_valaug_r$k --rot-random-seed 1000 --photo-seed 1000 --rot-deg $k
  X outputs/data_manifests/exp109_test3.csv exp109_test3_r$k --rot-deg $k
done
# ---- trainings (A: both branches; B: DINOv3 tokens16), 3 seeds
for s in 0 1 2; do
  for f in "dinov3_vitl16:tokens8 64x1024" "moge2_vitl:normals16~and 256x4"; do set -- $f
    echo ">>> $(date +%T) train A $1 seed=$s"
    G $PY scripts/train_probe_p3.py --exp $E --features $1 --decoder transformer --token-shape $2 --feat-device cpu --amp bf16 --targets soft \
      --aug-tags exp109_train_aug1,exp109_train_aug2 --aug-convention $CONV --seed $s || { echo TRAIN_FAILED; exit 1; }
  done
  echo ">>> $(date +%T) train B tokens16 seed=$s"
  G $PY scripts/train_probe_p3.py --exp $E --features dinov3_vitl16:tokens16 --decoder transformer --token-shape 256x1024 --feat-device cpu --amp bf16 \
    --targets soft --memmap outputs/shared_cache/tokens16_full_std.npy --seed $s || { echo TRAIN_FAILED; exit 1; }
done
echo ">>> $(date +%T) EXP-109 eval"; G $PY scripts/exp109_eval.py || echo EVAL109_FAILED
echo EXP109_DONE
# ---- EXP-108
echo ">>> $(date +%T) EXP-108 smoke"
G $PY scripts/train_pathflow.py --exp EXP-099_p3_smoke --features moge2_vitl:normals16~and --epochs 1 --eval-every 1 || { echo SMOKE108_FAILED; exit 1; }
for arm in "" "--no-geo"; do for s in 0 1 2; do for f in dinov3_vitl16:tokens8 moge2_vitl:normals16~and; do
  echo ">>> $(date +%T) EXP-108 train $f seed=$s $arm"; G $PY scripts/train_pathflow.py --features $f --seed $s $arm || { echo TRAIN108_FAILED; exit 1; }
done; done; done
echo ">>> $(date +%T) EXP-108 eval"; G $PY scripts/exp108_eval.py || echo EVAL108_FAILED
echo CHAIN108_DONE
# ---- EXP-107 jobs (last)
echo ">>> $(date +%T) EXP-107 sweep rerun nfe=10 m=8"
G $PY scripts/exp04x_fused_eval.py --exp EXP-042_image_conditioned_so3_posterior --run-exp EXP-107_paper_visuals_metrics_efficiency --seeds 0 --fused-only --feat-cpu --queries 1024 --nfe 10 --m 8 || echo SWEEP8_FAILED
echo ">>> $(date +%T) EXP-107 figures"; G $PY scripts/exp107_figures.py || echo FIGURES_FAILED
echo ">>> $(date +%T) EXP-107 timing"; G $PY scripts/exp107_timing.py || echo TIMING_FAILED
echo MASTER_DONE
