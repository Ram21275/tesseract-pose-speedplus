#!/usr/bin/env bash
# After EXP-064 (queue11 wrapper stopped 2026-10-19; user: "do 117 after 118"): EXP-118 6-DoF translation block (SPACE-HOP formulation; DEC-022, pre-registered).
# Arms J (joint) and F (frozen EXP-109 arm-A tree), both branches, seeds 0-2, EXP-109 arm-A recipe; then evaluation.
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; CONV=$(cat outputs/experiments/exp109_convention.txt)
G() { scripts/guarded.sh ${GMEM:-56} -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
block() { local name=$1; shift; echo "=== $(date +%T) BLOCK $name"; if "$@"; then echo "=== $(date +%T) BLOCK_DONE $name"; else echo "=== $(date +%T) BLOCK_FAILED $name"; fi; }
sleep 120; while pgrep -f '^bash scripts/run_exp064' >/dev/null; do sleep 60; done   # run_exp064.sh and run_exp064_symfix.sh
A=outputs/experiments/EXP-109_real_domain_aug_tokens16_orbit
declare -A AD=( [0]=$A/RUN-20261015-180909-seed0 [1]=$A/RUN-20261015-190709-seed1 [2]=$A/RUN-20261015-192852-seed2 )
declare -A AM=( [0]=$A/RUN-20261015-182205-seed0 [1]=$A/RUN-20261015-192302-seed1 [2]=$A/RUN-20261015-194214-seed2 )
b118() { for s in 0 1 2; do
    for spec in "dinov3_vitl16:tokens8 64x1024 ${AD[$s]}" "moge2_vitl:normals16~and 256x4 ${AM[$s]}"; do set -- $spec
      for arm in joint frozen; do
        echo ">>> $(date +%T) EXP-118 $arm $1 seed=$s"
        G $PY scripts/train_probe_p3.py --exp EXP-118_translation_6dof --features $1 --decoder transformer --token-shape $2 --feat-device cpu \
          --amp bf16 --targets soft --aug-tags exp109_train_aug1,exp109_train_aug2 --aug-convention $CONV --trans $arm \
          $( [ $arm = frozen ] && echo --init-from $3 ) --seed $s || return 1
  done; done; done
  G $PY scripts/exp118_eval.py; }
block EXP-118 b118
A109=tree:EXP-109_real_domain_aug_tokens16_orbit:exp109_train_aug1+exp109_train_aug2; TG=exp109_train_aug1+exp109_train_aug2
D117=outputs/experiments/EXP-117_orbit_distillation
b117() { for v in DINO PoE; do for s in 0 1 2; do
    [ -f $D117/teacher_${v}_seed$s.npz ] && { echo "teacher $v seed $s exists"; continue; }
    echo ">>> $(date +%T) EXP-117 teacher $v seed=$s"; G $PY scripts/exp117_teacher.py --variant $v --seed $s || return 1; done; done
  for s in 0 1 2; do
    for spec in "DINO dinov3_vitl16:tokens8 64x1024" "PoE dinov3_vitl16:tokens8 64x1024" "PoE moge2_vitl:normals16~and 256x4"; do set -- $spec
      echo ">>> $(date +%T) EXP-117 student teacher=$1 $2 seed=$s"
      GMEM=48 G $PY scripts/train_consistency.py --exp EXP-117_orbit_distillation --features $2 --token-shape $3 --mode distill --teacher-variant $1 \
        --teacher-targets $D117/teacher_$1_seed$s.npz --aug-convention $CONV --seed $s || return 1
  done; done
  G $PY scripts/orbit_eval.py --exp EXP-117_orbit_distillation --modes greedy,orbit_product,orbit_vote32 --variants DINO \
    --families $A109,tree:EXP-117_orbit_distillation:$TG:teacher_variant=DINO || echo "(DINO-only eval failed; continuing)"
  G $PY scripts/orbit_eval.py --exp EXP-117_orbit_distillation --modes greedy,orbit_product,orbit_vote32 --variants PoE \
    --families $A109,tree:EXP-117_orbit_distillation:$TG:teacher_variant=PoE; }
block EXP-117 b117
echo "=== $(date +%T) QUEUE12_DONE (EXP-118, EXP-117)"
