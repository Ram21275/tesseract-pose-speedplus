#!/usr/bin/env bash
# After queue11 (EXP-064 -> EXP-117): EXP-118 6-DoF translation block (SPACE-HOP formulation; DEC-022, pre-registered).
# Arms J (joint) and F (frozen EXP-109 arm-A tree), both branches, seeds 0-2, EXP-109 arm-A recipe; then evaluation.
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; CONV=$(cat outputs/experiments/exp109_convention.txt)
G() { scripts/guarded.sh ${GMEM:-56} -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
block() { local name=$1; shift; echo "=== $(date +%T) BLOCK $name"; if "$@"; then echo "=== $(date +%T) BLOCK_DONE $name"; else echo "=== $(date +%T) BLOCK_FAILED $name"; fi; }
while pgrep -f '^bash scripts/run_queue11.sh' >/dev/null; do sleep 60; done
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
echo "=== $(date +%T) QUEUE12_DONE"
