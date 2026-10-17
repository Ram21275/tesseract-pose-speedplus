#!/usr/bin/env bash
# DEC-021: after queue7 (EXP-115) -> EXP-116 (smoke, 18 trainings, eval) -> EXP-117 (teachers, 9 students, eval).
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; CONV=$(cat outputs/experiments/exp109_convention.txt)
G() { scripts/guarded.sh ${GMEM:-56} -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
block() { local name=$1; shift; echo "=== $(date +%T) BLOCK $name"; if "$@"; then echo "=== $(date +%T) BLOCK_DONE $name"; else echo "=== $(date +%T) BLOCK_FAILED $name"; fi; }
while pgrep -f '^bash scripts/run_queue7.sh' >/dev/null; do sleep 60; done
A109=tree:EXP-109_real_domain_aug_tokens16_orbit:exp109_train_aug1+exp109_train_aug2
TG=exp109_train_aug1+exp109_train_aug2
b116() { G $PY scripts/train_consistency.py --exp EXP-099_p3_smoke --features moge2_vitl:normals16~and --token-shape 256x4 --mode T \
           --aug-convention $CONV --epochs 1 --eval-every 1 || return 1
  for arm in T C1 C2; do for s in 0 1 2; do for f in "dinov3_vitl16:tokens8 64x1024" "moge2_vitl:normals16~and 256x4"; do set -- $f
    echo ">>> $(date +%T) EXP-116 $arm $1 seed=$s"
    G $PY scripts/train_consistency.py --exp EXP-116_roll_consistent_tree --features $1 --token-shape $2 --mode $arm --aug-convention $CONV --seed $s || return 1
  done; done; done
  G $PY scripts/orbit_eval.py --exp EXP-116_roll_consistent_tree --modes greedy,orbit_product,orbit_vote32 \
    --families $A109,tree:EXP-116_roll_consistent_tree:$TG:mode=T,tree:EXP-116_roll_consistent_tree:$TG:mode=C1,tree:EXP-116_roll_consistent_tree:$TG:mode=C2; }
b117() { mkdir -p outputs/experiments/EXP-117_orbit_distillation
  for v in DINO PoE; do for s in 0 1 2; do echo ">>> $(date +%T) EXP-117 teacher $v seed=$s"; G $PY scripts/exp117_teacher.py --variant $v --seed $s || return 1; done; done
  for s in 0 1 2; do
    for spec in "DINO dinov3_vitl16:tokens8 64x1024" "PoE dinov3_vitl16:tokens8 64x1024" "PoE moge2_vitl:normals16~and 256x4"; do set -- $spec
      echo ">>> $(date +%T) EXP-117 student teacher=$1 $2 seed=$s"
      G $PY scripts/train_consistency.py --exp EXP-117_orbit_distillation --features $2 --token-shape $3 --mode distill --teacher-variant $1 \
        --teacher-targets outputs/experiments/EXP-117_orbit_distillation/teacher_$1_seed$s.npz --aug-convention $CONV --seed $s || return 1
  done; done
  G $PY scripts/orbit_eval.py --exp EXP-117_orbit_distillation --modes greedy,orbit_product,orbit_vote32 --variants DINO \
    --families $A109,tree:EXP-117_orbit_distillation:$TG:teacher_variant=DINO || echo "(DINO-only eval failed; continuing)"
  G $PY scripts/orbit_eval.py --exp EXP-117_orbit_distillation --modes greedy,orbit_product,orbit_vote32 --variants PoE \
    --families $A109,tree:EXP-117_orbit_distillation:$TG:teacher_variant=PoE; }
block EXP-116 b116
block EXP-117 b117
echo "=== $(date +%T) QUEUE8_DONE"
