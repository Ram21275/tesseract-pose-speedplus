#!/usr/bin/env bash
# User 2026-10-17: start only after one of the user's 20 GB GPU jobs (liepose cmp.run PID 801420, jepa_probe PID 1184964)
# has finished. Then: EXP-117 (resume; finished teachers kept; lower caps: teacher/eval 40 GB, students 48 GB)
# -> EXP-064 (if scripts/run_exp064.sh exists by then) -> EXP-116. Non-blocking blocks; guarded.sh safety unchanged.
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; CONV=$(cat outputs/experiments/exp109_convention.txt)
G() { scripts/guarded.sh ${GMEM:-56} -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
block() { local name=$1; shift; echo "=== $(date +%T) BLOCK $name"; if "$@"; then echo "=== $(date +%T) BLOCK_DONE $name"; else echo "=== $(date +%T) BLOCK_FAILED $name"; fi; }
echo "=== $(date +%T) waiting for PID 801420 or 1184964 to finish"
while kill -0 801420 2>/dev/null && kill -0 1184964 2>/dev/null; do sleep 60; done
echo "=== $(date +%T) one of the user's 20 GB jobs finished; starting"
A109=tree:EXP-109_real_domain_aug_tokens16_orbit:exp109_train_aug1+exp109_train_aug2; TG=exp109_train_aug1+exp109_train_aug2
D117=outputs/experiments/EXP-117_orbit_distillation
b117() { mkdir -p $D117
  for v in DINO PoE; do for s in 0 1 2; do
    [ -f $D117/teacher_${v}_seed$s.npz ] && { echo "teacher $v seed $s exists"; continue; }
    echo ">>> $(date +%T) EXP-117 teacher $v seed=$s"; GMEM=40 G $PY scripts/exp117_teacher.py --variant $v --seed $s || return 1; done; done
  for s in 0 1 2; do
    for spec in "DINO dinov3_vitl16:tokens8 64x1024" "PoE dinov3_vitl16:tokens8 64x1024" "PoE moge2_vitl:normals16~and 256x4"; do set -- $spec
      echo ">>> $(date +%T) EXP-117 student teacher=$1 $2 seed=$s"
      GMEM=48 G $PY scripts/train_consistency.py --exp EXP-117_orbit_distillation --features $2 --token-shape $3 --mode distill --teacher-variant $1 \
        --teacher-targets $D117/teacher_$1_seed$s.npz --aug-convention $CONV --seed $s || return 1
  done; done
  GMEM=40 G $PY scripts/orbit_eval.py --exp EXP-117_orbit_distillation --modes greedy,orbit_product,orbit_vote32 --variants DINO \
    --families $A109,tree:EXP-117_orbit_distillation:$TG:teacher_variant=DINO || echo "(DINO-only eval failed; continuing)"
  GMEM=40 G $PY scripts/orbit_eval.py --exp EXP-117_orbit_distillation --modes greedy,orbit_product,orbit_vote32 --variants PoE \
    --families $A109,tree:EXP-117_orbit_distillation:$TG:teacher_variant=PoE; }
b116() { for arm in T C1 C2; do for s in 0 1 2; do for f in "dinov3_vitl16:tokens8 64x1024" "moge2_vitl:normals16~and 256x4"; do set -- $f
    echo ">>> $(date +%T) EXP-116 $arm $1 seed=$s"
    GMEM=48 G $PY scripts/train_consistency.py --exp EXP-116_roll_consistent_tree --features $1 --token-shape $2 --mode $arm --aug-convention $CONV --seed $s || return 1
  done; done; done
  GMEM=40 G $PY scripts/orbit_eval.py --exp EXP-116_roll_consistent_tree --modes greedy,orbit_product,orbit_vote32 \
    --families $A109,tree:EXP-116_roll_consistent_tree:$TG:mode=T,tree:EXP-116_roll_consistent_tree:$TG:mode=C1,tree:EXP-116_roll_consistent_tree:$TG:mode=C2; }
block EXP-117 b117
if [ -x scripts/run_exp064.sh ]; then block EXP-064 bash scripts/run_exp064.sh; else echo "=== $(date +%T) EXP-064 code not ready; skipped for now"; fi
block EXP-116 b116
echo "=== $(date +%T) QUEUE10_DONE"
