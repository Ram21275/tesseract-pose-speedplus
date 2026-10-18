#!/usr/bin/env bash
# EXP-117 resume after queue10 (EXP-116). The PoE teacher was stopped by memguard (RSS 34.2 GB > 34 GB watchdog of the
# lowered 40 GB cap); PoE teachers now run with the 56 GB cap. Finished teachers are kept. Then EXP-064 if ready.
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; CONV=$(cat outputs/experiments/exp109_convention.txt)
G() { scripts/guarded.sh ${GMEM:-56} -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
block() { local name=$1; shift; echo "=== $(date +%T) BLOCK $name"; if "$@"; then echo "=== $(date +%T) BLOCK_DONE $name"; else echo "=== $(date +%T) BLOCK_FAILED $name"; fi; }
while pgrep -f '^bash scripts/run_queue10.sh' >/dev/null; do sleep 60; done
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
if [ -x scripts/run_exp064.sh ]; then block EXP-064 bash scripts/run_exp064.sh; else echo "=== $(date +%T) EXP-064 code not ready"; fi
echo "=== $(date +%T) QUEUE11_DONE"
