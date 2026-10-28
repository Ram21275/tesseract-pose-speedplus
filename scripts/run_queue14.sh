#!/usr/bin/env bash
# DEC-024 resume queue (copy of queue13 helpers): EXP-122 (YCB-V + data_syn composites) -> EXP-106 (A1 / A2 on T-LESS + SYMSOL-T, A0 = EXP-064
# trees re-evaluated by the same evaluator) -> EXP-119 (orbit distillation on the EXP-106 base arm). GPU stages wait until
# the SPACE-HOP SymSol training (another project's job) has finished. A failed block does not stop later independent blocks.
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
G() { scripts/guarded.sh ${GMEM:-40} -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
block() { local name=$1; shift; echo "=== $(date +%T) BLOCK $name"; if "$@"; then echo "=== $(date +%T) BLOCK_DONE $name"; else echo "=== $(date +%T) BLOCK_FAILED $name"; fi; }
SD="dinov3_vitl16:tokens8 64x1024"; SM="moge2_vitl:normals16~and 256x4"
declare -A OBJS=( [tless]="1 4 5 11 18 21 27" [ycbv]="003_cracker_box 006_mustard_bottle 011_banana 025_mug 035_power_drill 037_scissors" [symsolt]="tet cube icosa cone cyl" )
floor() { [[ $1 == symsolt && $2 == moge2_vitl:* && " cube cyl icosa " == *" $3 "* ]] && echo "--sd-floor 1e-2"; }   # as EXP-064 A0

# DEC-024 resume (2026-10-28): queue13 died after the EXP-106 SYMSOL-T A2 trainings (tmux session gone ~08:50); EXP-106
# T-LESS A2 had failed at obj 5 / MoGe-2 / seed 1 with CUDA OOM (another project's job took the GPU). Resume: skip finished
# runs (scripts/run_done.py), retry a failed training up to 3 times 10 min apart, then the evaluations and EXP-119.
R() { local i; for i in 1 2 3; do G "$@" && return 0; echo "RETRY $i $(date +%T)"; sleep 600; done; return 1; }
done_() { $PY scripts/run_done.py "$@"; }

exp106_rest() { local ds=$1 lab=multi tag=A2 s o
  for s in 0 1 2; do for o in ${OBJS[$ds]}; do for f in "$SD" "$SM"; do set -- $f
    done_ EXP-106_tless_multipath_tree dataset=$ds obj=$o features=$1 seed=$s label=$lab && continue
    echo ">>> $(date +%T) EXP-106 $tag train $ds $o $1 seed=$s"
    R $PY scripts/train_xobj.py --dataset $ds --obj $o --features $1 --token-shape $2 --seed $s --label $lab $(floor $ds $1 $o) --exp EXP-106 --exp-name tless_multipath_tree || return 1
  done; done; done
  R $PY scripts/exp064_eval.py --dataset $ds --exp EXP-106 --exp-name tless_multipath_tree --runs-dir EXP-106_tless_multipath_tree --filter label=$lab --tag $tag; }

exp119() { local ds=$1 o s v sel
  sel=$($PY scripts/exp119_base_arm.py $ds) || return 1; eval "$sel"
  echo ">>> EXP-119 $ds base arm $BASE_ARM ($BASE_DIR, filter '$BASE_FILTER', label $BASE_LABEL)"
  for s in 0 1 2; do for o in ${OBJS[$ds]}; do
    for v in PoE DINO; do [[ -s outputs/experiments/EXP-119_xobj_orbit_distillation/teacher_${ds}_${o}_${v}_seed${s}.npz ]] || \
      R $PY scripts/xobj_teacher.py --dataset $ds --obj $o --variant $v --seed $s --runs-dir $BASE_DIR --filter "$BASE_FILTER" || return 1; done
    for spec in "DINO|$SD" "PoE|$SD" "PoE|$SM"; do IFS='|' read -r v f <<< "$spec"; set -- $f
      done_ EXP-119_xobj_orbit_distillation dataset=$ds obj=$o features=$1 seed=$s teacher_variant=$v && continue
      echo ">>> $(date +%T) EXP-119 student $ds $o $1 teacher=$v seed=$s"
      R $PY scripts/train_xobj.py --dataset $ds --obj $o --features $1 --token-shape $2 --seed $s --label $BASE_LABEL $(floor $ds $1 $o) \
        --teacher outputs/experiments/EXP-119_xobj_orbit_distillation/teacher_${ds}_${o}_${v}_seed${s}.npz --teacher-variant $v --beta 1 \
        --exp EXP-119 --exp-name xobj_orbit_distillation || return 1
    done; done; done
  R $PY scripts/exp064_eval.py --dataset $ds --exp EXP-119 --exp-name xobj_orbit_distillation --runs-dir EXP-119_xobj_orbit_distillation --filter teacher_variant=PoE --variants PoE --tag student_PoE || return 1
  R $PY scripts/exp064_eval.py --dataset $ds --exp EXP-119 --exp-name xobj_orbit_distillation --runs-dir EXP-119_xobj_orbit_distillation --filter teacher_variant=DINO --variants DINO --tag student_DINO; }

echo "=== $(date +%T) queue14 starts"
block EXP-106-symsolt-A2eval exp106_rest symsolt
block EXP-106-tless-A2 exp106_rest tless
for ds in symsolt tless; do block EXP-119-$ds exp119 $ds; done
echo "=== $(date +%T) QUEUE14_DONE"
