#!/usr/bin/env bash
# DEC-024 queue (2026-10-27): EXP-122 (YCB-V + data_syn composites) -> EXP-106 (A1 / A2 on T-LESS + SYMSOL-T, A0 = EXP-064
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
while pgrep -f 'src/symsol_train.py' >/dev/null; do sleep 300; done
echo "=== $(date +%T) GPU free of the SymSol job; queue13 starts"

exp122() {
  [[ -s outputs/data_manifests/exp064_ycbvsyn_all.csv ]] || return 1
  local x
  for x in "exp064_ycbvsyn_all|exp064_ycbvsyn_all.csv|" "exp064_ycbvsyn_aug1|exp064_ycbvsyn_train.csv|--rot-random-seed 1 --photo-seed 1" \
           "exp064_ycbvsyn_aug2|exp064_ycbvsyn_train.csv|--rot-random-seed 2 --photo-seed 2"; do
    IFS='|' read -r tag man extra <<< "$x"
    for bb in dinov3_vitl16 moge2_vitl; do echo ">>> $(date +%T) extract $tag $bb"
      G $PY scripts/extract_compact.py --manifest outputs/data_manifests/$man --root-key ycbvsyn_root --tag $tag --backbone $bb --rgb --key-cols obj,inst_id --roll-homography $extra || return 1; done
    G $PY scripts/extract_compact.py --manifest outputs/data_manifests/$man --root-key ycbvsyn_root --tag $tag --backbone consensus --rgb --key-cols obj,inst_id || return 1
  done
  for s in 0 1 2; do for o in ${OBJS[ycbv]}; do for f in "$SD" "$SM"; do set -- $f
    echo ">>> $(date +%T) EXP-122 train ycbv $o $1 seed=$s"
    G $PY scripts/train_xobj.py --dataset ycbv --obj $o --features $1 --token-shape $2 --seed $s --extra-train ycbvsyn --exp EXP-122 --exp-name ycbv_synthetic_training || return 1
  done; done; done
  G $PY scripts/exp064_eval.py --dataset ycbv --exp EXP-122 --exp-name ycbv_synthetic_training --runs-dir EXP-122_ycbv_synthetic_training --tag B; }

exp106() { local ds=$1 lab tag ok=0
  G $PY scripts/exp064_eval.py --dataset $ds --exp EXP-106 --exp-name tless_multipath_tree --runs-dir EXP-064_broader_object_validation --tag A0 || ok=1
  for lab in canonical multi; do tag=$([[ $lab == canonical ]] && echo A1 || echo A2)
    for s in 0 1 2; do for o in ${OBJS[$ds]}; do for f in "$SD" "$SM"; do set -- $f
      echo ">>> $(date +%T) EXP-106 $tag train $ds $o $1 seed=$s"
      G $PY scripts/train_xobj.py --dataset $ds --obj $o --features $1 --token-shape $2 --seed $s --label $lab $(floor $ds $1 $o) --exp EXP-106 --exp-name tless_multipath_tree || return 1
    done; done; done
    G $PY scripts/exp064_eval.py --dataset $ds --exp EXP-106 --exp-name tless_multipath_tree --runs-dir EXP-106_tless_multipath_tree --filter label=$lab --tag $tag || ok=1
  done; return $ok; }

exp119() { local ds=$1 o s v
  local sel; sel=$($PY scripts/exp119_base_arm.py $ds) || return 1; eval "$sel"
  echo ">>> EXP-119 $ds base arm $BASE_ARM ($BASE_DIR, filter '$BASE_FILTER', label $BASE_LABEL)"
  for s in 0 1 2; do for o in ${OBJS[$ds]}; do
    for v in PoE DINO; do G $PY scripts/xobj_teacher.py --dataset $ds --obj $o --variant $v --seed $s --runs-dir $BASE_DIR --filter "$BASE_FILTER" || return 1; done
    for spec in "DINO|$SD" "PoE|$SD" "PoE|$SM"; do IFS='|' read -r v f <<< "$spec"; set -- $f
      echo ">>> $(date +%T) EXP-119 student $ds $o $1 teacher=$v seed=$s"
      G $PY scripts/train_xobj.py --dataset $ds --obj $o --features $1 --token-shape $2 --seed $s --label $BASE_LABEL $(floor $ds $1 $o) \
        --teacher outputs/experiments/EXP-119_xobj_orbit_distillation/teacher_${ds}_${o}_${v}_seed${s}.npz --teacher-variant $v --beta 1 \
        --exp EXP-119 --exp-name xobj_orbit_distillation || return 1
    done; done; done
  G $PY scripts/exp064_eval.py --dataset $ds --exp EXP-119 --exp-name xobj_orbit_distillation --runs-dir EXP-119_xobj_orbit_distillation --filter teacher_variant=PoE --variants PoE --tag student_PoE || return 1
  G $PY scripts/exp064_eval.py --dataset $ds --exp EXP-119 --exp-name xobj_orbit_distillation --runs-dir EXP-119_xobj_orbit_distillation --filter teacher_variant=DINO --variants DINO --tag student_DINO; }

block EXP-122 exp122
for ds in tless symsolt; do block EXP-106-$ds exp106 $ds; done
for ds in tless symsolt; do block EXP-119-$ds exp119 $ds; done
echo "=== $(date +%T) QUEUE13_DONE"
