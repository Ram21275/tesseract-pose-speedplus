#!/usr/bin/env bash
# Resume of run_queue15.sh after the 2026-10-30 ~12:05 crash: EXP-124 and EXP-126 ycbv/tless are complete; finish EXP-126 symsolt
# (completed blocks are skipped by run_done), then remove QUEUE_PAUSE and write the QUEUE15_DONE marker (appended to queue15.log).
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GUARD_IGNORE_PAUSE=1
PY=.venv/bin/python; PAUSE=outputs/experiments/QUEUE_PAUSE
G() { GUARD_GPU_GB=${GG:-8} scripts/guarded.sh ${GMEM:-40} -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
R() { local i; for i in 1 2 3; do G "$@" && return 0; echo "RETRY $i $(date +%T)"; sleep 600; done; return 1; }
block() { local name=$1; shift; echo "=== $(date +%T) BLOCK $name"; if "$@"; then echo "=== $(date +%T) BLOCK_DONE $name"; else echo "=== $(date +%T) BLOCK_FAILED $name"; fi; }
SD="dinov3_vitl16:tokens8 64x1024"; SM="moge2_vitl:normals16~and 256x4"
declare -A OBJS=( [tless]="1 4 5 11 18 21 27" [ycbv]="003_cracker_box 006_mustard_bottle 011_banana 025_mug 035_power_drill 037_scissors" [symsolt]="tet cube icosa cone cyl" )
floor() { [[ $1 == symsolt && $2 == moge2_vitl:* && " cube cyl icosa " == *" $3 "* ]] && echo "--sd-floor 1e-2"; }
exp126() { local ds=$1 bdir bfilt extra="" s o init
  if [[ $ds == ycbv ]]; then bdir=EXP-122_ycbv_synthetic_training; bfilt=""; extra="--extra-train ycbvsyn"
  else local sel; sel=$($PY scripts/exp119_base_arm.py $ds) || return 1; eval "$sel"; bdir=$BASE_DIR; bfilt=$BASE_FILTER; fi
  echo ">>> EXP-126 $ds base trees: $bdir filter '$bfilt'"
  for s in 0 1 2; do for o in ${OBJS[$ds]}; do for f in "$SD" "$SM"; do set -- $f
    $PY scripts/run_done.py EXP-126_translation_xobj dataset=$ds obj=$o features=$1 seed=$s trans=frozen && continue
    init=$($PY scripts/find_run.py $bdir dataset=$ds obj=$o features=$1 seed=$s ${bfilt//,/ }) || return 1
    echo ">>> $(date +%T) EXP-126 trans $ds $o $1 seed=$s on $init"
    R $PY scripts/train_xobj.py --dataset $ds --obj $o --features $1 --token-shape $2 --seed $s $extra $(floor $ds $1 $o) \
      --trans frozen --init-from $init --exp EXP-126 --exp-name translation_xobj || return 1
  done; done; done
  R $PY scripts/exp064_eval.py --dataset $ds --exp EXP-126 --exp-name translation_xobj --runs-dir EXP-126_translation_xobj --filter trans=frozen --tag F; }
block EXP-126-symsolt exp126 symsolt
rm -f $PAUSE; echo "=== $(date +%T) QUEUE_PAUSE removed; queue14 resumes"
echo "=== $(date +%T) QUEUE15_DONE"
