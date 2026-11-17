#!/usr/bin/env bash
# DEC-030 queue (2026-11-17): EXP-131, the locked recipe on all objects of T-LESS (30) and YCB-V (21); SYMSOL-T is not rerun.
# Waits until the other project's SymSol training process (PID $WAIT_PID) has exited, then per dataset (T-LESS, then YCB-V):
# compact extraction to hdd3 -> per-object A2 trees (2 branches) + translation blocks, seed-major (all objects at seed 0 first,
# the DEC-012 cut order) -> evaluation (EXP-129/130 configurations, --save-preds) -> full BOP AR (VSD, MSSD, MSPD) and LL.
# Resumable: extraction skips finished caches, training skips finished runs (run_done.py). A failed dataset block does not stop
# the next one. Pre-registration: outputs/reports/EXP-131_all_objects.md.
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python; CR=/media/kavinder/hdd3/RAM/compact; EXP=EXP-131; EN=all_objects; RD=EXP-131_all_objects
WAIT_PID=${WAIT_PID:-2602866}
G() { GUARD_GPU_GB=${GG:-8} scripts/guarded.sh ${GMEM:-40} -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
R() { local i; for i in 1 2 3; do G "$@" && return 0; echo "RETRY $i $(date +%T)"; sleep 600; done; return 1; }
block() { local name=$1; shift; echo "=== $(date +%T) BLOCK $name"; if "$@"; then echo "=== $(date +%T) BLOCK_DONE $name"; else echo "=== $(date +%T) BLOCK_FAILED $name"; fi; }
SD="dinov3_vitl16:tokens8 64x1024"; SM="moge2_vitl:normals16~and 256x4"
YROOT=$($PY -c 'import yaml; print(yaml.safe_load(open("configs/paths.yaml"))["ycbv_root"])')
declare -A OBJS=( [tlessall]="$(seq -s' ' 1 30)" [ycbvall]="$(tr '\n' ' ' < "$YROOT/image_sets/classes.txt")" )
declare -A ROOT=( [tlessall]=tless_root [ycbvall]=ycbv_root [ycbvsynall]=ycbvsynall_root )
declare -A KEYC=( [tlessall]=obj_id,inst_id [ycbvall]=obj,inst_id [ycbvsynall]=obj,inst_id )
declare -A EXTRA=( [tlessall]="" [ycbvall]="--extra-train ycbvsynall" )
declare -A EVAL=( [tlessall]="--pose-t orbit_mean" [ycbvall]="--pose-t greedy" )
declare -A ARCFG=( [tlessall]="--variant PoE --mode orbit_product" [ycbvall]="--variant DINO --mode greedy" )
declare -A LLV=( [tlessall]=PoE [ycbvall]=DINO )

X() { local ds=$1 man=$2 tag=$3 bb; shift 3
  for bb in dinov3_vitl16 moge2_vitl; do echo ">>> $(date +%T) extract $tag $bb"
    R $PY scripts/extract_compact.py --manifest outputs/data_manifests/$man --root-key ${ROOT[$ds]} --tag $tag --backbone $bb --rgb \
      --key-cols ${KEYC[$ds]} --roll-homography --cache-root $CR "$@" || return 1; done
  R $PY scripts/extract_compact.py --manifest outputs/data_manifests/$man --root-key ${ROOT[$ds]} --tag $tag --backbone consensus --rgb --key-cols ${KEYC[$ds]} --cache-root $CR || return 1; }

extract() { local ds=$1 k d
  for d in $ds $([[ $ds == ycbvall ]] && echo ycbvsynall); do
    X $d exp064_${d}_all.csv exp064_${d}_all || return 1
    X $d exp064_${d}_train.csv exp064_${d}_aug1 --rot-random-seed 1 --photo-seed 1 || return 1
    X $d exp064_${d}_train.csv exp064_${d}_aug2 --rot-random-seed 2 --photo-seed 2 || return 1
  done
  for k in 90 180 270; do X $ds exp064_${ds}_eval.csv exp064_${ds}_eval_r$k --rot-deg $k || return 1; done; }

nan_in() { grep -qiE "val_sym_mean (nan|inf)" "outputs/experiments/$1/logs/train.log" 2>/dev/null; }

one() { local ds=$1 o=$2 s=$3 feat=$4 shape=$5 d0 dF init fl=""
  # tree (A2), NaN rule: retrain once with --sd-floor 1e-2
  if ! $PY scripts/run_done.py $RD dataset=$ds obj=$o features=$feat seed=$s label=multi trans=none; then
    echo ">>> $(date +%T) EXP-131 tree $ds $o $feat seed=$s"
    R $PY scripts/train_xobj.py --dataset $ds --obj $o --features $feat --token-shape $shape --seed $s --label multi ${EXTRA[$ds]} --exp $EXP --exp-name $EN || return 1
  fi
  d0=$($PY scripts/find_run.py $RD dataset=$ds obj=$o features=$feat seed=$s label=multi trans=none sd_floor=0.0) || return 1
  if nan_in $d0; then
    if ! dF=$($PY scripts/find_run.py $RD dataset=$ds obj=$o features=$feat seed=$s label=multi trans=none sd_floor=0.01); then
      echo ">>> $(date +%T) EXP-131 NaN rule: retrain $ds $o $feat seed=$s with --sd-floor 1e-2"
      R $PY scripts/train_xobj.py --dataset $ds --obj $o --features $feat --token-shape $shape --seed $s --label multi ${EXTRA[$ds]} --sd-floor 1e-2 --exp $EXP --exp-name $EN || return 1
      dF=$($PY scripts/find_run.py $RD dataset=$ds obj=$o features=$feat seed=$s label=multi trans=none sd_floor=0.01) || return 1
    fi
    init=$dF; fl="--sd-floor 1e-2"
  else init=$d0; fi
  # translation block on the frozen tree (EXP-126)
  $PY scripts/run_done.py $RD dataset=$ds obj=$o features=$feat seed=$s label=multi trans=frozen && return 0
  echo ">>> $(date +%T) EXP-131 trans $ds $o $feat seed=$s on $init"
  R $PY scripts/train_xobj.py --dataset $ds --obj $o --features $feat --token-shape $shape --seed $s --label multi ${EXTRA[$ds]} $fl \
    --trans frozen --init-from $init --exp $EXP --exp-name $EN; }

dataset() { local ds=$1 s o f ok=0
  extract $ds || return 1
  for s in 0 1 2; do for o in ${OBJS[$ds]}; do for f in "$SD" "$SM"; do set -- $f
    one $ds $o $s $1 $2 || return 1
  done; done; done
  R $PY scripts/exp064_eval.py --dataset $ds --exp $EXP --exp-name $EN --runs-dir $RD --filter trans=frozen --save-preds --modes greedy,orbit_product \
    --pose-t-variant DINO ${EVAL[$ds]} --tag F || return 1
  GG=0 R $PY scripts/exp131_ar.py --dataset $ds --src-exp $RD ${ARCFG[$ds]} --workers 6 || ok=1
  R $PY scripts/exp131_ll.py --dataset $ds --runs-dir $RD --filter trans=frozen --variant ${LLV[$ds]} || ok=1
  return $ok; }

echo "=== $(date +%T) queue21 waiting for PID $WAIT_PID (other project's SymSol training) to exit"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 300; done
echo "=== $(date +%T) PID $WAIT_PID has exited; queue21 starts"
for ds in ${EXP131_DATASETS:-tlessall ycbvall}; do block EXP-131-$ds dataset $ds; done
echo "=== $(date +%T) QUEUE21_DONE"
