#!/usr/bin/env bash
# DEC-032 queue (2026-11-18): EXP-132 Tesseract depth sweep, L = 6 and 7 (L = 5 = the existing trees), DINOv3 branch, A2 labels,
# rotation only (no translation blocks): T-LESS (30 objects) and YCB-V (21) on the EXP-131 data, SYMSOL-T (5 shapes) on the EXP-064 data.
# Waits until PID $WAIT_PID (the matching EXP-131 queue22 process) has exited, then per dataset in $EXP132_DATASETS:
# trees seed-major (seed 0 at L6 and L7 for every object first) -> evaluation at L5 / L6 / L7 with the same command -> LL at L5 / L6 / L7.
# Resumable (run_done.py, depth-aware). Pre-registration: outputs/reports/EXP-132_depth_sweep.md.
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python; EXP=EXP-132; EN=depth_sweep; RD=EXP-132_depth_sweep
WAIT_PID=${WAIT_PID:?set WAIT_PID}
G() { GUARD_GPU_GB=${GG:-8} scripts/guarded.sh ${GMEM:-40} -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
R() { local i; for i in 1 2 3; do G "$@" && return 0; echo "RETRY $i $(date +%T)"; sleep 600; done; return 1; }
block() { local name=$1; shift; echo "=== $(date +%T) BLOCK $name"; if "$@"; then echo "=== $(date +%T) BLOCK_DONE $name"; else echo "=== $(date +%T) BLOCK_FAILED $name"; fi; }
FEAT=dinov3_vitl16:tokens8; SHAPE=64x1024
YROOT=$($PY -c 'import yaml; print(yaml.safe_load(open("configs/paths.yaml"))["ycbv_root"])')
declare -A OBJS=( [tlessall]="$(seq -s' ' 1 30)" [ycbvall]="$(tr '\n' ' ' < "$YROOT/image_sets/classes.txt")" [symsolt]="cone cube cyl icosa tet" )
declare -A EXTRA=( [tlessall]="" [ycbvall]="--extra-train ycbvsynall" [symsolt]="" )
# depth-5 trees: EXP-131 (T-LESS / YCB-V, tree runs) and EXP-106 arm A2 (SYMSOL-T, the paper's SYMSOL-T trees)
declare -A L5DIR=( [tlessall]=EXP-131_all_objects [ycbvall]=EXP-131_all_objects [symsolt]=EXP-106_tless_multipath_tree )
nan_in() { grep -qiE "val_sym_mean (nan|inf)" "outputs/experiments/$1/logs/train.log" 2>/dev/null; }

one() { local ds=$1 o=$2 s=$3 L=$4 d0
  if ! $PY scripts/run_done.py $RD dataset=$ds obj=$o features=$FEAT seed=$s label=multi depth=$L sd_floor=0.0; then
    echo ">>> $(date +%T) EXP-132 tree $ds $o L$L seed=$s"
    R $PY scripts/train_xobj.py --dataset $ds --obj $o --features $FEAT --token-shape $SHAPE --seed $s --label multi --depth $L ${EXTRA[$ds]} --exp $EXP --exp-name $EN || return 1
  fi
  d0=$($PY scripts/find_run.py $RD dataset=$ds obj=$o features=$FEAT seed=$s label=multi depth=$L sd_floor=0.0) || return 1
  if nan_in $d0 && ! $PY scripts/run_done.py $RD dataset=$ds obj=$o features=$FEAT seed=$s label=multi depth=$L sd_floor=0.01; then
    # NaN rule (EXP-064): retrain once with --sd-floor 1e-2; evaluation then takes the newer run
    echo ">>> $(date +%T) EXP-132 NaN rule: retrain $ds $o L$L seed=$s with --sd-floor 1e-2"
    R $PY scripts/train_xobj.py --dataset $ds --obj $o --features $FEAT --token-shape $SHAPE --seed $s --label multi --depth $L ${EXTRA[$ds]} --sd-floor 1e-2 --exp $EXP --exp-name $EN || return 1
  fi; }

dataset() { local ds=$1 s o L ok=0 dir flt
  for s in 0 1 2; do for o in ${OBJS[$ds]}; do for L in 6 7; do one $ds $o $s $L || return 1; done; done; done
  for L in 5 6 7; do
    if [[ $L == 5 ]]; then dir=${L5DIR[$ds]}; flt="label=multi,trans=none"; else dir=$RD; flt="label=multi"; fi
    R $PY scripts/exp064_eval.py --dataset $ds --exp $EXP --exp-name depth_sweep_eval --runs-dir $dir --filter $flt --depth $L \
      --variants DINO --modes greedy,orbit_product --tag L$L || ok=1
    R $PY scripts/exp131_ll.py --dataset $ds --runs-dir $dir --filter $flt --depth $L --variant DINO --out EXP-132:depth_sweep_ll || ok=1
  done
  return $ok; }

echo "=== $(date +%T) queue23 waiting for PID $WAIT_PID to exit"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 300; done
echo "=== $(date +%T) PID $WAIT_PID has exited; queue23 starts"
for ds in ${EXP132_DATASETS:?set EXP132_DATASETS}; do block EXP-132-$ds dataset $ds; done
echo "=== $(date +%T) QUEUE23_DONE"
