#!/usr/bin/env bash
# EXP-064 (locked recipe, DEC-018 / DEC-021 amendments): convention gate -> per dataset (T-LESS, SYMSOL-T, YCB-V):
# compact extraction (clean, 2 augmented train copies, orbit views of val+test) -> per-object trees (2 branches x 3 seeds)
# -> evaluation. A failed dataset block does not stop the others. Caches on /media/kavinder/hdd2/RAM (compact_cache_root).
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
G() { scripts/guarded.sh ${GMEM:-40} -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
block() { local name=$1; shift; echo "=== $(date +%T) BLOCK $name"; if "$@"; then echo "=== $(date +%T) BLOCK_DONE $name"; else echo "=== $(date +%T) BLOCK_FAILED $name"; fi; }
GUARD_GPU_GB=0 G $PY scripts/exp064_convention.py || { echo "CONVENTION GATE FAILED"; exit 1; }
grep -q false outputs/experiments/exp064_convention.json && { echo "CONVENTION NOT VERIFIED"; cat outputs/experiments/exp064_convention.json; exit 1; }
declare -A ROOT=( [tless]=tless_root [ycbv]=ycbv_root [symsolt]=symsolt_root )
declare -A KEYC=( [tless]=obj_id,inst_id [ycbv]=obj,inst_id [symsolt]= )
declare -A OBJS=( [tless]="1 4 5 11 18 21 27" [ycbv]="003_cracker_box 006_mustard_bottle 011_banana 025_mug 035_power_drill 037_scissors" [symsolt]="tet cube icosa cone cyl" )
X() { local ds=$1 man=$2 tag=$3; shift 3; local hom=""; [ "$ds" != symsolt ] && hom="--roll-homography"
  for bb in dinov3_vitl16 moge2_vitl; do echo ">>> $(date +%T) extract $tag $bb"
    G $PY scripts/extract_compact.py --manifest outputs/data_manifests/$man --root-key ${ROOT[$ds]} --tag $tag --backbone $bb --rgb \
      ${KEYC[$ds]:+--key-cols ${KEYC[$ds]}} $hom "$@" || return 1; done
  G $PY scripts/extract_compact.py --manifest outputs/data_manifests/$man --root-key ${ROOT[$ds]} --tag $tag --backbone consensus --rgb ${KEYC[$ds]:+--key-cols ${KEYC[$ds]}} || return 1; }
dataset() { local ds=$1
  X $ds exp064_${ds}_all.csv exp064_${ds}_all || return 1
  X $ds exp064_${ds}_train.csv exp064_${ds}_aug1 --rot-random-seed 1 --photo-seed 1 || return 1
  X $ds exp064_${ds}_train.csv exp064_${ds}_aug2 --rot-random-seed 2 --photo-seed 2 || return 1
  for k in 90 180 270; do X $ds exp064_${ds}_eval.csv exp064_${ds}_eval_r$k --rot-deg $k || return 1; done
  for s in 0 1 2; do for o in ${OBJS[$ds]}; do for f in "dinov3_vitl16:tokens8 64x1024" "moge2_vitl:normals16~and 256x4"; do set -- $f
    echo ">>> $(date +%T) EXP-064 train $ds $o $1 seed=$s"
    G $PY scripts/train_xobj.py --dataset $ds --obj $o --features $1 --token-shape $2 --seed $s || return 1
  done; done; done
  G $PY scripts/exp064_eval.py --dataset $ds; }
for ds in ${EXP064_DATASETS:-tless symsolt ycbv}; do block EXP-064-$ds dataset $ds; done   # 2026-10-19: YCB-V re-run after the manifest path fix (.png -> .jpg)
echo "=== $(date +%T) EXP064_DONE"
