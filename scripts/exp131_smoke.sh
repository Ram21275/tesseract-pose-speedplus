#!/usr/bin/env bash
# EXP-131 smoke test (wiring only, numbers meaningless): tiny copies of the all-object manifests (one symmetric object per
# dataset), every queue21 stage end to end on the small spare GPU: extraction to a hdd3 SMOKE cache -> read back through the
# shared-cache symlinks -> 1-epoch tree (A2) -> translation block -> eval --save-preds -> AR (VSD/MSSD/MSPD) -> LL.
# Runs go to outputs/experiments/SMOKE_exp131 (never pushed); caches to /media/kavinder/hdd3/RAM/SMOKE_exp131.
set -eu; cd "$(dirname "$0")/.."
export TFPOSE_SMOKE=1 CUDA_VISIBLE_DEVICES=${SMOKE_GPU:-1} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GG=0
PY=.venv/bin/python; CR=/media/kavinder/hdd3/RAM/SMOKE_exp131; mkdir -p $CR
G() { GUARD_GPU_GB=0 scripts/guarded.sh 24 -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
$PY - <<'E'
import pandas as pd
D = "outputs/data_manifests/"
def take(ds, obj, oc, n=(48, 16, 16), syn=False):
    a = pd.read_csv(D + f"exp064_{ds}_all.csv"); a = a[a[oc] == obj]
    parts = [a[a.split == "train"].head(n[0])] + ([] if syn else [a[a.split == "validation"].head(n[1]), a[a.split == "test"].head(n[2])])
    s = pd.concat(parts); out = D + f"exp064_{ds}smoke"
    s.to_csv(out + "_all.csv", index=False, float_format="%.9g"); s[s.split == "train"].to_csv(out + "_train.csv", index=False, float_format="%.9g")
    s[s.split != "train"].to_csv(out + "_eval.csv", index=False, float_format="%.9g"); print(ds, obj, s.split.value_counts().to_dict())
take("tlessall", 27, "obj_id"); take("ycbvall", "024_bowl", "obj"); take("ycbvsynall", "024_bowl", "obj", syn=True)
E
declare -A ROOT=( [tlessallsmoke]=tless_root [ycbvallsmoke]=ycbv_root [ycbvsynallsmoke]=ycbvsynall_root )
declare -A KEYC=( [tlessallsmoke]=obj_id,inst_id [ycbvallsmoke]=obj,inst_id [ycbvsynallsmoke]=obj,inst_id )
X() { local ds=$1 man=$2 tag=$3 bb; shift 3
  for bb in dinov3_vitl16 moge2_vitl; do G $PY scripts/extract_compact.py --manifest outputs/data_manifests/$man --root-key ${ROOT[$ds]} --tag $tag --backbone $bb --rgb \
      --key-cols ${KEYC[$ds]} --roll-homography --cache-root $CR --batch 4 --workers 2 "$@"; done
  G $PY scripts/extract_compact.py --manifest outputs/data_manifests/$man --root-key ${ROOT[$ds]} --tag $tag --backbone consensus --rgb --key-cols ${KEYC[$ds]} --cache-root $CR; }
for d in tlessallsmoke ycbvallsmoke ycbvsynallsmoke; do
  X $d exp064_${d}_all.csv exp064_${d}_all
  X $d exp064_${d}_train.csv exp064_${d}_aug1 --rot-random-seed 1 --photo-seed 1
  X $d exp064_${d}_train.csv exp064_${d}_aug2 --rot-random-seed 2 --photo-seed 2
done
for d in tlessallsmoke ycbvallsmoke; do for k in 90 180 270; do X $d exp064_${d}_eval.csv exp064_${d}_eval_r$k --rot-deg $k; done; done
declare -A OBJ=( [tlessallsmoke]=27 [ycbvallsmoke]=024_bowl ); declare -A EXTRA=( [tlessallsmoke]="" [ycbvallsmoke]="--extra-train ycbvsynallsmoke" )
declare -A EV=( [tlessallsmoke]="--pose-t orbit_mean" [ycbvallsmoke]="--pose-t greedy" ); declare -A AR=( [tlessallsmoke]="--variant PoE --mode orbit_product" [ycbvallsmoke]="--variant DINO --mode greedy" )
declare -A LLV=( [tlessallsmoke]=PoE [ycbvallsmoke]=DINO )
for d in tlessallsmoke ycbvallsmoke; do o=${OBJ[$d]}
  for f in "dinov3_vitl16:tokens8 64x1024" "moge2_vitl:normals16~and 256x4"; do set -- $f
    G $PY scripts/train_xobj.py --dataset $d --obj $o --features $1 --token-shape $2 --seed 0 --label multi ${EXTRA[$d]} --epochs 2 --eval-every 1 --exp SMOKE --exp-name exp131
    init=$($PY scripts/find_run.py SMOKE_exp131 dataset=$d obj=$o features=$1 seed=0 label=multi trans=none)
    G $PY scripts/train_xobj.py --dataset $d --obj $o --features $1 --token-shape $2 --seed 0 --label multi ${EXTRA[$d]} --epochs 2 --eval-every 1 --trans frozen --init-from $init --exp SMOKE --exp-name exp131
  done
  G $PY scripts/exp064_eval.py --dataset $d --exp SMOKE --exp-name exp131 --runs-dir SMOKE_exp131 --filter trans=frozen --save-preds --modes greedy,orbit_product --pose-t-variant DINO ${EV[$d]} --tag F --seeds 0
  G $PY scripts/exp131_ar.py --dataset $d --src-exp SMOKE_exp131 ${AR[$d]} --workers 2
  G $PY scripts/exp131_ll.py --dataset $d --runs-dir SMOKE_exp131 --filter trans=frozen --variant ${LLV[$d]} --seeds 0
done
echo SMOKE_OK
