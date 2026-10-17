#!/usr/bin/env bash
# EXP-116 retry (first T run stopped by memguard at 14:30 on 2026-10-17: system available RAM hit the 16 GB floor under
# other users' load; trainer now avoids a 6.3 GB duplicate). Runs after queue8 (EXP-117). Protocol unchanged.
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; CONV=$(cat outputs/experiments/exp109_convention.txt)
G() { scripts/guarded.sh ${GMEM:-56} -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
block() { local name=$1; shift; echo "=== $(date +%T) BLOCK $name"; if "$@"; then echo "=== $(date +%T) BLOCK_DONE $name"; else echo "=== $(date +%T) BLOCK_FAILED $name"; fi; }
while pgrep -f '^bash scripts/run_queue8.sh' >/dev/null; do sleep 60; done
A109=tree:EXP-109_real_domain_aug_tokens16_orbit:exp109_train_aug1+exp109_train_aug2; TG=exp109_train_aug1+exp109_train_aug2
b116() { for arm in T C1 C2; do for s in 0 1 2; do for f in "dinov3_vitl16:tokens8 64x1024" "moge2_vitl:normals16~and 256x4"; do set -- $f
    echo ">>> $(date +%T) EXP-116 $arm $1 seed=$s"
    G $PY scripts/train_consistency.py --exp EXP-116_roll_consistent_tree --features $1 --token-shape $2 --mode $arm --aug-convention $CONV --seed $s || return 1
  done; done; done
  G $PY scripts/orbit_eval.py --exp EXP-116_roll_consistent_tree --modes greedy,orbit_product,orbit_vote32 \
    --families $A109,tree:EXP-116_roll_consistent_tree:$TG:mode=T,tree:EXP-116_roll_consistent_tree:$TG:mode=C1,tree:EXP-116_roll_consistent_tree:$TG:mode=C2; }
block EXP-116 b116
echo "=== $(date +%T) QUEUE9_DONE"
