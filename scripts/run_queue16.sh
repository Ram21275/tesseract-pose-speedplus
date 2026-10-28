#!/usr/bin/env bash
# DEC-026 / EXP-127: hierarchy vs flat over the same Tesseract leaves vs flat over a balanced Hopf grid (locked EXP-109 recipe).
# Starts only after the already-authorized queues have finished (QUEUE15_DONE and QUEUE14_DONE markers in their logs), never
# touches QUEUE_PAUSE (guarded.sh honours it). One GPU job at a time; every job through guarded.sh; completed runs are skipped.
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; X=EXP-127_hierarchy_vs_flat_same_grid; TAGS=exp109_train_aug1+exp109_train_aug2
G() { GUARD_GPU_GB=${GG:-8} scripts/guarded.sh ${GMEM:-56} -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
R() { local i; for i in 1 2 3; do G "$@" && return 0; echo "RETRY $i $(date +%T)"; sleep 600; done; return 1; }
block() { local name=$1; shift; echo "=== $(date +%T) BLOCK $name"; if "$@"; then echo "=== $(date +%T) BLOCK_DONE $name"; else echo "=== $(date +%T) BLOCK_FAILED $name"; return 1; fi; }
until grep -q QUEUE15_DONE outputs/experiments/queue15.log 2>/dev/null && grep -q QUEUE14_DONE outputs/experiments/queue14.log 2>/dev/null; do sleep 120; done
echo "=== $(date +%T) queue16 starts (queue14 and queue15 done)"

quant() { R $PY scripts/exp127_quant.py; }
smoke() { R $PY scripts/train_flat_tf.py --exp SMOKE_exp127 --grid tesseract_L5 --features moge2_vitl:normals16~and --token-shape 256x4 \
            --aug-convention pre- --epochs 2 --eval-every 1 --seed 0; }
train() { local s g f
  for s in 0 1 2; do for g in tesseract_L5 hopf1771x74; do for f in "dinov3_vitl16:tokens8 64x1024" "moge2_vitl:normals16~and 256x4"; do set -- $f
    $PY scripts/run_done.py $X features=$1 seed=$s grid=$g && continue
    echo ">>> $(date +%T) EXP-127 flat $g $1 seed=$s"
    R $PY scripts/train_flat_tf.py --exp $X --grid $g --features $1 --token-shape $2 --aug-convention pre- --seed $s || return 1
  done; done; done; }
evaluate() { R $PY scripts/orbit_eval.py --exp $X --modes greedy,orbit_product,orbit_vote32 --save-preds \
  --families tree:EXP-109_real_domain_aug_tokens16_orbit:$TAGS,flat:$X:$TAGS:grid=tesseract_L5,flat:$X:$TAGS:grid=hopf1771x74; }
timing() { R $PY scripts/exp127_timing.py; }
sources() { { echo "# EXP-127 source checkpoints (sha256  path)"
  for d in outputs/experiments/EXP-109_real_domain_aug_tokens16_orbit/RUN-* outputs/experiments/$X/RUN-*; do
    [[ -f $d/checkpoints/best.pt ]] && grep -qE 'tokens8|normals16~and' $d/config.yaml && grep -q 'exp109_train_aug1,exp109_train_aug2' $d/config.yaml && sha256sum $d/checkpoints/best.pt
  done; cp outputs/experiments/exp109_convention.txt outputs/experiments/$X/exp109_convention.txt; } > outputs/experiments/$X/source_checkpoints.txt; }

block EXP-127-smoke smoke && block EXP-127-train train && block EXP-127-eval evaluate && block EXP-127-timing timing && block EXP-127-sources sources
echo "=== $(date +%T) QUEUE16_DONE"
