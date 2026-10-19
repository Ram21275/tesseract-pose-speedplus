#!/usr/bin/env bash
# EXP-064 SYMSOL-T fix (2026-10-19): the 9 MoGe-2 trees (cube, cyl, icosa x seeds 0-2) trained to NaN loss (empty border
# cells in clean renders -> train sd ~1e-6 -> standardized augmented views overflow fp16). Re-train them with --sd-floor 1e-2,
# then re-evaluate SYMSOL-T (exp064_eval picks the newest run per object/seed/branch). Runs after the YCB-V block.
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
G() { scripts/guarded.sh ${GMEM:-40} -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
while pgrep -f '^bash scripts/run_exp064.sh' >/dev/null; do sleep 60; done
echo "=== $(date +%T) BLOCK EXP-064-symsolt-fix"
for s in 0 1 2; do for o in cube cyl icosa; do
  echo ">>> $(date +%T) EXP-064 retrain symsolt $o moge2 seed=$s (sd floor)"
  G $PY scripts/train_xobj.py --dataset symsolt --obj $o --features "moge2_vitl:normals16~and" --token-shape 256x4 --sd-floor 1e-2 --seed $s || echo "RETRAIN_FAILED $o $s"
done; done
G $PY scripts/exp064_eval.py --dataset symsolt && echo "=== $(date +%T) BLOCK_DONE EXP-064-symsolt-fix" || echo "=== $(date +%T) BLOCK_FAILED EXP-064-symsolt-fix"
