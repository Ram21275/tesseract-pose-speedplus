#!/usr/bin/env bash
# Resume of run_phase4.sh after its controller was stopped (2026-10-10 03:2x) while EXP-041 MoGe seed 0 was training.
# Waits for that job, then runs the remaining Phase-4 steps unchanged (same commands as run_phase4.sh).
set -u; cd "$(dirname "$0")/.."
G=scripts/guarded.sh; PY=.venv/bin/python; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ERR=outputs/experiments/EXP-040_tangent_residual_predicted_leaf/RUN-20261010-015411-seed0/tables/poe_synthval_greedy_err.csv
step() { echo ">>> $(date +%T) $*"; "$@" 2>&1 | grep -v -i warn | grep -E "seed|RUN|Error|Traceback|memguard|OutOfMemory|^fused|^\|" | tail -30; rc=${PIPESTATUS[0]}; echo "EXIT_CODE=$rc"; [ $rc -eq 0 ] || { echo CHAIN_FAILED; exit $rc; }; }
while pgrep -f '[t]rain_phase4.py --exp EXP-041_cell_conditioned_rfm --features moge2_vitl:normals16~and' >/dev/null; do sleep 20; done
echo ">>> $(date +%T) EXP-041 MoGe seed 0 finished: $(cat outputs/experiments/EXP-041_cell_conditioned_rfm/RUN-20261010-030942-seed0/status.json | grep -o '"state": "[a-z]*"')"
L41="--mode flow --base local --sigma-b-deg 3"
for s in 1 2; do for f in dinov3_vitl16:tokens8 moge2_vitl:normals16~and; do
  step $G 32 -- $PY scripts/train_phase4.py --exp EXP-041_cell_conditioned_rfm --features $f $L41 --centre mix --err-dist $ERR --seed $s
done; done
step $G 32 -- $PY scripts/exp04x_fused_eval.py --exp EXP-041_cell_conditioned_rfm
for s in 0 1 2; do for f in dinov3_vitl16:tokens8 moge2_vitl:normals16~and; do
  step $G 32 -- $PY scripts/train_phase4.py --exp EXP-042_image_conditioned_so3_posterior --features $f --mode flow --base uniform --centre mix --err-dist $ERR --seed $s
done; done
step $G 32 -- $PY scripts/exp04x_fused_eval.py --exp EXP-042_image_conditioned_so3_posterior --save-samples
echo CHAIN_DONE
