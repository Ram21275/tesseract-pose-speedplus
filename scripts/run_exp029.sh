#!/usr/bin/env bash
# EXP-029 (DEC-007): waits for the Phase-4 chain (run_phase4_resume.sh) to exit, then trains arms 2/4/5 x 3 seeds
# with the EXP-036 recipe and evaluates all arms. One guarded job at a time. Arm 1 reuses the EXP-036 DINO runs.
set -u; cd "$(dirname "$0")/.."
G=scripts/guarded.sh; PY=.venv/bin/python; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { echo ">>> $(date +%T) $*"; "$@" 2>&1 | grep -v -i warn | grep -E "seed|RUN|Error|Traceback|memguard|OutOfMemory|^\|" | tail -40; rc=${PIPESTATUS[0]}; echo "EXIT_CODE=$rc"; [ $rc -eq 0 ] || { echo CHAIN_FAILED; exit $rc; }; }
echo ">>> $(date +%T) waiting for the Phase-4 chain to finish"
while pgrep -f '[r]un_phase4_resume.sh' >/dev/null; do sleep 60; done
echo ">>> $(date +%T) Phase-4 chain exited; last line: $(tail -1 outputs/experiments/EXP-040_tangent_residual_predicted_leaf/phase4_chain.log)"
R="--exp EXP-029_panst3r_inspired_fusion --decoder transformer --targets soft --amp bf16 --feat-device gpu"
for s in 0 1 2; do
  step $G 40 -- $PY scripts/train_probe_p3.py $R --features "moge2_vitl:normals8~and" --token-shape 64x4 --seed $s
  step $G 40 -- $PY scripts/train_probe_p3.py $R --features "dinov3_vitl16:tokens8|moge2_vitl:normals8~and" --token-shape 64x1028 --fuse linear --seed $s
  step $G 40 -- $PY scripts/train_probe_p3.py $R --features "dinov3_vitl16:tokens8|moge2_vitl:normals8~and" --token-shape 64x1028 --fuse mlp --seed $s
done
step $G 40 -- $PY scripts/exp029_eval.py
echo CHAIN_DONE
