#!/usr/bin/env bash
# EXP-043: wait for the sample stage (exp04x_fused_eval --run-exp EXP-043...) to finish, then run the mass analysis.
set -u; cd "$(dirname "$0")/.."
D=outputs/experiments/EXP-043_flow_induced_tesseract_masses
while pgrep -f '[e]xp04x_fused_eval.py --exp EXP-042' >/dev/null; do sleep 30; done
S=$(grep -l '"state": "completed"' $D/RUN-*/status.json 2>/dev/null | head -1 | xargs -r dirname)
[ -n "$S" ] || { echo "SAMPLE_STAGE_FAILED"; exit 1; }
echo ">>> $(date +%T) samples in $S"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True scripts/guarded.sh 40 -- .venv/bin/python scripts/exp043_flow_masses.py --samples-run $S 2>&1 | grep -v -i warn | grep -v nn.Transformer
echo "EXIT_CODE=${PIPESTATUS[0]}"; echo CHAIN_DONE
