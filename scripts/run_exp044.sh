#!/usr/bin/env bash
# EXP-044 chain: teacher samples on synthetic train -> teacher diagnostic -> beta grid (seed 0) -> selection -> seeds 1,2 -> final eval.
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
E=EXP-044_distilled_tree; D=outputs/experiments/$E; PY=.venv/bin/python
G() { scripts/guarded.sh 40 -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
S=$(grep -l '"stage": "fused_eval"' $D/RUN-*/config.yaml 2>/dev/null | xargs -r -n1 dirname | while read d; do grep -q '"state": "completed"' $d/status.json && echo $d; done | head -1)
S=${S:-$(grep -l 'stage: fused_eval' $D/RUN-*/config.yaml 2>/dev/null | xargs -r -n1 dirname | while read d; do grep -q '"state": "completed"' $d/status.json && echo $d; done | head -1)}
if [ -z "$S" ]; then
  echo ">>> $(date +%T) teacher samples (synthetic train)"
  G $PY scripts/exp04x_fused_eval.py --exp EXP-042_image_conditioned_so3_posterior --run-exp $E --groups synthetic_train \
    --fused-only --save-samples --save-all-seeds --feat-cpu --queries 1024 || { echo TEACHER_FAILED; exit 1; }
  S=$(ls -d $D/RUN-* | tail -1)
fi
echo ">>> teacher run $S"
echo ">>> $(date +%T) teacher diagnostic (train masses)"
G $PY scripts/exp043_flow_masses.py --samples-run $S --groups synthetic_train --run-exp $E
train() {  # features token-shape beta seed
  echo ">>> $(date +%T) train $1 beta=$3 seed=$4"
  G $PY scripts/train_probe_p3.py --exp $E --features $1 --decoder transformer --token-shape $2 --feat-device cpu --amp bf16 \
    --targets soft --kd-samples $S --kd-beta $3 --seed $4
}
for b in 0 0.1 0.3 1.0; do
  train dinov3_vitl16:tokens8 64x1024 $b 0; train moge2_vitl:normals16~and 256x4 $b 0
done
echo ">>> $(date +%T) selection"
BEST=$(G $PY scripts/exp044_eval.py --stage select | tee /dev/stderr | grep '^BEST_BETA' | awk '{print $2}')
echo "BEST_BETA=$BEST"
if [ -n "$BEST" ] && [ "$BEST" != "0" ] && [ "$BEST" != "0.0" ]; then
  for s in 1 2; do train dinov3_vitl16:tokens8 64x1024 $BEST $s; train moge2_vitl:normals16~and 256x4 $BEST $s; done
  echo ">>> $(date +%T) final eval"
  G $PY scripts/exp044_eval.py --stage final --beta $BEST
fi
echo CHAIN_DONE
