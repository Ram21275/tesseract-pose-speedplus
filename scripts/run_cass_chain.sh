#!/usr/bin/env bash
# Options 2 -> 3 -> 4, strictly one job at a time, each memory-guarded.
set -u; cd "$(dirname "$0")/.."
G=scripts/guarded.sh; PY=.venv/bin/python
L18=outputs/experiments/EXP-018_cass_attention_injection
until grep -q "EXIT_CODE" $L18/extract.log 2>/dev/null; do sleep 20; done
grep -q "EXIT_CODE=0" $L18/extract.log || { echo "EXTRACTION_FAILED"; exit 1; }
echo ">>> EXP-018 mask eval"; $G 16 -- $PY scripts/exp018_eval_masks.py 2>&1 | grep -v -i warn | tail -25
for spec in moge2_vitl_cassdino:grid4 moge2_vitl_cassdino:normals16 dinov3_vitl16_cassmoge:grid4; do for s in 0 1 2; do
  $G 16 -- $PY scripts/train_probe.py --exp EXP-018_cass_attention_injection --features $spec --seed $s 2>&1 | grep -v -i warn | grep -E "seed|Error|Traceback|memguard" | head -3
  echo "EXIT_CODE=${PIPESTATUS[0]} $spec seed $s"
done; done
echo "EXP018_DONE"
mkdir -p outputs/experiments/EXP-019_spectral_part_pooling
for spec in dinov3_vitl16:parts4 dinov3_vitl16:grid2; do for s in 0 1 2; do
  $G 16 -- $PY scripts/train_probe.py --exp EXP-019_spectral_part_pooling --features $spec --seed $s 2>&1 | grep -v -i warn | grep -E "seed|Error|Traceback|memguard" | head -3
  echo "EXIT_CODE=${PIPESTATUS[0]} $spec seed $s"
done; done
echo "EXP019_DONE"
echo ">>> EXP-020"; $G 16 -- $PY scripts/exp020_spectral_complementarity.py 2>&1 | grep -v -i warn | tail -8
echo "CHAIN_DONE"
