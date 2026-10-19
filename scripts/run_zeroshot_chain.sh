#!/usr/bin/env bash
# EXP-064 zero-shot arm (DEC-022): T-LESS -> SYMSOL-T -> YCB-V, CPU, guarded (each job waits for RAM headroom; user:
# "if memory is available then go for it otherwise hold"). Each dataset starts once its eval_r270 consensus cache exists.
set -u; cd "$(dirname "$0")/.."
for ds in tless; do echo "=== $(date +%T) zero-shot $ds"; CUDA_VISIBLE_DEVICES= GUARD_GPU_GB=0 scripts/guarded.sh 40 -- .venv/bin/python scripts/exp064_zeroshot.py --dataset $ds 2>&1 | grep -v -i warn | tee outputs/experiments/exp064_zeroshot_$ds.log; done
for ds in symsolt ycbv; do
  until ls /media/kavinder/hdd2/RAM/compact/consensus_masks__subset_exp064_${ds}_eval_r270/masks.h5 >/dev/null 2>&1 && \
        ls /media/kavinder/hdd2/RAM/compact/moge2_vitl__subset_exp064_${ds}_eval_r270__*/manifest.json >/dev/null 2>&1; do sleep 300; done
  echo "=== $(date +%T) zero-shot $ds"
  CUDA_VISIBLE_DEVICES= GUARD_GPU_GB=0 scripts/guarded.sh 40 -- .venv/bin/python scripts/exp064_zeroshot.py --dataset $ds 2>&1 | grep -v -i warn | tee outputs/experiments/exp064_zeroshot_$ds.log
done
echo "=== $(date +%T) ZEROSHOT_CHAIN_DONE"
