#!/usr/bin/env bash
# DEC-025: EXP-124 GPU stages. Waits until queue14's EXP-106 T-LESS A2 evaluation is running, then sets QUEUE_PAUSE so that
# queue14 holds before EXP-119 (its running job finishes normally), waits for that job to end, runs EXP-124 with the pause
# ignored, and finally removes QUEUE_PAUSE so queue14 resumes. One GPU job at a time throughout.
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GUARD_IGNORE_PAUSE=1
PY=.venv/bin/python; PAUSE=outputs/experiments/QUEUE_PAUSE
G() { GUARD_GPU_GB=${GG:-8} scripts/guarded.sh ${GMEM:-40} -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
R() { local i; for i in 1 2 3; do G "$@" && return 0; echo "RETRY $i $(date +%T)"; sleep 600; done; return 1; }
block() { local name=$1; shift; echo "=== $(date +%T) BLOCK $name"; if "$@"; then echo "=== $(date +%T) BLOCK_DONE $name"; else echo "=== $(date +%T) BLOCK_FAILED $name"; fi; }
evalpid() { pgrep -f 'python scripts/exp064_eval.py --dataset tless --exp EXP-106' | head -1; }
until [[ -n "$(evalpid)" ]] || grep -q 'BLOCK_\(DONE\|FAILED\) EXP-106-tless-A2' outputs/experiments/queue14.log; do sleep 20; done
touch $PAUSE; echo "=== $(date +%T) QUEUE_PAUSE set; waiting for queue14's running job"
while [[ -n "$(evalpid)" ]] || pgrep -f 'python scripts/(train_xobj|xobj_teacher).py' >/dev/null; do sleep 30; done
echo "=== $(date +%T) GPU free; queue15 starts"

exp124() {
  R $PY scripts/exp124_eval.py --crops gt || return 1                                   # reference + evaluator check first
  while pgrep -f 'scripts/exp124_yolo_data.py' >/dev/null; do sleep 60; done
  [[ -s /media/kavinder/hdd2/RAM/speedplus_yolo/data.yaml ]] || return 1
  GG=14 R $PY scripts/exp124_train_det.py || return 1
  R $PY scripts/exp124_detect.py || return 1
  local k bb
  for k in 0 90 180 270; do
    for bb in dinov3_vitl16 moge2_vitl; do echo ">>> $(date +%T) extract exp124_det_r$k $bb"
      R $PY scripts/extract_compact.py --manifest outputs/data_manifests/exp124_det_test3.csv --root-key speedplus_root --tag exp124_det_r$k --backbone $bb $([[ $k != 0 ]] && echo --rot-deg $k) || return 1; done
    R $PY scripts/extract_compact.py --manifest outputs/data_manifests/exp124_det_test3.csv --root-key speedplus_root --tag exp124_det_r$k --backbone consensus || return 1
  done
  R $PY scripts/exp124_eval.py --crops det; }

block EXP-124 exp124
rm -f $PAUSE; echo "=== $(date +%T) QUEUE_PAUSE removed; queue14 resumes"
echo "=== $(date +%T) QUEUE15_DONE"
