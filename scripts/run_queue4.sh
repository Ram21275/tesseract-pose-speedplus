#!/usr/bin/env bash
# DEC-018: after the EXP-111 queue (queue3) -> EXP-112 accuracy, EXP-112 timing (non-blocking blocks).
set -u; cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
G() { scripts/guarded.sh ${GMEM:-56} -- "$@" 2>&1 | grep -v -i warn | grep -v nn.Transformer; return ${PIPESTATUS[0]}; }
block() { local name=$1; shift; echo "=== $(date +%T) BLOCK $name"; if "$@"; then echo "=== $(date +%T) BLOCK_DONE $name"; else echo "=== $(date +%T) BLOCK_FAILED $name"; fi; }
while pgrep -f '^bash scripts/run_queue3.sh' >/dev/null; do sleep 60; done
block EXP-112-accuracy G $PY scripts/exp112_eval.py
block EXP-112-timing G $PY scripts/exp112_timing.py
echo "=== $(date +%T) QUEUE4_DONE"
