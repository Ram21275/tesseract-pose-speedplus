#!/usr/bin/env bash
# Run a command inside a memory-capped cgroup (hard guard) so that a runaway
# job is killed by the kernel on its own instead of exhausting the shared
# machine. Usage: scripts/guarded.sh [MEM_MAX_GB] -- cmd args...
# Default cap: 40 GB RAM, no swap, at most 4096 tasks (guards against fork storms).
# The in-process watchdog (tfpose.memguard) is set to 85% of the cap so it fires
# first, records status "failed" with a readable reason, and exits before the
# kernel kill (which would leave status.json stuck at "running").
set -euo pipefail
MEM=40
if [[ "${1:-}" != "--" ]]; then MEM=$1; shift; fi
[[ "${1:-}" == "--" ]] && shift
export TFPOSE_MAX_RSS_GB=${TFPOSE_MAX_RSS_GB:-$((MEM*85/100))}
# Small per-image linear algebra is far faster single-threaded than with 24 BLAS
# threads fighting each other (and it leaves cores for other users).
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
# Be polite to other users' jobs: do not start until the machine can absorb this
# job's whole cap plus headroom (RAM) and the GPU has room (GUARD_GPU_GB, 0 = skip),
# then run at low CPU and I/O priority. Waits, never kills anything.
RAM_HEADROOM=${GUARD_RAM_HEADROOM_GB:-12}
GPU_NEED=${GUARD_GPU_GB:-8}
while :; do
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  gfree=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  gfree=$(( ${gfree:-999999} / 1024 ))
  if (( avail >= MEM + RAM_HEADROOM )) && (( GPU_NEED == 0 || gfree >= GPU_NEED )); then break; fi
  echo "[guarded] $(date +%T) waiting: RAM avail ${avail} GB (need $((MEM + RAM_HEADROOM))), GPU free ${gfree} GB (need ${GPU_NEED})" >&2
  sleep 120
done
set +e
nice -n 10 ionice -c 3 systemd-run --user --scope --quiet -p MemoryMax=${MEM}G -p MemorySwapMax=0 -p TasksMax=4096 -- "$@"
rc=$?
# A kernel kill (e.g. one allocation larger than the remaining headroom) leaves
# no status behind; record it so no run is left looking "running".
if [[ $rc -ne 0 ]]; then
  "$(dirname "$0")/../.venv/bin/python" "$(dirname "$0")/reap_dead_runs.py" --exit-code $rc || true
fi
exit $rc
