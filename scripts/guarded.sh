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
set +e
systemd-run --user --scope --quiet -p MemoryMax=${MEM}G -p MemorySwapMax=0 -p TasksMax=4096 -- "$@"
rc=$?
# A kernel kill (e.g. one allocation larger than the remaining headroom) leaves
# no status behind; record it so no run is left looking "running".
if [[ $rc -ne 0 ]]; then
  "$(dirname "$0")/../.venv/bin/python" "$(dirname "$0")/reap_dead_runs.py" --exit-code $rc || true
fi
exit $rc
