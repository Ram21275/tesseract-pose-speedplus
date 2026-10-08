#!/usr/bin/env bash
# Run a command inside a memory-capped cgroup (hard guard) so that a runaway
# job is killed by the kernel on its own instead of exhausting the shared
# machine. Usage: scripts/guarded.sh [MEM_MAX_GB] -- cmd args...
# Default cap: 40 GB RAM, no swap, at most 4096 tasks (guards against fork storms). The in-process watchdog (tfpose.memguard)
# is the soft guard and normally fires first with a readable reason.
set -euo pipefail
MEM=40
if [[ "${1:-}" != "--" ]]; then MEM=$1; shift; fi
[[ "${1:-}" == "--" ]] && shift
exec systemd-run --user --scope --quiet -p MemoryMax=${MEM}G -p MemorySwapMax=0 -p TasksMax=4096 -- "$@"
