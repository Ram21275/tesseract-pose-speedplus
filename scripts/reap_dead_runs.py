"""Mark runs whose process died without writing a final status as failed.

Called by scripts/guarded.sh when a guarded job exits abnormally (e.g. the
cgroup memory cap killed it before the in-process watchdog could react, which
happens for a single allocation larger than the remaining headroom).
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import psutil

REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--exit-code", type=int, required=True)
args = ap.parse_args()
for st in sorted((REPO / "outputs/experiments").glob("*/RUN-*/status.json")):
    try:
        s = json.loads(st.read_text())
    except Exception:
        continue
    pid = s.get("pid")
    if s.get("state") != "running" or pid is None:
        continue
    alive = psutil.pid_exists(pid)
    if alive:
        try:
            started = datetime.fromisoformat(s["start"]).timestamp()
            alive = psutil.Process(pid).create_time() <= started + 1
        except psutil.Error:
            alive = False
    if not alive:
        s.update(state="failed", end=datetime.now().isoformat(),
                 error=f"process {pid} died without writing a final status; guarded job exit code {args.exit_code} "
                       f"(137 = SIGKILL, most likely the cgroup memory cap in scripts/guarded.sh)")
        st.write_text(json.dumps(s, indent=2))
        print(f"[reap] marked failed: {st.parent.relative_to(REPO)}")
