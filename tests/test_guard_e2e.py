"""End-to-end check of scripts/guarded.sh + memguard + reaper (needs systemd user scopes)."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "outputs/experiments/EXP-999_guard_selftest"


@pytest.mark.skipif(shutil.which("systemd-run") is None, reason="systemd-run not available")
@pytest.mark.parametrize("mode", ["gradual", "oneshot"])
def test_guarded_job_never_left_running(mode):
    code = f"""
import sys, time, numpy as np
sys.path.insert(0, 'src')
from tfpose.runlog import Run
r = Run('EXP-999', 'guard_selftest', {{'mode': '{mode}'}}, seed=0)
if '{mode}' == 'gradual':
    xs = []
    for _ in range(40):
        xs.append(np.ones(2**24)); time.sleep(0.25)  # +128 MB per step (~0.5 GB/s)
else:
    x = np.ones((3 * 2**30) // 8); x += 1             # one 3 GB allocation
print('should not reach here')
"""
    try:
        r = subprocess.run(["scripts/guarded.sh", "2", "--", str(REPO / ".venv/bin/python"), "-c", code],
                           cwd=REPO, capture_output=True, text=True, timeout=120)
        assert r.returncode == 137, (r.returncode, r.stdout[-500:], r.stderr[-500:])
        runs = sorted(EXP.glob("RUN-*"))
        status = json.loads((runs[-1] / "status.json").read_text())
        assert status["state"] == "failed", status
        if mode == "gradual":
            assert "memguard" in status["error"], status   # soft guard fired first
        else:
            assert "died without writing" in status["error"], status   # hard cap + reaper
    finally:
        shutil.rmtree(EXP, ignore_errors=True)
