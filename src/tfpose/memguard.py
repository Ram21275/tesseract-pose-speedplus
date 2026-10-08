"""In-process memory watchdog (soft guard).

A background thread polls this process's RSS (including children) and the
system's available memory. If the process exceeds ``max_rss_gb`` or the
machine drops below ``min_avail_gb`` it writes a reason file, prints to
stderr and terminates the process with exit code 137, before the kernel
OOM / swap thrash can take the whole (shared) machine down.

The hard guard is the cgroup cap applied by ``scripts/guarded.sh``
(systemd-run --scope -p MemoryMax=...); this watchdog only makes failures
early and explainable. Limits can be overridden with the environment
variables TFPOSE_MAX_RSS_GB and TFPOSE_MIN_AVAIL_GB.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import psutil

DEFAULT_MAX_RSS_GB = 32.0
DEFAULT_MIN_AVAIL_GB = 16.0
_started = False


def _rss_gb(proc: psutil.Process) -> float:
    total = proc.memory_info().rss
    for c in proc.children(recursive=True):
        try:
            total += c.memory_info().rss
        except psutil.Error:
            pass
    return total / 2**30


def start(max_rss_gb: float | None = None, min_avail_gb: float | None = None,
          interval_s: float = 1.0, reason_file: Path | None = None, on_abort=None):
    """Start the watchdog once per process. Returns the limits in use."""
    global _started
    max_rss_gb = float(os.environ.get("TFPOSE_MAX_RSS_GB", max_rss_gb or DEFAULT_MAX_RSS_GB))
    min_avail_gb = float(os.environ.get("TFPOSE_MIN_AVAIL_GB", min_avail_gb or DEFAULT_MIN_AVAIL_GB))
    if _started:
        return {"max_rss_gb": max_rss_gb, "min_avail_gb": min_avail_gb}
    _started = True
    proc = psutil.Process()

    def loop():
        while True:
            rss = _rss_gb(proc)
            avail = psutil.virtual_memory().available / 2**30
            reason = None
            if rss > max_rss_gb:
                reason = f"process RSS {rss:.1f} GB > limit {max_rss_gb:.1f} GB"
            elif avail < min_avail_gb:
                reason = f"system available memory {avail:.1f} GB < floor {min_avail_gb:.1f} GB (process RSS {rss:.1f} GB)"
            if reason:
                msg = f"[memguard] aborting: {reason}"
                print(msg, file=sys.stderr, flush=True)
                if reason_file is not None:
                    try:
                        Path(reason_file).write_text(msg + "\n")
                    except Exception:
                        pass
                if on_abort is not None:
                    try:
                        on_abort(msg)
                    except Exception:
                        pass
                os._exit(137)
            time.sleep(interval_s)

    threading.Thread(target=loop, daemon=True, name="memguard").start()
    return {"max_rss_gb": max_rss_gb, "min_avail_gb": min_avail_gb}


def check_fits(n_bytes: float, what: str, headroom_gb: float | None = None):
    """Fail loudly before allocating ``n_bytes`` if it would breach the guard."""
    avail = psutil.virtual_memory().available / 2**30
    need = n_bytes / 2**30
    floor = float(os.environ.get("TFPOSE_MIN_AVAIL_GB", headroom_gb or DEFAULT_MIN_AVAIL_GB))
    if avail - need < floor:
        raise MemoryError(f"{what} needs {need:.1f} GB; only {avail:.1f} GB available (floor {floor:.1f} GB)")
