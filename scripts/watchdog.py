"""Queue watchdog (runs in tmux `ram:wd`; every 5 min; read-mostly, never touches other users' processes).

Auto-fixes (safe, only our own processes / files):
  - orphaned DataLoader workers (`pt_data_worker`, our uid, reparented to the user systemd = their main process died):
    killed, so a queue's output pipe closes and its retry logic runs (the 2026-10-29 EXP-124 stall);
  - runs whose status.json says "running" but whose process is gone: marked failed (scripts/reap_dead_runs.py).
Alerts (appended to outputs/experiments/watchdog_alerts.log; a heartbeat goes to watchdog.log):
  - a queue process (run_queueNN.sh) has died without printing its QUEUENN_DONE marker;
  - a queue's pipeline is stuck: only its `grep` filters are left as children for > 15 min (no job, no sleep);
  - a job of ours has used no CPU for > 30 min and holds no GPU memory;
  - QUEUE_PAUSE exists while no queue that ignores it (queue15) is alive: queue14 would wait forever;
  - system available memory below 12 GB.
Each distinct alert is reported once until it clears."""
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import psutil

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "outputs/experiments"
LOG, ALERTS = EXP / "watchdog.log", EXP / "watchdog_alerts.log"
QUEUES = {"14": EXP / "queue14.log", "15": EXP / "queue15.log", "16": EXP / "queue16.log"}
UID = psutil.Process().uids().real
INTERVAL, STALL_S, PIPE_S = 300, 1800, 900
cpu_seen: dict[int, tuple[float, float]] = {}      # pid -> (cpu seconds, time of last change)
pipe_seen: dict[int, float] = {}
active: set[str] = set()


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write(path, msg):
    with open(path, "a") as f:
        f.write(f"{now()} {msg}\n")


def alert(key, msg, current):
    current.add(key)
    if key not in active:
        write(ALERTS, "ALERT " + msg); write(LOG, "ALERT " + msg)


def ours(p):
    try:
        return p.uids().real == UID
    except psutil.Error:
        return False


def gpu_pids():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True).stdout
    return {int(x) for x in out.split() if x.strip().isdigit()}


def tick():
    current, fixes = set(), []
    user_systemd = {p.pid for p in psutil.process_iter(["name", "uids"]) if p.info["name"] == "systemd" and ours(p) and p.pid != 1}
    procs = [p for p in psutil.process_iter(["pid", "ppid", "name", "cmdline"]) if ours(p)]
    # 1. orphaned DataLoader workers -> kill
    for p in procs:
        if p.info["name"] == "pt_data_worker" and p.info["ppid"] in user_systemd:
            try:
                p.kill(); fixes.append(f"killed orphaned DataLoader worker {p.pid}")
            except psutil.Error:
                pass
    # 2. stale "running" runs -> failed
    stale = []
    for st in EXP.glob("*/RUN-*/status.json"):
        try:
            s = json.loads(st.read_text())
        except Exception:
            continue
        if s.get("state") == "running" and s.get("pid") and not psutil.pid_exists(s["pid"]):
            stale.append(str(st.parent.relative_to(EXP)))
    if stale:
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "scripts/reap_dead_runs.py"), "--exit-code", "137"], capture_output=True)
        fixes.append("marked dead runs failed: " + ", ".join(stale))
    # 3. queues
    qproc = {}
    for p in procs:
        cl = " ".join(p.info["cmdline"] or [])
        m = re.fullmatch(r"bash scripts/run_queue(\d+)r?\.sh", cl)
        if m:
            qproc[m.group(1)] = p
    gp = gpu_pids(); t = time.time(); status = []
    for q, log in QUEUES.items():
        txt = log.read_text(errors="ignore") if log.exists() else ""
        done = f"QUEUE{q}_DONE" in txt
        p = qproc.get(q)
        if p is None:
            status.append(f"q{q}:{'done' if done else 'NOT RUNNING'}")
            if not done and log.exists():
                alert(f"dead{q}", f"queue{q} is not running and has no QUEUE{q}_DONE marker (crash/reboot?); last block: "
                      + (re.findall(r"=== .*", txt) or ["-"])[-1], current)
            continue
        kids = p.children(recursive=True)
        names = [k.name() for k in kids if k.is_running()]
        jobs = [k for k in kids if k.name().startswith("python")]
        if kids and set(names) <= {"grep", "tee"}:
            pipe_seen.setdefault(p.pid, t)
            if t - pipe_seen[p.pid] > PIPE_S:
                alert(f"pipe{q}", f"queue{q}: only grep filters left for > {PIPE_S // 60} min (job exited but something holds the pipe)", current)
        else:
            pipe_seen.pop(p.pid, None)
        for j in jobs:
            c = sum(j.cpu_times()[:2]) + sum(sum(k.cpu_times()[:2]) for k in j.children(recursive=True) if k.is_running())
            last = cpu_seen.get(j.pid)
            if last is None or c > last[0] + 1.0:
                cpu_seen[j.pid] = (c, t)
            elif t - last[1] > STALL_S and j.pid not in gp:
                alert(f"stall{j.pid}", f"queue{q}: job {j.pid} ({' '.join(j.cmdline()[1:3])}) used no CPU for {int((t - last[1]) / 60)} min and holds no GPU memory", current)
        paused = any("guarded.sh" in " ".join(k.cmdline()) for k in kids if k.is_running()) and (EXP / "QUEUE_PAUSE").exists() and not jobs
        status.append(f"q{q}:{'paused' if paused else ('job ' + jobs[0].cmdline()[1].split('/')[-1] if jobs else 'waiting')}")
    if (EXP / "QUEUE_PAUSE").exists() and "15" not in qproc:
        alert("pause", "QUEUE_PAUSE exists but queue15 (the queue that removes it) is not running: queue14 will wait forever", current)
    avail = psutil.virtual_memory().available / 2**30
    if avail < 12:
        alert("mem", f"system available memory {avail:.1f} GB", current)
    for f in fixes:
        write(ALERTS, "FIXED " + f); write(LOG, "FIXED " + f)
    active.intersection_update(current); active.update(current)
    write(LOG, f"ok {' '.join(status)} avail={avail:.0f}GB alerts={len(current)}")


if __name__ == "__main__":
    write(LOG, "watchdog start")
    while True:
        try:
            tick()
        except Exception as e:                                  # never die silently
            write(LOG, f"watchdog error: {e!r}")
        time.sleep(INTERVAL)
