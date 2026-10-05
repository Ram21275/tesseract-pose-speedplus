"""Run-directory bookkeeping (ground rules sec. 9 and 12)."""
from __future__ import annotations

import csv
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "outputs"


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def environment_text() -> str:
    lines = [
        f"timestamp: {datetime.now().isoformat()}",
        f"host: {platform.node()}",
        f"python: {sys.version.split()[0]}",
        f"platform: {platform.platform()}",
        f"git_commit: {_git('rev-parse', 'HEAD')}",
        f"git_dirty_code: {bool(_git('status', '--porcelain', '--', 'src', 'scripts', 'configs', 'tests'))}",
    ]
    for mod in ["numpy", "scipy", "torch", "timm", "cv2", "sklearn", "vggt"]:
        try:
            m = __import__(mod)
            lines.append(f"{mod}: {getattr(m, '__version__', 'installed')}")
        except Exception:
            lines.append(f"{mod}: not installed")
    try:
        import torch
        lines.append(f"cuda: {torch.version.cuda}; cudnn: {torch.backends.cudnn.version()}")
        if torch.cuda.is_available():
            lines.append(f"gpu: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass
    vg = _vggt_commit()
    if vg:
        lines.append(f"vggt_commit: {vg}")
    return "\n".join(lines) + "\n"


def _vggt_commit():
    try:
        from importlib.metadata import distribution
        d = distribution("vggt")
        du = d.read_text("direct_url.json")
        if du:
            return json.loads(du).get("vcs_info", {}).get("commit_id")
    except Exception:
        return None


class Run:
    def __init__(self, exp_id: str, short: str, config: dict, seed: int = 0):
        self.exp = f"{exp_id}_{short}"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = OUTPUTS / "experiments" / self.exp / f"RUN-{stamp}-seed{seed}"
        if self.dir.exists():
            raise FileExistsError(self.dir)
        self.dir.mkdir(parents=True)
        self.t0 = time.time()
        self.config = dict(config, seed=seed)
        (self.dir / "config.yaml").write_text(yaml.safe_dump(self.config, sort_keys=False))
        (self.dir / "command.txt").write_text(" ".join([sys.executable, *sys.argv]) + "\n")
        (self.dir / "environment.txt").write_text(environment_text())
        self.status("running")

    def sub(self, name: str) -> Path:
        p = self.dir / name
        p.mkdir(exist_ok=True)
        return p

    def status(self, state: str, **extra):
        d = {"state": state, "start": datetime.fromtimestamp(self.t0).isoformat(),
             "wall_clock_s": round(time.time() - self.t0, 2),
             "peak_cpu_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
        try:
            import torch
            if torch.cuda.is_available():
                d["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        except Exception:
            pass
        d.update(extra)
        (self.dir / "status.json").write_text(json.dumps(d, indent=2))

    def write_metrics(self, metrics: dict | list[dict], name: str = "metrics"):
        """Write json + csv + tsv. ``metrics`` is a flat dict or a list of flat row dicts."""
        m = self.sub("metrics")
        (m / f"{name}.json").write_text(json.dumps(metrics, indent=2, default=float))
        rows = metrics if isinstance(metrics, list) else [metrics]
        keys = list(dict.fromkeys(k for r in rows for k in r))
        for ext, delim in [("csv", ","), ("tsv", "\t")]:
            with open(m / f"{name}.{ext}", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys, delimiter=delim)
                w.writeheader()
                w.writerows(rows)

    def done(self, **extra):
        self.status("completed", end=datetime.now().isoformat(), **extra)

    def fail(self, err: str):
        self.status("failed", error=err, end=datetime.now().isoformat())

    def rel(self) -> str:
        return os.path.relpath(self.dir, REPO)


def md_table(rows: list[dict], keys: list[str] | None = None, fmt: str = "{:.3f}") -> str:
    keys = keys or list(rows[0].keys())
    out = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        cells = []
        for k in keys:
            v = r.get(k, "")
            cells.append(fmt.format(v) if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)
