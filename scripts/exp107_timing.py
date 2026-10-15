"""EXP-107 / EXP-063 efficiency: per-component latency (batch 1) and throughput (batch 64), median of 5 repeats after warm-up,
torch.cuda.synchronize() around every timed block; peak GPU memory; parameter counts. Concurrent GPU load is recorded
(shared GPU), so every number is an upper bound on an unloaded GPU."""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import data as D, features as Fx, phase3 as P, so3flow as S  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

run = Run("EXP-107", "paper_visuals_metrics_efficiency", {"stage": "timing"}, seed=0)
REP = 5
try:
    dev = torch.device("cuda"); torch.manual_seed(0)
    paths = yaml.safe_load(open(REPO / "configs/paths.yaml")); root = Path(paths["speedplus_root"])
    df = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv"); rows64 = list(df[df.domain == "lightbox"].iloc[:64].itertuples())

    def gpu_load():
        o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        return o.strip().split("\n")[0]

    def timeit(fn, warm=2):
        for _ in range(warm):
            fn()
        ts = []
        for _ in range(REP):
            torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        return float(np.median(ts)) * 1000

    res = []
    def rec(comp, bs, ms, params=None):
        res.append({"component": comp, "batch": bs, "ms_per_batch": ms, "ms_per_image": ms / bs, "images_per_s": 1000 * bs / ms,
                    "params_M": params, "gpu_util_mem_at_start": gpu_load()})
        print(res[-1], flush=True)

    dino = Fx.DinoV3("vit_large_patch16_dinov3.lvd1689m", dev); moge = Fx.MoGe2Extractor(dev)
    npar = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
    for bs in (1, 64):
        xd = Fx.load_crops(root, rows64[:bs], dino.size); xm = Fx.load_crops(root, rows64[:bs], moge.size)
        torch.cuda.reset_peak_memory_stats(); rec("DINOv3-L features (frozen)", bs, timeit(lambda: dino(xd)), npar(dino.model))
        rec("MoGe-2 ViT-L features (frozen)", bs, timeit(lambda: moge(xm)), npar(moge.model))
    del dino, moge; torch.cuda.empty_cache()

    # trees + flow heads (seed 0), features as the trainer feeds them
    ts, heads = [], []
    for spec, tsh in (("dinov3_vitl16:tokens8", (64, 1024)), ("moge2_vitl:normals16~and", (256, 4))):
        for d in sorted((REPO / "outputs/experiments/EXP-042_image_conditioned_so3_posterior").glob("RUN-*-seed0")):
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            if cfg.get("features") == spec and cfg.get("mode") == "flow" and cfg.get("base") == "uniform" and (d / "checkpoints/best.pt").exists():
                sd = REPO / "outputs/experiments" / cfg["src_exp"] / (d / "source_runs.txt").read_text().split()[0]
                t = P.HierTransformer(tsh, 5).to(dev); t.load_state_dict(torch.load(sd / "checkpoints/best.pt", map_location=dev)["model"]); t.eval()
                h = S.FlowHead(layers=cfg["layers"]).to(dev); h.load_state_dict(torch.load(d / "checkpoints/best.pt", map_location=dev)["head"]); h.eval()
                ts.append(t); heads.append(h); break
    for bs in (1, 64):
        xs = [torch.randn(bs, 64 * 1024, device=dev), torch.randn(bs, 256 * 4, device=dev)]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            rec("tree A+B encode + per-level PoE + greedy decode", bs, timeit(lambda: fused_beam(ts, xs, "poe", 1)), npar(ts[0]) + npar(ts[1]))
            rec("tree A+B beam 8 (candidate set)", bs, timeit(lambda: fused_beam(ts, xs, "poe", 8, return_all=True)))
            _, _, hs = fused_beam(ts, xs, "poe", 1, return_h=True)
        mems = [h.float().repeat_interleave(32, 0) for h in hs]
        q0 = S.uniform_quats(bs * 32, dev); tau = torch.zeros(bs * 32, device=dev)
        with torch.no_grad():
            rec("flow: one velocity evaluation (2 heads, M = 32)", bs, timeit(lambda: [h(m, q0, tau, None) for h, m in zip(heads, mems)]), npar(heads[0]) + npar(heads[1]))
            vel = lambda qq, tt: torch.stack([h(m, qq, tt, None) for h, m in zip(heads, mems)]).mean(0)
            rec("flow: full sampling NFE 10, M 32", bs, timeit(lambda: S.integrate(vel, q0, 10)))
            smp = S.integrate(vel, q0, 10).view(bs, 32, 4)
            rec("KDE mode (M 32)", bs, timeit(lambda: S.kde_mode(smp)))
    run.write_metrics(res, "timing"); (run.sub("tables") / "timing.md").write_text(md_table(res))
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    run.done(gpu=gpu); print(md_table(res)); print("GPU", gpu); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
