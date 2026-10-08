"""EXP-101 part C: prediction-head latency, Tesseract tree (beam_search_fast) vs SPACE-HOP-style flat head.

Same trunk and input dimension for both (DINOv3-L grid4: 16384-d). Weights are random (timing does
not depend on them). Backbone (identical for both arms) is timed separately for context.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose.predictor import FlatAnchorMLP, HierMLP, beam_search_fast  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--d-in", type=int, default=16384)
ap.add_argument("--depths", default="2,3,4,5,6")
ap.add_argument("--devices", default="cuda:1,cpu,cuda:0")
ap.add_argument("--batches", default="1,256")
ap.add_argument("--warmup", type=int, default=50)
ap.add_argument("--iters", type=int, default=200)
ap.add_argument("--cpu-max-seconds", type=float, default=60.0, help="cap per CPU config; fewer iters recorded if hit")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
run = Run("EXP-101", "tesseract_vs_spacehop_hopf", dict(vars(args), part="C_timing"), seed=args.seed)
try:
    torch.set_num_threads(1)
    rows = []

    def bench(fn, dev, warm, iters, cap=None):
        sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
        for _ in range(warm):
            fn()
        sync()
        ts, t_start = [], time.perf_counter()
        for _ in range(iters):
            t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
            if cap and time.perf_counter() - t_start > cap:
                break
        ts = np.array(ts) * 1000
        return float(np.median(ts)), float(np.percentile(ts, 95)), len(ts)

    for dname in args.devices.split(","):
        dev = torch.device(dname)
        if dev.type == "cuda" and not torch.cuda.is_available():
            continue
        tag = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU (1 thread)"
        for B in [int(b) for b in args.batches.split(",")]:
            x = torch.randn(B, args.d_in, device=dev)
            for L in [int(l) for l in args.depths.split(",")]:
                K = 4 * 8 ** L
                cap = args.cpu_max_seconds if dev.type == "cpu" else None
                warm = args.warmup if dev.type == "cuda" else 3
                # Tesseract hierarchical head
                torch.manual_seed(args.seed)
                tm = HierMLP(args.d_in, L).to(dev).eval()
                p_tess = sum(p.numel() for p in tm.parameters())
                for beam in (1, 4):
                    with torch.no_grad():
                        med, p95, n = bench(lambda: beam_search_fast(tm, x, beam=beam), dev, warm, args.iters, cap)
                    rows.append({"device": tag, "batch": B, "depth_L": L, "hypotheses_K": K, "head": f"tesseract_beam{beam}",
                                 "params": p_tess, "median_ms": med, "p95_ms": p95, "per_image_ms": med / B, "iters": n})
                    print(rows[-1], flush=True)
                del tm
                # SPACE-HOP-style flat head over K anchors (argmax + anchor lookup)
                try:
                    fm = FlatAnchorMLP(args.d_in, K).to(dev).eval()
                    anchors = torch.nn.functional.normalize(torch.randn(K, 4, device=dev), dim=-1)
                    p_flat = sum(p.numel() for p in fm.parameters())
                    with torch.no_grad():
                        med, p95, n = bench(lambda: anchors[fm(x).argmax(-1)], dev, warm, args.iters, cap)
                    rows.append({"device": tag, "batch": B, "depth_L": L, "hypotheses_K": K, "head": "flat_argmax",
                                 "params": p_flat, "median_ms": med, "p95_ms": p95, "per_image_ms": med / B, "iters": n})
                    del fm, anchors
                except torch.OutOfMemoryError:
                    rows.append({"device": tag, "batch": B, "depth_L": L, "hypotheses_K": K, "head": "flat_argmax",
                                 "params": 512 * K + K, "median_ms": float("nan"), "p95_ms": float("nan"), "per_image_ms": float("nan"), "iters": 0,
                                 "note": "CUDA OOM"})
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                print(rows[-1], flush=True)
                run.write_metrics(rows, "timing")       # incremental, so partial results survive
    # backbone context (identical for both arms)
    from tfpose.features import DinoV3
    for dname in [d for d in args.devices.split(",") if d.startswith("cuda")]:
        dev = torch.device(dname)
        try:
            bb = DinoV3("vit_large_patch16_dinov3.lvd1689m", dev)
            for B in (1, 32):
                xi = torch.rand(B, 3, 256, 256)
                med, p95, n = bench(lambda: bb(xi), dev, 10, 50)
                rows.append({"device": torch.cuda.get_device_name(dev), "batch": B, "depth_L": "-", "hypotheses_K": "-",
                             "head": "backbone DINOv3-L/16 @256 (shared by both arms)", "params": sum(p.numel() for p in bb.model.parameters()),
                             "median_ms": med, "p95_ms": p95, "per_image_ms": med / B, "iters": n})
                print(rows[-1], flush=True)
            del bb; torch.cuda.empty_cache()
        except torch.OutOfMemoryError:
            rows.append({"device": dname, "head": "backbone", "note": "CUDA OOM"})
    run.write_metrics(rows, "timing")
    (run.sub("tables") / "timing.md").write_text(md_table(rows))
    run.done()
    print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc())
    raise
