"""EXP-128 deployment metrics of the downstream heads only (the frozen backbone is shared by every application and is not
counted or quantized; its cost is quoted once from EXP-112). Head = one tree per branch (DINOv3 64x1024 or MoGe-2 256x4
input) and, for 6-D, the translation block. Architecture is identical for every dataset/object, so these numbers apply
to every application; accuracy under each precision comes from the evaluators (--precision).
Per precision: parameters, bytes on disk (torch.save of the state dict; int8 = torch dynamic int8 model), latency of the
head work for greedy (beam 1) and orbit (4 views x beam 8 + teacher-forced scoring of 32 candidates per view) at batch 1
and 64, peak transient GPU memory. GPU: fp32 / bf16 / fp16 autocast. CPU: fp32 and real dynamic int8
(torch.ao.quantization.quantize_dynamic on nn.Linear, which does not convert the attention in-projection: the converted
fraction is reported; PyTorch's fused-MHA fast path is disabled for both CPU precisions because it rejects quantized
layers), 1 thread and all threads. Median of 7 after 3 warm-ups; random standardized inputs (timing does not
depend on values). Shared GPU: concurrent load is logged, times are upper bounds."""
import io
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import phase3 as P, precision as PR, tesseract as T, trans as TR  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

run = Run("EXP-128", "downstream_deployment_quantization", {"stage": "deploy"}, seed=0)
SHAPES = {"DINO": (64, 1024), "MoGe": (256, 4)}
try:
    torch.manual_seed(0)
    nbytes = lambda sd: len((lambda b: (torch.save(sd, b), b.getvalue())[1])(io.BytesIO()))
    rows_size = []
    for name, mk in (("tree DINOv3", lambda: P.HierTransformer(SHAPES["DINO"], 5)), ("tree MoGe-2", lambda: P.HierTransformer(SHAPES["MoGe"], 5)),
                     ("translation block", lambda: TR.TransHead())):
        m = mk().eval(); n = sum(p.numel() for p in m.parameters())
        q = torch.ao.quantization.quantize_dynamic(mk().eval(), {nn.Linear}, dtype=torch.qint8)
        conv = sum(1 for x in q.modules() if type(x).__module__.startswith("torch.ao.nn.quantized"))
        mm = [x for x in m.modules() if isinstance(x, (nn.Linear, nn.MultiheadAttention))]
        rows_size.append({"module": name, "params": n, "fp32_MB": nbytes(m.state_dict()) / 2**20, "fp16_MB": nbytes(m.half().state_dict()) / 2**20,
                          "int8_dynamic_MB": nbytes(q.state_dict()) / 2**20, "int8w_ideal_MB": (PR.fake_quant_int8_(mk())["int8_params"] * 1 + 4 * (n - PR.fake_quant_int8_(mk())["int8_params"])) / 2**20,
                          "linear_or_mha_modules": len(mm), "converted_by_dynamic_int8": conv})
        print(rows_size[-1], flush=True)

    def work(ms, xs, orbit):
        if not orbit:
            return fused_beam(ms, xs, "poe", 1)
        B = xs[0].shape[0]
        for _ in range(4):                                                   # 4 views: beam-8 candidates + scoring of 32
            c8, p8, _, _ = fused_beam(ms, xs, "poe", 8, return_all=True)
            cand = T.decode(c8.cpu().numpy().reshape(-1), p8.cpu().numpy().reshape(-1, 5))
            cand = np.repeat(cand.reshape(B, 8, 4), 4, 1).reshape(-1, 4)     # 32 candidates (mapping to the view omitted: CPU, tiny)
            cc, pp = T.encode(cand, 5); cc, pp = torch.as_tensor(cc, device=xs[0].device), torch.as_tensor(pp, device=xs[0].device)
            outs = [m.logits_tf(m.encode(x).repeat_interleave(32, 0), cc, pp) for m, x in zip(ms, xs)]
            sum(F.log_softmax(o[0].float(), -1) for o in outs)

    def bench(ms, xs, orbit, prec, dev, reps=7, warm=3):
        with torch.no_grad():
            for _ in range(warm):
                with PR.ctx(prec) if dev == "cuda" else torch.autocast("cpu", enabled=False):
                    work(ms, xs, orbit)
            if dev == "cuda":
                torch.cuda.synchronize(); base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                with PR.ctx(prec) if dev == "cuda" else torch.autocast("cpu", enabled=False):
                    work(ms, xs, orbit)
                if dev == "cuda":
                    torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            peak = (torch.cuda.max_memory_allocated() - base) / 2**20 if dev == "cuda" else float("nan")
        return float(np.median(ts)) * 1000, peak

    res = []
    load = lambda: subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    for variant, keys in (("DINO", ("DINO",)), ("PoE", ("DINO", "MoGe"))):
        for orbit in (False, True):
            for n in (1, 64):
                base_ms = [P.HierTransformer(SHAPES[k], 5).eval() for k in keys]
                xs_cpu = [torch.randn(n, SHAPES[k][0] * SHAPES[k][1]) for k in keys]
                for prec in ("fp32", "bf16", "fp16"):                                       # GPU
                    ms = [m.cuda() for m in base_ms]; xs = [x.cuda() for x in xs_cpu]
                    ms_, pk = bench(ms, xs, orbit, prec, "cuda")
                    res.append({"variant": variant, "inference": "orbit" if orbit else "greedy", "batch": n, "device": "GPU", "precision": prec, "threads": "",
                                "ms_per_batch": ms_, "ms_per_img": ms_ / n, "peak_transient_gpu_mb": pk, "gpu_util_mem_used": load()})
                    print(res[-1], flush=True)
                torch.backends.mha.set_fastpath_enabled(False)       # dynamic-int8 layers break the fused-MHA path check; same for both CPU rows
                for prec in ("fp32", "int8_dynamic"):                                        # CPU
                    ms = [m.cpu().float() for m in base_ms]
                    if prec == "int8_dynamic":
                        ms = [torch.ao.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8) for m in ms]
                    for th in (1, os.cpu_count()):
                        torch.set_num_threads(th)
                        ms_, _ = bench(ms, xs_cpu, orbit, "fp32", "cpu", reps=5 if n == 64 or orbit else 7, warm=2)
                        res.append({"variant": variant, "inference": "orbit" if orbit else "greedy", "batch": n, "device": "CPU", "precision": prec, "threads": th,
                                    "ms_per_batch": ms_, "ms_per_img": ms_ / n, "peak_transient_gpu_mb": float("nan"), "gpu_util_mem_used": ""})
                        print(res[-1], flush=True)
                    torch.set_num_threads(os.cpu_count())
                torch.backends.mha.set_fastpath_enabled(True)
    run.write_metrics(rows_size, "size"); run.write_metrics(res, "latency")
    (run.sub("tables") / "size.md").write_text(md_table(rows_size)); (run.sub("tables") / "latency.md").write_text(md_table(res))
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    cpu = subprocess.run(["bash", "-c", "lscpu | grep 'Model name' | head -1"], capture_output=True, text=True).stdout.strip()
    run.done(gpu=gpu, cpu=cpu, cpu_count=os.cpu_count()); print(md_table(rows_size)); print(md_table(res)); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
