"""EXP-112 end-to-end timing of the locked recipe (DEC-018) on real lightbox crops (images pre-decoded; JPEG decode excluded).
Per system and batch size (1, 64): in-plane crop warp(s) -> DINOv3 (-> MoGe-2 + EXP-015 consensus mask on CPU, PoE variant)
-> feature pooling + standardization -> tree(s) (greedy, or beam-8 per view + orbit scoring). Median of 5 after 2 warm-ups,
cuda-synchronized; per-component times; peak GPU memory. Shared GPU: concurrent load recorded, numbers are upper bounds."""
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import data as D, featsets, features as Fx, inplane as IP, masks as M, phase3 as P, tesseract as T  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

run = Run("EXP-112", "locked_recipe_eval_timing", {"stage": "timing"}, seed=0)
SD, SM = "dinov3_vitl16:tokens8", "moge2_vitl:normals16~and"
try:
    dev = torch.device("cuda"); torch.manual_seed(0)
    paths = yaml.safe_load(open(REPO / "configs/paths.yaml")); root = Path(paths["speedplus_root"])
    cam = D.load_camera(root); conv = (REPO / "outputs/experiments/exp109_convention.txt").read_text().strip()
    full = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv"); tr = ((full.domain == "synthetic") & (full.split == "train")).to_numpy()
    st = {}
    for sp in (SD, SM):
        _, x, _ = featsets.load(sp, "full"); mu, sd = featsets.train_stats(x, tr); del x
        st[sp] = (torch.as_tensor(mu, dtype=torch.float32, device=dev), torch.as_tensor(sd, dtype=torch.float32, device=dev))
    trees = []
    for sp, tsh in ((SD, (64, 1024)), (SM, (256, 4))):
        for d in sorted((REPO / "outputs/experiments/EXP-109_real_domain_aug_tokens16_orbit").glob("RUN-*-seed0")):
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            if cfg.get("features") == sp and cfg.get("aug_tags") and (d / "checkpoints/best.pt").exists():
                t = P.HierTransformer(tsh, 5).to(dev); t.load_state_dict(torch.load(d / "checkpoints/best.pt", map_location=dev)["model"]); t.eval(); trees.append(t); break
    dino = Fx.DinoV3("vit_large_patch16_dinov3.lvd1689m", dev); moge = Fx.MoGe2Extractor(dev)
    rows = full[full.domain == "lightbox"].iloc[:64]
    imgs = [D.read_gray(root, r.image_relpath) for r in rows.itertuples()]
    dray = IP.ray_undistorted(cam.K, cam.dist, rows.crop_cx.to_numpy(), rows.crop_cy.to_numpy())

    def crops(n, size, theta):
        out = [IP.crop_rotated(im, r.crop_cx, r.crop_cy, r.crop_side, size, theta) if theta else D.crop_square(im, r.crop_cx, r.crop_cy, r.crop_side, size)
               for im, r in zip(imgs[:n], rows.iloc[:n].itertuples())]
        return torch.from_numpy(np.stack(out)).float().div_(255.0)[:, None].expand(-1, 3, -1, -1).contiguous()

    def feats(n, theta, with_moge, T_):
        t0 = time.perf_counter(); xd = crops(n, dino.size, theta); xm = crops(n, moge.size, theta) if with_moge else None; T_["warp"] += time.perf_counter() - t0
        torch.cuda.synchronize(); t0 = time.perf_counter()
        od = dino(xd); g16 = od["last_grid"]                                           # (B,16,16,C)
        xD = ((F.adaptive_avg_pool2d(g16.permute(0, 3, 1, 2), 8).permute(0, 2, 3, 1).reshape(n, -1) - st[SD][0]) / st[SD][1])
        torch.cuda.synchronize(); T_["dino"] += time.perf_counter() - t0
        if not with_moge:
            return [xD]
        t0 = time.perf_counter(); om = moge(xm); torch.cuda.synchronize(); T_["moge"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        g = g16.half().float().cpu().numpy()
        fg = np.stack([(lambda sc: sc > M.otsu(sc))(M.dino_pca_score(gi)) for gi in g]).astype(np.float32)   # EXP-015 consensus (CPU)
        mask = (om["mask"][..., 0] > 0.5).float() * torch.as_tensor(fg, device=dev)
        nrm = om["normal"]; mm = mask[..., None]
        pool = lambda a: F.adaptive_avg_pool2d(a.permute(0, 3, 1, 2), 16).permute(0, 2, 3, 1)
        xM = ((torch.cat([pool(nrm * mm), pool(mm)], -1).reshape(n, -1) - st[SM][0]) / st[SM][1])
        torch.cuda.synchronize(); T_["moge_post"] += time.perf_counter() - t0
        return [xD, xM]

    def system(n, with_moge, orbit, T_):
        ms = trees if with_moge else trees[:1]
        if not orbit:
            xs = feats(n, 0.0, with_moge, T_)
            t0 = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                c, p = fused_beam(ms, xs, "poe", 1)
            torch.cuda.synchronize(); T_["tree"] += time.perf_counter() - t0
            return
        views, cands = [], []
        for k in (0, 90, 180, 270):
            xs = feats(n, float(k), with_moge, T_)
            t0 = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                c8, p8, _, _ = fused_beam(ms, xs, "poe", 8, return_all=True)
            qk = T.decode(c8.cpu().numpy().reshape(-1), p8.cpu().numpy().reshape(-1, 5))
            cands.append(IP.rotate_label(qk, -float(k), np.repeat(dray[:n], 8, 0), conv) if k else qk)
            views.append((k, xs)); torch.cuda.synchronize(); T_["tree"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        cand = np.concatenate([c.reshape(n, 8, 4) for c in cands], 1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for k, xs in views:
                mk = IP.rotate_label(cand.reshape(-1, 4), np.full(n * 32, float(k)), np.repeat(dray[:n], 32, 0), conv) if k else cand.reshape(-1, 4)
                cc, pp = T.encode(mk, 5); cc, pp = torch.as_tensor(cc, device=dev), torch.as_tensor(pp, device=dev)
                outs = [m.logits_tf(m.encode(x).repeat_interleave(32, 0), cc, pp) for m, x in zip(ms, xs)]
        torch.cuda.synchronize(); T_["orbit_scoring"] += time.perf_counter() - t0

    res = []
    for name, with_moge, orbit in (("A-DINO greedy", False, False), ("A-DINO + orbit", False, True),
                                   ("A-PoE greedy", True, False), ("A-PoE + orbit", True, True)):
        for n in (1, 64):
            for _ in range(2):
                system(n, with_moge, orbit, {k: 0.0 for k in ("warp", "dino", "moge", "moge_post", "tree", "orbit_scoring")})
            torch.cuda.reset_peak_memory_stats(); reps = []
            for _ in range(5):
                T_ = {k: 0.0 for k in ("warp", "dino", "moge", "moge_post", "tree", "orbit_scoring")}
                torch.cuda.synchronize(); t0 = time.perf_counter(); system(n, with_moge, orbit, T_); torch.cuda.synchronize()
                T_["total"] = time.perf_counter() - t0; reps.append(T_)
            med = {k: float(np.median([r[k] for r in reps])) * 1000 / n for k in reps[0]}
            load = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
            res.append({"system": name, "batch": n, **{f"ms_per_img_{k}": v for k, v in med.items()},
                        "fps_or_img_per_s": 1000.0 / med["total"], "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20, "gpu_util_mem_used": load})
            print(res[-1], flush=True)
    run.write_metrics(res, "timing"); (run.sub("tables") / "timing.md").write_text(md_table(res))
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    run.done(gpu=gpu); print(md_table(res, ["system", "batch", "ms_per_img_total", "fps_or_img_per_s", "ms_per_img_dino", "ms_per_img_moge",
                                            "ms_per_img_moge_post", "ms_per_img_tree", "ms_per_img_orbit_scoring", "peak_gpu_mb"])); print("GPU", gpu); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
