"""EXP-127 cost under identical conditions: tree (EXP-109 arm A) vs flat over the Tesseract L5 leaves vs flat over the
balanced Hopf 1771x74 grid, DINOv3-only and PoE, greedy and orbit (4 views, top-8 per view, orbit-product scoring).
Same 64 lightbox crops, batch 1 and 64, 2 warm-ups + median of 5, cuda-synchronized, families interleaved round by round so
that shared-GPU drift hits all of them alike (EXP-112 pipeline: crop warp -> DINOv3 (-> MoGe-2 + EXP-015 consensus on CPU)
-> pooling/standardization -> head). Reports per-image ms per component (head = per-view decode / top-8, orbit_scoring),
full-pipeline ms, inference memory per system = resident weights it uses (backbone(s) + its heads) + transient peak above the
allocated baseline (all models stay loaded in one process, so the raw peak is not per-system), trainable parameters per branch, and training peak GPU memory / ms per step
(20 AdamW steps at batch 256, bf16 autocast, each family's own loss on random standardized inputs). Seed-0 checkpoints."""
import subprocess
import sys
import time
from pathlib import Path

import cv2  # noqa: F401
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import data as D, featsets, features as Fx, flatgrid as FG, inplane as IP, masks as M, phase3 as P, tesseract as T  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

run = Run("EXP-127", "hierarchy_vs_flat_same_grid", {"stage": "timing"}, seed=0)
SD, SM = "dinov3_vitl16:tokens8", "moge2_vitl:normals16~and"
SHAPE = {SD: (64, 1024), SM: (256, 4)}
FAMS = {"tree": None, "flat_tesseract_L5": "tesseract_L5", "flat_hopf1771x74": "hopf1771x74"}
COMP = ("warp", "dino", "moge", "moge_post", "head", "orbit_scoring")
try:
    dev = torch.device("cuda"); torch.manual_seed(0)
    paths = yaml.safe_load(open(REPO / "configs/paths.yaml")); root = Path(paths["speedplus_root"])
    cam = D.load_camera(root); conv = (REPO / "outputs/experiments/exp109_convention.txt").read_text().strip()
    full = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv"); tr = ((full.domain == "synthetic") & (full.split == "train")).to_numpy()
    st = {}
    for sp in (SD, SM):
        _, x, _ = featsets.load(sp, "full"); mu, sd = featsets.train_stats(x, tr); del x
        st[sp] = (torch.as_tensor(mu, dtype=torch.float32, device=dev), torch.as_tensor(sd, dtype=torch.float32, device=dev))

    def find(exp, sp, grid):
        for d in sorted((REPO / "outputs/experiments" / exp).glob("RUN-*-seed0")):
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            if cfg.get("features") == sp and cfg.get("aug_tags") == "exp109_train_aug1,exp109_train_aug2" and (d / "checkpoints/best.pt").exists() \
                    and (grid is None or cfg.get("grid") == grid):
                return d
        raise FileNotFoundError((exp, sp, grid))

    MODELS, GRIDS, NPARAM = {}, {}, []
    for fam, grid in FAMS.items():
        if grid:
            GRIDS[fam] = torch.as_tensor(FG.make_grid(grid), dtype=torch.float32, device=dev)
        ms = []
        for sp in (SD, SM):
            d = find("EXP-109_real_domain_aug_tokens16_orbit" if grid is None else "EXP-127_hierarchy_vs_flat_same_grid", sp, grid)
            m = P.HierTransformer(SHAPE[sp], 5) if grid is None else FG.FlatTransformer(SHAPE[sp], len(GRIDS[fam]))
            m.load_state_dict(torch.load(d / "checkpoints/best.pt", map_location="cpu")["model"]); m.to(dev).eval(); ms.append(m)
            NPARAM.append({"family": fam, "features": sp, "source_run": str(d.relative_to(REPO)), "trainable_params": sum(p.numel() for p in m.parameters()),
                           "head_params": (m.flat_head.weight.numel() + m.flat_head.bias.numel()) if grid else 0})
        MODELS[fam] = ms
    dino = Fx.DinoV3("vit_large_patch16_dinov3.lvd1689m", dev); moge = Fx.MoGe2Extractor(dev)
    wmb = lambda mods: sum(sum(t.numel() * t.element_size() for t in (*m.parameters(), *m.buffers())) for m in mods) / 2**20
    _mods = lambda o: [v for v in vars(o).values() if isinstance(v, torch.nn.Module)] if not isinstance(o, torch.nn.Module) else [o]
    W_DINO, W_MOGE = wmb(_mods(dino)), wmb(_mods(moge)); print("backbone weights MB", W_DINO, W_MOGE, flush=True)
    assert W_DINO > 100 and W_MOGE > 100
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
        od = dino(xd); g16 = od["last_grid"]
        xD = ((F.adaptive_avg_pool2d(g16.permute(0, 3, 1, 2), 8).permute(0, 2, 3, 1).reshape(n, -1) - st[SD][0]) / st[SD][1])
        torch.cuda.synchronize(); T_["dino"] += time.perf_counter() - t0
        if not with_moge:
            return [xD]
        t0 = time.perf_counter(); om = moge(xm); torch.cuda.synchronize(); T_["moge"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        g = g16.half().float().cpu().numpy()
        fg = np.stack([(lambda sc: sc > M.otsu(sc))(M.dino_pca_score(gi)) for gi in g]).astype(np.float32)
        mask = (om["mask"][..., 0] > 0.5).float() * torch.as_tensor(fg, device=dev)
        nrm = om["normal"]; mm = mask[..., None]
        pool = lambda a: F.adaptive_avg_pool2d(a.permute(0, 3, 1, 2), 16).permute(0, 2, 3, 1)
        xM = ((torch.cat([pool(nrm * mm), pool(mm)], -1).reshape(n, -1) - st[SM][0]) / st[SM][1])
        torch.cuda.synchronize(); T_["moge_post"] += time.perf_counter() - t0
        return [xD, xM]

    def flat_lp(ms, xs):
        return F.log_softmax(sum(F.log_softmax(m.logits(m.encode(x)).float(), -1) for m, x in zip(ms, xs)), -1)

    def system(fam, n, with_moge, orbit, T_):
        ms = MODELS[fam] if with_moge else MODELS[fam][:1]; G = GRIDS.get(fam); gname = FAMS[fam]
        if not orbit:
            xs = feats(n, 0.0, with_moge, T_)
            t0 = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                if G is None:
                    c, p = fused_beam(ms, xs, "poe", 1); T.decode(c.cpu().numpy(), p.cpu().numpy())
                else:
                    G[flat_lp(ms, xs).argmax(-1)].cpu().numpy()
            torch.cuda.synchronize(); T_["head"] += time.perf_counter() - t0
            return
        views, cands, lps = [], [], []
        for k in (0, 90, 180, 270):
            xs = feats(n, float(k), with_moge, T_)
            t0 = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                if G is None:
                    c8, p8, _, _ = fused_beam(ms, xs, "poe", 8, return_all=True)
                    qk = T.decode(c8.cpu().numpy().reshape(-1), p8.cpu().numpy().reshape(-1, 5))
                else:
                    lp = flat_lp(ms, xs); lps.append(lp); qk = G[lp.topk(8, -1).indices].reshape(-1, 4).cpu().numpy()
            cands.append(IP.rotate_label(qk, -float(k), np.repeat(dray[:n], 8, 0), conv) if k else qk)
            views.append((k, xs)); torch.cuda.synchronize(); T_["head"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        cand = np.concatenate([c.reshape(n, 8, 4) for c in cands], 1); score = torch.zeros(n, 32, device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for vi, (k, xs) in enumerate(views):
                mk = IP.rotate_label(cand.reshape(-1, 4), np.full(n * 32, float(k)), np.repeat(dray[:n], 32, 0), conv) if k else cand.reshape(-1, 4)
                if G is None:
                    cc, pp = T.encode(mk, 5); cc, pp = torch.as_tensor(cc, device=dev), torch.as_tensor(pp, device=dev)
                    outs = [m.logits_tf(m.encode(x).repeat_interleave(32, 0), cc, pp) for m, x in zip(ms, xs)]
                    root = F.log_softmax(sum(F.log_softmax(o[0].float(), -1) for o in outs), -1)
                    child = F.log_softmax(sum(F.log_softmax(o[1].float(), -1) for o in outs), -1)
                    score += (root.gather(1, cc[:, None])[:, 0] + child.gather(2, pp[:, :, None])[..., 0].sum(1)).view(n, 32)
                else:
                    ai = FG.hard_anchor(torch.as_tensor(mk, dtype=torch.float32, device=dev), G, gname).view(n, 32)
                    score += lps[vi].gather(1, ai)
        cand[np.arange(n), score.argmax(1).cpu().numpy()]
        torch.cuda.synchronize(); T_["orbit_scoring"] += time.perf_counter() - t0

    SYSTEMS = [(f"{fam} {v} {'orbit' if o else 'greedy'}", fam, v == "PoE", o) for fam in FAMS for v in ("DINO", "PoE") for o in (False, True)]
    res = []
    for n in (1, 64):
        for _, fam, wm, o in SYSTEMS:                                  # warm-up, all systems
            for _ in range(2):
                system(fam, n, wm, o, {k: 0.0 for k in COMP})
        reps = {s[0]: [] for s in SYSTEMS}; peak = {s[0]: 0.0 for s in SYSTEMS}
        for _ in range(5):                                             # interleaved rounds
            for name, fam, wm, o in SYSTEMS:
                torch.cuda.synchronize(); base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats(); T_ = {k: 0.0 for k in COMP}
                t0 = time.perf_counter(); system(fam, n, wm, o, T_); torch.cuda.synchronize()
                T_["total"] = time.perf_counter() - t0; reps[name].append(T_); peak[name] = max(peak[name], (torch.cuda.max_memory_allocated() - base) / 2**20)
        load = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
        for name, fam, wm, o in SYSTEMS:
            med = {k: float(np.median([r[k] for r in reps[name]])) * 1000 / n for k in reps[name][0]}
            res.append({"system": name, "family": fam, "variant": "PoE" if wm else "DINO", "inference": "orbit" if o else "greedy", "batch": n,
                        **{f"ms_per_img_{k}": v for k, v in med.items()}, "ms_per_img_head_total": med["head"] + med["orbit_scoring"],
                        "fps_or_img_per_s": 1000.0 / med["total"], "weights_mb": W_DINO + (W_MOGE if wm else 0) + wmb(MODELS[fam] if wm else MODELS[fam][:1]),
                        "transient_peak_mb": peak[name], "gpu_util_mem_used": load})
            res[-1]["inference_mem_mb"] = res[-1]["weights_mb"] + res[-1]["transient_peak_mb"]
            print(res[-1], flush=True)
    del dino, moge; torch.cuda.empty_cache()

    # training-step memory / time per branch (20 steps, batch 256, the trainers' optimizer and loss)
    train = []
    for fam, grid in FAMS.items():
        for sp in (SD, SM):
            torch.manual_seed(0)
            m = (P.HierTransformer(SHAPE[sp], 5) if grid is None else FG.FlatTransformer(SHAPE[sp], len(GRIDS[fam]))).to(dev).train()
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05)
            q = torch.nn.functional.normalize(torch.randn(256, 4, dtype=torch.float64), dim=-1).numpy()
            x = torch.randn(256, SHAPE[sp][0] * SHAPE[sp][1], device=dev)
            if grid is None:
                c, p = T.encode(q, 5); rs, cs = P.soft_targets(q, c, p)
                c, p, rs, cs = (torch.as_tensor(a, device=dev) for a in (c, p, rs, cs))
            else:
                ai, w = FG.soft_anchor_targets(torch.as_tensor(q, dtype=torch.float32, device=dev), GRIDS[fam])
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); ts = []
            for it in range(20):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if grid is None:
                        rl, cl = m.logits_tf(m.encode(x), c, p)
                        loss = P.soft_ce(rl.float(), rs) + sum(P.soft_ce(cl[:, l].float(), cs[:, l]) for l in range(cl.shape[1]))
                    else:
                        loss = FG.soft_ce(m.logits(m.encode(x)), ai, w)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); torch.cuda.synchronize()
                if it >= 5:
                    ts.append(time.perf_counter() - t0)
            train.append({"family": fam, "features": sp, "train_peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20, "train_ms_per_step_b256": float(np.median(ts)) * 1000})
            print(train[-1], flush=True); del m, opt; torch.cuda.empty_cache()
    cost = pd.DataFrame(NPARAM).merge(pd.DataFrame(train), on=["family", "features"])
    run.write_metrics(res, "timing"); run.write_metrics(cost.to_dict("records"), "params_memory")
    (run.sub("tables") / "timing.md").write_text(md_table(res, ["system", "batch", "ms_per_img_total", "ms_per_img_head_total", "ms_per_img_head",
                                                               "ms_per_img_orbit_scoring", "fps_or_img_per_s", "weights_mb", "transient_peak_mb", "inference_mem_mb"]))
    (run.sub("tables") / "params_memory.md").write_text(md_table(cost.to_dict("records")))
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    run.done(gpu=gpu); print((run.sub("tables") / "timing.md").read_text()); print((run.sub("tables") / "params_memory.md").read_text()); print("GPU", gpu); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
