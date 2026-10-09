"""EXP-102: LoRA backbone adaptation (DINOv3-L or MoGe-2) + simple MLP Tesseract head, trained on synthetic train.

The backbone runs live on GT crops (images read and cropped on the fly; the dataset is never modified).
Features are built exactly like the frozen pipeline (DINOv3: last-layer 4x4 grid; MoGe-2: normals masked
by the cached EXP-015 consensus mask, 16x16) and standardized with the FROZEN features' train-split
statistics, so at initialization (LoRA B = 0) the model sees the frozen features.
After training, adapted features for the evaluation splits are cached for the PoE (fused) arm.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import data as D, featsets, tesseract as T  # noqa: E402
from tfpose.adapters import inject_lora, lora_parameters, lora_state_dict  # noqa: E402
from tfpose.features import DinoV3, MoGe2Extractor  # noqa: E402
from tfpose.metrics import path_metrics, rotation_metrics  # noqa: E402
from tfpose.predictor import HierMLP, parent_centres  # noqa: E402
from tfpose.runlog import Run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--branch", required=True, choices=["dino", "moge"])
ap.add_argument("--exp", default="EXP-102_lora_adapters")
ap.add_argument("--subset", default="full")
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--accum", type=int, default=1)
ap.add_argument("--lora-rank", type=int, default=8)
ap.add_argument("--lora-alpha", type=float, default=16.0)
ap.add_argument("--lr-lora", type=float, default=1e-4)
ap.add_argument("--lr-head", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--max-train", type=int, default=0, help="debug: limit training images")
ap.add_argument("--max-steps", type=int, default=0, help="debug: stop after this many optimizer steps")
ap.add_argument("--max-eval", type=int, default=0, help="debug: limit images per evaluation split")
ap.add_argument("--grad-ckpt", action="store_true", help="MoGe-2: gradient checkpointing (encoder, neck, heads) to fit memory")
ap.add_argument("--train-frac", type=float, default=1.0, help="fixed seeded fraction of synthetic train (pre-registered compute cap)")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
exp_id, short = args.exp.split("_", 1)
run = Run(exp_id, short, dict(vars(args), backbone_adaptation=True), seed=args.seed)
try:
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda")
    paths = yaml.safe_load(open(REPO / "configs/paths.yaml")); root = Path(paths["speedplus_root"])
    df = pd.read_csv(REPO / f"outputs/data_manifests/subset_{args.subset}.csv")
    q = df[[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    chart, path = T.encode(q, 5); pq = parent_centres(chart, path)
    dom = np.where(df.domain == "synthetic", np.where(df.split == "train", "synthetic_train", "synthetic_val"), df.domain)
    spec = "dinov3_vitl16:grid4" if args.branch == "dino" else "moge2_vitl:normals16~and"
    names, Xf, _ = featsets.load(spec, args.subset); assert names == df.image_relpath.tolist()
    trm = dom == "synthetic_train"
    mu = torch.tensor(Xf[trm].mean(0), device=dev); sd = torch.tensor(Xf[trm].std(0) + 1e-6, device=dev)
    np.savez(run.dir / "frozen_feature_stats.npz", mu=mu.cpu().numpy(), sd=sd.cpu().numpy())
    del Xf
    amask = featsets.consensus_mask(args.subset, names) if args.branch == "moge" else None      # (N,74,74,1), fixed

    # ---------------- backbone + LoRA ----------------
    if args.branch == "dino":
        ex = DinoV3("vit_large_patch16_dinov3.lvd1689m", dev); bb = ex.model; size = ex.size
        n_lora = inject_lora(bb, r=args.lora_rank, alpha=args.lora_alpha)
    else:
        ex = MoGe2Extractor(dev); bb = ex.model; size = 518
        n_lora = inject_lora(bb.encoder.backbone, r=args.lora_rank, alpha=args.lora_alpha)
        if args.grad_ckpt:
            bb.enable_gradient_checkpointing()
    bb.to(dev)

    def features(x, idx):
        """x: (B,3,H,W) in [0,1]; idx: dataset indices (for the fixed consensus mask). Returns standardized feature."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if args.branch == "dino":
                _, inter = bb.forward_intermediates((x - ex.mean) / ex.std, indices=[23], return_prefix_tokens=True, norm=True, output_fmt="NLC")
                g = inter[-1][0].float().reshape(-1, 16, 16, 1024).permute(0, 3, 1, 2)
                f = F.adaptive_avg_pool2d(g, 4).permute(0, 2, 3, 1).reshape(len(x), -1)
            else:
                nrm = bb.forward(x, num_tokens=37 * 37)["normal"].float()                       # (B,518,518,3), unit normals
                nrm = F.adaptive_avg_pool2d(nrm.permute(0, 3, 1, 2), 74)                         # (B,3,74,74)
                m = torch.as_tensor(amask[idx], device=dev).permute(0, 3, 1, 2)                 # (B,1,74,74)
                f = torch.cat([F.adaptive_avg_pool2d(nrm * m, 16), F.adaptive_avg_pool2d(m, 16)], 1).permute(0, 2, 3, 1).reshape(len(x), -1)
        return (f.float() - mu) / sd

    class Crops(torch.utils.data.Dataset):
        def __init__(self, idx): self.idx = idx
        def __len__(self): return len(self.idx)
        def __getitem__(self, k):
            i = self.idx[k]; r = df.iloc[i]
            img = D.crop_square(D.read_gray(root, r.image_relpath), r.crop_cx, r.crop_cy, r.crop_side, size)
            return torch.from_numpy(img).float().div_(255.0)[None].expand(3, -1, -1).contiguous(), i

    def loader(idx, shuffle):
        return torch.utils.data.DataLoader(Crops(idx), batch_size=args.batch, shuffle=shuffle, num_workers=args.workers,
                                           pin_memory=True, drop_last=shuffle, persistent_workers=False)

    head = HierMLP(int(mu.numel()), 5).to(dev)
    Ct, Pt, PQt = (torch.as_tensor(a, device=dev) for a in (chart, path, pq.astype(np.float32)))
    tr_idx = np.flatnonzero(trm)
    if args.train_frac < 1.0:   # fixed subset (seed 0 for every run, so all seeds/arms see the same images)
        tr_idx = np.sort(np.random.default_rng(0).choice(tr_idx, int(round(args.train_frac * len(tr_idx))), replace=False))
    if args.max_train:
        tr_idx = np.random.default_rng(args.seed).choice(tr_idx, args.max_train, replace=False)
    steps_per_epoch = len(tr_idx) // (args.batch * args.accum)
    total = args.epochs * steps_per_epoch if not args.max_steps else args.max_steps
    opt = torch.optim.AdamW([{"params": lora_parameters(bb), "lr": args.lr_lora, "weight_decay": 0.0},
                             {"params": head.parameters(), "lr": args.lr_head, "weight_decay": args.wd}])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr_lora, args.lr_head], total_steps=max(total, 4), pct_start=max(0.05, 2.5 / max(total, 4)))

    @torch.no_grad()
    def evaluate(idx, keep_feats=False):
        bb.eval(); head.eval(); cs, ps, fs = [], [], []
        for x, i in loader(idx, False):
            f = features(x.to(dev, non_blocking=True), i.numpy())
            c, p, _, _ = head.beam_search(f, beam=1)
            cs.append(c[:, 0].cpu().numpy()); ps.append(p[:, 0].cpu().numpy())
            if keep_feats:
                fs.append(f.half().cpu().numpy())
        cp, pp = np.concatenate(cs), np.concatenate(ps)
        res = {**rotation_metrics(T.decode(cp, pp), q[idx]), **path_metrics(cp, pp, chart[idx], path[idx])}
        return res, cp, pp, (np.concatenate(fs) if keep_feats else None)

    cap = (lambda a: a[:args.max_eval]) if args.max_eval else (lambda a: a)
    val_idx = cap(np.flatnonzero(dom == "synthetic_val"))
    log = open(run.sub("logs") / "train.log", "w"); best = (np.inf, None, 0); step = 0; t0 = time.time()
    print(f"trainable LoRA params {n_lora:,}; head params {sum(p.numel() for p in head.parameters()):,}; "
          f"{steps_per_epoch} steps/epoch", file=log, flush=True)
    for ep in range(1, args.epochs + 1):
        bb.train(); head.train(); tot, nb = 0.0, 0
        for k, (x, i) in enumerate(loader(tr_idx, True)):
            b = torch.as_tensor(i, device=dev)
            f = features(x.to(dev, non_blocking=True), i.numpy())
            loss, _ = head.loss(f, Ct[b], Pt[b], PQt[b])
            (loss / args.accum).backward(); tot += loss.item(); nb += 1
            if (k + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(lora_parameters(bb) + list(head.parameters()), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
                if step % 50 == 0:
                    print(f"ep {ep} step {step}/{total} loss {tot / nb:.4f} {(time.time() - t0) / step:.2f} s/step", file=log, flush=True)
                if args.max_steps and step >= args.max_steps:
                    break
        vm, _, _, _ = evaluate(val_idx)
        print(f"ep {ep} train_loss {tot / max(nb, 1):.4f} val_mean_deg {vm['mean_deg']:.3f} val_median {vm['median_deg']:.3f} "
              f"elapsed {(time.time() - t0) / 60:.1f} min", file=log, flush=True)
        if vm["mean_deg"] < best[0]:
            best = (vm["mean_deg"], ({k: v.detach().clone() for k, v in lora_state_dict(bb).items()},
                                     {k: v.detach().clone() for k, v in head.state_dict().items()}), ep)
        if args.max_steps and step >= args.max_steps:
            break
    bb.load_state_dict(best[1][0], strict=False); head.load_state_dict(best[1][1])
    torch.save({"lora": best[1][0], "head": best[1][1]}, run.sub("checkpoints") / "best.pt")
    rows = []
    cache = REPO / f"outputs/shared_cache/adapted__{run.exp}__{args.branch}__seed{args.seed}__{run.dir.name}"
    cache.mkdir(parents=True, exist_ok=False)
    with h5py.File(cache / "features.h5", "w") as fh:
        for g in ["synthetic_val", "lightbox", "sunlamp"]:
            idx = cap(np.flatnonzero(dom == g))
            r, cp, pp, fs = evaluate(idx, keep_feats=True)
            rows.append({"branch": args.branch, "seed": args.seed, "domain": g, **r})
            fh.create_dataset(f"{g}/feat", data=fs); fh.create_dataset(f"{g}/index", data=idx)
    (cache / "manifest.json").write_text(json.dumps({"what": "adapted, frozen-standardized features for eval splits", "spec": spec,
                                                     "producer_run": run.rel()}, indent=2))
    run.write_metrics(rows, "metrics")
    run.done(best_epoch=best[2], lora_params=n_lora, head_params=sum(p.numel() for p in head.parameters()), adapted_cache=str(cache.relative_to(REPO)))
    print(f"adapter {args.branch} seed {args.seed} " + " | ".join(f"{r['domain']}: mean {r['mean_deg']:.2f} med {r['median_deg']:.2f}" for r in rows))
    print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc())
    raise
