"""EXP-113 trainer: matched flat Hopf classifier (src/tfpose/flatgrid.py) with the locked EXP-109 recipe: EXP-036
budget (100 epochs, batch 256, AdamW 1e-3, wd 0.05, OneCycle, bf16), same frozen features, same augmented views
(clean + exp109_train_aug1/2, one per sample per step, labels rotated by the derived convention). Loss: CE against
geodesic soft targets over the 32 nearest anchors (tau = the tree's leaf temperature, 2.51 deg). Checkpoint: lowest
clean synthetic-val mean of the argmax anchor (as the tree's greedy rule)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import data as D, featsets, flatgrid as FG, inplane as IP  # noqa: E402
from tfpose.metrics import rotation_metrics  # noqa: E402
from tfpose.runlog import Run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default="EXP-113_tesseract_vs_flat_hopf")
ap.add_argument("--features", required=True)
ap.add_argument("--token-shape", required=True)
ap.add_argument("--aug-tags", default="exp109_train_aug1,exp109_train_aug2")
ap.add_argument("--aug-convention", required=True)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--eval-every", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
exp_id, short = args.exp.split("_", 1)
run = Run(exp_id, short, {**vars(args), "model": "flat_hopf", "n_anchors": FG.HOPF_POINTS * FG.HOPF_ROLLS}, seed=args.seed)
try:
    torch.manual_seed(args.seed); np.random.seed(args.seed); dev = torch.device("cuda")
    df = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv"); q = df[[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    tr = ((df.domain == "synthetic") & (df.split == "train")).to_numpy()
    groups = {"synthetic_val": ((df.domain == "synthetic") & (df.split == "validation")).to_numpy(),
              "lightbox": (df.domain == "lightbox").to_numpy(), "sunlamp": (df.domain == "sunlamp").to_numpy()}
    names, X, info = featsets.load(args.features, "full"); assert names == df.image_relpath.tolist()
    (run.dir / "feature_source.json").write_text(json.dumps(info, indent=2))
    stats = featsets.train_stats(X, tr); Xt = featsets.standardize_to_tensor(X, tr, "cpu", torch.float16, stats=stats); d_in = X.shape[1]; del X
    G = torch.as_tensor(FG.matched_hopf_grid(), dtype=torch.float32, device=dev)
    tr_rows = np.flatnonzero(tr); POS = torch.full((len(df),), -1, dtype=torch.long, device=dev)
    POS[torch.as_tensor(tr_rows, device=dev)] = torch.arange(len(tr_rows), device=dev)
    cam = D.load_camera(Path(yaml.safe_load(open(REPO / "configs/paths.yaml"))["speedplus_root"]))
    dray = IP.ray_undistorted(cam.K, cam.dist, df.crop_cx.to_numpy()[tr_rows], df.crop_cy.to_numpy()[tr_rows])
    VIEWS = [t for t in args.aug_tags.split(",") if t]; VX, VQ = [], [torch.as_tensor(q[tr_rows], dtype=torch.float32, device=dev)]
    for tag in VIEWS:
        vn, vx, vinfo = featsets.load(args.features, tag); assert vn == df.image_relpath.values[tr_rows].tolist(), tag
        VX.append(featsets.standardize_to_tensor(vx, None, "cpu", torch.float16, stats=stats)); del vx
        ang = pd.read_csv(REPO / "outputs/shared_cache" / Path(vinfo["cache"]).name / "aug.csv").rot_deg.to_numpy()
        VQ.append(torch.as_tensor(IP.rotate_label(q[tr_rows], ang, dray, args.aug_convention), dtype=torch.float32, device=dev))
    VQ = torch.stack(VQ)
    ts = tuple(int(v) for v in args.token_shape.split("x")); assert ts[0] * ts[1] == d_in
    model = FG.FlatTransformer(ts, len(G), dropout=args.dropout).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * int(np.ceil(len(tr_rows) / args.batch)), pct_start=0.05)

    @torch.no_grad()
    def predict(mask):
        model.eval(); idx = np.flatnonzero(mask); out = []
        for s in range(0, len(idx), 1024):
            x = Xt[torch.as_tensor(idx[s:s + 1024])].to(dev).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(G[model.logits(model.encode(x)).float().argmax(-1)].cpu().numpy())
        return idx, np.concatenate(out)

    log = open(run.sub("logs") / "train.log", "w"); best = (np.inf, None, 0)
    for ep in range(1, args.epochs + 1):
        model.train(); perm = np.random.permutation(tr_rows); tot = 0.0
        for s in range(0, len(perm), args.batch):
            b = torch.as_tensor(perm[s:s + args.batch], device=dev); x = Xt[b.cpu()].to(dev).float(); pb = POS[b]
            v = torch.randint(0, len(VIEWS) + 1, (len(b),), device=dev)
            for j in range(len(VIEWS)):
                m = v == j + 1
                if m.any():
                    x[m] = VX[j][pb[m].cpu()].to(dev).float()
            qi, w = FG.soft_anchor_targets(VQ[v, pb], G)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = FG.soft_ce(model.logits(model.encode(x)), qi, w)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(b)
        if ep % args.eval_every == 0 or ep == args.epochs:
            idx, qp = predict(groups["synthetic_val"]); vm = rotation_metrics(qp, q[idx])
            print(f"ep {ep} loss {tot / len(perm):.4f} val_mean {vm['mean_deg']:.3f} median {vm['median_deg']:.3f}", file=log, flush=True)
            if vm["mean_deg"] < best[0]:
                best = (vm["mean_deg"], {k: t.detach().clone() for k, t in model.state_dict().items()}, ep)
    model.load_state_dict(best[1]); torch.save({"model": best[1]}, run.sub("checkpoints") / "best.pt")
    rows = []
    for g, m in groups.items():
        idx, qp = predict(m); rows.append({"features": args.features, "model": "flat_hopf", "seed": args.seed, "domain": g, **rotation_metrics(qp, q[idx])})
    run.write_metrics(rows, "metrics"); run.done(best_epoch=best[2], n_params=n_params, trainable_params=n_params)
    print(args.features, "flat seed", args.seed, " | ".join(f"{r['domain']}: {r['mean_deg']:.2f}" for r in rows)); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
