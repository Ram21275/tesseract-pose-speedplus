"""Phase-3 probe trainer: EXP-030 MLP tree + options --decoder {mlp,gru}, --targets {hard,soft}, --residual.

With --decoder mlp --targets hard and no residual this is the EXP-030 configuration. Evaluation
reports greedy/beam decoding, and (with --residual) both the discrete leaf centre and the refined rotation.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import featsets, phase3 as P, tesseract as T  # noqa: E402
from tfpose.metrics import path_metrics, rotation_metrics  # noqa: E402
from tfpose.predictor import HierMLP, decode_torch, parent_centres  # noqa: E402
from tfpose.runlog import Run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--features", required=True)
ap.add_argument("--subset", default="full")
ap.add_argument("--decoder", default="mlp", choices=["mlp", "gru", "transformer"])
ap.add_argument("--token-shape", default=None, help="TxC token grid for --decoder transformer, e.g. 64x1024 or 256x4")
ap.add_argument("--fuse", default="linear", choices=["linear", "mlp"], help="transformer input layer (EXP-029 arms 4/5)")
ap.add_argument("--feat-device", default="gpu", choices=["gpu", "cpu"], help="cpu = pinned host memory, batches copied to GPU")
ap.add_argument("--targets", default="hard", choices=["hard", "soft"])
ap.add_argument("--residual", action="store_true")
ap.add_argument("--res-weight", type=float, default=1.0)
ap.add_argument("--depth", type=int, default=5)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--hidden", type=int, default=512)
ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--eval-every", type=int, default=5)
ap.add_argument("--beams", default="1,4")
ap.add_argument("--gpu-dtype", default="fp16", choices=["fp32", "fp16"])
ap.add_argument("--amp", default="none", choices=["none", "bf16"], help="autocast for the probe forward pass (speed); features and losses as before")
ap.add_argument("--train-frac", type=float, default=1.0, help="fixed seeded (rng 0) fraction of synthetic train, as in train_adapter.py")
ap.add_argument("--clip", type=float, default=0.0, help="gradient-norm clip (0 = off)")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
exp_id, short = args.exp.split("_", 1)
run = Run(exp_id, short, vars(args), seed=args.seed)
try:
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda")
    df = pd.read_csv(REPO / f"outputs/data_manifests/subset_{args.subset}.csv")
    names, X, cache_info = featsets.load(args.features, args.subset)
    if names != df.image_relpath.tolist():
        raise ValueError("feature cache order does not match subset manifest")
    (run.dir / "feature_source.json").write_text(json.dumps(cache_info, indent=2))
    q = df[[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    chart, path = T.encode(q, args.depth)
    pq = parent_centres(chart, path)
    groups = {"synthetic_train": (df.domain == "synthetic") & (df.split == "train"),
              "synthetic_val": (df.domain == "synthetic") & (df.split == "validation"),
              "lightbox": df.domain == "lightbox", "sunlamp": df.domain == "sunlamp"}
    tr = groups["synthetic_train"].to_numpy()
    Xt = featsets.standardize_to_tensor(X, tr, dev if args.feat_device == "gpu" else "cpu",
                                        torch.float16 if args.gpu_dtype == "fp16" else torch.float32, pin=False)  # pinned host memory rounds up to a power of two (9 GB -> 16 GB)
    d_in = X.shape[1]; del X

    def getx(b):  # b: index tensor on GPU
        return (Xt[b] if args.feat_device == "gpu" else Xt[b.cpu()].to(dev, non_blocking=True)).float()
    Ct, Pt, PQt = torch.as_tensor(chart, device=dev), torch.as_tensor(path, device=dev), torch.as_tensor(pq, dtype=torch.float32, device=dev)
    if args.targets == "soft":
        rs, cs = P.soft_targets(q, chart, path)
        RSt, CSt = torch.as_tensor(rs, device=dev), torch.as_tensor(cs, device=dev)
    if args.residual:
        Dt = torch.as_tensor(P.leaf_rotvec_targets(q, chart, path), device=dev)
        QLt = torch.as_tensor(T.decode(chart, path), dtype=torch.float32, device=dev)
    if args.decoder == "transformer":
        if args.residual:
            raise ValueError("--residual is not defined for the transformer decoder")
        ts = tuple(int(v) for v in args.token_shape.split("x")); assert ts[0] * ts[1] == d_in, (ts, d_in)
        model = P.HierTransformer(ts, args.depth, dropout=args.dropout, fuse=args.fuse).to(dev)
    elif args.decoder == "gru":
        model = P.HierGRU(d_in, args.depth, hidden=args.hidden, dropout=args.dropout).to(dev)
    else:
        model = HierMLP(d_in, args.depth, hidden=args.hidden, dropout=args.dropout).to(dev)
    res = P.ResidualHead(args.hidden).to(dev) if args.residual else None
    params = list(model.parameters()) + (list(res.parameters()) if res else [])
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    tr_idx = np.flatnonzero(tr)
    if args.train_frac < 1.0:
        tr_idx = np.sort(np.random.default_rng(0).choice(tr_idx, int(round(args.train_frac * len(tr_idx))), replace=False))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * int(np.ceil(len(tr_idx) / args.batch)), pct_start=0.05)
    res_scale = np.radians(P.TAU_DEG[args.depth])                     # residual loss in units of the L5 cell size

    def losses(b):
        x = getx(b); h = model.encode(x)
        if args.decoder == "transformer":
            rl, cl = model.logits_tf(h, Ct[b], Pt[b])
        elif args.decoder == "mlp":
            B, L = Pt[b].shape
            hl = h[:, None].expand(B, L, h.shape[-1]).reshape(B * L, -1)
            lev = torch.arange(L, device=dev).repeat(B)
            cl = model.child_logits(hl, lev, PQt[b].reshape(B * L, 4)).view(B, L, 8)
        else:
            cl = model.child_logits_tf(h, Ct[b], Pt[b], PQt[b])
        if args.decoder != "transformer":
            rl = model.root(h)
        if args.targets == "soft":
            loss = P.soft_ce(rl, RSt[b]) + sum(P.soft_ce(cl[:, l], CSt[b][:, l]) for l in range(cl.shape[1]))
        else:
            loss = F.cross_entropy(rl, Ct[b]) + sum(F.cross_entropy(cl[:, l], Pt[b][:, l]) for l in range(cl.shape[1]))
        if res is not None:   # teacher-forced on the GT leaf
            loss = loss + args.res_weight * (((res(h, QLt[b]) - Dt[b]) / res_scale) ** 2).sum(-1).mean()
        return loss

    @torch.no_grad()
    def evaluate(mask, beam):
        model.eval(); res is not None and res.eval()
        idx = np.flatnonzero(mask); cs_, ps_, qr_, hits = [], [], [], []
        for s in range(0, len(idx), 1024):
            b = torch.as_tensor(idx[s:s + 1024], device=dev); x = getx(b)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp == "bf16"):
                if args.decoder == "mlp":
                    c, p, _, _ = model.beam_search(x, beam=beam)
                else:
                    c, p, _ = model.beam_search(x, beam=beam)
            c0, p0 = c[:, 0], p[:, 0]
            if res is not None:
                ql = decode_torch(c0, p0)
                qr_.append(P.qmul(ql, P.rotvec_to_quat(res(model.encode(x), ql))).cpu().numpy())
            cs_.append(c0.cpu().numpy()); ps_.append(p0.cpu().numpy())
            gt_leaf = T.leaf_id(chart[idx[s:s + 1024]], path[idx[s:s + 1024]])
            cand = np.stack([T.leaf_id(c[:, k].cpu().numpy(), p[:, k].cpu().numpy()) for k in range(c.shape[1])], 1)
            hits.append(np.any(cand == gt_leaf[:, None], 1))
        cp, pp = np.concatenate(cs_), np.concatenate(ps_)
        out = {"beam": beam, **path_metrics(cp, pp, chart[idx], path[idx]), f"top{beam}_leaf_recall": float(np.mean(np.concatenate(hits)))}
        disc = rotation_metrics(T.decode(cp, pp), q[idx])
        out.update({f"discrete_{k}": v for k, v in disc.items()})
        if res is not None:
            out.update(rotation_metrics(np.concatenate(qr_), q[idx]))
        else:
            out.update(disc)
        return out, cp, pp, (np.concatenate(qr_) if res is not None else T.decode(cp, pp))

    log = open(run.sub("logs") / "train.log", "w")
    best = (np.inf, None, 0)
    for ep in range(1, args.epochs + 1):
        model.train(); res is not None and res.train()
        perm = np.random.permutation(tr_idx); tot = 0.0
        for s in range(0, len(perm), args.batch):
            b = torch.as_tensor(perm[s:s + args.batch], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp == "bf16"):
                loss = losses(b)
            opt.zero_grad(set_to_none=True); loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.clip)
            opt.step(); sched.step()
            tot += loss.item() * len(b)
        if ep % args.eval_every == 0 or ep == args.epochs:
            vm, _, _, _ = evaluate(groups["synthetic_val"].to_numpy(), 1)
            print(f"ep {ep} train_loss {tot / len(perm):.4f} val_mean_deg {vm['mean_deg']:.3f} val_median {vm['median_deg']:.3f} "
                  f"discrete_mean {vm['discrete_mean_deg']:.3f}", file=log, flush=True)
            if vm["mean_deg"] < best[0]:
                best = (vm["mean_deg"], ({k: v.detach().clone() for k, v in model.state_dict().items()},
                                         {k: v.detach().clone() for k, v in res.state_dict().items()} if res else None), ep)
    model.load_state_dict(best[1][0])
    if res is not None:
        res.load_state_dict(best[1][1])
    torch.save({"model": best[1][0], "residual": best[1][1]}, run.sub("checkpoints") / "best.pt")
    rows, pred_rows = [], []
    for beam in [int(b) for b in args.beams.split(",")]:
        for g, m in groups.items():
            if g == "synthetic_train":
                continue
            r, cp, pp, qp = evaluate(m.to_numpy(), beam)
            rows.append({"features": args.features, "decoder": args.decoder, "targets": args.targets, "residual": args.residual,
                         "seed": args.seed, "domain": g, **r})
            if beam == 1:
                idx = np.flatnonzero(m.to_numpy())
                err = np.degrees(2 * np.arccos(np.clip(np.abs((qp * q[idx]).sum(1)), 0, 1)))
                pred_rows.append(pd.DataFrame({"image_relpath": df.image_relpath.values[idx], "domain": g, "chart_pred": cp,
                                               "path_pred": ["".join(map(str, x)) for x in pp],
                                               **{f"q_pred_{k}": qp[:, i] for i, k in enumerate("wxyz")}, "err_deg": err}))
    run.write_metrics(rows, "metrics")
    pd.concat(pred_rows).to_csv(run.sub("predictions") / "predictions_greedy.csv", index=False, float_format="%.6g")
    run.done(best_epoch=best[2], n_params=n_params, trainable_params=n_params, feature_dim=int(d_in))
    g1 = [r for r in rows if r["beam"] == 1]
    print(f"{args.features} {args.decoder}/{args.targets}/res={args.residual} seed {args.seed} " +
          " | ".join(f"{r['domain']}: mean {r['mean_deg']:.2f} med {r['median_deg']:.2f} (disc {r['discrete_mean_deg']:.2f})" for r in g1))
    print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc())
    raise
