"""Arm K trainer (EXP-105 / EXP-106 amendment): original Kurz 2017 corner formulation as an 8-facet Transformer tree.

Same encoder/decoder, optimiser, budget and checkpoint rule as the EXP-036 recipe (train_probe_p3.py --decoder
transformer); only the root (8 facets), the targets (multi-path corner targets, tfpose.kurz8.corner_targets) and the
decoding (top-8 corner vote) change. Loss = the EXP-044 level-marginal cross-entropy with the corner target leaves in
place of flow samples: -(1/(L+1)) sum_l log p_l(prefix_l(leaf)), averaged over --k of the target leaves drawn
uniformly per image per step (an unbiased estimate of the uniform-target cross-entropy).
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
from tfpose import featsets, kurz8 as K8  # noqa: E402
from tfpose.metrics import rotation_metrics  # noqa: E402
from tfpose.runlog import Run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--features", required=True)
ap.add_argument("--token-shape", required=True)
ap.add_argument("--subset", default="full")
ap.add_argument("--feat-device", default="cpu", choices=["gpu", "cpu"])
ap.add_argument("--depth", type=int, default=5)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--eval-every", type=int, default=5)
ap.add_argument("--k", type=int, default=8, help="target leaves teacher-forced per image per step")
ap.add_argument("--vote-beam", type=int, default=8)
ap.add_argument("--amp", default="bf16", choices=["none", "bf16"])
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
    groups = {"synthetic_train": (df.domain == "synthetic") & (df.split == "train"),
              "synthetic_val": (df.domain == "synthetic") & (df.split == "validation"),
              "lightbox": df.domain == "lightbox", "sunlamp": df.domain == "sunlamp"}
    tr = groups["synthetic_train"].to_numpy()
    Xt = featsets.standardize_to_tensor(X, tr, dev if args.feat_device == "gpu" else "cpu", torch.float16, pin=False)
    d_in = X.shape[1]; del X

    def getx(b):
        return (Xt[b] if args.feat_device == "gpu" else Xt[b.cpu()].to(dev, non_blocking=True)).float()
    tc, tp, tv = K8.corner_targets(q, args.depth)
    TCt, TPt, TVt = (torch.as_tensor(a, device=dev) for a in (tc, tp, tv))
    ts = tuple(int(v) for v in args.token_shape.split("x")); assert ts[0] * ts[1] == d_in, (ts, d_in)
    model = K8.HierTransformer8(ts, args.depth, dropout=args.dropout).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    tr_idx = np.flatnonzero(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * int(np.ceil(len(tr_idx) / args.batch)), pct_start=0.05)

    def loss_fn(b):
        h = model.encode(getx(b)); B, Kk = len(b), args.k
        w = TVt[b].float()                                                           # uniform over valid targets
        m = torch.multinomial(w, Kk, replacement=True)                                # (B, k)
        sc = TCt[b].gather(1, m).reshape(-1); sp = TPt[b].gather(1, m[..., None].expand(-1, -1, args.depth)).reshape(B * Kk, -1)
        rl, cl = model.logits_tf(h.repeat_interleave(Kk, 0), sc, sp)
        step = torch.cat([F.log_softmax(rl.float(), -1).gather(1, sc[:, None]),
                          F.log_softmax(cl.float(), -1).gather(2, sp[:, :, None])[..., 0]], 1)    # (Bk, L+1)
        return -step.cumsum(1).mean()

    @torch.no_grad()
    def evaluate(mask):
        model.eval(); idx = np.flatnonzero(mask); qv, qg = [], []
        for s in range(0, len(idx), 1024):
            b = torch.as_tensor(idx[s:s + 1024], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp == "bf16"):
                c, p, sc = K8.beam8([model], [getx(b)], "single", args.vote_beam)
            qv.append(K8.corner_vote(c, p, sc).cpu().numpy()); qg.append(K8.decode8_torch(c[:, 0], p[:, 0]).cpu().numpy())
        qv, qg = np.concatenate(qv), np.concatenate(qg)
        out = rotation_metrics(qv, q[idx]); out.update({f"top1centre_{k}": v for k, v in rotation_metrics(qg, q[idx]).items()})
        return out, qv

    log = open(run.sub("logs") / "train.log", "w")
    best = (np.inf, None, 0)
    for ep in range(1, args.epochs + 1):
        model.train(); perm = np.random.permutation(tr_idx); tot = 0.0
        for s in range(0, len(perm), args.batch):
            b = torch.as_tensor(perm[s:s + args.batch], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp == "bf16"):
                loss = loss_fn(b)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(b)
        if ep % args.eval_every == 0 or ep == args.epochs:
            vm, _ = evaluate(groups["synthetic_val"].to_numpy())
            print(f"ep {ep} train_loss {tot / len(perm):.4f} val_vote_mean {vm['mean_deg']:.3f} val_vote_median {vm['median_deg']:.3f} "
                  f"top1centre_mean {vm['top1centre_mean_deg']:.3f}", file=log, flush=True)
            if vm["mean_deg"] < best[0]:
                best = (vm["mean_deg"], {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
    model.load_state_dict(best[1])
    torch.save({"model": best[1]}, run.sub("checkpoints") / "best.pt")
    rows, pred_rows = [], []
    for g, m in groups.items():
        if g == "synthetic_train":
            continue
        r, qp = evaluate(m.to_numpy()); idx = np.flatnonzero(m.to_numpy())
        rows.append({"features": args.features, "arm": "K_kurz8", "seed": args.seed, "domain": g, **r})
        err = np.degrees(2 * np.arccos(np.clip(np.abs((qp * q[idx]).sum(1)), 0, 1)))
        pred_rows.append(pd.DataFrame({"image_relpath": df.image_relpath.values[idx], "domain": g,
                                       **{f"q_pred_{k}": qp[:, i] for i, k in enumerate("wxyz")}, "err_deg": err}))
    run.write_metrics(rows, "metrics")
    pd.concat(pred_rows).to_csv(run.sub("predictions") / "predictions_vote.csv", index=False, float_format="%.6g")
    run.done(best_epoch=best[2], n_params=n_params, trainable_params=n_params, feature_dim=int(d_in))
    print(f"{args.features} K seed {args.seed} " + " | ".join(f"{r['domain']}: vote {r['mean_deg']:.2f} med {r['median_deg']:.2f} "
                                                             f"(top1 centre {r['top1centre_mean_deg']:.2f})" for r in rows))
    print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc())
    raise
