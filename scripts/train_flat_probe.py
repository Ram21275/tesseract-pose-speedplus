"""EXP-101 part B: SPACE-HOP-style flat anchor classifier on frozen features (same trunk/budget as train_probe.py).

Target = closest anchor (SPACE-HOP get_closest_anchor); prediction = argmax anchor; no continuous offset.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import featsets  # noqa: E402
from tfpose.grids import nearest, spacehop_hopf_grid  # noqa: E402
from tfpose.metrics import rotation_metrics  # noqa: E402
from tfpose.predictor import FlatAnchorMLP  # noqa: E402
from tfpose.runlog import Run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default="EXP-101_tesseract_vs_spacehop_hopf")
ap.add_argument("--features", default="dinov3_vitl16:grid4")
ap.add_argument("--subset", default="v1")
ap.add_argument("--hopf-points", type=int, default=256)
ap.add_argument("--hopf-rolls", type=int, default=12)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--hidden", type=int, default=512)
ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--eval-every", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
exp_id, short = args.exp.split("_", 1)
run = Run(exp_id, short, dict(vars(args), head="flat_spacehop_hopf"), seed=args.seed)
try:
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda")
    df = pd.read_csv(REPO / f"outputs/data_manifests/subset_{args.subset}.csv")
    names, X, cache_info = featsets.load(args.features, args.subset)
    if names != df.image_relpath.tolist():
        raise ValueError("feature cache order does not match subset manifest")
    (run.dir / "feature_source.json").write_text(json.dumps(cache_info, indent=2))
    q = df[[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    anchors = spacehop_hopf_grid(args.hopf_points, args.hopf_rolls)
    label, qerr = nearest(q, anchors)                                   # closest anchor = CE target
    groups = {"synthetic_train": (df.domain == "synthetic") & (df.split == "train"),
              "synthetic_val": (df.domain == "synthetic") & (df.split == "validation"),
              "lightbox": df.domain == "lightbox", "sunlamp": df.domain == "sunlamp"}
    tr = groups["synthetic_train"].to_numpy()
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, device=dev)
    Yt = torch.as_tensor(label, device=dev)
    model = FlatAnchorMLP(X.shape[1], len(anchors), hidden=args.hidden, dropout=args.dropout).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    tr_idx = np.flatnonzero(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * int(np.ceil(len(tr_idx) / args.batch)), pct_start=0.05)

    @torch.no_grad()
    def evaluate(mask):
        model.eval()
        idx = np.flatnonzero(mask)
        preds, top5 = [], []
        for s in range(0, len(idx), 1024):
            lg = model(Xt[idx[s:s + 1024]])
            preds.append(lg.argmax(-1).cpu().numpy())
            top5.append((lg.topk(5, -1).indices == Yt[idx[s:s + 1024]][:, None]).any(-1).cpu().numpy())
        p = np.concatenate(preds)
        out = rotation_metrics(anchors[p], q[idx])
        out.update({"anchor_acc": float(np.mean(p == label[idx])), "top5_anchor_recall": float(np.mean(np.concatenate(top5))),
                    "quantization_floor_mean_deg": float(np.degrees(qerr[idx]).mean())})
        return out, p

    log = open(run.sub("logs") / "train.log", "w")
    best = (np.inf, None, 0)
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(tr_idx); tot = 0.0
        for s in range(0, len(perm), args.batch):
            b = torch.as_tensor(perm[s:s + args.batch], device=dev)
            loss = F.cross_entropy(model(Xt[b]), Yt[b])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(b)
        if ep % args.eval_every == 0 or ep == args.epochs:
            vm, _ = evaluate(groups["synthetic_val"].to_numpy())
            print(f"ep {ep} train_loss {tot / len(perm):.4f} val_mean_deg {vm['mean_deg']:.2f} val_median {vm['median_deg']:.2f}", file=log, flush=True)
            if vm["mean_deg"] < best[0]:
                best = (vm["mean_deg"], {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
    model.load_state_dict(best[1])
    rows, pred_rows = [], []
    for g, m in groups.items():
        if g == "synthetic_train":
            continue
        res, p = evaluate(m.to_numpy())
        rows.append({"features": args.features, "head": f"flat_hopf_{args.hopf_points}x{args.hopf_rolls}", "n_anchors": len(anchors),
                     "seed": args.seed, "domain": g, **res})
        idx = np.flatnonzero(m.to_numpy())
        err = np.degrees(2 * np.arccos(np.clip(np.abs((anchors[p] * q[idx]).sum(1)), 0, 1)))
        pred_rows.append(pd.DataFrame({"image_relpath": df.image_relpath.values[idx], "domain": g, "anchor_gt": label[idx],
                                       "anchor_pred": p, "err_deg": err}))
    run.write_metrics(rows, "metrics")
    pd.concat(pred_rows).to_csv(run.sub("predictions") / "predictions_argmax.csv", index=False, float_format="%.6g")
    run.done(best_epoch=best[2], n_params=n_params, trainable_params=n_params, feature_dim=int(X.shape[1]), n_anchors=len(anchors))
    print(f"flat_hopf {len(anchors)} seed {args.seed} " + " | ".join(f"{r['domain']}: mean {r['mean_deg']:.1f} med {r['median_deg']:.1f}" for r in rows))
    print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc())
    raise
