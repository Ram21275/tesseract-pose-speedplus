"""Phase-1 control: common hierarchical MLP Tesseract predictor on frozen cached features.

Splits (fixed): train = synthetic/train subset; model selection = synthetic/validation
subset; lightbox and sunlamp subsets are scored only (never used for selection).
One run directory per (feature spec, seed).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import featsets, tesseract as T  # noqa: E402
from tfpose.metrics import path_metrics, rotation_metrics  # noqa: E402
from tfpose.predictor import HierMLP, parent_centres  # noqa: E402
from tfpose.runlog import Run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)            # e.g. EXP-010_dinov3_baseline
ap.add_argument("--features", required=True)       # e.g. dinov3_vitb16:l11_cls+mean
ap.add_argument("--subset", default="v1")
ap.add_argument("--depth", type=int, default=5)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--hidden", type=int, default=512)
ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--eval-every", type=int, default=5)
ap.add_argument("--beams", default="1,2,4,8")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

exp_id, short = args.exp.split("_", 1)
run = Run(exp_id, short, vars(args), seed=args.seed)
try:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
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
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6  # train-split statistics only
    Xt = torch.tensor((X - mu) / sd, device=dev)
    Ct, Pt, PQt = (torch.tensor(a, device=dev) for a in (chart, path, pq.astype(np.float32)))

    model = HierMLP(X.shape[1], args.depth, hidden=args.hidden, dropout=args.dropout).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    tr_idx = np.flatnonzero(tr)
    steps = args.epochs * int(np.ceil(len(tr_idx) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)

    def evaluate(mask, beam):
        model.eval()
        idx = np.flatnonzero(mask)
        cs, ps, hits = [], [], {k: [] for k in (1, 5) if k <= beam}
        t0 = time.time()
        for s in range(0, len(idx), 512):
            b = idx[s:s + 512]
            c, p, _, nodes = model.beam_search(Xt[b], beam=beam)
            c, p = c.cpu().numpy(), p.cpu().numpy()
            cs.append(c[:, 0]); ps.append(p[:, 0])
            gt_leaf = T.leaf_id(chart[b], path[b])
            cand = np.stack([T.leaf_id(c[:, k], p[:, k]) for k in range(c.shape[1])], 1)
            for k in hits:
                hits[k].append(np.any(cand[:, :k] == gt_leaf[:, None], 1))
        torch.cuda.synchronize()
        ms = (time.time() - t0) / len(idx) * 1000
        cp, pp = np.concatenate(cs), np.concatenate(ps)
        out = {"beam": beam, "nodes_scored_per_image": nodes, "head_ms_per_image": ms}
        out.update(path_metrics(cp, pp, chart[idx], path[idx], {k: np.concatenate(v) for k, v in hits.items()}))
        for l in range(1, args.depth + 1):
            qp = T.decode(cp, pp[:, :l])
            rm = rotation_metrics(qp, q[idx])
            out[f"L{l}_mean_deg"], out[f"L{l}_median_deg"] = rm["mean_deg"], rm["median_deg"]
        out.update(rotation_metrics(T.decode(cp, pp), q[idx]))
        return out, cp, pp

    log = open(run.sub("logs") / "train.log", "w")
    best = (np.inf, None)
    step = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(tr_idx)
        tot = 0.0
        for s in range(0, len(perm), args.batch):
            b = torch.as_tensor(perm[s:s + args.batch], device=dev)
            loss, _ = model.loss(Xt[b], Ct[b], Pt[b], PQt[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step(); step += 1
            tot += loss.item() * len(b)
        if ep % args.eval_every == 0 or ep == args.epochs:
            vm, _, _ = evaluate(groups["synthetic_val"].to_numpy(), 1)
            msg = f"ep {ep} train_loss {tot / len(perm):.4f} val_mean_deg {vm['mean_deg']:.2f} val_median {vm['median_deg']:.2f} val_root {vm['root_acc']:.3f}"
            print(msg, file=log, flush=True)
            if vm["mean_deg"] < best[0]:  # model selection on synthetic validation only
                best = (vm["mean_deg"], {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
    model.load_state_dict(best[1])
    torch.save(best[1], run.sub("checkpoints") / "best.pt")

    rows, pred_rows = [], []
    for beam in [int(b) for b in args.beams.split(",")]:
        for g, m in groups.items():
            if g == "synthetic_train":
                continue
            res, cp, pp = evaluate(m.to_numpy(), beam)
            rows.append({"features": args.features, "seed": args.seed, "domain": g, **res})
            if beam == 1:
                idx = np.flatnonzero(m.to_numpy())
                qp = T.decode(cp, pp)
                err = np.degrees(2 * np.arccos(np.clip(np.abs((qp * q[idx]).sum(1)), 0, 1)))
                pred_rows.append(pd.DataFrame({"image_relpath": df.image_relpath.values[idx], "domain": g,
                                               "chart_gt": chart[idx], "path_gt": ["".join(map(str, x)) for x in path[idx]],
                                               "chart_pred": cp, "path_pred": ["".join(map(str, x)) for x in pp],
                                               **{f"q_pred_{k}": qp[:, i] for i, k in enumerate("wxyz")},
                                               "err_deg": err}))
    run.write_metrics(rows, "metrics")
    pd.concat(pred_rows).to_csv(run.sub("predictions") / "predictions_greedy.csv", index=False, float_format="%.6g")
    run.done(best_epoch=best[2], n_params=n_params, trainable_params=n_params, feature_dim=int(X.shape[1]))
    g1 = [r for r in rows if r["beam"] == 1]
    print(args.features, "seed", args.seed, " | ".join(f"{r['domain']}: mean {r['mean_deg']:.1f} med {r['median_deg']:.1f} root {r['root_acc']:.2f}" for r in g1))
    print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc())
    raise
