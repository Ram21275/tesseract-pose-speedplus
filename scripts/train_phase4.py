"""Phase-4 continuous heads on a FROZEN EXP-036 Tesseract Transformer (one branch, one seed).

--mode residual  (EXP-040): delta = head(mem, q_c);  q = q_c (x) Exp(delta);  Huber loss in units of the L5 cell size.
--mode flow      (EXP-041/042): omega(R_tau, tau | mem[, q_c]) by flow matching on SO(3) (geodesic path, body frame).
    --base uniform  : R_0 ~ Haar(SO(3))                         (EXP-042 image-conditioned posterior)
    --base local    : R_0 = q_c (x) Exp(n), n ~ N(0, sigma_b^2)  (EXP-041 cell-conditioned refiner; head also sees q_c)

Training centre q_c (--centre):
    gt       : GT leaf centre
    perturb  : leaf centre of GT (x) Exp(e), e random axis with angle drawn from the frozen predictor's
               synthetic-val greedy-error distribution (EXP-040 stage-0 diagnostic; synthetic only)
    mix      : per sample, the GT leaf with prob. 1/2, else perturb
At inference q_c is always the frozen predictor's greedy leaf (never the GT path).
The frozen Transformer is in eval mode with requires_grad False (asserted); only the head trains.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fnn
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import featsets, phase3 as P, so3flow as S, tesseract as T  # noqa: E402
from tfpose.data import eval_groups  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.metrics import rotation_metrics  # noqa: E402
from tfpose.predictor import decode_torch  # noqa: E402
from tfpose.runlog import Run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--features", required=True)
ap.add_argument("--src-exp", default="EXP-036_transformer_soft_targets")
ap.add_argument("--mode", required=True, choices=["residual", "flow"])
ap.add_argument("--base", default="uniform", choices=["uniform", "local"])
ap.add_argument("--sigma-b-deg", type=float, default=5.0, help="local base std per axis (deg)")
ap.add_argument("--centre", default="mix", choices=["gt", "perturb", "mix"])
ap.add_argument("--err-dist", default=None, help="csv with column err_deg (synthetic-val greedy errors) for --centre perturb/mix")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch", type=int, default=128, help="images per step")
ap.add_argument("--k", type=int, default=16, help="(tau, R_0) pairs per image per step (flow)")
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--eval-every", type=int, default=5)
ap.add_argument("--nfe", type=int, default=10)
ap.add_argument("--m", type=int, default=32, help="endpoint samples per image at evaluation (flow)")
ap.add_argument("--val-cap", type=int, default=4000, help="images of synthetic val used for checkpoint selection")
ap.add_argument("--train-frac", type=float, default=1.0, help="seeded fraction of synthetic train (smoke tests)")
ap.add_argument("--eval-cap", type=int, default=0, help="cap images per evaluation group (smoke tests; 0 = all)")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
exp_id, short = args.exp.split("_", 1)
run = Run(exp_id, short, vars(args), seed=args.seed)
try:
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda")
    df = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv")
    names, X, cache_info = featsets.load(args.features, "full")
    if names != df.image_relpath.tolist():
        raise ValueError("feature cache order does not match subset manifest")
    (run.dir / "feature_source.json").write_text(json.dumps(cache_info, indent=2))
    q = df[[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    chart, path = T.encode(q, 5)
    tr = ((df.domain == "synthetic") & (df.split == "train")).to_numpy()
    groups = eval_groups(df)
    Xt = featsets.standardize_to_tensor(X, tr, dev, torch.float16, pin=False); del X
    Qt = torch.as_tensor(q, dtype=torch.float32, device=dev)
    QLt = torch.as_tensor(T.decode(chart, path), dtype=torch.float32, device=dev)

    # ---- frozen EXP-036 predictor of the same branch and seed ----
    src = None
    for d in sorted((REPO / "outputs/experiments" / args.src_exp).glob(f"RUN-*-seed{args.seed}")):
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        if cfg.get("features") == args.features and (d / "checkpoints/best.pt").exists():
            src = (d, cfg)
    if src is None:
        raise FileNotFoundError(args.src_exp, args.features, args.seed)
    d, cfg = src
    tree = P.HierTransformer(tuple(int(v) for v in cfg["token_shape"].split("x")), cfg["depth"], dropout=cfg["dropout"]).to(dev)
    tree.load_state_dict(torch.load(d / "checkpoints/best.pt", map_location=dev)["model"]); tree.eval()
    for p_ in tree.parameters():
        p_.requires_grad_(False)
    assert not any(p_.requires_grad for p_ in tree.parameters())
    (run.dir / "source_runs.txt").write_text(d.name + "\n")

    def mem_of(b):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return tree.encode(Xt[b].float()).float()

    # ---- head ----
    if args.mode == "residual":
        head = S.ResidualHead040(layers=args.layers).to(dev)
    else:
        head = S.FlowHead(layers=args.layers, cell=args.base == "local").to(dev)
    n_params = sum(p_.numel() for p_ in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    tr_idx = np.flatnonzero(tr)
    if args.train_frac < 1.0:
        tr_idx = np.sort(np.random.default_rng(0).choice(tr_idx, int(round(args.train_frac * len(tr_idx))), replace=False))
    steps = args.epochs * int(np.ceil(len(tr_idx) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    cell = np.radians(P.TAU_DEG[5])
    sig_b = np.radians(args.sigma_b_deg)
    errs = None
    if args.centre in ("perturb", "mix"):
        errs = torch.as_tensor(np.radians(pd.read_csv(args.err_dist).err_deg.to_numpy()), dtype=torch.float32, device=dev)

    def train_centre(b):
        if args.centre == "gt" or errs is None:
            return QLt[b]
        ax = torch.randn(len(b), 3, device=dev); ax = ax / ax.norm(dim=-1, keepdim=True)
        ang = errs[torch.randint(len(errs), (len(b),), device=dev)]
        qp = S.qcanon(P.qmul(Qt[b], P.rotvec_to_quat(ax * ang[:, None]))).cpu().numpy()
        c, pth = T.encode(qp, 5)
        qc = torch.as_tensor(T.decode(c, pth), dtype=torch.float32, device=dev)
        if args.centre == "mix":
            keep = torch.rand(len(b), device=dev) < 0.5
            qc = torch.where(keep[:, None], QLt[b], qc)
        return qc

    def loss_fn(b):
        mem = mem_of(b); qc = train_centre(b)
        if args.mode == "residual":
            target = S.relative_rotvec(qc, Qt[b])
            ok = target.norm(dim=-1) < S.CUT_LOCUS
            return Fnn.huber_loss(head(mem, qc)[ok] / cell, target[ok] / cell, delta=1.0)
        K = args.k
        memk = mem.repeat_interleave(K, 0); q1 = Qt[b].repeat_interleave(K, 0); qck = qc.repeat_interleave(K, 0)
        q0 = S.uniform_quats(len(q1), dev) if args.base == "uniform" else S.local_quats(qck, sig_b)
        tau = torch.rand(len(q1), device=dev)
        qt, a, ok = S.fm_pairs(q0, q1, tau)
        v = head(memk, qt, tau, qck if args.base == "local" else None)
        return ((v - a) ** 2).sum(-1)[ok].mean()

    # ---- inference ----
    @torch.no_grad()
    def greedy_leaf(b):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            c, p = fused_beam([tree], [Xt[b].float()], "single", 1)
        return decode_torch(c, p).float()

    @torch.no_grad()
    def infer(idx, m):
        head.eval(); out = {"qc": [], "pt": [], "samples": []}
        bs = 512 if args.mode == "residual" else max(8, 4096 // m)
        for s in range(0, len(idx), bs):
            b = torch.as_tensor(idx[s:s + bs], device=dev)
            mem = mem_of(b); qc = greedy_leaf(b)
            out["qc"].append(qc)
            if args.mode == "residual":
                out["pt"].append(S.qcanon(P.qmul(qc, P.rotvec_to_quat(head(mem, qc)))))
                continue
            memk = mem.repeat_interleave(m, 0); qck = qc.repeat_interleave(m, 0)
            q0 = S.uniform_quats(len(qck), dev) if args.base == "uniform" else S.local_quats(qck, sig_b)
            cond = qck if args.base == "local" else None
            smp = S.integrate(lambda qq, tt: head(memk, qq, tt, cond), q0, args.nfe).view(len(b), m, 4)
            out["samples"].append(smp); out["pt"].append(S.kde_mode(smp))
        return {k: torch.cat(v) for k, v in out.items() if v}

    sel = np.flatnonzero(groups["synthetic_val"])
    sel = np.sort(np.random.default_rng(0).choice(sel, min(args.val_cap, len(sel)), replace=False))
    log = open(run.sub("logs") / "train.log", "w")
    best = (np.inf, None, 0)
    for ep in range(1, args.epochs + 1):
        head.train(); perm = np.random.permutation(tr_idx); tot = 0.0; t0 = time.time()
        for s in range(0, len(perm), args.batch):
            b = torch.as_tensor(perm[s:s + args.batch], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(b)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step(); sched.step(); tot += loss.item() * len(b)
        msg = f"ep {ep} loss {tot / len(perm):.4f} {time.time() - t0:.0f}s"
        if ep % args.eval_every == 0 or ep == args.epochs:
            r = infer(sel, min(args.m, 16))
            e = S.geo_deg(r["pt"], Qt[torch.as_tensor(sel, device=dev)]).cpu().numpy()
            msg += f" | synth-val(sel {len(sel)}) point mean {e.mean():.3f} median {np.median(e):.3f}"
            if e.mean() < best[0]:
                best = (e.mean(), {k: v.detach().clone() for k, v in head.state_dict().items()}, ep)
        print(msg, file=log, flush=True)
    head.load_state_dict(best[1])
    torch.save({"head": best[1], "args": vars(args)}, run.sub("checkpoints") / "best.pt")

    # ---- final evaluation on all DEC-005 groups ----
    rows, preds = [], []
    for g, msk in groups.items():
        if g == "lightbox":          # union of lightbox_val and lightbox_test; recomputed from predictions
            continue
        idx = np.flatnonzero(msk)
        if args.eval_cap:
            idx = idx[:args.eval_cap]
        t0 = time.time()
        r = infer(idx, args.m); torch.cuda.synchronize(); dt = (time.time() - t0) / len(idx)
        qg = Qt[torch.as_tensor(idx, device=dev)]
        ept = S.geo_deg(r["pt"], qg).cpu().numpy(); ec = S.geo_deg(r["qc"], qg).cpu().numpy()
        row = {"features": args.features, "mode": args.mode, "base": args.base, "seed": args.seed, "domain": g, "n": len(idx),
               "latency_ms_per_image": 1000 * dt, **rotation_metrics(r["pt"].cpu().numpy(), q[idx]),
               "leaf_mean_deg": float(ec.mean()), "leaf_median_deg": float(np.median(ec))}
        if "samples" in r:
            smp = r["samples"]; es = S.geo_deg(smp, qg[:, None]).cpu().numpy()        # (N,M)
            dmode = S.geo_deg(smp, r["pt"][:, None]).cpu().numpy()
            row.update({"nfe": args.nfe, "M": args.m, "sample_mean_deg": float(es.mean()), "sample_median_deg": float(np.median(es)),
                        "best_of_M_mean_deg": float(es.min(1).mean()), "best_of_M_median_deg": float(np.median(es.min(1))),
                        **{f"any_sample_within_{t}": float((es.min(1) <= t).mean()) for t in (5, 10, 20)},
                        "spread_deg": float(dmode.mean())})
            for a in (0.5, 0.8, 0.9, 0.95):                                         # ball around the mode, radius = a-quantile
                rad = np.quantile(dmode, a, axis=1)
                row[f"coverage@{a}"] = float((ept <= rad).mean())
        rows.append(row)
        preds.append(pd.DataFrame({"image_relpath": df.image_relpath.values[idx], "group": g, "err_deg": ept, "leaf_err_deg": ec,
                                   **{f"q_pred_{k}": r["pt"][:, i].cpu().numpy() for i, k in enumerate("wxyz")}}))
        if "samples" in r and g in ("synthetic_val", "lightbox_val"):
            np.save(run.sub("samples") / f"{g}_samples_fp16.npy", r["samples"].half().cpu().numpy())
    run.write_metrics(rows, "metrics")
    pd.concat(preds).to_csv(run.sub("predictions") / "predictions.csv", index=False, float_format="%.6g")
    run.done(best_epoch=best[2], n_params=n_params, trainable_params=n_params)
    print(f"{args.features} {args.mode}/{args.base} seed {args.seed} " + " | ".join(
        f"{r['domain']}: {r['mean_deg']:.2f}/{r['median_deg']:.2f} (leaf {r['leaf_mean_deg']:.2f})" for r in rows))
    print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
