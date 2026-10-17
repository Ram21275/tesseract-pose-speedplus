"""EXP-116 / EXP-117 trainer (one Tesseract tree per branch; EXP-109 arm-A recipe otherwise: EXP-036 architecture and
budget, clean + exp109_train_aug1/2 views, labels rotated by the derived "pre-" convention).

--mode
  T        (116) two distinct views u, v per image; L_GT(u) + L_GT(v) + lam * [H(K p_u -> p_v) + H(K p_v -> p_u)],
           H = level-marginal CE (EXP-044 form) of the other view's tree on this view's beam-8 leaves transported by the
           known roll (c' = Exp(-(theta_v - theta_u) d) (x) c), weights = softmax of the beam path log-probs (stop-grad).
  C1       (116) the same two views and GT losses, lam = 0 (data-matched control).
  C2       (116) like T but the consistency term on the leaf level only (output-level equivariance control).
  distill  (117) one view v per image; L_GT(v) + beta * H(teacher -> v), teacher = precomputed orbit candidates
           (24 rotations in the clean frame + weights) transported into v's frame.
Checkpoint: lowest clean synthetic-val greedy mean (as A)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import data as D, featsets, inplane as IP, phase3 as P, tesseract as T  # noqa: E402
from tfpose.cachepaths import CACHE  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.metrics import rotation_metrics  # noqa: E402
from tfpose.runlog import Run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--features", required=True)
ap.add_argument("--token-shape", required=True)
ap.add_argument("--mode", required=True, choices=["T", "C1", "C2", "distill"])
ap.add_argument("--lam", type=float, default=1.0)
ap.add_argument("--beta", type=float, default=1.0)
ap.add_argument("--beam", type=int, default=8)
ap.add_argument("--teacher-variant", default=None, choices=[None, "PoE", "DINO"], help="EXP-117: which orbit teacher (recorded for selection)")
ap.add_argument("--teacher-targets", default=None, help="EXP-117: .npz with idx (rows of subset_full), cand (N,24,4), w (N,24)")
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
run = Run(exp_id, short, vars(args), seed=args.seed)
L = 5
try:
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed); dev = torch.device("cuda")
    df = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv"); q = df[[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    tr = ((df.domain == "synthetic") & (df.split == "train")).to_numpy(); va = ((df.domain == "synthetic") & (df.split == "validation")).to_numpy()
    names, X, info = featsets.load(args.features, "full"); assert names == df.image_relpath.tolist()
    (run.dir / "feature_source.json").write_text(json.dumps(info, indent=2))
    stats = featsets.train_stats(X, tr); Xc = featsets.standardize_to_tensor(X, tr, "cpu", torch.float16, stats=stats); del X
    tr_rows = np.flatnonzero(tr); NT = len(tr_rows)
    cam = D.load_camera(Path(yaml.safe_load(open(REPO / "configs/paths.yaml"))["speedplus_root"]))
    dray = IP.ray_undistorted(cam.K, cam.dist, df.crop_cx.to_numpy()[tr_rows], df.crop_cy.to_numpy()[tr_rows])
    VIEWS = [t for t in args.aug_tags.split(",") if t]
    VX = [Xc[torch.as_tensor(tr_rows)]]; ANG = [np.zeros(NT)]
    for tag in VIEWS:
        vn, vx, vinfo = featsets.load(args.features, tag); assert vn == df.image_relpath.values[tr_rows].tolist(), tag
        VX.append(featsets.standardize_to_tensor(vx, None, "cpu", torch.float16, stats=stats)); del vx
        ANG.append(pd.read_csv(CACHE / Path(vinfo["cache"]).name / "aug.csv").rot_deg.to_numpy())
    ANG = np.stack(ANG)                                                                     # (V, NT)
    V = len(VX)
    QV = np.stack([IP.rotate_label(q[tr_rows], ANG[v], dray, args.aug_convention) for v in range(V)])   # (V, NT, 4)
    CP = [T.encode(QV[v], L) for v in range(V)]
    RS, CS = zip(*[P.soft_targets(QV[v], CP[v][0], CP[v][1]) for v in range(V)])
    Ct = torch.as_tensor(np.stack([c for c, _ in CP]), device=dev); Pt = torch.as_tensor(np.stack([p for _, p in CP]), device=dev)
    RSt = torch.as_tensor(np.stack(RS), device=dev); CSt = torch.as_tensor(np.stack(CS), device=dev)
    if args.mode == "distill":
        tt = np.load(REPO / args.teacher_targets); pos = {r: i for i, r in enumerate(tr_rows)}
        tix = np.array([pos[r] for r in tt["idx"]]); assert len(tix) == NT and (np.sort(tix) == np.arange(NT)).all()
        TC = np.empty((NT,) + tt["cand"].shape[1:]); TW = np.empty((NT, tt["w"].shape[1]))
        TC[tix] = tt["cand"]; TW[tix] = tt["w"]
        (run.dir / "source_runs.txt").write_text(f"teacher_targets {args.teacher_targets}\n")
    ts = tuple(int(v) for v in args.token_shape.split("x"))
    model = P.HierTransformer(ts, L, dropout=args.dropout).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * int(np.ceil(NT / args.batch)), pct_start=0.05)

    def view_x(vsel, pb):
        x = torch.empty((len(pb),) + tuple(VX[0].shape[1:]), dtype=torch.float32, device=dev)
        for v in range(V):
            m = vsel == v
            if m.any():
                x[torch.as_tensor(m, device=dev)] = VX[v][torch.as_tensor(pb[m])].to(dev).float()
        return x

    def gt_loss(h, vsel, pb):
        vi, pi = torch.as_tensor(vsel, device=dev), torch.as_tensor(pb, device=dev)
        rl, cl = model.logits_tf(h, Ct[vi, pi], Pt[vi, pi])
        return P.soft_ce(rl, RSt[vi, pi]) + sum(P.soft_ce(cl[:, l], CSt[vi, pi][:, l]) for l in range(L))

    def level_ce(h, rot, w, leaf_only=False):
        """-sum_j w_j * mean_l log p^(l)(prefix_l(path(rot_j))) for the tree on memory h; rot (B,K,4) numpy, w (B,K) torch."""
        B, K, _ = rot.shape
        c, p = T.encode(rot.reshape(-1, 4), L); c, p = torch.as_tensor(c, device=dev), torch.as_tensor(p, device=dev)
        rl, cl = model.logits_tf(h.repeat_interleave(K, 0), c, p)
        step = torch.cat([F.log_softmax(rl.float(), -1).gather(1, c[:, None]), F.log_softmax(cl.float(), -1).gather(2, p[:, :, None])[..., 0]], 1)
        cum = step.cumsum(1).view(B, K, L + 1)
        per = cum[..., -1] if leaf_only else cum.mean(-1)
        return -(w * per).sum(-1).mean()

    def beam_support(h_x, x):
        with torch.no_grad():
            c8, p8, sc, _ = fused_beam([model], [x], "single", args.beam, return_all=True)
        rot = T.decode(c8.cpu().numpy().reshape(-1), p8.cpu().numpy().reshape(-1, L)).reshape(len(x), -1, 4)
        return rot, torch.softmax(sc.float(), -1)

    @torch.no_grad()
    def evaluate(mask):
        model.eval(); idx = np.flatnonzero(mask); out = []
        for s in range(0, len(idx), 1024):
            x = Xc[torch.as_tensor(idx[s:s + 1024])].to(dev).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                c, p = fused_beam([model], [x], "single", 1)
            out.append(T.decode(c.cpu().numpy(), p.cpu().numpy()))
        return idx, np.concatenate(out)

    log = open(run.sub("logs") / "train.log", "w"); best = (np.inf, None, 0)
    for ep in range(1, args.epochs + 1):
        model.train(); perm = rng.permutation(NT); tot = tc = 0.0
        for s in range(0, NT, args.batch):
            pb = perm[s:s + args.batch]; B = len(pb)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.mode == "distill":
                    vs = rng.integers(0, V, B); x = view_x(vs, pb); h = model.encode(x)
                    loss = gt_loss(h, vs, pb)
                    cand = TC[pb]; K = cand.shape[1]
                    rot = IP.rotate_label(cand.reshape(-1, 4), np.repeat(ANG[vs, pb], K), np.repeat(dray[pb], K, 0), args.aug_convention).reshape(B, K, 4)
                    cons = level_ce(h, rot, torch.as_tensor(TW[pb], dtype=torch.float32, device=dev))
                    loss = loss + args.beta * cons
                else:
                    vu = rng.integers(0, V, B); vv = (vu + rng.integers(1, V, B)) % V          # two distinct views
                    xu, xv = view_x(vu, pb), view_x(vv, pb); hu, hv = model.encode(xu), model.encode(xv)
                    loss = gt_loss(hu, vu, pb) + gt_loss(hv, vv, pb); cons = torch.zeros((), device=dev)
                    if args.mode in ("T", "C2"):
                        ru, wu = beam_support(hu, xu); rv, wv = beam_support(hv, xv)
                        K = ru.shape[1]; dd = np.repeat(dray[pb], K, 0)
                        ru2v = IP.rotate_label(ru.reshape(-1, 4), np.repeat(ANG[vv, pb] - ANG[vu, pb], K), dd, args.aug_convention).reshape(B, K, 4)
                        rv2u = IP.rotate_label(rv.reshape(-1, 4), np.repeat(ANG[vu, pb] - ANG[vv, pb], K), dd, args.aug_convention).reshape(B, K, 4)
                        lo = args.mode == "C2"
                        cons = level_ce(hv, ru2v, wu, lo) + level_ce(hu, rv2u, wv, lo)
                        loss = loss + args.lam * cons
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * B; tc += float(cons) * B
        if ep % args.eval_every == 0 or ep == args.epochs:
            idx, qp = evaluate(va); vm = rotation_metrics(qp, q[idx])
            print(f"ep {ep} loss {tot / NT:.4f} cons {tc / NT:.4f} val_mean {vm['mean_deg']:.3f} median {vm['median_deg']:.3f}", file=log, flush=True)
            if vm["mean_deg"] < best[0]:
                best = (vm["mean_deg"], {k: t.detach().clone() for k, t in model.state_dict().items()}, ep)
    model.load_state_dict(best[1]); torch.save({"model": best[1]}, run.sub("checkpoints") / "best.pt")
    rows = []
    for g, m in (("synthetic_val", va), ("lightbox", (df.domain == "lightbox").to_numpy()), ("sunlamp", (df.domain == "sunlamp").to_numpy())):
        idx, qp = evaluate(m); rows.append({"features": args.features, "mode": args.mode, "seed": args.seed, "domain": g, **rotation_metrics(qp, q[idx])})
    run.write_metrics(rows, "metrics"); run.done(best_epoch=best[2], n_params=n_params, trainable_params=n_params)
    print(args.features, args.mode, "seed", args.seed, " | ".join(f"{r['domain']}: {r['mean_deg']:.2f}" for r in rows)); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
