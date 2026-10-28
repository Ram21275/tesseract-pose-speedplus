"""EXP-064 per-object trainer, locked recipe (DEC-018): one Tesseract tree per branch and object, EXP-036 architecture and
budget, trained on clean + 2 augmented views (photometric + in-plane roll; T-LESS / YCB-V rolled by the exact homography),
labels in the dataset's target convention rolled exactly (tfpose.xobj.roll_target). Normalization statistics from the
object's clean train rows. Checkpoint: lowest symmetry-aware validation greedy mean (DEC-018 amendment for EXP-064).

Options added 2026-10-27 (defaults reproduce EXP-064 exactly):
  --extra-train DS2   (EXP-122) append DS2's train rows of the same object (clean + aug1/2 views), e.g. ycbvsyn composites;
                      normalization statistics then come from the combined clean train rows.
  --label raw|canonical|multi   (EXP-106 arms A0 / A1 / A2). canonical: in every view, the equivalent rotation R.S closest to
                      identity. multi: the recipe's soft-target CE averaged over equivalents R.S (all of them if |S| <= --sym-k,
                      else --sym-k drawn per row and step); identical to raw when |S| = 1.
  --trans frozen --init-from RUN   (EXP-126) SPACE-HOP translation block (tfpose.trans, dataset image size) trained on the
                      frozen tree of RUN (same features / normalization); targets u standardized per dimension with train
                      statistics; checkpoint = lowest validation mean translation error (view 0).
  --teacher NPZ --beta B   (EXP-119) + B * level-marginal CE to orbit-teacher candidates (clean target frame, mapped into the
                      view's frame exactly), as EXP-117."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import featsets, phase3 as P, tesseract as T, trans as TR, xobj as XO  # noqa: E402
from tfpose.cachepaths import CACHE  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.allo import mat_to_quat  # noqa: E402
from tfpose.rotations import quat_multiply, quat_to_matrix  # noqa: E402
from tfpose.runlog import Run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=list(XO.DATASETS))
ap.add_argument("--obj", required=True)
ap.add_argument("--features", required=True)
ap.add_argument("--token-shape", required=True)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--eval-every", type=int, default=5)
ap.add_argument("--sd-floor", type=float, default=0.0, help="floor on the per-dimension train sd (2026-10-19 fix: SYMSOL-T MoGe-2 cells empty in clean renders -> sd~1e-6 -> fp16 overflow -> NaN)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--exp", default="EXP-064")
ap.add_argument("--exp-name", default="broader_object_validation")
ap.add_argument("--extra-train", default=None, choices=[None] + list(XO.DATASETS))
ap.add_argument("--label", default="raw", choices=["raw", "canonical", "multi"])
ap.add_argument("--sym-k", type=int, default=8)
ap.add_argument("--teacher", default=None)
ap.add_argument("--teacher-variant", default=None, choices=[None, "PoE", "DINO"])
ap.add_argument("--beta", type=float, default=1.0)
ap.add_argument("--trans", default="none", choices=["none", "frozen"])
ap.add_argument("--init-from", default=None, help="EXP-126: tree run folder (relative to outputs/experiments) to freeze")
args = ap.parse_args()
run = Run(args.exp, args.exp_name, {**vars(args), "stage": "train"}, seed=args.seed)
L = 5
try:
    torch.manual_seed(args.seed); np.random.seed(args.seed); dev = torch.device("cuda"); ds = args.dataset
    ALL = pd.read_csv(REPO / f"outputs/data_manifests/exp064_{ds}_all.csv"); TRN = pd.read_csv(REPO / f"outputs/data_manifests/exp064_{ds}_train.csv")
    oc = XO.DATASETS[ds]["obj_col"]; obj = type(ALL[oc].iloc[0])(args.obj) if not isinstance(ALL[oc].iloc[0], str) else args.obj
    a_tr = np.flatnonzero(((ALL.split == "train") & (ALL[oc] == obj)).to_numpy()); a_va = np.flatnonzero(((ALL.split == "validation") & (ALL[oc] == obj)).to_numpy())
    t_tr = np.flatnonzero((TRN[oc] == obj).to_numpy()); assert len(t_tr) == len(a_tr)
    SRC = [(ds, a_tr, t_tr, TRN)]
    if args.extra_train:
        e = args.extra_train; EA = pd.read_csv(REPO / f"outputs/data_manifests/exp064_{e}_all.csv"); ET = pd.read_csv(REPO / f"outputs/data_manifests/exp064_{e}_train.csv")
        ea = np.flatnonzero(((EA.split == "train") & (EA[oc] == obj)).to_numpy()); et = np.flatnonzero((ET[oc] == obj).to_numpy()); assert len(ea) == len(et) > 0
        SRC.append((e, ea, et, ET))
    Xtr = np.concatenate([featsets.load_rows(args.features, f"exp064_{d_}_all", a_)[1] for d_, a_, _, _ in SRC]); _, Xva = featsets.load_rows(args.features, f"exp064_{ds}_all", a_va)
    mu, sd = featsets.train_stats(Xtr, np.ones(len(Xtr), bool)); sd = np.maximum(sd, args.sd_floor); st = (mu, sd)
    if args.trans == "frozen":                                                         # EXP-126: the frozen tree's own normalization
        ck0 = torch.load(REPO / "outputs/experiments" / args.init_from / "checkpoints/best.pt", map_location="cpu", weights_only=False)
        assert np.allclose(ck0["mu"], mu, atol=1e-4) and np.allclose(ck0["sd"], sd, atol=1e-4), "init run used other train rows"
        mu, sd = ck0["mu"], ck0["sd"]; st = (mu, sd); (run.dir / "source_runs.txt").write_text(f"{args.init_from}\n")
    VX = [featsets.standardize_to_tensor(Xtr, None, "cpu", torch.float16, stats=st)]; del Xtr; Xva = featsets.standardize_to_tensor(Xva, None, "cpu", torch.float16, stats=st)
    dtr = pd.concat([T_.iloc[t_] for _, _, t_, T_ in SRC], ignore_index=True); K = XO.intrinsics(ds, dtr); q_ego, tvec = XO.pose_ego(ds, dtr); dray = XO.crop_ray(K, dtr)
    QV = [XO.target(ds, dtr)]; ANG = [np.zeros(len(dtr))]
    for j in (1, 2):
        xs, angs = [], []
        for d_, _, t_, _ in SRC:
            tag = f"exp064_{d_}_aug{j}"; xs.append(featsets.load_rows(args.features, tag, t_)[1])
            angs.append(pd.read_csv(featsets.find_cache(args.features.split(":")[0], tag) / "aug.csv").rot_deg.to_numpy()[t_])
        VX.append(featsets.standardize_to_tensor(np.concatenate(xs), None, "cpu", torch.float16, stats=st)); del xs
        ANG.append(np.concatenate(angs)); QV.append(XO.roll_target(ds, q_ego, tvec, ANG[-1], dray))
    S = XO.symmetry_set(ds, obj); SQ = mat_to_quat(S)                                    # object-frame symmetries; target' = target . s
    if args.label == "canonical":                                                      # A1: equivalent closest to identity, per view
        def canon(qv):
            qe = quat_multiply(qv[:, None], SQ[None]); return qe[np.arange(len(qv)), np.abs(qe[..., 0]).argmax(1)]
        QV = [canon(qv) for qv in QV]
    V = len(VX); CP = [T.encode(qv, L) for qv in QV]; SO = [P.soft_targets(QV[v], *CP[v]) for v in range(V)]
    if args.teacher:                                                                   # EXP-119: teacher candidates (clean frame) -> each view
        tz = np.load(REPO / args.teacher); assert np.array_equal(tz["idx"], a_tr) and not args.extra_train
        Kc = tz["cand"].shape[1]; TW = torch.as_tensor(tz["w"], dtype=torch.float32); TCV = []
        for v in range(V):
            tt = np.repeat(tvec, Kc, 0); dd = np.repeat(dray, Kc, 0)
            TCV.append(XO.roll_target(ds, XO.from_target(ds, tz["cand"].reshape(-1, 4).astype(np.float64), tt), tt, np.repeat(ANG[v], Kc), dd).reshape(len(dtr), Kc, 4))
        TCV = np.stack(TCV); (run.dir / "source_runs.txt").write_text(f"teacher {args.teacher}\n")
    Ct = torch.as_tensor(np.stack([c for c, _ in CP]), device=dev); Pt = torch.as_tensor(np.stack([p for _, p in CP]), device=dev)
    RSt = torch.as_tensor(np.stack([r for r, _ in SO]), device=dev); CSt = torch.as_tensor(np.stack([c for _, c in SO]), device=dev)
    dva = ALL.iloc[a_va]; qva = XO.target(ds, dva)
    Rva_eq = np.einsum("nij,sjk->nsik", quat_to_matrix(qva), S)                       # (N,S,3,3) equivalent GT rotations
    ts = tuple(int(v) for v in args.token_shape.split("x"))
    model = P.HierTransformer(ts, L, dropout=args.dropout).to(dev); n_params = sum(p.numel() for p in model.parameters())
    NT = len(dtr); head = None; train_params = list(model.parameters())
    if args.trans == "frozen":
        model.load_state_dict(ck0["model"]); model.eval(); [p_.requires_grad_(False) for p_ in model.parameters()]
        W_, H_ = XO.DATASETS[ds]["img_wh"]; box = (dtr.crop_cx.to_numpy(), dtr.crop_cy.to_numpy(), dtr.crop_side.to_numpy(), dtr.crop_side.to_numpy())
        UV = np.stack([TR.to_u(XO.roll_pose(q_ego, tvec, ANG[v], dray)[1], K, *box, W_, H_) for v in range(V)])   # (V,N,3) per view
        u_mu, u_sd = UV[0].mean(0), UV[0].std(0) + 1e-8; UVt = torch.as_tensor((UV - u_mu) / u_sd, dtype=torch.float32, device=dev)
        Kva = XO.intrinsics(ds, dva); _, tva = XO.pose_ego(ds, dva); bva = (dva.crop_cx.to_numpy(), dva.crop_cy.to_numpy(), dva.crop_side.to_numpy(), dva.crop_side.to_numpy())
        head = TR.TransHead(model.root_tok.shape[-1], dropout=args.dropout).to(dev); train_params = list(head.parameters())
        n_params = sum(p.numel() for p in train_params)
    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * int(np.ceil(NT / args.batch)), pct_start=0.05)

    @torch.no_grad()
    def val_trans():
        head.eval(); out = []
        for s0 in range(0, len(Xva), 1024):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(head(model.encode(Xva[s0:s0 + 1024].to(dev).float())).float().cpu().numpy())
        u = np.concatenate(out) * u_sd + u_mu
        return float(np.linalg.norm(TR.from_u(u, Kva, *bva, W_, H_) - tva, axis=1).mean())

    def sym_err(qp):
        Rp = quat_to_matrix(qp); c = np.einsum("nij,nsij->ns", Rp, Rva_eq)               # trace(Rp^T Rgt S)
        return np.degrees(np.arccos(np.clip((c.max(1) - 1) / 2, -1, 1)))

    @torch.no_grad()
    def val_mean():
        model.eval(); out = []
        for s in range(0, len(Xva), 1024):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                c, p = fused_beam([model], [Xva[s:s + 1024].to(dev).float()], "single", 1)
            out.append(T.decode(c.cpu().numpy(), p.cpu().numpy()))
        return float(sym_err(np.concatenate(out)).mean())

    def multi_loss(h, v, pb):
        """A2: mean over equivalents R.s of the recipe's soft-target CE (teacher-forced along each equivalent's path)."""
        B, Sn = len(pb), len(SQ); k = min(Sn, args.sym_k)
        j = np.tile(np.arange(Sn), (B, 1)) if Sn <= args.sym_k else np.argsort(np.random.rand(B, Sn), 1)[:, :k]
        q = np.stack([QV[vv][p] for vv, p in zip(v, pb)]); qe = quat_multiply(np.repeat(q, k, 0), SQ[j.reshape(-1)])
        c, p = T.encode(qe, L); rs, cs = P.soft_targets(qe, c, p)
        rl, cl = model.logits_tf(h.repeat_interleave(k, 0), torch.as_tensor(c, device=dev), torch.as_tensor(p, device=dev))
        rs, cs = torch.as_tensor(rs, device=dev), torch.as_tensor(cs, device=dev)
        return P.soft_ce(rl, rs) + sum(P.soft_ce(cl[:, l], cs[:, l]) for l in range(L))

    def level_ce(h, rot, w):
        """EXP-117 level-marginal CE: -sum_j w_j * mean_l log p^(l)(prefix_l(path(rot_j))); rot (B,K,4) numpy, w (B,K) torch."""
        B, Kc = rot.shape[:2]
        c, p = T.encode(rot.reshape(-1, 4), L); c, p = torch.as_tensor(c, device=dev), torch.as_tensor(p, device=dev)
        rl, cl = model.logits_tf(h.repeat_interleave(Kc, 0), c, p)
        step = torch.cat([F.log_softmax(rl.float(), -1).gather(1, c[:, None]), F.log_softmax(cl.float(), -1).gather(2, p[:, :, None])[..., 0]], 1)
        return -(w * step.cumsum(1).view(B, Kc, L + 1).mean(-1)).sum(-1).mean()

    log = open(run.sub("logs") / "train.log", "w"); best = (np.inf, None, 0)
    for ep in range(1, args.epochs + 1):
        model.train(args.trans != "frozen"); perm = np.random.permutation(NT); tot = 0.0
        if head is not None:
            head.train()
        for s in range(0, NT, args.batch):
            pb = perm[s:s + args.batch]; v = np.random.randint(0, V, len(pb))
            x = torch.empty((len(pb),) + tuple(VX[0].shape[1:]), dtype=torch.float32, device=dev)
            for j in range(V):
                m = v == j
                if m.any():
                    x[torch.as_tensor(m, device=dev)] = VX[j][torch.as_tensor(pb[m])].to(dev).float()
            vi, pi = torch.as_tensor(v, device=dev), torch.as_tensor(pb, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.trans == "frozen":                                            # EXP-126: translation block only
                    with torch.no_grad():
                        h = model.encode(x)
                    loss = ((head(h).float() - UVt[vi, pi]) ** 2).sum(-1).mean()
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(pb)
                    continue
                h = model.encode(x)
                if args.label == "multi":
                    loss = multi_loss(h, v, pb)
                else:
                    rl, cl = model.logits_tf(h, Ct[vi, pi], Pt[vi, pi])
                    loss = P.soft_ce(rl, RSt[vi, pi]) + sum(P.soft_ce(cl[:, l], CSt[vi, pi][:, l]) for l in range(L))
                if args.teacher:
                    loss = loss + args.beta * level_ce(h, TCV[v, pb], TW[pb].to(dev))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(pb)
        if ep % args.eval_every == 0 or ep == args.epochs:
            vm = val_trans() if head is not None else val_mean()
            print(f"ep {ep} loss {tot / NT:.4f} {'val_trans_mean' if head is not None else 'val_sym_mean'} {vm:.4f}", file=log, flush=True)
            if vm < best[0]:
                best = (vm, {k: t.detach().clone() for k, t in (head if head is not None else model).state_dict().items()}, ep)
    ck = {"model": best[1] if head is None else model.state_dict(), "mu": np.asarray(mu, np.float32), "sd": np.asarray(sd, np.float32)}
    if head is not None:
        ck.update(trans=best[1], u_mu=u_mu, u_sd=u_sd, uz_train_mean=float(UV[0][:, 2].mean()), img_wh=(W_, H_))
    torch.save(ck, run.sub("checkpoints") / "best.pt")
    run.done(best_epoch=best[2], best_val_sym_mean=best[0], n_train=NT, n_val=len(a_va), n_params=n_params, trainable_params=n_params)
    print(ds, args.obj, args.features, "seed", args.seed, f"val_sym_mean {best[0]:.2f}"); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
