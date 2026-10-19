"""EXP-064 per-object trainer, locked recipe (DEC-018): one Tesseract tree per branch and object, EXP-036 architecture and
budget, trained on clean + 2 augmented views (photometric + in-plane roll; T-LESS / YCB-V rolled by the exact homography),
labels in the dataset's target convention rolled exactly (tfpose.xobj.roll_target). Normalization statistics from the
object's clean train rows. Checkpoint: lowest symmetry-aware validation greedy mean (DEC-018 amendment for EXP-064)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import featsets, phase3 as P, tesseract as T, xobj as XO  # noqa: E402
from tfpose.cachepaths import CACHE  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.rotations import quat_to_matrix  # noqa: E402
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
args = ap.parse_args()
run = Run("EXP-064", "broader_object_validation", {**vars(args), "stage": "train"}, seed=args.seed)
L = 5
try:
    torch.manual_seed(args.seed); np.random.seed(args.seed); dev = torch.device("cuda"); ds = args.dataset
    ALL = pd.read_csv(REPO / f"outputs/data_manifests/exp064_{ds}_all.csv"); TRN = pd.read_csv(REPO / f"outputs/data_manifests/exp064_{ds}_train.csv")
    oc = XO.DATASETS[ds]["obj_col"]; obj = type(ALL[oc].iloc[0])(args.obj) if not isinstance(ALL[oc].iloc[0], str) else args.obj
    a_tr = np.flatnonzero(((ALL.split == "train") & (ALL[oc] == obj)).to_numpy()); a_va = np.flatnonzero(((ALL.split == "validation") & (ALL[oc] == obj)).to_numpy())
    t_tr = np.flatnonzero((TRN[oc] == obj).to_numpy()); assert len(t_tr) == len(a_tr)
    _, Xtr = featsets.load_rows(args.features, f"exp064_{ds}_all", a_tr); _, Xva = featsets.load_rows(args.features, f"exp064_{ds}_all", a_va)
    mu, sd = featsets.train_stats(Xtr, np.ones(len(Xtr), bool)); sd = np.maximum(sd, args.sd_floor); st = (mu, sd)
    VX = [featsets.standardize_to_tensor(Xtr, None, "cpu", torch.float16, stats=st)]; Xva = featsets.standardize_to_tensor(Xva, None, "cpu", torch.float16, stats=st)
    dtr = TRN.iloc[t_tr]; K = XO.intrinsics(ds, dtr); q_ego, tvec = XO.pose_ego(ds, dtr); dray = XO.crop_ray(K, dtr)
    QV = [XO.target(ds, dtr)]
    for j in (1, 2):
        tag = f"exp064_{ds}_aug{j}"
        _, xv = featsets.load_rows(args.features, tag, t_tr); VX.append(featsets.standardize_to_tensor(xv, None, "cpu", torch.float16, stats=st)); del xv
        ang = pd.read_csv(featsets.find_cache(args.features.split(":")[0], tag) / "aug.csv").rot_deg.to_numpy()[t_tr]
        QV.append(XO.roll_target(ds, q_ego, tvec, ang, dray))
    V = len(VX); CP = [T.encode(qv, L) for qv in QV]; SO = [P.soft_targets(QV[v], *CP[v]) for v in range(V)]
    Ct = torch.as_tensor(np.stack([c for c, _ in CP]), device=dev); Pt = torch.as_tensor(np.stack([p for _, p in CP]), device=dev)
    RSt = torch.as_tensor(np.stack([r for r, _ in SO]), device=dev); CSt = torch.as_tensor(np.stack([c for _, c in SO]), device=dev)
    dva = ALL.iloc[a_va]; qva = XO.target(ds, dva); S = XO.symmetry_set(ds, obj)
    Rva_eq = np.einsum("nij,sjk->nsik", quat_to_matrix(qva), S)                       # (N,S,3,3) equivalent GT rotations
    ts = tuple(int(v) for v in args.token_shape.split("x"))
    model = P.HierTransformer(ts, L, dropout=args.dropout).to(dev); n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd); NT = len(t_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * int(np.ceil(NT / args.batch)), pct_start=0.05)

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

    log = open(run.sub("logs") / "train.log", "w"); best = (np.inf, None, 0)
    for ep in range(1, args.epochs + 1):
        model.train(); perm = np.random.permutation(NT); tot = 0.0
        for s in range(0, NT, args.batch):
            pb = perm[s:s + args.batch]; v = np.random.randint(0, V, len(pb))
            x = torch.empty((len(pb),) + tuple(VX[0].shape[1:]), dtype=torch.float32, device=dev)
            for j in range(V):
                m = v == j
                if m.any():
                    x[torch.as_tensor(m, device=dev)] = VX[j][torch.as_tensor(pb[m])].to(dev).float()
            vi, pi = torch.as_tensor(v, device=dev), torch.as_tensor(pb, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rl, cl = model.logits_tf(model.encode(x), Ct[vi, pi], Pt[vi, pi])
                loss = P.soft_ce(rl, RSt[vi, pi]) + sum(P.soft_ce(cl[:, l], CSt[vi, pi][:, l]) for l in range(L))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(pb)
        if ep % args.eval_every == 0 or ep == args.epochs:
            vm = val_mean(); print(f"ep {ep} loss {tot / NT:.4f} val_sym_mean {vm:.3f}", file=log, flush=True)
            if vm < best[0]:
                best = (vm, {k: t.detach().clone() for k, t in model.state_dict().items()}, ep)
    torch.save({"model": best[1], "mu": mu.astype(np.float32), "sd": sd.astype(np.float32)}, run.sub("checkpoints") / "best.pt")
    run.done(best_epoch=best[2], best_val_sym_mean=best[0], n_train=NT, n_val=len(a_va), n_params=n_params, trainable_params=n_params)
    print(ds, args.obj, args.features, "seed", args.seed, f"val_sym_mean {best[0]:.2f}"); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
