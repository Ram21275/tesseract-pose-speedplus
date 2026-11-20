"""Blind prediction for EXP-132 (depth sweep) from the depth-5 errors only (CPU, no GPU, no training).

Model (selection-resolution decomposition). For each object, a fraction pi of test images is "in mode": the tree's leaf is
the tesseract cell that contains q_gt * Exp(w), w ~ N(0, sigma^2 I_3) (per-axis, degrees); the error is the symmetry-aware
angle between that leaf centre and the GT. The other images are mode-selection errors whose distribution does not depend
on the depth. Then, exactly within the model, going from depth 5 to depth L changes
    mean error  by  pi * (E_in(L) - E_in(5)),        R@tau  by  pi * (F_in(tau; L) - F_in(tau; 5)),
and, if the tree's in-mode density is calibrated (a Gaussian of the same sigma around its estimate), the mean
log-likelihood at the GT by pi * (D_in(L) - D_in(5)), D_in(L) = E log[P(GT leaf) / vol(leaf)].
Fit per object on the depth-5 test errors: (pi, sigma) by least squares between the empirical CDF and pi * F_in(x; sigma, 5)
on x = 0.5..12 deg (errors above 12 deg are counted as selection errors only through pi).
Inputs: EXP-131 eval predictions (DINOv3, depth 5, the same trees EXP-132 re-evaluates at L5). Output: JSON + markdown table.
"""
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tfpose import tesseract as T  # noqa: E402
from tfpose.rotations import matrix_to_quat, quat_multiply, normalize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pred-dir", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--mode", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--depths", default="5,6,7")
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)
DEPTHS = [int(x) for x in args.depths.split(",")]
XG = np.arange(0.5, 12.01, 0.5); SIG = np.arange(0.5, 8.01, 0.25); TAUS = (2.0, 5.0, 10.0)


def expq(w_deg):
    w = np.deg2rad(w_deg); a = np.linalg.norm(w, axis=1, keepdims=True); ax = w / np.maximum(a, 1e-12)
    return np.concatenate([np.cos(a / 2), np.sin(a / 2) * ax], 1)


def sym_err(qh, qg, sq):
    # min over symmetries s of the angle between qh and qg*s (q ~ -q)
    best = np.full(len(qh), 180.0)
    for s in sq:
        qs = quat_multiply(qg, np.broadcast_to(s, qg.shape))
        d = np.degrees(2 * np.arccos(np.clip(np.abs((qh * qs).sum(1)), 0, 1)))
        best = np.minimum(best, d)
    return best


def in_mode_err(qg, sq, sigma, L, reps=2):
    qg = np.repeat(qg, reps, 0)
    qn = normalize(quat_multiply(qg, expq(rng.normal(0, sigma, (len(qg), 3)))))
    c, p = T.encode(qn, L)
    return sym_err(T.decode(c, p), qg, sq)


def leaf_logvol(chart, path):  # identical to scripts/exp131_ll.py
    lo, hi = T._cell_bounds(chart, path); c = 0.5 * (lo + hi); side = (hi - lo)[:, 0]
    return np.log(side ** 3) - 2 * np.log1p((c ** 2).sum(1))


def in_mode_logdens(qg, sigma, L, M=3000):
    # calibrated tree: estimate mu = q*Exp(e), predictive N(mu, sigma^2); log P(GT leaf) - log vol(GT leaf), mean over GT
    out = []
    for q in qg:
        mu = normalize(quat_multiply(q[None], expq(rng.normal(0, sigma, (1, 3)))))
        x = normalize(quat_multiply(np.repeat(mu, M, 0), expq(rng.normal(0, sigma, (M, 3)))))
        cg, pg = T.encode(q[None], L); cx, px = T.encode(x, L)
        hit = ((cx == cg[0]) & (px == pg[0]).all(1)).mean()
        out.append(np.log(max(hit, 0.5 / M)) - leaf_logvol(cg, pg)[0])
    return float(np.mean(out))


files = sorted(glob.glob(os.path.join(args.pred_dir, f"{args.dataset}_*_seed*_DINO.npz")))
objs = sorted({os.path.basename(f)[len(args.dataset) + 1:].rsplit("_seed", 1)[0] for f in files})
rows = []
for o in objs:
    E, Q, S = [], None, None
    for f in sorted(glob.glob(os.path.join(args.pred_dir, f"{args.dataset}_{o}_seed*_DINO.npz"))):
        z = np.load(f, allow_pickle=True); m = z["split"] == args.split
        E.append(z[f"err_{args.mode}"][m]); Q = z["q_tgt_gt"][m]; S = z["sym"]
    e = np.concatenate(E); sq = normalize(np.stack([matrix_to_quat(s) for s in S]))
    Femp = np.array([(e <= x).mean() for x in XG])
    best = None
    for sg in SIG:
        ein = in_mode_err(Q, sq, sg, 5); Fin = np.array([(ein <= x).mean() for x in XG])
        pi = float(np.clip(Femp @ Fin / max(Fin @ Fin, 1e-12), 0, 1)); r = float(((Femp - pi * Fin) ** 2).sum())
        if best is None or r < best[0]:
            best = (r, sg, pi)
    _, sg, pi = best
    r = {"obj": o, "n": int(len(e)), "n_sym": int(len(S)), "sigma": float(sg), "pi": pi,
         "L5_mean": float(e.mean()), **{f"L5_R@{t:g}": float((e <= t).mean()) for t in TAUS}}
    ein = {L: in_mode_err(Q, sq, sg, L, reps=8) for L in DEPTHS}
    sub = Q[rng.choice(len(Q), min(200, len(Q)), replace=False)]
    dens = {L: in_mode_logdens(sub, sg, L) for L in DEPTHS}
    for L in DEPTHS:
        r[f"Ein_mean_L{L}"] = float(ein[L].mean())
        r[f"pred_mean_L{L}"] = r["L5_mean"] + pi * float(ein[L].mean() - ein[5].mean())
        for t in TAUS:
            r[f"pred_R@{t:g}_L{L}"] = r[f"L5_R@{t:g}"] + pi * float((ein[L] <= t).mean() - (ein[5] <= t).mean())
        r[f"pred_dLL_L{L}"] = pi * (dens[L] - dens[5])
    rows.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if "Ein" not in k}, flush=True)

agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("obj",)}
res = {"dataset": args.dataset, "mode": args.mode, "split": args.split, "pred_dir": args.pred_dir, "objects": rows,
       "object_balanced": agg}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(res, open(args.out, "w"), indent=1)
print("| L | mean | R@2 | R@5 | R@10 | dLL vs L5 |\n|---|---|---|---|---|---|")
for L in DEPTHS:
    print(f"| {L} | {agg[f'pred_mean_L{L}']:.2f} | {100*agg[f'pred_R@2_L{L}']:.1f} | {100*agg[f'pred_R@5_L{L}']:.1f} | "
          f"{100*agg[f'pred_R@10_L{L}']:.1f} | {agg[f'pred_dLL_L{L}']:+.2f} |")
print(f"object-balanced sigma {agg['sigma']:.2f} deg, pi {agg['pi']:.3f}")
