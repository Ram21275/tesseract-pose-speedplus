"""EXP-043: flow-induced Tesseract masses from the SAME endpoint samples (no training).

Input: fused EXP-042 endpoint samples per seed and group (N, M=32, 4), saved by
exp04x_fused_eval.py --save-samples --save-all-seeds into this experiment's samples/ stage.
For every level l = 0 (root chart) .. 5 (leaf):
  flow mass  mu_l(c) = (1/M) #{m : prefix_l(R_m) = c}   (exact parent-child consistency by construction; verified)
  tree mass  p_l(GT prefix) = prod of PoE conditional probabilities along the GT path (teacher-forced, EXP-036 trees)
Reported per level: mass on the GT cell (flow vs tree), recall (flow mass > 0), top-1 (most-massive cell = GT;
tree: greedy prefix = GT), occupancy (distinct cells with mass), parent-child consistency error, mass
self-consistency vs M (TV distance of m-subsets to the full 32), seam cases (GT chart margin < 0.05).
Point estimates (descriptive comparison): tree greedy leaf, KDE mode (EXP-042), flow-mass greedy descent.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import featsets, phase3 as P, so3flow as S, tesseract as T  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.predictor import decode_torch  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--samples-run", required=True, help="EXP-043 sample-stage run dir (holds samples/*_fused_seed{s}_fp16.npy)")
ap.add_argument("--tree-exp", default="EXP-036_transformer_soft_targets")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--groups", default="synthetic_val,lightbox,sunlamp")
ap.add_argument("--run-exp", default="EXP-043_flow_induced_tesseract_masses", help="experiment dir for this analysis run")
args = ap.parse_args()
run = Run(args.run_exp.split("_", 1)[0], args.run_exp.split("_", 1)[1], dict(vars(args), stage="analysis"), seed=0)
L = 5
SPECS = ["dinov3_vitl16:tokens8", "moge2_vitl:normals16~and"]
try:
    dev = torch.device("cuda")
    df = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv")
    q = df[[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    chart, path = T.encode(q, L)
    tr = ((df.domain == "synthetic") & (df.split == "train")).to_numpy()
    sdir = REPO / args.samples_run / "samples"
    groups = args.groups.split(",")
    X = {}
    for sp in SPECS:
        names, x, _ = featsets.load(sp, "full"); assert names == df.image_relpath.tolist()
        X[sp] = featsets.standardize_to_tensor(x, tr, "cpu", torch.float16); del x   # host memory; batches moved to GPU

    def tree(spec, seed):
        for d in sorted((REPO / "outputs/experiments" / args.tree_exp).glob(f"RUN-*-seed{seed}")):
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            if cfg.get("features") == spec and (d / "checkpoints/best.pt").exists():
                m = P.HierTransformer(tuple(int(v) for v in cfg["token_shape"].split("x")), cfg["depth"], dropout=cfg["dropout"]).to(dev)
                m.load_state_dict(torch.load(d / "checkpoints/best.pt", map_location=dev)["model"]); m.eval()
                return m
        raise FileNotFoundError(spec, seed)

    def prefix_ids(c, p, l):                       # unique id of the level-l cell (l = 0: chart)
        return c.astype(np.int64) if l == 0 else T.leaf_id(c, p[:, :l])

    def tv(a_ids, b_ids):                          # TV distance between empirical distributions per image (rows)
        out = np.empty(len(a_ids))
        for i, (a, b) in enumerate(zip(a_ids, b_ids)):
            ua, ca = np.unique(a, return_counts=True); ub, cb = np.unique(b, return_counts=True)
            pa = dict(zip(ua, ca / len(a))); pb = dict(zip(ub, cb / len(b)))
            out[i] = 0.5 * sum(abs(pa.get(k, 0) - pb.get(k, 0)) for k in set(pa) | set(pb))
        return out

    rows, prow = [], []
    for seed in [int(s) for s in args.seeds.split(",")]:
        mA, mB = tree(SPECS[0], seed), tree(SPECS[1], seed)
        for g in groups:
            idx = np.load(sdir / f"{g}_index.npy"); smp = np.load(sdir / f"{g}_fused_seed{seed}_fp16.npy").astype(np.float64)
            N, M, _ = smp.shape
            smp /= np.linalg.norm(smp, axis=-1, keepdims=True)
            sc, sp_ = T.encode(smp.reshape(-1, 4), L)                       # (N*M,), (N*M, L)
            sc, sp_ = sc.reshape(N, M), sp_.reshape(N, M, L)
            gc, gp = chart[idx], path[idx]
            # ---- tree: teacher-forced PoE mass on the GT prefix + greedy prefix ----
            lp_gt = np.zeros((N, L + 1)); gr_c, gr_p = [], []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for s in range(0, N, 1024):
                    b = torch.as_tensor(idx[s:s + 1024], device=dev)
                    xs = [X[SPECS[0]][b.cpu()].to(dev).float(), X[SPECS[1]][b.cpu()].to(dev).float()]
                    ct, pt = torch.as_tensor(gc[s:s + 1024], device=dev), torch.as_tensor(gp[s:s + 1024], device=dev)
                    outs = [m.logits_tf(m.encode(x), ct, pt) for m, x in zip((mA, mB), xs)]
                    root = F.log_softmax(sum(F.log_softmax(o[0].float(), -1) for o in outs), -1)
                    child = F.log_softmax(sum(F.log_softmax(o[1].float(), -1) for o in outs), -1)       # (B,L,8)
                    step = torch.cat([root.gather(1, ct[:, None]), child.gather(2, pt[:, :, None])[..., 0]], 1)  # (B,L+1)
                    lp_gt[s:s + 1024] = step.cumsum(1).cpu().numpy()
                    c, p = fused_beam([mA, mB], xs, "poe", 1)
                    gr_c.append(c.cpu().numpy()); gr_p.append(p.cpu().numpy())
            gr_c, gr_p = np.concatenate(gr_c), np.concatenate(gr_p)
            # ---- flow masses per level ----
            for l in range(L + 1):
                ids = np.stack([prefix_ids(sc[:, m], sp_[:, m], l) for m in range(M)], 1)          # (N,M)
                gt_id = prefix_ids(gc, gp, l)
                mass_gt = (ids == gt_id[:, None]).mean(1)
                # most-massive cell (ties -> smallest id)
                top = np.array([np.unique(r, return_counts=True) for r in ids], dtype=object)
                top1 = np.array([u[np.argmax(c_)] for u, c_ in top]); occ = np.array([len(u) for u, _ in top])
                tree_mass = np.exp(lp_gt[:, l]); tree_top1 = prefix_ids(gr_c, gr_p, l) == gt_id
                cons = 0.0
                if l < L:                                                                        # parent-child consistency
                    kids = np.stack([prefix_ids(sc[:, m], sp_[:, m], l + 1) for m in range(M)], 1)
                    for i in range(0, N, max(1, N // 200)):                                      # ~200 images checked exhaustively
                        for u in np.unique(ids[i]):
                            par = (ids[i] == u).sum(); ch = sum((kids[i] == k).sum() for k in np.unique(kids[i][ids[i] == u]))
                            cons = max(cons, abs(par - ch) / M)
                tvs = {f"tv_vs_M32_m{m_}": float(tv(ids[:, :m_], ids).mean()) for m_ in (4, 8, 16)}
                srt = np.sort(np.abs(q[idx]), 1); seam = (srt[:, -1] - srt[:, -2]) < 0.05
                rows.append({"seed": seed, "domain": g, "level": l, "n": N,
                             "flow_mass_on_gt": float(mass_gt.mean()), "tree_mass_on_gt": float(tree_mass.mean()),
                             "flow_recall": float((mass_gt > 0).mean()), "flow_top1": float((top1 == gt_id).mean()),
                             "tree_top1_greedy": float(tree_top1.mean()), "flow_occupancy": float(occ.mean()),
                             "parent_child_max_err": cons, **tvs,
                             "seam_n": int(seam.sum()), "seam_flow_top1": float((top1 == gt_id)[seam].mean()) if seam.any() else np.nan,
                             "seam_tree_top1": float(tree_top1[seam].mean()) if seam.any() else np.nan})
            # ---- point estimates: tree greedy vs flow-mass greedy descent ----
            fc = np.empty(N, np.int64); fp = np.empty((N, L), np.int64)
            for i in range(N):
                keep = np.ones(M, bool)
                u, c_ = np.unique(sc[i], return_counts=True); fc[i] = u[np.argmax(c_)]; keep &= sc[i] == fc[i]
                for l in range(L):
                    u, c_ = np.unique(sp_[i, keep, l], return_counts=True); fp[i, l] = u[np.argmax(c_)]; keep &= sp_[i, :, l] == fp[i, l]
            def err(qq):
                return np.degrees(2 * np.arccos(np.clip(np.abs((qq * q[idx]).sum(1)), 0, 1)))
            e_tree, e_flow = err(T.decode(gr_c, gr_p)), err(T.decode(fc, fp))
            kde = S.kde_mode(torch.as_tensor(smp, dtype=torch.float32)).numpy(); e_kde = err(kde)
            for nm, e in [("tree greedy (PoE, EXP-036)", e_tree), ("flow KDE mode (EXP-042)", e_kde), ("flow-mass greedy descent", e_flow)]:
                prow.append({"seed": seed, "domain": g, "estimate": nm, "mean_deg": e.mean(), "median_deg": np.median(e),
                             "acc@10": float((e <= 10).mean()), "frac_gt90": float((e > 90).mean())})
            print(seed, g, f"tree {e_tree.mean():.2f} kde {e_kde.mean():.2f} flowgreedy {e_flow.mean():.2f}", flush=True)
    per = pd.DataFrame(rows); run.write_metrics(rows, "per_level_per_seed")
    num = [c for c in per.columns if c not in ("seed", "domain", "level")]
    agg = per.groupby(["domain", "level"])[num].mean().reset_index()
    run.write_metrics(agg.to_dict("records"), "per_level"); (run.sub("tables") / "per_level.md").write_text(md_table(agg.to_dict("records")))
    pp = pd.DataFrame(prow); run.write_metrics(prow, "point_estimates_per_seed")
    pa = pp.groupby(["domain", "estimate"])[["mean_deg", "median_deg", "acc@10", "frac_gt90"]].mean().join(
        pp.groupby(["domain", "estimate"])[["mean_deg"]].std().rename(columns={"mean_deg": "mean_deg_std"})).reset_index()
    run.write_metrics(pa.to_dict("records"), "point_estimates"); (run.sub("tables") / "point_estimates.md").write_text(md_table(pa.to_dict("records")))
    run.done()
    print(md_table(agg.to_dict("records"), ["domain", "level", "flow_mass_on_gt", "tree_mass_on_gt", "flow_recall", "flow_top1", "tree_top1_greedy", "flow_occupancy", "parent_child_max_err", "tv_vs_M32_m8"]))
    print(md_table(pa.to_dict("records")))
    print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
