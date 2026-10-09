"""Phase-4 stage-0 diagnostic on the frozen EXP-036 predictor (no training).

For single branches and the PoE (DEC-003), per evaluation group incl. a 12k sample of synthetic train:
- greedy leaf accuracy and the distribution of greedy-leaf -> GT rotation error;
- beam-K oracle: GT leaf among the K beams, best-of-K rotation error, recall within 5/10/20 deg.
Sets the conditioning distribution for the EXP-040 residual (predicted vs GT leaf) and the
width of the EXP-041 local base distribution. Groups follow DEC-005.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import featsets, phase3 as P, tesseract as T  # noqa: E402
from tfpose.data import eval_groups  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.predictor import decode_torch  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src-exp", default="EXP-036_transformer_soft_targets")
ap.add_argument("--a", default="dinov3_vitl16:tokens8")
ap.add_argument("--b", default="moge2_vitl:normals16~and")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--beam", type=int, default=8)
ap.add_argument("--n-train", type=int, default=12000)
ap.add_argument("--run-exp", default="EXP-040_tangent_residual_predicted_leaf")
args = ap.parse_args()
run = Run(args.run_exp.split("_", 1)[0], args.run_exp.split("_", 1)[1], dict(vars(args), stage="diag"), seed=0)
BINS = [0, 2.5, 5, 10, 20, 45, 90, 180.1]
try:
    dev = torch.device("cuda")
    df = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv")
    q = df[[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    chart, path = T.encode(q, 5)
    gt_leaf = T.leaf_id(chart, path)
    tr = ((df.domain == "synthetic") & (df.split == "train")).to_numpy()
    groups = eval_groups(df)
    tr_s = np.zeros(len(df), bool); tr_s[np.random.default_rng(0).choice(np.flatnonzero(tr), args.n_train, replace=False)] = True
    groups = {"synthetic_train_sample": tr_s, **groups}
    feats = {}

    def feat(spec):
        if spec not in feats:
            names, X, _ = featsets.load(spec, "full"); assert names == df.image_relpath.tolist()
            feats[spec] = featsets.standardize_to_tensor(X, tr, "cpu" if X.nbytes > 4e9 else dev, torch.float16, pin=False)
            del X
        return feats[spec]

    def load(spec, seed):
        for d in sorted((REPO / "outputs/experiments" / args.src_exp).glob(f"RUN-*-seed{seed}")):
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            if cfg.get("features") != spec or not (d / "checkpoints/best.pt").exists():
                continue
            m = P.HierTransformer(tuple(int(v) for v in cfg["token_shape"].split("x")), cfg["depth"], dropout=cfg["dropout"]).to(dev)
            m.load_state_dict(torch.load(d / "checkpoints/best.pt", map_location=dev)["model"]); m.eval()
            return m, feat(spec), d.name
        raise FileNotFoundError(spec, seed)

    def err_deg(qa, qb):
        return np.degrees(2 * np.arccos(np.clip(np.abs((qa * qb).sum(-1)), 0, 1)))

    rows, used, preds = [], [], []
    for seed in [int(s) for s in args.seeds.split(",")]:
        mA, xA, nA = load(args.a, seed); mB, xB, nB = load(args.b, seed); used += [nA, nB]
        for name, ms, xs, rule in [("A", [mA], [xA], "single"), ("B", [mB], [xB], "single"), ("PoE(A,B)", [mA, mB], [xA, xB], "poe")]:
            for g, msk in groups.items():
                idx = np.flatnonzero(msk); C, Pp = [], []
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for s in range(0, len(idx), 1024):
                        b = torch.as_tensor(idx[s:s + 1024], device=dev)
                        c, p, _, _ = fused_beam(ms, [(x[b] if x.is_cuda else x[b.cpu()].to(dev, non_blocking=True)).float() for x in xs],
                                                rule, args.beam, return_all=True)
                        C.append(c.cpu().numpy()); Pp.append(p.cpu().numpy())
                C, Pp = np.concatenate(C), np.concatenate(Pp)               # (N,K), (N,K,5)
                K = C.shape[1]
                qk = np.stack([T.decode(C[:, k], Pp[:, k]) for k in range(K)], 1)
                ek = err_deg(qk, q[idx][:, None])                           # (N,K)
                lk = np.stack([T.leaf_id(C[:, k], Pp[:, k]) for k in range(K)], 1)
                e1, best = ek[:, 0], ek.min(1)
                hist = np.histogram(e1, BINS)[0] / len(e1)
                row = {"seed": seed, "method": name, "domain": g, "n": len(idx),
                       "greedy_mean": e1.mean(), "greedy_median": np.median(e1), "greedy_leaf_acc": np.mean(lk[:, 0] == gt_leaf[idx]),
                       **{f"greedy_frac_{BINS[i]}-{BINS[i + 1]:.0f}": hist[i] for i in range(len(hist))},
                       f"top{K}_leaf_recall": np.mean((lk == gt_leaf[idx][:, None]).any(1)),
                       f"best_of_{K}_mean": best.mean(), f"best_of_{K}_median": np.median(best),
                       **{f"best_of_{K}_within_{t}": np.mean(best <= t) for t in (5, 10, 20)},
                       **{f"greedy_within_{t}": np.mean(e1 <= t) for t in (5, 10, 20)},
                       f"best_beam_rank_mean_if_not_0": float(ek.argmin(1)[ek.argmin(1) > 0].mean()) if (ek.argmin(1) > 0).any() else 0.0}
                rows.append(row)
                if seed == 0 and name == "PoE(A,B)":
                    preds.append(pd.DataFrame({"image_relpath": df.image_relpath.values[idx], "group": g,
                                               **{f"err_beam{k}": ek[:, k] for k in range(K)}}))
                print(name, seed, g, f"greedy {row['greedy_mean']:.2f}  leafacc {row['greedy_leaf_acc']:.3f}  best{K} {row[f'best_of_{K}_mean']:.2f}"
                      f"  within10 {row['greedy_within_10']:.3f}->{row[f'best_of_{K}_within_10']:.3f}", flush=True)
    per = pd.DataFrame(rows); run.write_metrics(rows, "per_seed")
    num = [c for c in per.columns if c not in ("seed", "method", "domain")]
    agg = per.groupby(["method", "domain"])[num].mean().reset_index()
    run.write_metrics(agg.to_dict("records"), "metrics"); (run.sub("tables") / "metrics.md").write_text(md_table(agg.to_dict("records")))
    pd.concat(preds).to_csv(run.sub("predictions") / "poe_seed0_beam_errors.csv", index=False, float_format="%.4g")
    (run.dir / "source_runs.txt").write_text("\n".join(used) + "\n"); run.done()
    print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
