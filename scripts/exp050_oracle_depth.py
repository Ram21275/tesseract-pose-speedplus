"""EXP-050: oracle adaptive-depth targets for the frozen EXP-036 PoE tree (reduced Phase 5, DEC-012).

For every image, decode the PoE greedy path and record the error of the prefix cell centre at each depth
l = 0..5 (e_l), the error of the best of the beam-8 leaves (BRANCH oracle), and the oracle depth
l* = argmin_l [e_l + lambda * l] for lambda in {0, 0.1, 0.5, 1.0} deg per level. Headroom = mean e_5 - mean e_{l*}.
Targets (STOP at l* / DESCEND_ONE / BRANCH) are saved for the training split, as the plan requires.
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
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base-exp", default="EXP-036_transformer_soft_targets")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--lambdas", default="0,0.1,0.5,1.0")
args = ap.parse_args()
run = Run("EXP-050", "oracle_adaptive_depth", vars(args), seed=0)
SPECS = {"dinov3_vitl16:tokens8": "64x1024", "moge2_vitl:normals16~and": "256x4"}
L = 5
LAM = [float(v) for v in args.lambdas.split(",")]
try:
    dev = torch.device("cuda")
    df = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv")
    q = df[[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    tr = ((df.domain == "synthetic") & (df.split == "train")).to_numpy()
    groups = {"synthetic_train": tr, "synthetic_val": ((df.domain == "synthetic") & (df.split == "validation")).to_numpy(),
              "lightbox": (df.domain == "lightbox").to_numpy(), "sunlamp": (df.domain == "sunlamp").to_numpy()}
    X = {}
    for sp in SPECS:
        names, x, _ = featsets.load(sp, "full"); assert names == df.image_relpath.tolist()
        X[sp] = featsets.standardize_to_tensor(x, tr, "cpu", torch.float16); del x

    def geo(a, b):
        return np.degrees(2 * np.arccos(np.clip(np.abs((a * b).sum(-1)), 0, 1)))

    def find(spec, seed):
        for d in sorted((REPO / "outputs/experiments" / args.base_exp).glob(f"RUN-*-seed{seed}")):
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            if cfg.get("features") == spec and (d / "checkpoints/best.pt").exists():
                return d, cfg
        raise FileNotFoundError(spec, seed)

    rows, used, targets = [], [], {}
    for seed in [int(s) for s in args.seeds.split(",")]:
        ms = []
        for spec, tsh in SPECS.items():
            d, cfg = find(spec, seed)
            m = P.HierTransformer(tuple(int(v) for v in tsh.split("x")), cfg["depth"], dropout=cfg["dropout"]).to(dev)
            m.load_state_dict(torch.load(d / "checkpoints/best.pt", map_location=dev)["model"]); m.eval(); ms.append(m)
            used.append(f"{args.base_exp}/{d.name}")
        for g, msk in groups.items():
            idx = np.flatnonzero(msk); C, Pp, B8 = [], [], []
            for s in range(0, len(idx), 1024):
                b = torch.as_tensor(idx[s:s + 1024]); xs = [X[sp][b].to(dev).float() for sp in SPECS]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    c, p = fused_beam(ms, xs, "poe", 1)
                    c8, p8, _, _ = fused_beam(ms, xs, "poe", 8, return_all=True)
                C.append(c.cpu().numpy()); Pp.append(p.cpu().numpy())
                cen8 = T.decode(c8.cpu().numpy().reshape(-1), p8.cpu().numpy().reshape(-1, L)).reshape(len(b), -1, 4)
                B8.append(geo(cen8, q[idx[s:s + 1024]][:, None]).min(1))
            C, Pp, B8 = np.concatenate(C), np.concatenate(Pp), np.concatenate(B8)
            E = np.stack([geo(T.decode(C, Pp[:, :l]), q[idx]) for l in range(L + 1)], 1)       # (N, 6) prefix-centre errors
            r = {"seed": seed, "domain": g, "n": len(idx), **{f"e{l}_mean": float(E[:, l].mean()) for l in range(L + 1)},
                 "branch_beam8_oracle_mean": float(B8.mean())}
            for lam in LAM:
                ls = (E + lam * np.arange(L + 1)).argmin(1); eo = E[np.arange(len(E)), ls]
                r[f"oracle_lam{lam:g}_mean"] = float(eo.mean()); r[f"oracle_lam{lam:g}_mean_depth"] = float(ls.mean())
                r[f"oracle_lam{lam:g}_frac_stop_early"] = float((ls < L).mean())
            r["headroom_stop_lam0"] = r["e5_mean"] - r["oracle_lam0_mean"]
            r["headroom_branch"] = r["e5_mean"] - r["branch_beam8_oracle_mean"]
            rows.append(r)
            if g == "synthetic_train" and seed == 0:
                ls = E.argmin(1)
                tgt = np.where(B8 < E[:, L] - 1e-6, 2, np.where(ls < L, 0, 1))        # 2 BRANCH, 0 STOP early, 1 DESCEND to leaf
                targets = {"index": idx, "prefix_err": E.astype(np.float32), "beam8_best_err": B8.astype(np.float32),
                           "oracle_depth_lam0": ls.astype(np.int8), "target": tgt.astype(np.int8)}
            print(seed, g, f"e5 {r['e5_mean']:.3f} oracle-stop {r['oracle_lam0_mean']:.3f} beam8 {r['branch_beam8_oracle_mean']:.3f}", flush=True)
        del ms; torch.cuda.empty_cache()
    per = pd.DataFrame(rows); run.write_metrics(rows, "per_seed")
    cols = [c for c in per.columns if c not in ("seed", "domain", "n")]
    agg = per.groupby("domain")[cols].mean().join(per.groupby("domain")[["e5_mean", "headroom_stop_lam0"]].std().add_suffix("_std")).reset_index()
    run.write_metrics(agg.to_dict("records"), "metrics"); (run.sub("tables") / "metrics.md").write_text(md_table(agg.to_dict("records")))
    np.savez_compressed(run.sub("predictions") / "train_targets_seed0.npz", **targets)
    (run.dir / "source_runs.txt").write_text("\n".join(used) + "\n"); run.done()
    print(md_table(agg.to_dict("records"), ["domain", "e5_mean", "e5_mean_std", "oracle_lam0_mean", "headroom_stop_lam0", "headroom_stop_lam0_std",
                                            "oracle_lam0_mean_depth", "branch_beam8_oracle_mean"]))
    print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
