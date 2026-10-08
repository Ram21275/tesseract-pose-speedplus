"""Phase-3 evaluation: single branches and equal-weight PoE (DEC-003) for a probe variant, several beam widths.

Loads checkpoints written by train_probe.py (EXP-030) or train_probe_p3.py (EXP-031/033/035).
With residual heads, the fused rotation is q_leaf (x) Exp(mean of both branches' residuals at the fused leaf).
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
from tfpose.metrics import rotation_metrics  # noqa: E402
from tfpose.predictor import HierMLP, decode_torch  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True, help="experiment holding the probe runs")
ap.add_argument("--a", default="dinov3_vitl16:grid4")
ap.add_argument("--b", default="moge2_vitl:normals16~and")
ap.add_argument("--subset", default="full")
ap.add_argument("--beams", default="1,4")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--run-exp", required=True)
args = ap.parse_args()
run = Run(args.run_exp.split("_", 1)[0], args.run_exp.split("_", 1)[1], dict(vars(args), stage="fused_eval"), seed=0)
try:
    dev = torch.device("cuda")
    df = pd.read_csv(REPO / f"outputs/data_manifests/subset_{args.subset}.csv")
    q = df[[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    tr = ((df.domain == "synthetic") & (df.split == "train")).to_numpy()
    groups = {"synthetic_val": ((df.domain == "synthetic") & (df.split == "validation")).to_numpy(),
              "lightbox": (df.domain == "lightbox").to_numpy(), "sunlamp": (df.domain == "sunlamp").to_numpy()}
    feats = {}

    def feat(spec):
        if spec not in feats:
            names, X, _ = featsets.load(spec, args.subset); assert names == df.image_relpath.tolist()
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            feats[spec] = torch.tensor((X - mu) / sd, device=dev, dtype=torch.float16)
        return feats[spec]

    def load(spec, seed):
        for d in sorted((REPO / "outputs/experiments" / args.exp).glob(f"RUN-*-seed{seed}")):
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            if cfg.get("features") != spec or cfg.get("epochs") != 100 or not (d / "checkpoints/best.pt").exists():
                continue
            x = feat(spec)
            ck = torch.load(d / "checkpoints/best.pt", map_location=dev)
            sd_model = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
            if cfg.get("decoder", "mlp") == "gru":
                m = P.HierGRU(x.shape[1], cfg["depth"], hidden=cfg["hidden"], dropout=cfg["dropout"]).to(dev)
            else:
                m = HierMLP(x.shape[1], cfg["depth"], hidden=cfg["hidden"], dropout=cfg["dropout"]).to(dev)
            m.load_state_dict(sd_model); m.eval()
            res = None
            if isinstance(ck, dict) and ck.get("residual") is not None:
                res = P.ResidualHead(cfg["hidden"]).to(dev); res.load_state_dict(ck["residual"]); res.eval()
            return m, res, x, d.name
        raise FileNotFoundError(spec, seed)

    def nodes(beam, L=5):
        k, n = min(beam, 4), 4
        for _ in range(L):
            n += 8 * k; k = min(beam, k * 8)
        return n

    rows, used = [], []
    for seed in [int(s) for s in args.seeds.split(",")]:
        mA, rA, xA, nA = load(args.a, seed); mB, rB, xB, nB = load(args.b, seed); used += [nA, nB]
        for beam in [int(b) for b in args.beams.split(",")]:
            for name, ms, rs, xs, rule in [("A", [mA], [rA], [xA], "single"), ("B", [mB], [rB], [xB], "single"),
                                           ("PoE(A,B)", [mA, mB], [rA, rB], [xA, xB], "poe")]:
                for g, msk in groups.items():
                    idx = np.flatnonzero(msk); qd, qr = [], []
                    with torch.no_grad():
                        for s in range(0, len(idx), 2048):
                            b = torch.as_tensor(idx[s:s + 2048], device=dev)
                            c, p, hs = fused_beam(ms, [x[b].float() for x in xs], rule, beam, return_h=True)
                            ql = decode_torch(c, p); qd.append(ql.cpu().numpy())
                            if all(r is not None for r in rs):
                                delta = torch.stack([r(h, ql) for r, h in zip(rs, hs)]).mean(0)
                                qr.append(P.qmul(ql, P.rotvec_to_quat(delta)).cpu().numpy())
                    disc = rotation_metrics(np.concatenate(qd), q[msk])
                    row = {"seed": seed, "beam": beam, "nodes_per_branch": nodes(beam), "method": name, "domain": g,
                           **{f"discrete_{k}": v for k, v in disc.items()}}
                    row.update(rotation_metrics(np.concatenate(qr), q[msk]) if qr else disc)
                    rows.append(row)
    per = pd.DataFrame(rows); run.write_metrics(rows, "per_seed")
    g = per.groupby(["beam", "method", "domain"])
    cols = ["mean_deg", "median_deg", "acc@1", "acc@5", "acc@10", "acc@20", "discrete_mean_deg", "nodes_per_branch"]
    agg = g[cols].mean().join(g[["mean_deg"]].std().rename(columns={"mean_deg": "mean_deg_std"})).reset_index()
    run.write_metrics(agg.to_dict("records"), "metrics"); (run.sub("tables") / "metrics.md").write_text(md_table(agg.to_dict("records")))
    (run.dir / "source_runs.txt").write_text("\n".join(used) + "\n"); run.done()
    print(md_table(agg.to_dict("records"))); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
