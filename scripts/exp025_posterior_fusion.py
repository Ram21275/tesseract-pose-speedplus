"""EXP-025: training-free posterior fusion of two Tesseract probes (PoE / per-level average) vs single branches."""
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
from tfpose import featsets, tesseract as T  # noqa: E402
from tfpose.metrics import rotation_metrics  # noqa: E402
from tfpose.predictor import HierMLP, decode_torch  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--a", default="EXP-010_dinov3_baseline:dinov3_vitl16:grid4")
ap.add_argument("--b", default="EXP-016_consensus_pose:moge2_vitl:normals16~and")
ap.add_argument("--subset", default="v1")
ap.add_argument("--seeds", default="0,1,2")
args = ap.parse_args()
run = Run("EXP-025", "posterior_fusion_controls", vars(args), seed=0)
try:
    dev = torch.device("cuda")
    df = pd.read_csv(REPO / f"outputs/data_manifests/subset_{args.subset}.csv")
    q = df[[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    groups = {"synthetic_val": ((df.domain == "synthetic") & (df.split == "validation")).to_numpy(),
              "lightbox": (df.domain == "lightbox").to_numpy(), "sunlamp": (df.domain == "sunlamp").to_numpy()}
    tr = ((df.domain == "synthetic") & (df.split == "train")).to_numpy()

    def load_branch(spec, seed):
        exp, feat = spec.split(":", 1)
        for d in sorted((REPO / "outputs/experiments" / exp).glob(f"RUN-*-seed{seed}")):
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            if cfg.get("features") == feat and cfg.get("epochs") == 100 and (d / "checkpoints/best.pt").exists():
                names, X, _ = featsets.load(feat, args.subset)
                assert names == df.image_relpath.tolist()
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6           # same standardization as training
                m = HierMLP(X.shape[1], cfg["depth"], hidden=cfg["hidden"], dropout=cfg["dropout"]).to(dev)
                m.load_state_dict(torch.load(d / "checkpoints/best.pt", map_location=dev)); m.eval()
                return m, torch.tensor((X - mu) / sd, device=dev), d.name
        raise FileNotFoundError(spec, seed)

    @torch.no_grad()
    def fused_beam(models, xs, rule, beam):
        """Beam search where each level's log-probs combine several probes."""
        hs = [m.encode(x) for m, x in zip(models, xs)]

        def comb(lps):
            if rule == "poe":
                return F.log_softmax(sum(lps), -1)                      # renormalized product
            if rule == "avg":
                return torch.logsumexp(torch.stack(lps), 0) - np.log(len(lps))
            return lps[0]
        B = hs[0].shape[0]
        lp = comb([F.log_softmax(m.root(h).float(), -1) for m, h in zip(models, hs)])
        k0 = min(beam, 4)
        score, chart = lp.topk(k0, -1)
        path = torch.zeros(B, k0, 0, dtype=torch.long, device=dev)
        for l in range(models[0].depth):
            K = chart.shape[1]
            pq = decode_torch(chart.reshape(-1), path.reshape(B * K, l))
            lev = torch.full((B * K,), l, device=dev, dtype=torch.long)
            clps = [F.log_softmax(m.child_logits(h[:, None].expand(B, K, h.shape[-1]).reshape(B * K, -1), lev, pq).float(), -1)
                    for m, h in zip(models, hs)]
            tot = (score[:, :, None] + comb(clps).view(B, K, 8)).view(B, K * 8)
            score, idx = tot.topk(min(beam, K * 8), -1)
            src, child = idx // 8, idx % 8
            chart = chart.gather(1, src)
            path = torch.cat([path.gather(1, src[:, :, None].expand(-1, -1, l)), child[:, :, None]], -1)
        return chart[:, 0], path[:, 0]

    rows, used = [], []
    for seed in [int(s) for s in args.seeds.split(",")]:
        mA, xA, rA = load_branch(args.a, seed); mB, xB, rB = load_branch(args.b, seed)
        used += [rA, rB]
        for beam in (1, 4):
            for name, models, xs, rule in [("A_dino", [mA], [xA], "single"), ("B_consensus_normals", [mB], [xB], "single"),
                                           ("PoE(A,B)", [mA, mB], [xA, xB], "poe"), ("AVG(A,B)", [mA, mB], [xA, xB], "avg")]:
                for g, msk in groups.items():
                    idx = torch.as_tensor(np.flatnonzero(msk), device=dev)
                    c, p = fused_beam(models, [x[idx] for x in xs], rule, beam)
                    qp = T.decode(c.cpu().numpy(), p.cpu().numpy())
                    rows.append({"seed": seed, "beam": beam, "method": name, "domain": g, **rotation_metrics(qp, q[msk])})
    per = pd.DataFrame(rows)
    run.write_metrics(rows, "per_seed")
    g = per.groupby(["beam", "method", "domain"])
    agg = g[["mean_deg", "median_deg", "acc@10", "acc@20"]].mean().join(g[["mean_deg"]].std().rename(columns={"mean_deg": "mean_deg_std"})).reset_index()
    run.write_metrics(agg.to_dict("records"), "metrics")
    (run.sub("tables") / "metrics.md").write_text(md_table(agg.to_dict("records")))
    (run.dir / "source_runs.txt").write_text("\n".join(used) + "\n")
    run.done()
    print(md_table(agg.to_dict("records"))); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc())
    raise
