"""EXP-026: reliability-weighted posterior fusion (entropy, margin, agreement gate) vs equal PoE."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import featsets, tesseract as T  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.metrics import rotation_metrics  # noqa: E402
from tfpose.predictor import HierMLP  # noqa: E402
from tfpose.rotations import geodesic_distance_deg  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

A, B, SUBSET, AGREE_DEG = "EXP-010_dinov3_baseline:dinov3_vitl16:grid4", "EXP-016_consensus_pose:moge2_vitl:normals16~and", "v1", 20.0
run = Run("EXP-026", "reliability_weighting", {"a": A, "b": B, "subset": SUBSET, "agree_deg": AGREE_DEG}, seed=0)
try:
    dev = torch.device("cuda")
    df = pd.read_csv(REPO / f"outputs/data_manifests/subset_{SUBSET}.csv")
    q = df[[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    tr = ((df.domain == "synthetic") & (df.split == "train")).to_numpy()
    groups = {"synthetic_val": ((df.domain == "synthetic") & (df.split == "validation")).to_numpy(),
              "lightbox": (df.domain == "lightbox").to_numpy(), "sunlamp": (df.domain == "sunlamp").to_numpy()}
    feats = {}

    def load(spec, seed):
        exp, feat = spec.split(":", 1)
        if feat not in feats:
            names, X, _ = featsets.load(feat, SUBSET); assert names == df.image_relpath.tolist()
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            feats[feat] = torch.tensor((X - mu) / sd, device=dev)
        for d in sorted((REPO / "outputs/experiments" / exp).glob(f"RUN-*-seed{seed}")):
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            if cfg.get("features") == feat and cfg.get("epochs") == 100 and (d / "checkpoints/best.pt").exists():
                m = HierMLP(feats[feat].shape[1], cfg["depth"], hidden=cfg["hidden"], dropout=cfg["dropout"]).to(dev)
                m.load_state_dict(torch.load(d / "checkpoints/best.pt", map_location=dev)); return m.eval(), feats[feat], d.name
        raise FileNotFoundError(spec, seed)

    rows, used = [], []
    for seed in (0, 1, 2):
        mA, xA, rA = load(A, seed); mB, xB, rB = load(B, seed); used += [rA, rB]
        for g, msk in groups.items():
            idx = torch.as_tensor(np.flatnonzero(msk), device=dev)
            dec = lambda c, p: T.decode(c.cpu().numpy(), p.cpu().numpy())
            qa = dec(*fused_beam([mA], [xA[idx]], "single")); qb = dec(*fused_beam([mB], [xB[idx]], "single"))
            qpoe = dec(*fused_beam([mA, mB], [xA[idx], xB[idx]], "poe"))
            out = {"A_dino": qa, "PoE": qpoe,
                   "ent": dec(*fused_beam([mA, mB], [xA[idx], xB[idx]], "ent")),
                   "margin": dec(*fused_beam([mA, mB], [xA[idx], xB[idx]], "margin"))}
            agree = geodesic_distance_deg(qa, qb) < AGREE_DEG
            out["agree_gate"] = np.where(agree[:, None], qpoe, qa)
            for name, qp in out.items():
                rows.append({"seed": seed, "method": name, "domain": g, "agree_frac": float(agree.mean()), **rotation_metrics(qp, q[msk])})
    per = pd.DataFrame(rows); run.write_metrics(rows, "per_seed")
    g = per.groupby(["method", "domain"])
    agg = g[["mean_deg", "median_deg", "acc@10", "acc@20"]].mean().join(g[["mean_deg"]].std().rename(columns={"mean_deg": "mean_deg_std"})).reset_index()
    run.write_metrics(agg.to_dict("records"), "metrics"); (run.sub("tables") / "metrics.md").write_text(md_table(agg.to_dict("records")))
    (run.dir / "source_runs.txt").write_text("\n".join(used) + "\n"); run.done()
    print(md_table(agg.to_dict("records"))); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
