"""EXP-105 arm K evaluation: equal-weight per-level PoE of the two 8-facet Kurz trees, top-8 corner-vote decoding,
vs the EXP-036 PoE (greedy, recomputed here with the same code path as exp044_eval.py). DEC-006: synthetic val
selects; lightbox / sunlamp descriptive.
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
from tfpose import featsets, kurz8 as K8, phase3 as P, tesseract as T  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.metrics import rotation_metrics  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--k-exp", default="EXP-105_candidate_expansion_best_fit")
ap.add_argument("--base-exp", default="EXP-036_transformer_soft_targets")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--vote-beam", type=int, default=8)
args = ap.parse_args()
run = Run("EXP-105", "candidate_expansion_best_fit", {**vars(args), "stage": "armK_eval"}, seed=0)
SPECS = {"dinov3_vitl16:tokens8": "64x1024", "moge2_vitl:normals16~and": "256x4"}
try:
    dev = torch.device("cuda")
    df = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv")
    q = df[[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    tr = ((df.domain == "synthetic") & (df.split == "train")).to_numpy()
    groups = {"synthetic_val": ((df.domain == "synthetic") & (df.split == "validation")).to_numpy(),
              "lightbox": (df.domain == "lightbox").to_numpy(), "sunlamp": (df.domain == "sunlamp").to_numpy()}
    X = {}
    for sp in SPECS:
        names, x, _ = featsets.load(sp, "full"); assert names == df.image_relpath.tolist()
        X[sp] = featsets.standardize_to_tensor(x, tr, "cpu", torch.float16); del x

    def find(exp, spec, seed, k8):
        for d in sorted((REPO / "outputs/experiments" / exp).glob(f"RUN-*-seed{seed}")):
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            if cfg.get("features") == spec and (d / "checkpoints/best.pt").exists() and ("vote_beam" in cfg) == k8:
                return d, cfg
        raise FileNotFoundError(exp, spec, seed)

    rows, used = [], []
    for seed in [int(s) for s in args.seeds.split(",")]:
        km, bm = [], []
        for spec, tsh in SPECS.items():
            ts = tuple(int(v) for v in tsh.split("x"))
            d, cfg = find(args.k_exp, spec, seed, True)
            m = K8.HierTransformer8(ts, cfg["depth"], dropout=cfg["dropout"]).to(dev)
            m.load_state_dict(torch.load(d / "checkpoints/best.pt", map_location=dev)["model"]); m.eval(); km.append(m)
            d2, cfg2 = find(args.base_exp, spec, seed, False)
            b = P.HierTransformer(ts, cfg2["depth"], dropout=cfg2["dropout"]).to(dev)
            b.load_state_dict(torch.load(d2 / "checkpoints/best.pt", map_location=dev)["model"]); b.eval(); bm.append(b)
            used += [f"{args.k_exp}/{d.name}", f"{args.base_exp}/{d2.name}"]
        for g, msk in groups.items():
            idx = np.flatnonzero(msk); out = {k: [] for k in ["K_poe_vote", "K_poe_top1centre", "K_dino_vote", "K_moge_vote", "EXP036_poe_greedy"]}
            for s in range(0, len(idx), 512):
                b = torch.as_tensor(idx[s:s + 512], device=dev); xs = [X[sp][b.cpu()].to(dev).float() for sp in SPECS]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    c, p, sc = K8.beam8(km, xs, "poe", args.vote_beam)
                    out["K_poe_vote"].append(K8.corner_vote(c, p, sc)); out["K_poe_top1centre"].append(K8.decode8_torch(c[:, 0], p[:, 0]))
                    for name, m, x in (("K_dino_vote", km[0], xs[0]), ("K_moge_vote", km[1], xs[1])):
                        c1, p1, s1 = K8.beam8([m], [x], "single", args.vote_beam); out[name].append(K8.corner_vote(c1, p1, s1))
                    c0, p0 = fused_beam(bm, xs, "poe", 1)
                out["EXP036_poe_greedy"].append(torch.as_tensor(T.decode(c0.cpu().numpy(), p0.cpu().numpy())))
            for name, v in out.items():
                rows.append({"seed": seed, "domain": g, "arm": name, **rotation_metrics(torch.cat([t.float().cpu() for t in v]).numpy(), q[idx])})
            print(seed, g, " | ".join(f"{r['arm']} {r['mean_deg']:.3f}" for r in rows[-len(out):]), flush=True)
        del km, bm; torch.cuda.empty_cache()
    per = pd.DataFrame(rows); run.write_metrics(rows, "armK_per_seed")
    cols = ["mean_deg", "median_deg", "p95_deg", "acc@5", "acc@10", "acc@20"]
    agg = per.groupby(["domain", "arm"])[cols].mean().join(per.groupby(["domain", "arm"])[["mean_deg"]].std().rename(columns={"mean_deg": "mean_deg_std"})).reset_index()
    run.write_metrics(agg.to_dict("records"), "armK"); (run.sub("tables") / "armK.md").write_text(md_table(agg.to_dict("records")))
    (run.dir / "source_runs.txt").write_text("\n".join(used) + "\n"); run.done()
    print(md_table(agg.to_dict("records"), ["domain", "arm", "mean_deg", "mean_deg_std", "median_deg", "acc@10"]))
    print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
