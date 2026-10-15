"""EXP-108 evaluation: per-level PoE of the two branch revisers, K-step top-down revision of the PoE greedy path,
K in {0,1,2,4,8}; 3 seeds; per-domain metrics, change / improve / harm rates, latency."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import featsets, pathflow as PF, phase3 as P, tesseract as T  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

E108, E036 = "EXP-108_path_flow_revision", "EXP-036_transformer_soft_targets"
run = Run("EXP-108", "path_flow_revision", {"stage": "eval", "K": [0, 1, 2, 4, 8]}, seed=0)
SPECS = ["dinov3_vitl16:tokens8", "moge2_vitl:normals16~and"]
KS = [0, 1, 2, 4, 8]
try:
    dev = torch.device("cuda")
    df = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv"); q = df[[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    tr = ((df.domain == "synthetic") & (df.split == "train")).to_numpy()
    groups = {"synthetic_val": ((df.domain == "synthetic") & (df.split == "validation")).to_numpy(),
              "lightbox": (df.domain == "lightbox").to_numpy(), "sunlamp": (df.domain == "sunlamp").to_numpy()}
    X = {}
    for sp in SPECS:
        names, x, _ = featsets.load(sp, "full"); assert names == df.image_relpath.tolist()
        X[sp] = featsets.standardize_to_tensor(x, tr, "cpu", torch.float16); del x
    rows, used = [], []
    for seed in (0, 1, 2):
        trees, revs = [], []
        for sp in SPECS:
            d = [d for d in sorted((REPO / "outputs/experiments" / E108).glob(f"RUN-*-seed{seed}"))
                 if yaml.safe_load((d / "config.yaml").read_text()).get("features") == sp and (d / "checkpoints/best.pt").exists()][-1]
            src = (d / "source_runs.txt").read_text().split()[0]; scfg = yaml.safe_load((REPO / "outputs/experiments" / src / "config.yaml").read_text())
            t = P.HierTransformer(tuple(int(v) for v in scfg["token_shape"].split("x")), scfg["depth"], dropout=scfg["dropout"]).to(dev)
            t.load_state_dict(torch.load(REPO / "outputs/experiments" / src / "checkpoints/best.pt", map_location=dev)["model"]); t.eval()
            r = PF.PathReviser(5).to(dev); r.load_state_dict(torch.load(d / "checkpoints/best.pt", map_location=dev)["reviser"]); r.eval()
            trees.append(t); revs.append(r); used += [src, f"{E108}/{d.name}"]
        for g, msk in groups.items():
            idx = np.flatnonzero(msk); errs = {k: [] for k in KS}; lat = {k: 0.0 for k in KS}
            for s in range(0, len(idx), 1024):
                b = idx[s:s + 1024]; xs = [X[sp][torch.as_tensor(b)].to(dev).float() for sp in SPECS]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    c, p, hs = fused_beam(trees, xs, "poe", 1, return_h=True)
                    for k in KS:
                        torch.cuda.synchronize(); t0 = time.perf_counter()
                        ck, pk, _ = PF.revise(revs, hs, c, p, k, "poe") if k else (c, p, None)
                        torch.cuda.synchronize(); lat[k] += time.perf_counter() - t0
                        errs[k].append(np.degrees(2 * np.arccos(np.clip(np.abs((T.decode(ck.cpu().numpy(), pk.cpu().numpy()) * q[b]).sum(1)), 0, 1))))
            e0 = np.concatenate(errs[0])
            for k in KS:
                e = np.concatenate(errs[k]); ch = np.abs(e - e0) > 1e-6
                rows.append({"seed": seed, "domain": g, "K": k, "mean_deg": e.mean(), "median_deg": np.median(e), "p95_deg": np.percentile(e, 95),
                             "acc@5": float((e <= 5).mean()), "acc@10": float((e <= 10).mean()), "frac_gt90": float((e > 90).mean()),
                             "frac_gt150": float((e > 150).mean()), "frac_changed": float(ch.mean()),
                             "frac_improved": float((e < e0 - 1e-6).mean()), "frac_harmed": float((e > e0 + 1e-6).mean()),
                             "flips_repaired": float(((e0 > 90) & (e <= 90)).mean()), "flips_created": float(((e0 <= 90) & (e > 90)).mean()),
                             "added_ms_per_image": 1000 * lat[k] / len(idx)})
            print(seed, g, " ".join(f"K{r['K']} {r['mean_deg']:.3f}" for r in rows[-len(KS):]), flush=True)
    per = pd.DataFrame(rows); run.write_metrics(rows, "per_seed")
    cols = [c for c in per.columns if c not in ("seed", "domain", "K")]
    agg = per.groupby(["domain", "K"])[cols].mean().join(per.groupby(["domain", "K"])[["mean_deg"]].std().rename(columns={"mean_deg": "mean_deg_std"})).reset_index()
    run.write_metrics(agg.to_dict("records"), "metrics"); (run.sub("tables") / "metrics.md").write_text(md_table(agg.to_dict("records")))
    (run.dir / "source_runs.txt").write_text("\n".join(used) + "\n"); run.done()
    print(md_table(agg.to_dict("records"), ["domain", "K", "mean_deg", "mean_deg_std", "median_deg", "frac_gt90", "frac_changed", "frac_improved", "frac_harmed", "flips_repaired", "flips_created"]))
    print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
