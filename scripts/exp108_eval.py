"""EXP-108 (DEC-015) evaluation: discrete flow matching on the Tesseract path, 3 seeds.
For every image: M = 8 sources x0 ~ PoE tree (temperature 1), CTMC Euler to t = 1 with NFE in {1, 2, 4, 8}; point estimate =
KDE mode (5 deg) of the M decoded leaves (pre-declared system: NFE 4, M 8; M = 1 also reported). Baseline per image = EXP-036
PoE greedy. Change rates are counted from actual decisions (chosen sample's path vs the greedy path); flip transitions are
reported with denominators."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import featsets, pathflow as PF, phase3 as P, so3flow as S, tesseract as T  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

E108, E036 = "EXP-108_path_flow_revision", "EXP-036_transformer_soft_targets"
SPECS = {"dinov3_vitl16:tokens8": "64x1024", "moge2_vitl:normals16~and": "256x4"}
NFES, M = [1, 2, 4, 8], 8
run = Run("EXP-108", "path_flow_revision", {"stage": "eval_dfm", "nfe": NFES, "M": M}, seed=0)
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
    geo = lambda a, b: np.degrees(2 * np.arccos(np.clip(np.abs((a * b).sum(-1)), 0, 1)))
    rows, used = [], []
    for seed in (0, 1, 2):
        d = [d for d in sorted((REPO / "outputs/experiments" / E108).glob(f"RUN-*-seed{seed}"))
             if yaml.safe_load((d / "config.yaml").read_text()).get("method") == "dfm" and (d / "checkpoints/best.pt").exists()][-1]
        ck = torch.load(d / "checkpoints/best.pt", map_location=dev)
        models = []
        for st in ck["models"]:
            m = PF.PathDFM(5).to(dev); m.load_state_dict(st); m.eval(); models.append(m)
        trees = []
        for src, (sp, tsh) in zip((d / "source_runs.txt").read_text().split(), SPECS.items()):
            cfg = yaml.safe_load((REPO / "outputs/experiments" / src / "config.yaml").read_text())
            t = P.HierTransformer(tuple(int(v) for v in tsh.split("x")), 5, dropout=cfg["dropout"]).to(dev)
            t.load_state_dict(torch.load(REPO / "outputs/experiments" / src / "checkpoints/best.pt", map_location=dev)["model"]); t.eval(); trees.append(t)
        used.append(f"{E108}/{d.name}")
        gen = torch.Generator(device=dev).manual_seed(1000 + seed)
        for g, msk in groups.items():
            idx = np.flatnonzero(msk); E0, ER = [], {(n, mm): [] for n in NFES for mm in (1, M)}
            RC = {n: [] for n in NFES}; CC = {n: [] for n in NFES}; AC = {n: [] for n in NFES}; lat = {n: 0.0 for n in NFES}
            for s in range(0, len(idx), 512):
                b = idx[s:s + 512]; B = len(b); xs = [X[sp][torch.as_tensor(b)].to(dev).float() for sp in SPECS]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    cg, pg, hs = fused_beam(trees, xs, "poe", 1, return_h=True)
                    E0.append(geo(T.decode(cg.cpu().numpy(), pg.cpu().numpy()), q[b]))
                    hm = [h.repeat_interleave(M, 0) for h in hs]
                    c0, p0 = PF.sample_tree_paths(trees, hm, 1.0, gen)
                    for n in NFES:
                        torch.cuda.synchronize(); t0 = time.perf_counter()
                        c1, p1 = PF.dfm_sample(models, hm, c0, p0, n, "poe", gen)
                        torch.cuda.synchronize(); lat[n] += time.perf_counter() - t0
                        qs = torch.as_tensor(T.decode(c1.cpu().numpy(), p1.cpu().numpy()), dtype=torch.float32, device=dev).view(B, M, 4)
                        sc = torch.exp(-0.5 * (torch.rad2deg(2 * torch.arccos(torch.einsum("bmd,bnd->bmn", qs, qs).abs().clamp(max=1.0))) / 5.0) ** 2).sum(-1)
                        pick = sc.argmax(1)
                        qk = qs[torch.arange(B), pick].cpu().numpy(); qb = q[b]
                        ER[(n, M)].append(geo(qk, qb)); ER[(n, 1)].append(geo(qs[:, 0].cpu().numpy(), qb))
                        cs = c1.view(B, M)[torch.arange(B), pick]; ps = p1.view(B, M, -1)[torch.arange(B), pick]
                        RC[n].append((cs != cg).cpu().numpy()); CC[n].append((ps != pg).cpu().numpy()); AC[n].append(((cs != cg) | (ps != pg).any(1)).cpu().numpy())
            e0 = np.concatenate(E0); n150 = int((e0 > 150).sum()); n20 = int((e0 < 20).sum())
            for n in NFES:
                for mm in (1, M):
                    e = np.concatenate(ER[(n, mm)])
                    r = {"seed": seed, "domain": g, "nfe": n, "M": mm, "n": len(e), "mean_deg": e.mean(), "median_deg": np.median(e),
                         "p95_deg": np.percentile(e, 95), "acc@5": float((e <= 5).mean()), "acc@10": float((e <= 10).mean()),
                         "frac_gt90": float((e > 90).mean()), "frac_gt150": float((e > 150).mean()),
                         "greedy_mean_deg": e0.mean(), "n_init_gt150": n150, "n_gt150_to_lt20": int(((e0 > 150) & (e < 20)).sum()),
                         "frac_gt150_to_lt20": ((e0 > 150) & (e < 20)).sum() / max(n150, 1), "n_init_lt20": n20,
                         "n_lt20_to_gt150": int(((e0 < 20) & (e > 150)).sum()), "frac_lt20_to_gt150": ((e0 < 20) & (e > 150)).sum() / max(n20, 1),
                         "ms_per_image_all_M": 1000 * lat[n] / len(idx)}
                    if mm == M:
                        rc, cc, ac = np.concatenate(RC[n]), np.concatenate(CC[n]), np.concatenate(AC[n])
                        r.update({"frac_path_changed": float(ac.mean()), "frac_root_changed": float(rc.mean()),
                                  **{f"frac_child{l}_changed": float(cc[:, l].mean()) for l in range(cc.shape[1])}})
                    rows.append(r)
            print(seed, g, f"greedy {e0.mean():.3f} | " + " ".join(f"nfe{n}/M{M} {np.concatenate(ER[(n, M)]).mean():.3f}" for n in NFES), flush=True)
    per = pd.DataFrame(rows); run.write_metrics(rows, "per_seed")
    cols = [c for c in per.columns if c not in ("seed", "domain", "nfe", "M")]
    agg = per.groupby(["domain", "nfe", "M"])[cols].mean().join(per.groupby(["domain", "nfe", "M"])[["mean_deg"]].std().rename(columns={"mean_deg": "mean_deg_std"})).reset_index()
    run.write_metrics(agg.to_dict("records"), "metrics"); (run.sub("tables") / "metrics.md").write_text(md_table(agg.to_dict("records")))
    (run.dir / "source_runs.txt").write_text("\n".join(used) + "\n"); run.done()
    print(md_table(agg.to_dict("records"), ["domain", "nfe", "M", "mean_deg", "mean_deg_std", "greedy_mean_deg", "frac_gt150", "frac_path_changed",
                                            "frac_root_changed", "n_init_gt150", "frac_gt150_to_lt20", "n_init_lt20", "frac_lt20_to_gt150"]))
    print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
