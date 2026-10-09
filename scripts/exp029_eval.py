"""EXP-029 evaluation: PanSt3R-inspired feature fusion vs training-free per-level PoE (greedy decoding).

Arms (3 seeds each, same recipe as EXP-036: Transformer decoder, depth 5, soft targets):
  1 A      DINOv3 tokens8 (64x1024)                         -- the EXP-036 DINO runs (identical config, reused)
  2 G8     MoGe-2 normals8~and (64x4)                       -- EXP-029
  3 PoE8   per-level PoE of 1 and 2 (DEC-003 rule)
  4 Flin   [S_i || G_i] (64x1028) -> LN -> Linear -> one Transformer   -- EXP-029, --fuse linear
  5 Fmlp   [S_i || G_i] -> LN -> Linear-GELU-Linear -> one Transformer -- EXP-029, --fuse mlp
  ref PoE16 historical EXP-036 PoE (DINO tokens8 + MoGe normals16~and 256x4)
Concatenated inputs are rebuilt from the separately standardised sources (per-dimension train-split
standardisation is separable, so this equals standardising the concatenation; geometry is rounded to fp16 first,
as in featsets.load_concat). Lightbox and sunlamp are descriptive (DEC-006).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import featsets, phase3 as P, tesseract as T  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.predictor import decode_torch  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default="EXP-029_panst3r_inspired_fusion")
ap.add_argument("--ref-exp", default="EXP-036_transformer_soft_targets")
ap.add_argument("--seeds", default="0,1,2")
args = ap.parse_args()
run = Run("EXP-029", "panst3r_inspired_fusion", dict(vars(args), stage="eval"), seed=0)
DINO, G8, G16 = "dinov3_vitl16:tokens8", "moge2_vitl:normals8~and", "moge2_vitl:normals16~and"
CAT = f"{DINO}|{G8}"
try:
    dev = torch.device("cuda")
    df = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv")
    q = df[[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    tr = ((df.domain == "synthetic") & (df.split == "train")).to_numpy()
    groups = {"synthetic_val": ((df.domain == "synthetic") & (df.split == "validation")).to_numpy(),
              "lightbox": (df.domain == "lightbox").to_numpy(), "sunlamp": (df.domain == "sunlamp").to_numpy()}
    X = {}
    for spec in (DINO, G8, G16):
        names, x, _ = featsets.load(spec, "full"); assert names == df.image_relpath.tolist()
        if spec == G8:
            x = x.astype(np.float16).astype(np.float32)
        X[spec] = featsets.standardize_to_tensor(x, tr, dev, torch.float16); del x

    def batch(spec, b):
        if spec == CAT:
            B = len(b)
            return torch.cat([X[DINO][b].view(B, 64, 1024), X[G8][b].view(B, 64, 4)], -1).reshape(B, -1).float()
        return X[spec][b].float()

    def load(exp, spec, seed, fuse="linear"):
        for d in sorted((REPO / "outputs/experiments" / exp).glob(f"RUN-*-seed{seed}")):
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            if cfg.get("features") != spec or cfg.get("fuse", "linear") != fuse or cfg.get("targets") != "soft" \
                    or cfg.get("decoder") != "transformer" or not (d / "checkpoints/best.pt").exists():
                continue
            m = P.HierTransformer(tuple(int(v) for v in cfg["token_shape"].split("x")), cfg["depth"], dropout=cfg["dropout"],
                                  fuse=cfg.get("fuse", "linear")).to(dev)
            m.load_state_dict(torch.load(d / "checkpoints/best.pt", map_location=dev)["model"]); m.eval()
            return m, f"{exp}/{d.name}"
        raise FileNotFoundError(exp, spec, seed, fuse)

    def p95(e):
        return float(np.percentile(e, 95))

    rows, errs, used = [], {}, []
    for seed in [int(s) for s in args.seeds.split(",")]:
        mA, nA = load(args.ref_exp, DINO, seed); m16, n16 = load(args.ref_exp, G16, seed)
        mG, nG = load(args.exp, G8, seed); mL, nL = load(args.exp, CAT, seed, "linear"); mM, nM = load(args.exp, CAT, seed, "mlp")
        used += [nA, n16, nG, nL, nM]
        arms = [("1 A (DINOv3)", [mA], [DINO], "single"), ("2 G8 (geometry 8x8)", [mG], [G8], "single"),
                ("3 PoE8 (A,G8)", [mA, mG], [DINO, G8], "poe"), ("4 Flin (concat+linear)", [mL], [CAT], "single"),
                ("5 Fmlp (concat+MLP)", [mM], [CAT], "single"), ("ref PoE16 (EXP-036)", [mA, m16], [DINO, G16], "poe")]
        for name, ms, specs, rule in arms:
            n_params = sum(p.numel() for m in ms for p in m.parameters())
            for g, msk in groups.items():
                idx = np.flatnonzero(msk); qs = []
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated(); t0 = time.time()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for s in range(0, len(idx), 1024):
                        b = torch.as_tensor(idx[s:s + 1024], device=dev)
                        c, p = fused_beam(ms, [batch(sp, b) for sp in specs], rule, 1)
                        qs.append(decode_torch(c, p).float().cpu().numpy())
                torch.cuda.synchronize(); dt = (time.time() - t0) / len(idx)
                peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
                qp = np.concatenate(qs); e = np.degrees(2 * np.arccos(np.clip(np.abs((qp * q[idx]).sum(1)), 0, 1)))
                errs[(name, seed, g)] = e
                rows.append({"arm": name, "seed": seed, "domain": g, "mean_deg": e.mean(), "median_deg": np.median(e), "p95_deg": p95(e),
                             **{f"acc@{t}": float((e <= t).mean()) for t in (5, 10, 20)},
                             "frac_gt90": float((e > 90).mean()), "frac_gt150": float((e > 150).mean()),
                             "params": n_params, "latency_ms_per_image_b1024": 1000 * dt, "peak_infer_mem_mb_b1024": peak})
                print(name, seed, g, f"{e.mean():.3f}", flush=True)
    per = pd.DataFrame(rows); run.write_metrics(rows, "per_seed")
    num = [c for c in per.columns if c not in ("arm", "seed", "domain")]
    agg = per.groupby(["arm", "domain"])[num].mean().join(per.groupby(["arm", "domain"])[["mean_deg"]].std().rename(columns={"mean_deg": "mean_deg_std"})).reset_index()
    run.write_metrics(agg.to_dict("records"), "metrics"); (run.sub("tables") / "metrics.md").write_text(md_table(agg.to_dict("records")))
    # paired per-image comparisons vs PoE8 (and vs the historical PoE16), per seed and on seed-averaged errors
    prs = []
    seeds = sorted({k[1] for k in errs})
    for arm in ["4 Flin (concat+linear)", "5 Fmlp (concat+MLP)", "ref PoE16 (EXP-036)", "1 A (DINOv3)"]:
        for ctl in ["3 PoE8 (A,G8)", "ref PoE16 (EXP-036)"]:
            if arm == ctl:
                continue
            for g in groups:
                for sd in seeds + ["seed-mean"]:
                    if sd == "seed-mean":
                        a = np.mean([errs[(arm, s, g)] for s in seeds], 0); c = np.mean([errs[(ctl, s, g)] for s in seeds], 0)
                    else:
                        a, c = errs[(arm, sd, g)], errs[(ctl, sd, g)]
                    d = a - c
                    prs.append({"arm": arm, "control": ctl, "domain": g, "seed": sd, "mean_diff_deg": d.mean(), "median_diff_deg": np.median(d),
                                "frac_better_by_1deg": float((d < -1).mean()), "frac_worse_by_1deg": float((d > 1).mean()),
                                "frac_fixed_gt90": float(((c > 90) & (a <= 90)).mean()), "frac_broken_gt90": float(((c <= 90) & (a > 90)).mean())})
    run.write_metrics(prs, "paired"); (run.sub("tables") / "paired.md").write_text(md_table(prs))
    np.savez_compressed(run.sub("predictions") / "per_image_errors.npz",
                        **{f"{a.split()[0]}_{a.split()[1]}_s{s}_{g}": e for (a, s, g), e in errs.items()},
                        **{f"index_{g}": np.flatnonzero(m) for g, m in groups.items()})
    (run.dir / "source_runs.txt").write_text("\n".join(used) + "\n"); run.done()
    print(md_table(agg.to_dict("records"), ["arm", "domain", "mean_deg", "mean_deg_std", "median_deg", "p95_deg", "acc@10", "frac_gt90", "params"]))
    print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
