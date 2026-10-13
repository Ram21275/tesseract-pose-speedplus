"""EXP-044 evaluation: distilled trees (GT loss + beta * flow-mass distillation), PoE of the two branches, greedy.

--stage select : seed 0, every beta present in EXP-044 (beta = 0 rerun included); prints BEST_BETA by synthetic-val
                 PoE greedy mean (DEC-006: real domains descriptive only).
--stage final  : beta = 0 (EXP-036, 3 seeds) vs the selected beta (EXP-044, 3 seeds), plus the beta = 0 rerun (seed 0).
Per arm, seed and group: PoE greedy errors, top-8 beam leaf recall, latency. On synthetic val also per level: PoE tree
mass on the GT cell, greedy top-1, and the distillation fit = -(1/(L+1)) sum_l log p_l(prefix_l(R_m)) averaged over
the 32 EXP-042 flow samples of that val image (EXP-043 sample stage; evaluation only, never used for training).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import featsets, phase3 as P, tesseract as T  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.predictor import decode_torch  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True, choices=["select", "final"])
ap.add_argument("--beta", type=float, default=None, help="selected beta (--stage final)")
ap.add_argument("--exp", default="EXP-044_distilled_tree")
ap.add_argument("--ref-exp", default="EXP-036_transformer_soft_targets")
ap.add_argument("--val-samples", default="outputs/experiments/EXP-043_flow_induced_tesseract_masses/RUN-20261013-004254-seed0")
args = ap.parse_args()
run = Run("EXP-044", "distilled_tree", dict(vars(args), stage=f"eval_{args.stage}"), seed=0)
L = 5
SPECS = ["dinov3_vitl16:tokens8", "moge2_vitl:normals16~and"]
try:
    dev = torch.device("cuda")
    df = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv")
    q = df[[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    chart, path = T.encode(q, L)
    tr = ((df.domain == "synthetic") & (df.split == "train")).to_numpy()
    groups = {"synthetic_val": ((df.domain == "synthetic") & (df.split == "validation")).to_numpy(),
              "lightbox": (df.domain == "lightbox").to_numpy(), "sunlamp": (df.domain == "sunlamp").to_numpy()}
    X = {}
    for sp in SPECS:
        names, x, _ = featsets.load(sp, "full"); assert names == df.image_relpath.tolist()
        X[sp] = featsets.standardize_to_tensor(x, tr, "cpu", torch.float16); del x

    def find(exp, spec, seed, beta):
        """Run dir of the tree for (spec, seed); beta=None = EXP-036 (no distillation key)."""
        for d in sorted((REPO / "outputs/experiments" / exp).glob(f"RUN-*-seed{seed}")):
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            if cfg.get("features") != spec or cfg.get("targets") != "soft" or cfg.get("decoder") != "transformer" \
                    or not (d / "checkpoints/best.pt").exists():
                continue
            if beta is not None and abs(float(cfg.get("kd_beta", -1)) - beta) > 1e-9:
                continue
            return d, cfg
        return None, None

    def load(d, cfg):
        m = P.HierTransformer(tuple(int(v) for v in cfg["token_shape"].split("x")), cfg["depth"], dropout=cfg["dropout"]).to(dev)
        m.load_state_dict(torch.load(d / "checkpoints/best.pt", map_location=dev)["model"]); m.eval()
        return m

    def betas_seed0():
        out = set()
        for d in (REPO / "outputs/experiments" / args.exp).glob("RUN-*-seed0"):
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            if cfg.get("features") in SPECS and "kd_beta" in cfg and (d / "checkpoints/best.pt").exists():
                out.add(float(cfg["kd_beta"]))
        return sorted(b for b in out if all(find(args.exp, s, 0, b)[0] for s in SPECS))

    def xb(sp, b):
        return X[sp][b.cpu()].to(dev).float()

    def step_lp(ms, xs, c, p):
        """PoE per-step log-probs along given paths (B, L+1); xs already repeated to len(c)."""
        outs = [m.logits_tf(m.encode(x), c, p) for m, x in zip(ms, xs)]
        root = F.log_softmax(sum(F.log_softmax(o[0].float(), -1) for o in outs), -1)
        child = F.log_softmax(sum(F.log_softmax(o[1].float(), -1) for o in outs), -1)
        return torch.cat([root.gather(1, c[:, None]), child.gather(2, p[:, :, None])[..., 0]], 1)

    def evaluate(arm, seed, ms):
        rows, lev = [], []
        for g, msk in groups.items():
            idx = np.flatnonzero(msk); qs, cs, ps, hit = [], [], [], []
            torch.cuda.synchronize(); t0 = time.time()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for s in range(0, len(idx), 1024):
                    b = torch.as_tensor(idx[s:s + 1024], device=dev); xs = [xb(sp, b) for sp in SPECS]
                    c, p = fused_beam(ms, xs, "poe", 1)
                    qs.append(decode_torch(c, p).float().cpu().numpy()); cs.append(c.cpu().numpy()); ps.append(p.cpu().numpy())
            torch.cuda.synchronize(); dt = (time.time() - t0) / len(idx)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for s in range(0, len(idx), 512):
                    b = torch.as_tensor(idx[s:s + 512], device=dev)
                    c8, p8, _, _ = fused_beam(ms, [xb(sp, b) for sp in SPECS], "poe", 8, return_all=True)
                    gl = T.leaf_id(chart[idx[s:s + 512]], path[idx[s:s + 512]])
                    cand = np.stack([T.leaf_id(c8[:, k].cpu().numpy(), p8[:, k].cpu().numpy()) for k in range(c8.shape[1])], 1)
                    hit.append((cand == gl[:, None]).any(1))
            qp = np.concatenate(qs); e = np.degrees(2 * np.arccos(np.clip(np.abs((qp * q[idx]).sum(1)), 0, 1)))
            rows.append({"arm": arm, "seed": seed, "domain": g, "mean_deg": e.mean(), "median_deg": np.median(e),
                         "p95_deg": float(np.percentile(e, 95)), **{f"acc@{t}": float((e <= t).mean()) for t in (5, 10, 20)},
                         "frac_gt90": float((e > 90).mean()), "frac_gt150": float((e > 150).mean()),
                         "top8_leaf_recall": float(np.concatenate(hit).mean()), "latency_ms_per_image_b1024": 1000 * dt})
            print(arm, seed, g, f"{e.mean():.3f}", flush=True)
            if g != "synthetic_val":
                continue
            # ---- per level on synthetic val: GT mass, greedy top-1, distillation fit vs the val flow samples ----
            gc, gp = chart[idx], path[idx]; gr_c, gr_p = np.concatenate(cs), np.concatenate(ps)
            lp_gt = np.zeros((len(idx), L + 1)); fit = np.zeros((len(idx), L + 1))
            sdir = REPO / args.val_samples / "samples"
            vidx = np.load(sdir / "synthetic_val_index.npy"); assert np.array_equal(vidx, idx)
            smp = np.load(sdir / f"synthetic_val_fused_seed{seed}_fp16.npy").astype(np.float64)
            N, M, _ = smp.shape; smp /= np.linalg.norm(smp, axis=-1, keepdims=True)
            sc, spp = T.encode(smp.reshape(-1, 4), L); sc, spp = sc.reshape(N, M), spp.reshape(N, M, L); del smp
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for s in range(0, N, 128):
                    b = torch.as_tensor(idx[s:s + 128], device=dev); B = len(b); xs = [xb(sp, b) for sp in SPECS]
                    lp_gt[s:s + B] = step_lp(ms, xs, torch.as_tensor(gc[s:s + B], device=dev),
                                             torch.as_tensor(gp[s:s + B], device=dev)).cumsum(1).cpu().numpy()
                    st = step_lp(ms, [x.repeat_interleave(M, 0) for x in xs], torch.as_tensor(sc[s:s + B].reshape(-1), device=dev),
                                 torch.as_tensor(spp[s:s + B].reshape(B * M, L), device=dev))
                    fit[s:s + B] = -st.cumsum(1).view(B, M, L + 1).mean(1).cpu().numpy()
            for l in range(L + 1):
                pid = (lambda c_, p_: c_.astype(np.int64) if l == 0 else T.leaf_id(c_, p_[:, :l]))
                lev.append({"arm": arm, "seed": seed, "level": l, "tree_mass_on_gt": float(np.exp(lp_gt[:, l]).mean()),
                            "greedy_top1": float((pid(gr_c, gr_p) == pid(gc, gp)).mean()), "distill_fit_ce": float(fit[:, l].mean())})
            lev.append({"arm": arm, "seed": seed, "level": "mean", "distill_fit_ce": float(fit.mean())})
        return rows, lev

    def ms_for(exp, seed, beta):
        found = [find(exp, s, seed, beta) for s in SPECS]
        if any(d is None for d, _ in found):
            raise FileNotFoundError(exp, seed, beta)
        return [load(d, c) for d, c in found], [f"{exp}/{d.name}" for d, _ in found]

    arms = []
    if args.stage == "select":
        for b in betas_seed0():
            arms.append((f"beta={b:g} (EXP-044)", args.exp, 0, b))
        arms.append(("beta=0 (EXP-036)", args.ref_exp, 0, None))
    else:
        for sd in (0, 1, 2):
            arms += [("beta=0 (EXP-036)", args.ref_exp, sd, None), (f"beta={args.beta:g} (EXP-044)", args.exp, sd, args.beta)]
        arms.append(("beta=0 rerun (EXP-044)", args.exp, 0, 0.0))
    rows, lev, used = [], [], []
    for name, exp, sd, b in arms:
        ms, srcs = ms_for(exp, sd, b); used += srcs
        r, lv = evaluate(name, sd, ms); rows += r; lev += lv
        del ms; torch.cuda.empty_cache()
    per = pd.DataFrame(rows); run.write_metrics(rows, "per_seed"); run.write_metrics(lev, "per_level_per_seed")
    num = [c for c in per.columns if c not in ("arm", "seed", "domain")]
    agg = per.groupby(["arm", "domain"])[num].mean().join(
        per.groupby(["arm", "domain"])[["mean_deg"]].std().rename(columns={"mean_deg": "mean_deg_std"})).reset_index()
    run.write_metrics(agg.to_dict("records"), "metrics"); (run.sub("tables") / "metrics.md").write_text(md_table(agg.to_dict("records")))
    lv = pd.DataFrame(lev); lv["level"] = lv["level"].astype(str)
    lagg = lv.groupby(["arm", "level"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    run.write_metrics(lagg.to_dict("records"), "per_level"); (run.sub("tables") / "per_level.md").write_text(md_table(lagg.to_dict("records")))
    (run.dir / "source_runs.txt").write_text("\n".join(used) + "\n")
    print(md_table(agg.to_dict("records"), ["arm", "domain", "mean_deg", "mean_deg_std", "median_deg", "acc@10", "frac_gt90", "top8_leaf_recall"]))
    print(md_table(lagg.to_dict("records")))
    if args.stage == "select":
        sv = per[(per.domain == "synthetic_val") & per.arm.str.contains("EXP-044")]
        best = sv.loc[sv.mean_deg.idxmin()]
        run.done(best_beta=float(best.arm.split("=")[1].split()[0]))
        print("BEST_BETA", best.arm.split("=")[1].split()[0])
    else:
        run.done()
    print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
