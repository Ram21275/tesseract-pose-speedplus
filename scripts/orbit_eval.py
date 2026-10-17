"""Shared evaluator for EXP-113 / 114 / 115 (locked recipe, DEC-018). For every model family (tree / flat), variant
(PoE = DINOv3 + MoGe-2, DINO = DINOv3 only), seed and domain, the SAME 4 in-plane views and the SAME candidate pool
(top-8 per view, mapped back: 32 candidates) are scored by every inference rule:
  greedy         : view 0 only (tree greedy path / flat argmax anchor)
  orbit_product  : ours, s(c) = sum_k log p_k(map_k(c))  (tree: teacher-forced leaf log-prob; flat: nearest-anchor log-prob)
  orbit_maxconf  : the candidate with the highest confidence in its own view
  orbit_vote1    : rotation-cluster voting over the 4 back-mapped view top-1s (5-deg Gaussian kernel; ties by confidence)
  orbit_vote32   : weighted kernel voting over all 32 candidates (weights = per-view normalized probabilities)
Domains: synthetic val, synthetic val-aug (EXP-109 seed-1000 views), lightbox, sunlamp. Flip transitions vs the same
family+variant greedy, with denominators."""
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
from tfpose import data as D, featsets, flatgrid as FG, inplane as IP, phase3 as P, tesseract as T  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True, help="run directory label, e.g. EXP-114_orbit_scoring_rules")
ap.add_argument("--families", required=True, help="comma list of family:experiment:aug_tags, e.g. tree:EXP-109_real_domain_aug_tokens16_orbit:exp109_train_aug1+exp109_train_aug2")
ap.add_argument("--modes", default="greedy,orbit_product,orbit_maxconf,orbit_vote1,orbit_vote32")
ap.add_argument("--variants", default="PoE,DINO")
ap.add_argument("--seeds", default="0,1,2")
args = ap.parse_args()
exp_id, short = args.exp.split("_", 1)
run = Run(exp_id, short, vars(args), seed=0)
SD, SM = "dinov3_vitl16:tokens8", "moge2_vitl:normals16~and"
SHAPE = {SD: (64, 1024), SM: (256, 4)}
MODES = args.modes.split(","); VIEWS = (0, 90, 180, 270)
try:
    dev = torch.device("cuda")
    conv = (REPO / "outputs/experiments/exp109_convention.txt").read_text().strip()
    full = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv"); tr = ((full.domain == "synthetic") & (full.split == "train")).to_numpy()
    t3 = pd.read_csv(REPO / "outputs/data_manifests/exp109_test3.csv"); va = pd.read_csv(REPO / "outputs/data_manifests/exp109_syn_val.csv")
    pos3 = pd.Series(np.arange(len(full)), index=full.image_relpath)[t3.image_relpath].to_numpy()
    cam = D.load_camera(Path(yaml.safe_load(open(REPO / "configs/paths.yaml"))["speedplus_root"]))
    q3 = t3[[f"q_can_{k}" for k in "wxyz"]].to_numpy(); d3 = IP.ray_undistorted(cam.K, cam.dist, t3.crop_cx.to_numpy(), t3.crop_cy.to_numpy())
    dom3 = np.where(t3.domain == "synthetic", "synthetic_val", t3.domain)
    qv = va[[f"q_can_{k}" for k in "wxyz"]].to_numpy(); dv = IP.ray_undistorted(cam.K, cam.dist, va.crop_cx.to_numpy(), va.crop_cy.to_numpy())
    ang_va = pd.read_csv(featsets.find_cache("dinov3_vitl16", "exp109_valaug") / "aug.csv").rot_deg.to_numpy()
    qva = IP.rotate_label(qv, ang_va, dv, conv)
    Fz = {}
    for sp in (SD, SM):
        _, xf, _ = featsets.load(sp, "full"); st = featsets.train_stats(xf, tr)
        Fz[(sp, "test", 0)] = featsets.standardize_to_tensor(xf[pos3], None, "cpu", torch.float16, stats=st); del xf
        for k in (90, 180, 270):
            n, x, _ = featsets.load(sp, f"exp109_test3_r{k}"); assert n == t3.image_relpath.tolist()
            Fz[(sp, "test", k)] = featsets.standardize_to_tensor(x, None, "cpu", torch.float16, stats=st)
        for k in VIEWS:
            n, x, _ = featsets.load(sp, "exp109_valaug" if k == 0 else f"exp109_valaug_r{k}"); assert n == va.image_relpath.tolist()
            Fz[(sp, "valaug", k)] = featsets.standardize_to_tensor(x, None, "cpu", torch.float16, stats=st)
    G = torch.as_tensor(FG.matched_hopf_grid(), dtype=torch.float32, device=dev)

    def find_runs(exp, seed, family, tags, cfgfilter=None):
        out = {}
        for d in sorted((REPO / "outputs/experiments" / exp).glob(f"RUN-*-seed{seed}")):
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            if not (d / "checkpoints/best.pt").exists() or cfg.get("features") not in SHAPE:
                continue
            if (cfg.get("aug_tags") or "") != tags or (family == "flat") != (cfg.get("model") == "flat_hopf"):
                continue
            if cfgfilter and any(str(cfg.get(k)) != v for k, v in cfgfilter.items()):
                continue
            out[cfg["features"]] = d
        return out

    def load(family, d, sp):
        st = torch.load(d / "checkpoints/best.pt", map_location=dev)["model"]
        m = FG.FlatTransformer(SHAPE[sp], len(G)).to(dev) if family == "flat" else P.HierTransformer(SHAPE[sp], 5).to(dev)
        m.load_state_dict(st); m.eval(); return m

    geo = lambda a, b: np.degrees(2 * np.arccos(np.clip(np.abs((a * b).sum(-1)), 0, 1)))

    def tree_lp(ms, xs, cand):
        B, C, _ = cand.shape
        c, p = T.encode(cand.reshape(-1, 4), 5); c, p = torch.as_tensor(c, device=dev), torch.as_tensor(p, device=dev)
        outs = [m.logits_tf(m.encode(x).repeat_interleave(C, 0), c, p) for m, x in zip(ms, xs)]
        root = F.log_softmax(sum(F.log_softmax(o[0].float(), -1) for o in outs), -1)
        child = F.log_softmax(sum(F.log_softmax(o[1].float(), -1) for o in outs), -1)
        return (root.gather(1, c[:, None])[:, 0] + child.gather(2, p[:, :, None])[..., 0].sum(1)).view(B, C).cpu().numpy()

    @torch.no_grad()
    def evaluate(family, ms, keys, setname, n_rows, base_ang, dray):
        res = {m: [] for m in MODES}
        for s in range(0, n_rows, 256):
            r = np.arange(s, min(s + 256, n_rows)); B = len(r)
            cands, confs, views, flat_lp = [], [], [], []
            for k in VIEWS:
                xs = [Fz[(kk, setname, k)][torch.as_tensor(r)].to(dev).float() for kk in keys]; views.append((k, xs))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if family == "tree":
                        c8, p8, sc, _ = fused_beam(ms, xs, "poe", 8, return_all=True)
                        qk = T.decode(c8.cpu().numpy().reshape(-1), p8.cpu().numpy().reshape(-1, 5)).reshape(B, 8, 4); cf = sc.float().cpu().numpy()
                    else:
                        lp = F.log_softmax(sum(F.log_softmax(m.logits(m.encode(x)).float(), -1) for m, x in zip(ms, xs)), -1)
                        top = lp.topk(8, -1); qk = G[top.indices].cpu().numpy(); cf = top.values.cpu().numpy(); flat_lp.append(lp)
                back = IP.rotate_label(qk.reshape(-1, 4), -np.full(B * 8, float(k)), np.repeat(dray[r], 8, 0), conv).reshape(B, 8, 4) if k else qk
                cands.append(back); confs.append(cf)
            cand = np.concatenate(cands, 1); conf = np.concatenate(confs, 1)                     # (B,32,4), (B,32)
            if "greedy" in MODES:
                if family == "tree":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        c1, p1 = fused_beam(ms, views[0][1], "poe", 1)
                    res["greedy"].append(T.decode(c1.cpu().numpy(), p1.cpu().numpy()))
                else:
                    res["greedy"].append(cands[0][:, 0])
            if "orbit_product" in MODES:
                score = np.zeros(cand.shape[:2])
                for vi, (k, xs) in enumerate(views):
                    mk = IP.rotate_label(cand.reshape(-1, 4), np.full(B * 32, float(k)), np.repeat(dray[r], 32, 0), conv) if k else cand.reshape(-1, 4)
                    if family == "tree":
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            score += tree_lp(ms, xs, mk.reshape(B, 32, 4))
                    else:
                        ai = FG.nearest_anchor(torch.as_tensor(mk, dtype=torch.float32, device=dev), G).view(B, 32)
                        score += flat_lp[vi].gather(1, ai).cpu().numpy()
                res["orbit_product"].append(cand[np.arange(B), score.argmax(1)])
            if "orbit_maxconf" in MODES:
                res["orbit_maxconf"].append(cand[np.arange(B), conf.argmax(1)])
            dd = None
            if "orbit_vote1" in MODES:
                tops = cand[:, ::8]; tconf = conf[:, ::8]                                          # (B,4,4) view top-1s
                d4 = geo(tops[:, :, None], tops[:, None]); sup = np.exp(-0.5 * (d4 / 5.0) ** 2).sum(-1) + 1e-6 * (tconf - tconf.min(1, keepdims=True))
                res["orbit_vote1"].append(tops[np.arange(B), sup.argmax(1)])
            if "orbit_vote32" in MODES:
                w = np.exp(conf.reshape(B, 4, 8) - conf.reshape(B, 4, 8).max(-1, keepdims=True)); w = (w / w.sum(-1, keepdims=True)).reshape(B, 32)
                dd = geo(cand[:, :, None], cand[:, None]); sup = (np.exp(-0.5 * (dd / 5.0) ** 2) * w[:, None, :]).sum(-1)
                res["orbit_vote32"].append(cand[np.arange(B), sup.argmax(1)])
            v0 = cand[:, 0]; vk = cand[:, 8::8]                                             # view top-1s, back-mapped
            res.setdefault("_consistency", []).append((geo(vk, v0[:, None]) <= 10).mean(1))
        return {m: np.concatenate(v) for m, v in res.items()}

    rows = []
    for spec in args.families.split(","):
        parts = spec.split(":"); family, exp, tags = parts[:3]; tags = tags.replace("+", ",")
        cfgfilter = dict(kv.split("=") for kv in parts[3].split(";")) if len(parts) > 3 and parts[3] else None    # e.g. mode=T
        for seed in [int(x) for x in args.seeds.split(",")]:
            runs = find_runs(exp, seed, family, tags, cfgfilter)
            for variant in args.variants.split(","):
                keys = (SD, SM) if variant == "PoE" else (SD,)
                if any(k not in runs for k in keys):
                    print("missing runs", family, exp, tags, seed, variant, "- skipped", flush=True); continue
                ms = [load(family, runs[k], k) for k in keys]
                P3 = evaluate(family, ms, keys, "test", len(t3), np.zeros(len(t3)), d3)
                PV = evaluate(family, ms, keys, "valaug", len(va), ang_va, dv)
                cons3, consv = P3.pop("_consistency"), PV.pop("_consistency")
                for mode in MODES:
                    ERR = {g: geo(P3[mode][dom3 == g], q3[dom3 == g]) for g in ("synthetic_val", "lightbox", "sunlamp")}
                    ERR["synthetic_val_aug"] = geo(PV[mode], qva)
                    BASE = {g: (geo(P3["greedy"][dom3 == g], q3[dom3 == g]) if g != "synthetic_val_aug" else geo(PV["greedy"], qva)) for g in ERR} if "greedy" in MODES else ERR
                    for g, e in ERR.items():
                        e0 = BASE[g]; n150, n20 = int((e0 > 150).sum()), int((e0 < 20).sum())
                        rows.append({"family": family + (":" + parts[3] if len(parts) > 3 and parts[3] else ""), "train_aug": tags, "variant": variant, "mode": mode, "seed": seed, "domain": g,
                                     "mean_deg": e.mean(), "median_deg": np.median(e), "acc@10": float((e <= 10).mean()),
                                     "frac_gt90": float((e > 90).mean()), "frac_gt150": float((e > 150).mean()),
                                     "n_greedy_gt150": n150, "frac_gt150_to_lt20": ((e0 > 150) & (e < 20)).sum() / max(n150, 1),
                                     "n_greedy_lt20": n20, "frac_lt20_to_gt150": ((e0 < 20) & (e > 150)).sum() / max(n20, 1),
                                     "view_consistency@10": float((consv if g == "synthetic_val_aug" else cons3[dom3 == g]).mean())})
                print(family, tags[:30], seed, variant, " | ".join(f"{m}: " + "/".join(f"{np.mean(geo(P3[m][dom3 == g], q3[dom3 == g])):.2f}" for g in ("synthetic_val", "lightbox", "sunlamp")) for m in MODES), flush=True)
                del ms; torch.cuda.empty_cache()
    per = pd.DataFrame(rows); run.write_metrics(rows, "per_seed")
    keys_ = ["domain", "family", "train_aug", "variant", "mode"]
    cols = [c for c in per.columns if c not in keys_ + ["seed"]]
    agg = per.groupby(keys_)[cols].mean().join(per.groupby(keys_)[["mean_deg"]].std().rename(columns={"mean_deg": "mean_deg_std"})).reset_index()
    run.write_metrics(agg.to_dict("records"), "metrics"); (run.sub("tables") / "metrics.md").write_text(md_table(agg.to_dict("records")))
    run.done(); print(md_table(agg.to_dict("records"), keys_ + ["mean_deg", "mean_deg_std", "median_deg", "frac_gt150", "frac_gt150_to_lt20", "frac_lt20_to_gt150"])); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
