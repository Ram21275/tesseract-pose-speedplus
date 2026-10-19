"""EXP-118 evaluation (pre-registered 2026-10-19): 6-DoF on SPEED+ (synthetic val, lightbox, sunlamp; EXP-109 test3 rows and
their 90/180/270 rolled views). Arms: J (joint tree + translation block), F (translation block on the frozen EXP-109 arm-A
tree), B (box-only: U_x = U_y = 0, U_z = training mean). Variants PoE (rotation per-level PoE; translation = mean of the
branches' u) and DINO-only. Rotation: greedy / orbit_product / orbit_vote32 (as orbit_eval.py). Translation: greedy (view 0)
and orbit (each view's t back-rolled to view 0, averaged). Metrics: E_T (m), E_R (deg), e_t = E_T/|t|, ESA score."""
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
from tfpose import data as D, featsets, inplane as IP, phase3 as P, tesseract as T, trans as TR  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="0,1,2")
args = ap.parse_args()
run = Run("EXP-118", "translation_6dof", {**vars(args), "stage": "eval"}, seed=0)
SD, SM = "dinov3_vitl16:tokens8", "moge2_vitl:normals16~and"; SHAPE = {SD: (64, 1024), SM: (256, 4)}
VIEWS = (0, 90, 180, 270); MODES = ("greedy", "orbit_product", "orbit_vote32")
geo = lambda a, b: np.degrees(2 * np.arccos(np.clip(np.abs((a * b).sum(-1)), 0, 1)))
try:
    dev = torch.device("cuda")
    conv = (REPO / "outputs/experiments/exp109_convention.txt").read_text().strip()
    full = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv"); tr = ((full.domain == "synthetic") & (full.split == "train")).to_numpy()
    t3 = pd.read_csv(REPO / "outputs/data_manifests/exp109_test3.csv")
    pos3 = pd.Series(np.arange(len(full)), index=full.image_relpath)[t3.image_relpath].to_numpy()
    cam = D.load_camera(Path(yaml.safe_load(open(REPO / "configs/paths.yaml"))["speedplus_root"]))
    q3 = t3[[f"q_can_{k}" for k in "wxyz"]].to_numpy(); tv3 = t3[["t_x", "t_y", "t_z"]].to_numpy()
    BOX = (t3.crop_cx.to_numpy(), t3.crop_cy.to_numpy(), t3.crop_side.to_numpy(), t3.crop_side.to_numpy())
    d3 = IP.ray_undistorted(cam.K, cam.dist, t3.crop_cx.to_numpy(), t3.crop_cy.to_numpy())
    dom3 = np.where(t3.domain == "synthetic", "synthetic_val", t3.domain); DOMS = ("synthetic_val", "lightbox", "sunlamp")
    ftr = full[tr]; uz_mean = TR.to_u(ftr[["t_x", "t_y", "t_z"]].to_numpy(), cam.K, ftr.crop_cx.to_numpy(), ftr.crop_cy.to_numpy(),
                                      ftr.crop_side.to_numpy(), ftr.crop_side.to_numpy())[:, 2].mean()
    Fz = {}
    for sp in (SD, SM):
        _, xf, _ = featsets.load(sp, "full"); st = featsets.train_stats(xf, tr)
        Fz[(sp, 0)] = featsets.standardize_to_tensor(xf[pos3], None, "cpu", torch.float16, stats=st); del xf
        for k in VIEWS[1:]:
            n, x, _ = featsets.load(sp, f"exp109_test3_r{k}"); assert n == t3.image_relpath.tolist()
            Fz[(sp, k)] = featsets.standardize_to_tensor(x, None, "cpu", torch.float16, stats=st)
    back = {k: IP.rotate_label(np.tile([1.0, 0, 0, 0], (len(t3), 1)), -np.full(len(t3), float(k)), d3, conv) for k in VIEWS[1:]}   # view k -> view 0

    def find(arm, seed):
        out = {}
        for d in sorted((REPO / "outputs/experiments/EXP-118_translation_6dof").glob(f"RUN-*-seed{seed}")):
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            if cfg.get("trans") == arm and (d / "checkpoints/best.pt").exists():
                out[cfg["features"]] = d
        return out

    def load(d, sp):
        ck = torch.load(d / "checkpoints/best.pt", map_location=dev)
        m = P.HierTransformer(SHAPE[sp], 5).to(dev); m.load_state_dict(ck["model"]); m.eval()
        h = TR.TransHead(m.root_tok.shape[-1]).to(dev); h.load_state_dict(ck["trans"]); h.eval(); return m, h

    def tree_lp(ms, xs, cand):
        B, C, _ = cand.shape
        c, p = T.encode(cand.reshape(-1, 4), 5); c, p = torch.as_tensor(c, device=dev), torch.as_tensor(p, device=dev)
        outs = [m.logits_tf(m.encode(x).repeat_interleave(C, 0), c, p) for m, x in zip(ms, xs)]
        root = F.log_softmax(sum(F.log_softmax(o[0].float(), -1) for o in outs), -1)
        child = F.log_softmax(sum(F.log_softmax(o[1].float(), -1) for o in outs), -1)
        return (root.gather(1, c[:, None])[:, 0] + child.gather(2, p[:, :, None])[..., 0].sum(1)).view(B, C).float().cpu().numpy()

    @torch.no_grad()
    def evaluate(ms, hs, keys, rot=True):
        R = {m: [] for m in MODES}; Tg, To = [], []
        for s in range(0, len(t3), 256):
            r = np.arange(s, min(s + 256, len(t3))); B = len(r); cands, confs, views, tks = [], [], [], []
            for k in VIEWS:
                xs = [Fz[(kk, k)][torch.as_tensor(r)].to(dev).float() for kk in keys]; views.append((k, xs))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    u = sum(h(m.encode(x)).float() for m, h, x in zip(ms, hs, xs)).cpu().numpy() / len(keys)
                tk = TR.from_u(u, cam.K, *(b[r] for b in BOX)); tks.append(tk if k == 0 else TR.quat_rotate(back[k][r], tk))
                if rot:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        c8, p8, sc, _ = fused_beam(ms, xs, "poe", 8, return_all=True)
                    qk = T.decode(c8.cpu().numpy().reshape(-1), p8.cpu().numpy().reshape(-1, 5)).reshape(B, 8, 4)
                    cands.append(IP.rotate_label(qk.reshape(-1, 4), -np.full(B * 8, float(k)), np.repeat(d3[r], 8, 0), conv).reshape(B, 8, 4) if k else qk)
                    confs.append(sc.float().cpu().numpy())
            Tg.append(tks[0]); To.append(np.mean(tks, 0))
            if not rot:
                continue
            cand = np.concatenate(cands, 1); conf = np.concatenate(confs, 1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                c1, p1 = fused_beam(ms, views[0][1], "poe", 1)
            R["greedy"].append(T.decode(c1.cpu().numpy(), p1.cpu().numpy()))
            score = np.zeros((B, 32))
            for k, xs in views:
                mk = IP.rotate_label(cand.reshape(-1, 4), np.full(B * 32, float(k)), np.repeat(d3[r], 32, 0), conv) if k else cand.reshape(-1, 4)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    score += tree_lp(ms, xs, mk.reshape(B, 32, 4))
            R["orbit_product"].append(cand[np.arange(B), score.argmax(1)])
            w = np.exp(conf.reshape(B, 4, 8) - conf.reshape(B, 4, 8).max(-1, keepdims=True)); w = (w / w.sum(-1, keepdims=True)).reshape(B, 32)
            sup = (np.exp(-0.5 * (geo(cand[:, :, None], cand[:, None]) / 5.0) ** 2) * w[:, None, :]).sum(-1)
            R["orbit_vote32"].append(cand[np.arange(B), sup.argmax(1)])
        return ({m: np.concatenate(v) for m, v in R.items()} if rot else None), np.concatenate(Tg), np.concatenate(To)

    def score_rows(arm, variant, seed, rmode, tmode, qp, tp):
        out = []
        for g in DOMS:
            m = dom3 == g; et = np.linalg.norm(tp[m] - tv3[m], axis=1); etn = et / np.linalg.norm(tv3[m], axis=1)
            er = geo(qp[m], q3[m]) if qp is not None else np.full(m.sum(), np.nan)
            eq = np.radians(er); eqs = np.where(er < 0.169, 0, eq); ets = np.where(etn < 2.173e-3, 0, etn)
            out.append({"arm": arm, "variant": variant, "seed": seed, "rot_mode": rmode, "trans_mode": tmode, "domain": g,
                        "E_T_mean_m": et.mean(), "E_T_median_m": np.median(et), "e_t_norm_mean": etn.mean(),
                        "E_R_mean_deg": np.nanmean(er), "E_R_median_deg": np.nanmedian(er), "esa_score": float(np.nanmean(eqs + ets))})
        return out

    rows = []
    tB = TR.from_u(np.stack([np.zeros(len(t3)), np.zeros(len(t3)), np.full(len(t3), uz_mean)], 1), cam.K, *BOX)
    rows += score_rows("B_box_only", "-", 0, "-", "box", None, tB)
    for seed in [int(s) for s in args.seeds.split(",")]:
        for arm, cfgname in (("J_joint", "joint"), ("F_frozen_tree", "frozen")):
            runs = find(cfgname, seed)
            for variant, keys in (("PoE", (SD, SM)), ("DINO", (SD,))):
                if any(k not in runs for k in keys):
                    print("missing", arm, seed, variant, flush=True); continue
                mh = [load(runs[k], k) for k in keys]; ms, hs = [a for a, _ in mh], [b for _, b in mh]
                R, Tg, To = evaluate(ms, hs, keys)
                for rm in MODES:
                    rows += score_rows(arm, variant, seed, rm, "greedy", R[rm], Tg)
                    rows += score_rows(arm, variant, seed, rm, "orbit_mean", R[rm], To)
                print(arm, seed, variant, " | ".join(f"{g}: E_T {np.linalg.norm(To[dom3 == g] - tv3[dom3 == g], axis=1).mean():.3f} m, "
                                                     f"E_R {geo(R['orbit_product'][dom3 == g], q3[dom3 == g]).mean():.2f}" for g in DOMS), flush=True)
                del mh, ms, hs; torch.cuda.empty_cache()
    per = pd.DataFrame(rows); run.write_metrics(rows, "per_seed")
    keys_ = ["domain", "arm", "variant", "rot_mode", "trans_mode"]; cols = ["E_T_mean_m", "E_T_median_m", "e_t_norm_mean", "E_R_mean_deg", "E_R_median_deg", "esa_score"]
    agg = per.groupby(keys_)[cols].mean().join(per.groupby(keys_)[["E_T_mean_m", "E_R_mean_deg"]].std().add_suffix("_std")).reset_index()
    run.write_metrics(agg.to_dict("records"), "metrics"); (run.sub("tables") / "metrics.md").write_text(md_table(agg.to_dict("records")))
    run.done(); print(md_table(agg.to_dict("records"))); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
