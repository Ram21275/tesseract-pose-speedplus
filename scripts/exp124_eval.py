"""EXP-124 evaluation (pre-registered 2026-10-28): SPEED+ 6-DoF on the full lightbox (6,740) / sunlamp (2,791) test sets and
synthetic/validation, with GT crops (reference) or EXP-124 detected crops, scored with the official SPEED+ metric.
Systems (fixed before results):
  S1 (primary): rotation = EXP-117 PoE students (DINOv3 + MoGe-2, PoE teacher), orbit_product over 4 in-plane views;
                translation = EXP-118 joint (J) DINOv3 block, orbit mean.
  S2: EXP-118 J PoE trees + their translation blocks (one model family), orbit_product / orbit mean.
  S3 (light): EXP-117 DINOv3 student (DINO teacher), orbit_product; translation EXP-118 J DINOv3, orbit mean.
Greedy and orbit_vote32 rotations, and single-view translation, are also reported.
Score: e_q = 2 arccos|<q, q*>| (rad), e_t = |t - t*| / |t*|; per-component zeroing below 0.169 deg / 2.173e-3
(SPNv2 Eq. 9 / Kelvins) and, as a check, the SPEED+ Eq. 4 joint rule; mean per domain."""
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
ap.add_argument("--crops", choices=["gt", "det"], required=True)
ap.add_argument("--seeds", default="0,1,2")
args = ap.parse_args()
run = Run("EXP-124", "speedplus_detected_official_score", {**vars(args), "stage": "eval"}, seed=0)
SD, SM = "dinov3_vitl16:tokens8", "moge2_vitl:normals16~and"; SHAPE = {SD: (64, 1024), SM: (256, 4)}
VIEWS = (0, 90, 180, 270); MODES = ("greedy", "orbit_product", "orbit_vote32")
geo = lambda a, b: np.degrees(2 * np.arccos(np.clip(np.abs((a * b).sum(-1)), 0, 1)))
EXPD = REPO / "outputs/experiments"
try:
    dev = torch.device("cuda")
    conv = (EXPD / "exp109_convention.txt").read_text().strip()
    full = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv"); tr = ((full.domain == "synthetic") & (full.split == "train")).to_numpy()
    man = "exp109_test3.csv" if args.crops == "gt" else "exp124_det_test3.csv"
    t3 = pd.read_csv(REPO / "outputs/data_manifests" / man)
    ref = pd.read_csv(REPO / "outputs/data_manifests/exp109_test3.csv"); assert (t3.image_relpath.values == ref.image_relpath.values).all()
    pos3 = pd.Series(np.arange(len(full)), index=full.image_relpath)[t3.image_relpath].to_numpy()
    cam = D.load_camera(Path(yaml.safe_load(open(REPO / "configs/paths.yaml"))["speedplus_root"]))
    q3 = t3[[f"q_can_{k}" for k in "wxyz"]].to_numpy(); tv3 = t3[["t_x", "t_y", "t_z"]].to_numpy()
    BOX = (t3.crop_cx.to_numpy(), t3.crop_cy.to_numpy(), t3.crop_side.to_numpy(), t3.crop_side.to_numpy())
    d3 = IP.ray_undistorted(cam.K, cam.dist, t3.crop_cx.to_numpy(), t3.crop_cy.to_numpy())
    dom3 = np.where(t3.domain == "synthetic", "synthetic_val", t3.domain); DOMS = ("synthetic_val", "lightbox", "sunlamp")
    Fz = {}
    for sp in (SD, SM):
        _, xf, _ = featsets.load(sp, "full"); st = featsets.train_stats(xf, tr)
        if args.crops == "gt":
            Fz[(sp, 0)] = featsets.standardize_to_tensor(xf[pos3], None, "cpu", torch.float16, stats=st)
        del xf
        for k in (VIEWS if args.crops == "det" else VIEWS[1:]):
            tag = f"exp109_test3_r{k}" if args.crops == "gt" else f"exp124_det_r{k}"
            n, x, _ = featsets.load(sp, tag); assert n == t3.image_relpath.tolist(), tag
            Fz[(sp, k)] = featsets.standardize_to_tensor(x, None, "cpu", torch.float16, stats=st)
    back = {k: IP.rotate_label(np.tile([1.0, 0, 0, 0], (len(t3), 1)), -np.full(len(t3), float(k)), d3, conv) for k in VIEWS[1:]}

    def find(exp, seed, **flt):
        out = {}
        for d in sorted((EXPD / exp).glob(f"RUN-*-seed{seed}")):
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            if (d / "checkpoints/best.pt").exists() and cfg.get("features") in SHAPE and all(str(cfg.get(k)) == v for k, v in flt.items()):
                out[cfg["features"]] = d
        return out

    def load_tree(d, sp, with_head=False):
        ck = torch.load(d / "checkpoints/best.pt", map_location=dev)
        m = P.HierTransformer(SHAPE[sp], 5).to(dev); m.load_state_dict(ck["model"]); m.eval()
        if not with_head:
            return m
        h = TR.TransHead(m.root_tok.shape[-1]).to(dev); h.load_state_dict(ck["trans"]); h.eval(); return m, h

    def tree_lp(ms, xs, cand):
        B, C, _ = cand.shape
        c, p = T.encode(cand.reshape(-1, 4), 5); c, p = torch.as_tensor(c, device=dev), torch.as_tensor(p, device=dev)
        outs = [m.logits_tf(m.encode(x).repeat_interleave(C, 0), c, p) for m, x in zip(ms, xs)]
        root = F.log_softmax(sum(F.log_softmax(o[0].float(), -1) for o in outs), -1)
        child = F.log_softmax(sum(F.log_softmax(o[1].float(), -1) for o in outs), -1)
        return (root.gather(1, c[:, None])[:, 0] + child.gather(2, p[:, :, None])[..., 0].sum(1)).view(B, C).float().cpu().numpy()

    @torch.no_grad()
    def rotate(ms, keys):
        R = {m: [] for m in MODES}
        for s in range(0, len(t3), 256):
            r = np.arange(s, min(s + 256, len(t3))); B = len(r); cands, confs, views = [], [], []
            for k in VIEWS:
                xs = [Fz[(kk, k)][torch.as_tensor(r)].to(dev).float() for kk in keys]; views.append((k, xs))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    c8, p8, sc, _ = fused_beam(ms, xs, "poe", 8, return_all=True)
                qk = T.decode(c8.cpu().numpy().reshape(-1), p8.cpu().numpy().reshape(-1, 5)).reshape(B, 8, 4)
                cands.append(IP.rotate_label(qk.reshape(-1, 4), -np.full(B * 8, float(k)), np.repeat(d3[r], 8, 0), conv).reshape(B, 8, 4) if k else qk)
                confs.append(sc.float().cpu().numpy())
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
        return {m: np.concatenate(v) for m, v in R.items()}

    @torch.no_grad()
    def translate(mh, keys):
        Tg, To = [], []
        for s in range(0, len(t3), 512):
            r = np.arange(s, min(s + 512, len(t3))); tks = []
            for k in VIEWS:
                xs = [Fz[(kk, k)][torch.as_tensor(r)].to(dev).float() for kk in keys]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    u = sum(h(m.encode(x)).float() for (m, h), x in zip(mh, xs)).cpu().numpy() / len(keys)
                tk = TR.from_u(u, cam.K, *(b[r] for b in BOX)); tks.append(tk if k == 0 else TR.quat_rotate(back[k][r], tk))
            Tg.append(tks[0]); To.append(np.mean(tks, 0))
        return {"greedy": np.concatenate(Tg), "orbit_mean": np.concatenate(To)}

    def score_rows(system, seed, rmode, tmode, qp, tp):
        out = []
        for g in DOMS:
            m = dom3 == g; et = np.linalg.norm(tp[m] - tv3[m], axis=1); etn = et / np.linalg.norm(tv3[m], axis=1)
            er = geo(qp[m], q3[m]); eq = np.radians(er)
            sq, stt = np.where(er < 0.169, 0, eq), np.where(etn < 2.173e-3, 0, etn)
            joint = np.where((er < 0.169) & (etn < 2.173e-3), 0, eq + etn)
            out.append({"system": system, "crops": args.crops, "seed": seed, "rot_mode": rmode, "trans_mode": tmode, "domain": g, "n": int(m.sum()),
                        "E_R_mean_deg": er.mean(), "E_R_median_deg": np.median(er), "E_T_mean_m": et.mean(), "E_T_median_m": np.median(et),
                        "e_t_norm_mean": etn.mean(), "score_q": sq.mean(), "score_t": stt.mean(), "score": float((sq + stt).mean()),
                        "score_joint_rule": float(joint.mean()), "frac_ER_gt150": float((er > 150).mean())})
        return out

    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        st_poe = find("EXP-117_orbit_distillation", seed, mode="distill", teacher_variant="PoE")
        st_dino = find("EXP-117_orbit_distillation", seed, mode="distill", teacher_variant="DINO")
        jj = find("EXP-118_translation_6dof", seed, trans="joint")
        (run.dir / "source_runs.txt").open("a").write("".join(f"seed{seed} {n} {k} {d.parent.name}/{d.name}\n" for n, dd in
                                                               (("student_PoE", st_poe), ("student_DINO", st_dino), ("J", jj)) for k, d in dd.items()))
        tJD = translate([load_tree(jj[SD], SD, True)], (SD,))
        tJP = translate([load_tree(jj[k], k, True) for k in (SD, SM)], (SD, SM))
        for system, rot_ms, rkeys, trn in (("S1_studentPoE_rot+J_DINO_trans", [load_tree(st_poe[k], k) for k in (SD, SM)], (SD, SM), tJD),
                                           ("S2_J_PoE", [load_tree(jj[k], k) for k in (SD, SM)], (SD, SM), tJP),
                                           ("S3_studentDINO_rot+J_DINO_trans", [load_tree(st_dino[SD], SD)], (SD,), tJD)):
            R = rotate(rot_ms, rkeys)
            for rm in MODES:
                for tm in ("greedy", "orbit_mean"):
                    rows += score_rows(system, seed, rm, tm, R[rm], trn[tm])
            print(system, seed, " | ".join(f"{g}: E_R {geo(R['orbit_product'][dom3 == g], q3[dom3 == g]).mean():.2f} "
                                           f"E_T {np.linalg.norm(trn['orbit_mean'][dom3 == g] - tv3[dom3 == g], axis=1).mean():.3f}" for g in DOMS), flush=True)
            torch.cuda.empty_cache()
    per = pd.DataFrame(rows); run.write_metrics(rows, "per_seed")
    keys_ = ["crops", "system", "rot_mode", "trans_mode", "domain"]
    cols = ["E_R_mean_deg", "E_R_median_deg", "E_T_mean_m", "E_T_median_m", "e_t_norm_mean", "score_q", "score_t", "score", "score_joint_rule", "frac_ER_gt150"]
    agg = per.groupby(keys_)[cols].mean().join(per.groupby(keys_)[["score", "E_R_mean_deg", "E_T_mean_m"]].std().add_suffix("_std")).reset_index()
    run.write_metrics(agg.to_dict("records"), "metrics"); (run.sub("tables") / "metrics.md").write_text(md_table(agg.to_dict("records")))
    run.done(); print(md_table(agg[(agg.rot_mode == "orbit_product") & (agg.trans_mode == "orbit_mean")].to_dict("records"))); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
