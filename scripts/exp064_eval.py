"""EXP-064 evaluation (locked recipe, DEC-018): per dataset, object, seed; variants PoE (DINOv3 + MoGe-2) and DINO-only;
inference greedy / orbit_product / orbit_vote32 over 4 in-plane views (0 / 90 / 180 / 270; T-LESS / YCB-V by the exact
roll homography; candidates mapped with tfpose.xobj using the GT translation, as in the GT-crop protocol).
Errors: symmetry-aware (min over the object's symmetry set) and raw; per object and object-balanced; validation and test."""
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
from tfpose import featsets, phase3 as P, tesseract as T, trans as TR, xobj as XO  # noqa: E402
from tfpose import precision as PR  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.rotations import quat_to_matrix  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=list(XO.DATASETS))
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--modes", default="greedy,orbit_product,orbit_vote32")
ap.add_argument("--exp", default="EXP-064")
ap.add_argument("--exp-name", default="broader_object_validation")
ap.add_argument("--runs-dir", default="EXP-064_broader_object_validation", help="experiment folder holding the train runs")
ap.add_argument("--filter", default="", help="k=v,... on the train-run config (missing key counts as its default: label=raw)")
ap.add_argument("--variants", default="PoE,DINO")
ap.add_argument("--tag", default="", help="label stored in every metrics row (arm name)")
ap.add_argument("--precision", default="bf16", choices=PR.PRECISIONS, help="EXP-128: head inference precision (default = all earlier results)")
args = ap.parse_args()
ds = args.dataset
run = Run(args.exp, args.exp_name, {**vars(args), "stage": "eval"}, seed=0)
FILT = dict(kv.split("=", 1) for kv in args.filter.split(",") if kv); DEFAULTS = {"label": "raw", "extra_train": "None", "teacher_variant": "None", "trans": "none"}
SD, SM = "dinov3_vitl16:tokens8", "moge2_vitl:normals16~and"; SHAPE = {SD: (64, 1024), SM: (256, 4)}
VIEWS = (0, 90, 180, 270); MODES = args.modes.split(",")
try:
    dev = torch.device("cuda")
    ALL = pd.read_csv(REPO / f"outputs/data_manifests/exp064_{ds}_all.csv"); EV = pd.read_csv(REPO / f"outputs/data_manifests/exp064_{ds}_eval.csv")
    oc = XO.DATASETS[ds]["obj_col"]
    runs = {}
    for d in sorted((REPO / "outputs/experiments" / args.runs_dir).glob("RUN-*")):
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        if any(str(cfg.get(k, DEFAULTS.get(k))) != v for k, v in FILT.items()):
            continue
        if cfg.get("stage") == "train" and cfg.get("dataset") == ds and (d / "checkpoints/best.pt").exists():
            runs[(str(cfg["obj"]), cfg["seed"], cfg["features"])] = d
    objs = sorted({k[0] for k in runs})
    (run.dir / "source_runs.txt").write_text("".join(f"{k}\t{args.runs_dir}/{d.name}\n" for k, d in sorted(runs.items())))
    geo = lambda a, b: np.degrees(2 * np.arccos(np.clip(np.abs((a * b).sum(-1)), 0, 1)))

    def tree_lp(ms, xs, cand):
        B, C, _ = cand.shape
        c, p = T.encode(cand.reshape(-1, 4), 5); c, p = torch.as_tensor(c, device=dev), torch.as_tensor(p, device=dev)
        outs = [m.logits_tf(m.encode(x).repeat_interleave(C, 0), c, p) for m, x in zip(ms, xs)]
        root = F.log_softmax(sum(F.log_softmax(o[0].float(), -1) for o in outs), -1)
        child = F.log_softmax(sum(F.log_softmax(o[1].float(), -1) for o in outs), -1)
        return (root.gather(1, c[:, None])[:, 0] + child.gather(2, p[:, :, None])[..., 0].sum(1)).view(B, C).float().cpu().numpy()

    rows, trows = [], []
    for obj in objs:
        objv = obj if isinstance(ALL[oc].iloc[0], str) else type(ALL[oc].iloc[0])(obj)
        a_ev = np.flatnonzero(((ALL.split != "train") & (ALL[oc] == objv)).to_numpy()); e_ev = np.flatnonzero((EV[oc] == objv).to_numpy())
        de = EV.iloc[e_ev]; assert len(de) == len(a_ev)
        K = XO.intrinsics(ds, de); q_ego, tvec = XO.pose_ego(ds, de); dray = XO.crop_ray(K, de); qgt = XO.target(ds, de)
        S = XO.symmetry_set(ds, objv); Req = np.einsum("nij,sjk->nsik", quat_to_matrix(qgt), S)
        TV = {k: XO.roll_pose(q_ego, tvec, float(k), dray)[1] for k in VIEWS}           # translation in each view
        def to_view(c, k):          # view-0 target rotations (N,M,4) -> view-k target frame
            n, m = c.shape[:2]; tt = np.repeat(tvec, m, 0); dd = np.repeat(dray, m, 0)
            return XO.roll_target(ds, XO.from_target(ds, c.reshape(-1, 4), tt), tt, float(k), dd).reshape(n, m, 4) if k else c
        def to_zero(c, k):          # view-k target rotations -> view-0 frame
            n, m = c.shape[:2]; tk = np.repeat(TV[k], m, 0); dd = np.repeat(dray, m, 0)
            return XO.roll_target(ds, XO.from_target(ds, c.reshape(-1, 4), tk), tk, -float(k), dd).reshape(n, m, 4) if k else c
        def sym_err(qp):
            c = np.einsum("nij,nsij->ns", quat_to_matrix(qp), Req); return np.degrees(np.arccos(np.clip((c.max(1) - 1) / 2, -1, 1)))
        for seed in [int(s) for s in args.seeds.split(",")]:
            VAR = [(n, k) for n, k in (("PoE", (SD, SM)), ("DINO", (SD,))) if n in args.variants.split(",")]
            NEED = sorted({k for _, ks in VAR for k in ks})
            if any((obj, seed, sp) not in runs for sp in NEED):
                print("missing", obj, seed, flush=True); continue
            MS, FX, HD = {}, {}, {}
            for sp in NEED:
                ck = torch.load(runs[(obj, seed, sp)] / "checkpoints/best.pt", map_location=dev, weights_only=False)
                m = P.HierTransformer(SHAPE[sp], 5).to(dev); m.load_state_dict(ck["model"]); m.eval(); MS[sp] = PR.prepare(m, args.precision)
                if "trans" in ck:                                                    # EXP-126 translation block
                    h_ = TR.TransHead(m.root_tok.shape[-1]).to(dev); h_.load_state_dict(ck["trans"]); h_.eval(); HD[sp] = (h_, ck)
                st = (ck["mu"], ck["sd"])
                _, x0 = featsets.load_rows(sp, f"exp064_{ds}_all", a_ev); FX[(sp, 0)] = featsets.standardize_to_tensor(x0, None, "cpu", torch.float16, stats=st)
                for k in VIEWS[1:]:
                    _, xk = featsets.load_rows(sp, f"exp064_{ds}_eval_r{k}", e_ev); FX[(sp, k)] = featsets.standardize_to_tensor(xk, None, "cpu", torch.float16, stats=st)
            for variant, keys in VAR:
                ms = [MS[k] for k in keys]; preds = {m: [] for m in MODES}
                if all(k in HD for k in keys):                                       # EXP-126: translation, view 0 and orbit mean
                    W_, H_ = XO.DATASETS[ds]["img_wh"]; box = (de.crop_cx.to_numpy(), de.crop_cy.to_numpy(), de.crop_side.to_numpy(), de.crop_side.to_numpy())
                    TK = []
                    with torch.no_grad():
                        for k in VIEWS:
                            u = np.concatenate([sum((HD[kk][0](MS[kk].encode(FX[(kk, k)][s0:s0 + 512].to(dev).float())).float().cpu().numpy()
                                                     * HD[kk][1]["u_sd"] + HD[kk][1]["u_mu"]) for kk in keys) / len(keys) for s0 in range(0, len(de), 512)])
                            tk = TR.from_u(u, K, *box, W_, H_)
                            TK.append(tk if k == 0 else XO.roll_pose(q_ego, tk, -float(k), dray)[1])   # back to view 0
                    tb = TR.from_u(np.stack([np.zeros(len(de)), np.zeros(len(de)), np.full(len(de), np.mean([HD[kk][1]["uz_train_mean"] for kk in keys]))], 1), K, *box, W_, H_)
                    cm = XO.DATASETS[ds]["t_to_cm"]
                    for tmode, tp in (("greedy", TK[0]), ("orbit_mean", np.mean(TK, 0)), ("box_only", tb)):
                        et = np.linalg.norm(tp - tvec, axis=1)
                        for split in ("validation", "test"):
                            msk = (de.split == split).to_numpy()
                            if msk.any():
                                trows.append({"dataset": ds, "arm": args.tag, "obj": obj, "seed": seed, "variant": variant, "trans_mode": tmode, "split": split,
                                              "n": int(msk.sum()), "E_T_mean": et[msk].mean(), "E_T_median": np.median(et[msk]),
                                              "e_t_norm_mean": float((et[msk] / np.linalg.norm(tvec[msk], axis=1)).mean()),
                                              **({f"T@{c}cm": float((et[msk] * cm <= c).mean()) for c in (2, 5, 10)} if cm else {})})
                with torch.no_grad():
                    for s0 in range(0, len(de), 256):
                        r = np.arange(s0, min(s0 + 256, len(de))); B = len(r); cands, confs, views = [], [], []
                        for k in VIEWS:
                            xs = [FX[(kk, k)][torch.as_tensor(r)].to(dev).float() for kk in keys]; views.append((k, xs))
                            with PR.ctx(args.precision):
                                c8, p8, sc, _ = fused_beam(ms, xs, "poe", 8, return_all=True)
                            qk = T.decode(c8.cpu().numpy().reshape(-1), p8.cpu().numpy().reshape(-1, 5)).reshape(B, 8, 4)
                            cz = qk if k == 0 else XO.roll_target(ds, XO.from_target(ds, qk.reshape(-1, 4), np.repeat(TV[k][r], 8, 0)),
                                                                  np.repeat(TV[k][r], 8, 0), -float(k), np.repeat(dray[r], 8, 0)).reshape(B, 8, 4)
                            cands.append(cz); confs.append(sc.float().cpu().numpy())
                        cand = np.concatenate(cands, 1); conf = np.concatenate(confs, 1)
                        if "greedy" in MODES:
                            with PR.ctx(args.precision):
                                c1, p1 = fused_beam(ms, views[0][1], "poe", 1)
                            preds["greedy"].append(T.decode(c1.cpu().numpy(), p1.cpu().numpy()))
                        if "orbit_product" in MODES:
                            score = np.zeros(cand.shape[:2])
                            for k, xs in views:
                                tt = np.repeat(tvec[r], 32, 0); dd = np.repeat(dray[r], 32, 0)
                                mk = cand if k == 0 else XO.roll_target(ds, XO.from_target(ds, cand.reshape(-1, 4), tt), tt, float(k), dd).reshape(B, 32, 4)
                                with PR.ctx(args.precision):
                                    score += tree_lp(ms, xs, mk)
                            preds["orbit_product"].append(cand[np.arange(B), score.argmax(1)])
                        if "orbit_vote32" in MODES:
                            w = np.exp(conf.reshape(B, 4, 8) - conf.reshape(B, 4, 8).max(-1, keepdims=True)); w = (w / w.sum(-1, keepdims=True)).reshape(B, 32)
                            dd2 = geo(cand[:, :, None], cand[:, None]); sup = (np.exp(-0.5 * (dd2 / 5.0) ** 2) * w[:, None, :]).sum(-1)
                            preds["orbit_vote32"].append(cand[np.arange(B), sup.argmax(1)])
                for mode in MODES:
                    qp = np.concatenate(preds[mode]); es = sym_err(qp); er = geo(qp, qgt)
                    for split in ("validation", "test"):
                        msk = (de.split == split).to_numpy()
                        if not msk.any():
                            continue
                        e = es[msk]
                        rows.append({"dataset": ds, "arm": args.tag, "obj": obj, "seed": seed, "variant": variant, "mode": mode, "split": split, "n": int(msk.sum()),
                                     "sym_mean": e.mean(), "sym_median": np.median(e), **{f"sym_acc@{a}": float((e <= a).mean()) for a in (2, 5, 10, 15, 30)},
                                     "sym_frac_gt90": float((e > 90).mean()), "raw_mean": er[msk].mean()})
                print(ds, obj, seed, variant, " | ".join(f"{m} {np.mean(sym_err(np.concatenate(preds[m]))):.2f}" for m in MODES), flush=True)
            del MS, FX; torch.cuda.empty_cache()
    if trows:
        tp_ = pd.DataFrame(trows); run.write_metrics(trows, "trans_per_object_seed"); tc = [c for c in tp_.columns if c.startswith(("E_T", "e_t", "T@"))]
        tob = tp_.groupby(["split", "variant", "trans_mode", "seed"])[tc].mean().reset_index()
        tagg = tob.groupby(["split", "variant", "trans_mode"])[tc].mean().join(tob.groupby(["split", "variant", "trans_mode"])[["E_T_mean"]].std().add_suffix("_std")).reset_index()
        run.write_metrics(tagg.to_dict("records"), "trans_object_balanced"); print(md_table(tagg.to_dict("records")))
        run.write_metrics(tp_.groupby(["split", "variant", "trans_mode", "obj"])[tc].mean().reset_index().to_dict("records"), "trans_per_object")
    per = pd.DataFrame(rows); run.write_metrics(rows, "per_object_seed")
    po = per.groupby(["split", "variant", "mode", "obj"]).agg(sym_mean=("sym_mean", "mean"), sym_mean_std=("sym_mean", "std"), sym_median=("sym_median", "mean"),
                                                           acc10=("sym_acc@10", "mean"), raw_mean=("raw_mean", "mean")).reset_index()
    run.write_metrics(po.to_dict("records"), "per_object"); (run.sub("tables") / "per_object.md").write_text(md_table(po.to_dict("records")))
    ob = per.groupby(["split", "variant", "mode", "seed"])[["sym_mean", "sym_median", "sym_acc@10", "sym_frac_gt90", "raw_mean"]].mean().reset_index()   # object-balanced per seed
    agg = ob.groupby(["split", "variant", "mode"]).agg(sym_mean=("sym_mean", "mean"), sym_mean_std=("sym_mean", "std"), sym_median=("sym_median", "mean"),
                                                    acc10=("sym_acc@10", "mean"), frac_gt90=("sym_frac_gt90", "mean"), raw_mean=("raw_mean", "mean")).reset_index()
    run.write_metrics(agg.to_dict("records"), "object_balanced"); (run.sub("tables") / "object_balanced.md").write_text(md_table(agg.to_dict("records")))
    run.done(); print(md_table(agg.to_dict("records"))); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
