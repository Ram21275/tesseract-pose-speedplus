"""EXP-064 zero-shot arm (amendment 2026-10-19, pre-registered before this run): the frozen SPEED+ trees (EXP-109 arm A,
both branches, seeds 0-2) on the EXP-064 evaluation rows, no retraining. Scored in the trees' own convention (camera-frame
rotation; candidates mapped between orbit views by the camera roll about the crop ray, no GT translation).
Metrics: (a) raw symmetry-aware error; (b) chance (uniform random rotations, same metric); (c) frame-free relative-rotation
error over random same-object test pairs + its chance; (d) 3-DOF calibrated (one offset per object chosen on validation,
reported on test only). Standardization: SPEED+ synthetic-train statistics (primary) and the target object's own clean
train statistics (secondary, label-free). Validation is subsampled to <= 1000 rows per object (rng 0) to keep the CPU run
short; test is complete. CPU only."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import featsets, phase3 as P, tesseract as T, xobj as XO  # noqa: E402
from tfpose.cachepaths import COMPACT  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.allo import mat_to_quat  # noqa: E402
from tfpose.rotations import quat_to_matrix  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=list(XO.DATASETS))
ap.add_argument("--threads", type=int, default=12)
ap.add_argument("--val-cap", type=int, default=1000)
args = ap.parse_args()
ds = args.dataset
run = Run("EXP-064", "broader_object_validation", {**vars(args), "stage": "zeroshot"}, seed=0)
SD, SM = "dinov3_vitl16:tokens8", "moge2_vitl:normals16~and"; SHAPE = {SD: (64, 1024), SM: (256, 4)}
VIEWS = (0, 90, 180, 270)
torch.set_num_threads(args.threads); dev = torch.device("cpu")
geo = lambda a, b: np.degrees(2 * np.arccos(np.clip(np.abs((a * b).sum(-1)), 0, 1)))
try:
    # --- frozen SPEED+ trees (EXP-109 arm A) ---
    trees = {}
    for d in sorted((REPO / "outputs/experiments/EXP-109_real_domain_aug_tokens16_orbit").glob("RUN-*")):
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        if (d / "checkpoints/best.pt").exists() and cfg.get("features") in SHAPE and (cfg.get("aug_tags") or "") == "exp109_train_aug1,exp109_train_aug2":
            m = P.HierTransformer(SHAPE[cfg["features"]], 5); m.load_state_dict(torch.load(d / "checkpoints/best.pt", map_location=dev)["model"]); m.eval()
            trees[(cfg["features"], int(cfg["seed"]))] = m
    assert len(trees) == 6, sorted(trees)
    # --- SPEED+ standardization statistics (synthetic-train rows of subset full, as orbit_eval.py), cached ---
    full = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv")
    tr = np.flatnonzero(((full.domain == "synthetic") & (full.split == "train")).to_numpy())
    SPS = {}
    for sp in (SD, SM):
        f = COMPACT / f"speedplus_trainstats__{sp.replace(':', '_').replace('~', '-')}.npz"
        if not f.exists():
            s1 = s2 = 0.0                          # chunked (the full cache stores the 16x16 grid: ~25 GB if loaded at once)
            for c0 in range(0, len(tr), 2000):
                _, x = featsets.load_rows(sp, "full", tr[c0:c0 + 2000]); x = x.astype(np.float64); s1 = s1 + x.sum(0); s2 = s2 + (x ** 2).sum(0); del x
            mu = s1 / len(tr); sd = np.sqrt(np.maximum(s2 / len(tr) - mu ** 2, 0)) + 1e-6       # = featsets.train_stats
            np.savez(f, mu=mu, sd=sd)
        z = np.load(f); SPS[sp] = (z["mu"], z["sd"])
    ALL = pd.read_csv(REPO / f"outputs/data_manifests/exp064_{ds}_all.csv"); EV = pd.read_csv(REPO / f"outputs/data_manifests/exp064_{ds}_eval.csv")
    oc = XO.DATASETS[ds]["obj_col"]; rng = np.random.default_rng(0)
    objs = sorted(ALL[oc].unique().tolist(), key=str)

    def rot_view(q, ang, d):        # camera-frame rotation after an in-plane roll by ang (cv2 sign) about ray d
        return XO.roll_pose(q, np.zeros((len(q), 3)), ang, d)[0]

    def tree_lp(ms, xs, cand):
        B, C, _ = cand.shape
        c, p = T.encode(cand.reshape(-1, 4), 5); c, p = torch.as_tensor(c), torch.as_tensor(p)
        outs = [m.logits_tf(m.encode(x).repeat_interleave(C, 0), c, p) for m, x in zip(ms, xs)]
        F = torch.nn.functional
        root = F.log_softmax(sum(F.log_softmax(o[0].float(), -1) for o in outs), -1)
        child = F.log_softmax(sum(F.log_softmax(o[1].float(), -1) for o in outs), -1)
        return (root.gather(1, c[:, None])[:, 0] + child.gather(2, p[:, :, None])[..., 0].sum(1)).view(B, C).numpy()

    def symerr(qp, Req):            # (N,4), (N,S,3,3) -> deg
        c = np.einsum("nij,nsij->ns", quat_to_matrix(qp), Req); return np.degrees(np.arccos(np.clip((c.max(1) - 1) / 2, -1, 1)))

    def rand_q(n, g):
        q = g.normal(size=(n, 4)); return q / np.linalg.norm(q, axis=1, keepdims=True)

    def pair_err(qp, Rg, S, pairs):  # min_S angle(Rp_i Rp_j^T, Rg_i S Rg_j^T)
        Rp = quat_to_matrix(qp); i, j = pairs[:, 0], pairs[:, 1]
        A = Rp[i] @ np.transpose(Rp[j], (0, 2, 1)); out = np.empty(len(i))
        for s0 in range(0, len(i), 500):
            sl = slice(s0, s0 + 500)
            B = np.einsum("nij,sjk,nlk->nsil", Rg[i[sl]], S, Rg[j[sl]])          # Rg_i S Rg_j^T
            c = np.einsum("nij,nsij->ns", A[sl], B); out[sl] = np.degrees(np.arccos(np.clip((c.max(1) - 1) / 2, -1, 1)))
        return out

    rows = []
    for obj in objs:
        a_ev = np.flatnonzero(((ALL.split != "train") & (ALL[oc] == obj)).to_numpy()); e_ev = np.flatnonzero((EV[oc] == obj).to_numpy())
        de = EV.iloc[e_ev]; assert len(de) == len(a_ev)
        isv = (de.split == "validation").to_numpy(); vi = np.flatnonzero(isv); ti = np.flatnonzero(~isv)
        if len(vi) > args.val_cap:
            vi = np.sort(rng.choice(vi, args.val_cap, replace=False))
        keep = np.sort(np.concatenate([vi, ti])); de = de.iloc[keep]; a_k = a_ev[keep]; e_k = e_ev[keep]
        isv = (de.split == "validation").to_numpy()
        K = XO.intrinsics(ds, de); q_ego, _ = XO.pose_ego(ds, de); dray = XO.crop_ray(K, de)
        S = XO.symmetry_set(ds, obj); Rg = quat_to_matrix(q_ego); Req = np.einsum("nij,sjk->nsik", Rg, S)
        tst = np.flatnonzero(~isv); pairs = tst[rng.integers(0, len(tst), (2000, 2))] if len(tst) > 1 else None
        if pairs is not None:
            pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        g = np.random.default_rng(1); qr = rand_q(len(de), g)
        chance = {"sym": symerr(qr, Req), "pair": pair_err(qr, Rg, S, pairs) if pairs is not None else None,
                  "const_pair": pair_err(np.tile([1.0, 0, 0, 0], (len(de), 1)), Rg, S, pairs) if pairs is not None else None}   # post-hoc control (T-LESS write-up)
        # features per view (raw), standardized per stats source
        RAW = {}
        for sp in (SD, SM):
            _, RAW[(sp, 0)] = featsets.load_rows(sp, f"exp064_{ds}_all", a_k)
            for k in VIEWS[1:]:
                _, RAW[(sp, k)] = featsets.load_rows(sp, f"exp064_{ds}_eval_r{k}", e_k)
        OWN = {}
        for sp in (SD, SM):
            a_tr = np.flatnonzero(((ALL.split == "train") & (ALL[oc] == obj)).to_numpy())
            _, xt = featsets.load_rows(sp, f"exp064_{ds}_all", a_tr); OWN[sp] = featsets.train_stats(xt, np.ones(len(xt), bool)); del xt
        for stats_src in ("speedplus", "target_own"):
            X = {(sp, k): featsets.standardize_to_tensor(RAW[(sp, k)], None, "cpu", torch.float32, stats=SPS[sp] if stats_src == "speedplus" else OWN[sp])
                 for sp in (SD, SM) for k in VIEWS}
            modes = ("greedy", "orbit_product", "orbit_vote32") if stats_src == "speedplus" else ("greedy",)
            for seed in (0, 1, 2):
                for variant, keys in (("PoE", (SD, SM)), ("DINO", (SD,))):
                    ms = [trees[(k, seed)] for k in keys]; preds = {m: [] for m in modes}
                    with torch.no_grad():
                        for s0 in range(0, len(de), 128):
                            r = np.arange(s0, min(s0 + 128, len(de))); B = len(r)
                            c1, p1 = fused_beam(ms, [X[(k, 0)][r] for k in keys], "poe" if len(keys) > 1 else "single", 1)
                            preds["greedy"].append(T.decode(c1.numpy(), p1.numpy()))
                            if len(modes) == 1:
                                continue
                            cands, confs = [], []
                            for k in VIEWS:
                                c8, p8, sc, _ = fused_beam(ms, [X[(kk, k)][r] for kk in keys], "poe" if len(keys) > 1 else "single", 8, return_all=True)
                                qk = T.decode(c8.numpy().reshape(-1), p8.numpy().reshape(-1, 5))
                                cands.append(rot_view(qk, -float(k), np.repeat(dray[r], 8, 0)).reshape(B, 8, 4) if k else qk.reshape(B, 8, 4))
                                confs.append(sc.float().numpy())
                            cand = np.concatenate(cands, 1); conf = np.concatenate(confs, 1); score = np.zeros((B, 32))
                            for k in VIEWS:
                                mk = rot_view(cand.reshape(-1, 4), float(k), np.repeat(dray[r], 32, 0)).reshape(B, 32, 4) if k else cand
                                score += tree_lp(ms, [X[(kk, k)][r] for kk in keys], mk)
                            preds["orbit_product"].append(cand[np.arange(B), score.argmax(1)])
                            w = np.exp(conf.reshape(B, 4, 8) - conf.reshape(B, 4, 8).max(-1, keepdims=True)); w = (w / w.sum(-1, keepdims=True)).reshape(B, 32)
                            sup = (np.exp(-0.5 * (geo(cand[:, :, None], cand[:, None]) / 5.0) ** 2) * w[:, None, :]).sum(-1)
                            preds["orbit_vote32"].append(cand[np.arange(B), sup.argmax(1)])
                    for mode in modes:
                        qp = np.concatenate(preds[mode]); es = symerr(qp, Req)
                        # (d) calibration: candidates A = Rp_i^T Rg_i S from <= 200 val rows, chosen by mean val error
                        v = np.flatnonzero(isv); cal = np.nan
                        if len(v) and len(tst):
                            Rp = quat_to_matrix(qp); cs = v[np.random.default_rng(2).choice(len(v), min(200, len(v)), replace=False)]
                            Ss = S if len(S) <= 24 else S[np.linspace(0, len(S) - 1, 24).astype(int)]
                            Ac = np.einsum("nji,njk,skl->nsil", Rp[cs], Rg[cs], Ss).reshape(-1, 3, 3)
                            best, bA = np.inf, None
                            for c0 in range(0, len(Ac), 256):
                                Ab = Ac[c0:c0 + 256]; RA = np.einsum("nij,ajk->anik", Rp[v], Ab)       # (a,n,3,3)
                                cc = np.einsum("anij,nsij->ans", RA, Req[v]).max(-1)
                                e = np.degrees(np.arccos(np.clip((cc - 1) / 2, -1, 1))).mean(1)
                                if e.min() < best:
                                    best, bA = e.min(), Ab[e.argmin()]
                            RAt = Rp[tst] @ bA; cc = np.einsum("nij,nsij->ns", RAt, Req[tst]).max(1); cal = float(np.degrees(np.arccos(np.clip((cc - 1) / 2, -1, 1))).mean())
                        pe = pair_err(qp, Rg, S, pairs) if pairs is not None else np.array([np.nan])
                        for split, msk in (("validation", isv), ("test", ~isv)):
                            if not msk.any():
                                continue
                            rows.append({"dataset": ds, "obj": str(obj), "stats": stats_src, "seed": seed, "variant": variant, "mode": mode, "split": split,
                                         "n": int(msk.sum()), "sym_mean": float(es[msk].mean()), "sym_median": float(np.median(es[msk])),
                                         "acc10": float((es[msk] <= 10).mean()), "chance_sym_mean": float(chance["sym"][msk].mean()),
                                         "pair_mean": float(pe.mean()) if split == "test" else np.nan,
                                         "chance_pair_mean": float(chance["pair"].mean()) if (split == "test" and chance["pair"] is not None) else np.nan,
                                         "const_pair_mean": float(chance["const_pair"].mean()) if (split == "test" and chance["const_pair"] is not None) else np.nan,
                                         "calibrated_test_mean": cal if split == "test" else np.nan})
                        print(ds, obj, stats_src, seed, variant, mode, f"test sym {es[~isv].mean():.1f} (chance {chance['sym'][~isv].mean():.1f}) "
                              f"pair {pe.mean():.1f} (chance {np.nanmean(chance['pair']) if chance['pair'] is not None else np.nan:.1f}) cal {cal:.1f}", flush=True)
        del RAW, X
    per = pd.DataFrame(rows); run.write_metrics(rows, "per_object_seed")
    cols = ["sym_mean", "sym_median", "acc10", "chance_sym_mean", "pair_mean", "chance_pair_mean", "const_pair_mean", "calibrated_test_mean"]
    ob = per.groupby(["stats", "split", "variant", "mode", "seed"])[cols].mean().reset_index()
    agg = ob.groupby(["stats", "split", "variant", "mode"]).agg(**{c: (c, "mean") for c in cols}, sym_mean_std=("sym_mean", "std"), pair_mean_std=("pair_mean", "std")).reset_index()
    run.write_metrics(agg.to_dict("records"), "object_balanced"); (run.sub("tables") / "object_balanced.md").write_text(md_table(agg.to_dict("records")))
    po = per[per.split == "test"].groupby(["stats", "variant", "mode", "obj"])[cols].mean().reset_index()
    (run.sub("tables") / "per_object_test.md").write_text(md_table(po.to_dict("records")))
    run.done(); print(md_table(agg.to_dict("records"))); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
