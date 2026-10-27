"""EXP-119 teacher targets (cross-dataset version of exp117_teacher.py): orbit over each training crop's three available
views (clean + exp064_<ds>_aug1/2, known roll angles) with the frozen trees of one object, seed and variant (PoE = DINOv3 +
MoGe-2, or DINO only) taken from the EXP-106 base arm (--runs-dir / --filter). Candidates: beam-8 leaves of each view mapped
to the clean target frame exactly (tfpose.xobj, GT translation); weights = softmax of the orbit-product score (sum over views
of the PoE leaf log-prob of the candidate mapped into that view). Saved as .npz (git-ignored)."""
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
from tfpose import featsets, phase3 as P, tesseract as T, xobj as XO  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.runlog import Run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=list(XO.DATASETS))
ap.add_argument("--obj", required=True)
ap.add_argument("--variant", choices=["PoE", "DINO"], required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--runs-dir", required=True)
ap.add_argument("--filter", default="")
args = ap.parse_args()
run = Run("EXP-119", "xobj_orbit_distillation", {"stage": "teacher", **vars(args)}, seed=args.seed)
SD, SM = "dinov3_vitl16:tokens8", "moge2_vitl:normals16~and"; SHAPE = {SD: (64, 1024), SM: (256, 4)}
KEYS = (SD, SM) if args.variant == "PoE" else (SD,)
FILT = dict(kv.split("=", 1) for kv in args.filter.split(",") if kv); DEFAULTS = {"label": "raw", "extra_train": "None", "teacher_variant": "None"}
try:
    dev = torch.device("cuda"); ds = args.dataset; L = 5
    ALL = pd.read_csv(REPO / f"outputs/data_manifests/exp064_{ds}_all.csv"); TRN = pd.read_csv(REPO / f"outputs/data_manifests/exp064_{ds}_train.csv")
    oc = XO.DATASETS[ds]["obj_col"]; obj = args.obj if isinstance(ALL[oc].iloc[0], str) else type(ALL[oc].iloc[0])(args.obj)
    a_tr = np.flatnonzero(((ALL.split == "train") & (ALL[oc] == obj)).to_numpy()); t_tr = np.flatnonzero((TRN[oc] == obj).to_numpy())
    dtr = TRN.iloc[t_tr]; K = XO.intrinsics(ds, dtr); q_ego, tvec = XO.pose_ego(ds, dtr); dray = XO.crop_ray(K, dtr); NT = len(dtr)
    ms, FV, ANG = [], {}, [np.zeros(NT)]
    for sp in KEYS:
        found = None
        for d in sorted((REPO / "outputs/experiments" / args.runs_dir).glob("RUN-*")):             # newest matching run wins
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            if (cfg.get("stage") == "train" and cfg.get("dataset") == ds and str(cfg.get("obj")) == str(args.obj) and cfg.get("seed") == args.seed
                    and cfg.get("features") == sp and (d / "checkpoints/best.pt").exists() and all(str(cfg.get(k, DEFAULTS.get(k))) == v for k, v in FILT.items())):
                found = d
        assert found is not None, (sp, args)
        ck = torch.load(found / "checkpoints/best.pt", map_location=dev, weights_only=False); st = (ck["mu"], ck["sd"])
        m = P.HierTransformer(SHAPE[sp], L).to(dev); m.load_state_dict(ck["model"]); m.eval(); ms.append(m)
        (run.dir / "source_runs.txt").open("a").write(f"{args.runs_dir}/{found.name}\n")
        FV[(sp, 0)] = featsets.standardize_to_tensor(featsets.load_rows(sp, f"exp064_{ds}_all", a_tr)[1], None, "cpu", torch.float16, stats=st)
        for j in (1, 2):
            tag = f"exp064_{ds}_aug{j}"
            FV[(sp, j)] = featsets.standardize_to_tensor(featsets.load_rows(sp, tag, t_tr)[1], None, "cpu", torch.float16, stats=st)
            if sp == SD:
                ANG.append(pd.read_csv(featsets.find_cache(sp.split(":")[0], tag) / "aug.csv").rot_deg.to_numpy()[t_tr])
    ANG = np.stack(ANG); TV = [XO.roll_pose(q_ego, tvec, ANG[v], dray)[1] for v in range(3)]   # translation in each view

    def tree_lp(xs, cand):
        B, C, _ = cand.shape
        c, p = T.encode(cand.reshape(-1, 4), L); c, p = torch.as_tensor(c, device=dev), torch.as_tensor(p, device=dev)
        outs = [m.logits_tf(m.encode(x).repeat_interleave(C, 0), c, p) for m, x in zip(ms, xs)]
        root = F.log_softmax(sum(F.log_softmax(o[0].float(), -1) for o in outs), -1)
        child = F.log_softmax(sum(F.log_softmax(o[1].float(), -1) for o in outs), -1)
        return (root.gather(1, c[:, None])[:, 0] + child.gather(2, p[:, :, None])[..., 0].sum(1)).view(B, C).float().cpu().numpy()

    CAND = np.empty((NT, 24, 4), np.float32); W = np.empty((NT, 24), np.float32)
    with torch.no_grad():
        for s in range(0, NT, 256):
            r = np.arange(s, min(s + 256, NT)); B = len(r); views, cands = [], []
            for v in range(3):
                xs = [FV[(sp, v)][torch.as_tensor(r)].to(dev).float() for sp in KEYS]; views.append((v, xs))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    c8, p8, _, _ = fused_beam(ms, xs, "poe", 8, return_all=True)
                qk = T.decode(c8.cpu().numpy().reshape(-1), p8.cpu().numpy().reshape(-1, L))
                tk = np.repeat(TV[v][r], 8, 0)                                              # view-v target -> clean frame
                cands.append(XO.roll_target(ds, XO.from_target(ds, qk, tk), tk, -np.repeat(ANG[v, r], 8), np.repeat(dray[r], 8, 0)).reshape(B, 8, 4))
            cand = np.concatenate(cands, 1); score = np.zeros((B, 24)); tt = np.repeat(tvec[r], 24, 0); dd = np.repeat(dray[r], 24, 0)
            for v, xs in views:
                mk = XO.roll_target(ds, XO.from_target(ds, cand.reshape(-1, 4), tt), tt, np.repeat(ANG[v, r], 24), dd).reshape(B, 24, 4)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    score += tree_lp(xs, mk)
            CAND[r] = cand; W[r] = torch.softmax(torch.as_tensor(score), -1).numpy()
    out = REPO / f"outputs/experiments/EXP-119_xobj_orbit_distillation/teacher_{ds}_{args.obj}_{args.variant}_seed{args.seed}.npz"
    out.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(out, idx=a_tr, cand=CAND, w=W)
    qt = XO.target(ds, dtr); g = np.degrees(2 * np.arccos(np.clip(np.abs((CAND[np.arange(NT), W.argmax(1)] * qt).sum(-1)), 0, 1)))
    run.done(teacher=str(out.relative_to(REPO)), mean_max_weight=float(W.max(1).mean()), teacher_top_raw_mean_deg=float(g.mean()))
    print("saved", out, "mean max weight", W.max(1).mean(), "top-candidate raw err", g.mean())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
