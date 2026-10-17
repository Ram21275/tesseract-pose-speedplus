"""EXP-117 teacher targets: orbit over each synthetic-train image's available views (clean + exp109_train_aug1/2, known
roll angles) with the frozen EXP-109 arm-A trees of one seed and one variant (PoE = DINOv3 + MoGe-2, or DINO only).
Candidates: beam-8 leaves of each view mapped to the clean frame (24); weights = softmax of the orbit-product score
(sum over views of the PoE leaf log-prob of the candidate mapped into that view). Saved as a small .npz in the repo."""
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
from tfpose import data as D, featsets, inplane as IP, phase3 as P, tesseract as T  # noqa: E402
from tfpose.cachepaths import CACHE  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.runlog import Run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["PoE", "DINO"], required=True)
ap.add_argument("--seed", type=int, required=True)
args = ap.parse_args()
run = Run("EXP-117", "orbit_distillation", {"stage": "teacher", **vars(args)}, seed=args.seed)
SD, SM = "dinov3_vitl16:tokens8", "moge2_vitl:normals16~and"; SHAPE = {SD: (64, 1024), SM: (256, 4)}
KEYS = (SD, SM) if args.variant == "PoE" else (SD,)
TAGS = ["exp109_train_aug1", "exp109_train_aug2"]
try:
    dev = torch.device("cuda"); conv = (REPO / "outputs/experiments/exp109_convention.txt").read_text().strip()
    df = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv"); tr = ((df.domain == "synthetic") & (df.split == "train")).to_numpy()
    rows = np.flatnonzero(tr); NT = len(rows)
    cam = D.load_camera(Path(yaml.safe_load(open(REPO / "configs/paths.yaml"))["speedplus_root"]))
    dray = IP.ray_undistorted(cam.K, cam.dist, df.crop_cx.to_numpy()[rows], df.crop_cy.to_numpy()[rows])
    FV, ANG = {}, [np.zeros(NT)]
    for sp in KEYS:
        _, xf, _ = featsets.load(sp, "full"); st = featsets.train_stats(xf, tr)
        FV[(sp, 0)] = featsets.standardize_to_tensor(xf[rows], None, "cpu", torch.float16, stats=st); del xf
        for j, tag in enumerate(TAGS, 1):
            n, x, info = featsets.load(sp, tag); assert n == df.image_relpath.values[rows].tolist()
            FV[(sp, j)] = featsets.standardize_to_tensor(x, None, "cpu", torch.float16, stats=st)
            if sp == SD:
                ANG.append(pd.read_csv(CACHE / Path(info["cache"]).name / "aug.csv").rot_deg.to_numpy())
    ANG = np.stack(ANG)
    ms = []
    for sp in KEYS:
        for d in sorted((REPO / "outputs/experiments/EXP-109_real_domain_aug_tokens16_orbit").glob(f"RUN-*-seed{args.seed}")):
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            if cfg.get("features") == sp and cfg.get("aug_tags") and (d / "checkpoints/best.pt").exists():
                m = P.HierTransformer(SHAPE[sp], 5).to(dev); m.load_state_dict(torch.load(d / "checkpoints/best.pt", map_location=dev)["model"]); m.eval(); ms.append(m)
                (run.dir / "source_runs.txt").open("a").write(f"EXP-109_real_domain_aug_tokens16_orbit/{d.name}\n"); break

    def tree_lp(xs, cand):
        B, C, _ = cand.shape
        c, p = T.encode(cand.reshape(-1, 4), 5); c, p = torch.as_tensor(c, device=dev), torch.as_tensor(p, device=dev)
        outs = [m.logits_tf(m.encode(x).repeat_interleave(C, 0), c, p) for m, x in zip(ms, xs)]
        root = F.log_softmax(sum(F.log_softmax(o[0].float(), -1) for o in outs), -1)
        child = F.log_softmax(sum(F.log_softmax(o[1].float(), -1) for o in outs), -1)
        return (root.gather(1, c[:, None])[:, 0] + child.gather(2, p[:, :, None])[..., 0].sum(1)).view(B, C).float().cpu().numpy()

    CAND = np.empty((NT, 8 * len(ANG), 4), np.float32); W = np.empty((NT, 8 * len(ANG)), np.float32)
    with torch.no_grad():
        for s in range(0, NT, 256):
            r = np.arange(s, min(s + 256, NT)); B = len(r); views, cands = [], []
            for v in range(len(ANG)):
                xs = [FV[(sp, v)][torch.as_tensor(r)].to(dev).float() for sp in KEYS]; views.append((v, xs))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    c8, p8, _, _ = fused_beam(ms, xs, "poe", 8, return_all=True)
                qk = T.decode(c8.cpu().numpy().reshape(-1), p8.cpu().numpy().reshape(-1, 5))
                cands.append(IP.rotate_label(qk, -np.repeat(ANG[v, r], 8), np.repeat(dray[r], 8, 0), conv).reshape(B, 8, 4))   # to clean frame
            cand = np.concatenate(cands, 1); K = cand.shape[1]; score = np.zeros((B, K))
            for v, xs in views:
                mk = IP.rotate_label(cand.reshape(-1, 4), np.repeat(ANG[v, r], K), np.repeat(dray[r], K, 0), conv).reshape(B, K, 4)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    score += tree_lp(xs, mk)
            CAND[r] = cand; W[r] = torch.softmax(torch.as_tensor(score), -1).numpy()
            if s // 256 % 40 == 0:
                print(f"{s}/{NT}", flush=True)
    out = REPO / f"outputs/experiments/EXP-117_orbit_distillation/teacher_{args.variant}_seed{args.seed}.npz"
    np.savez_compressed(out, idx=rows, cand=CAND, w=W)
    run.done(teacher=str(out.relative_to(REPO)), mean_max_weight=float(W.max(1).mean())); print("saved", out, "mean max weight", W.max(1).mean())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
