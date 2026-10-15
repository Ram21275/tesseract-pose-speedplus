"""EXP-110: weighted per-level PoE of the frozen EXP-036 DINOv3 / MoGe-2 trees, greedy, w in a fixed grid, 3 seeds.
Domains: synthetic val, synthetic val-aug (EXP-109 proxy; selection), lightbox, sunlamp."""
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
from tfpose.runlog import Run, md_table  # noqa: E402

W = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
SPECS = {"dinov3_vitl16:tokens8": "64x1024", "moge2_vitl:normals16~and": "256x4"}
run = Run("EXP-110", "weighted_poe", {"w": W}, seed=0)
try:
    dev = torch.device("cuda")
    full = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv"); tr = ((full.domain == "synthetic") & (full.split == "train")).to_numpy()
    va = pd.read_csv(REPO / "outputs/data_manifests/exp109_syn_val.csv")
    conv = (REPO / "outputs/experiments/exp109_convention.txt").read_text().strip()
    cam = D.load_camera(Path(yaml.safe_load(open(REPO / "configs/paths.yaml"))["speedplus_root"]))
    ang = pd.read_csv(featsets.find_cache("dinov3_vitl16", "exp109_valaug") / "aug.csv").rot_deg.to_numpy()
    qva = IP.rotate_label(va[[f"q_can_{k}" for k in "wxyz"]].to_numpy(), ang, IP.ray_undistorted(cam.K, cam.dist, va.crop_cx.to_numpy(), va.crop_cy.to_numpy()), conv)
    q = full[[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    sets = {"synthetic_val": np.flatnonzero(((full.domain == "synthetic") & (full.split == "validation")).to_numpy()),
            "lightbox": np.flatnonzero((full.domain == "lightbox").to_numpy()), "sunlamp": np.flatnonzero((full.domain == "sunlamp").to_numpy())}
    Xf, Xa = {}, {}
    for sp in SPECS:
        _, x, _ = featsets.load(sp, "full"); st = featsets.train_stats(x, tr)
        Xf[sp] = featsets.standardize_to_tensor(x, tr, "cpu", torch.float16, stats=st); del x
        n, xa, _ = featsets.load(sp, "exp109_valaug"); assert n == va.image_relpath.tolist()
        Xa[sp] = featsets.standardize_to_tensor(xa, None, "cpu", torch.float16, stats=st)

    @torch.no_grad()
    def greedy(ms, xs, w):
        hs = [m.encode(x) for m, x in zip(ms, xs)]; wt = (w, 1.0 - w)
        lp = F.log_softmax(sum(a * F.log_softmax(m.root(h).float(), -1) for a, m, h in zip(wt, ms, hs)), -1)
        c = lp.argmax(-1); p = torch.zeros(len(c), 0, dtype=torch.long, device=dev)
        for _ in range(5):
            lp = F.log_softmax(sum(a * F.log_softmax(m.child_logits_prefix(h, c, p).float(), -1) for a, m, h in zip(wt, ms, hs)), -1)
            p = torch.cat([p, lp.argmax(-1, keepdim=True)], 1)
        return T.decode(c.cpu().numpy(), p.cpu().numpy())

    geo = lambda a, b: np.degrees(2 * np.arccos(np.clip(np.abs((a * b).sum(1)), 0, 1)))
    rows = []
    for seed in (0, 1, 2):
        ms = []
        for sp, tsh in SPECS.items():
            for d in sorted((REPO / "outputs/experiments/EXP-036_transformer_soft_targets").glob(f"RUN-*-seed{seed}")):
                cfg = yaml.safe_load((d / "config.yaml").read_text())
                if cfg.get("features") == sp and (d / "checkpoints/best.pt").exists():
                    m = P.HierTransformer(tuple(int(v) for v in tsh.split("x")), 5, dropout=cfg["dropout"]).to(dev)
                    m.load_state_dict(torch.load(d / "checkpoints/best.pt", map_location=dev)["model"]); m.eval(); ms.append(m); break
        E = {}
        for w in W:
            for g, idx in list(sets.items()) + [("synthetic_val_aug", None)]:
                preds = []
                rng_ = range(0, len(idx) if idx is not None else len(va), 1024)
                for s in rng_:
                    if idx is None:
                        xs = [Xa[sp][s:s + 1024].to(dev).float() for sp in SPECS]
                    else:
                        xs = [Xf[sp][torch.as_tensor(idx[s:s + 1024])].to(dev).float() for sp in SPECS]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        preds.append(greedy(ms, xs, w))
                E[(w, g)] = geo(np.concatenate(preds), q[idx] if idx is not None else qva)
        for (w, g), e in E.items():
            e0 = E[(0.5, g)]; n150 = int((e0 > 150).sum()); n20 = int((e0 < 20).sum())
            rows.append({"seed": seed, "w_dino": w, "domain": g, "mean_deg": e.mean(), "median_deg": np.median(e), "acc@10": float((e <= 10).mean()),
                         "frac_gt90": float((e > 90).mean()), "frac_gt150": float((e > 150).mean()),
                         "n_w05_gt150": n150, "frac_gt150_to_lt20_vs_w05": ((e0 > 150) & (e < 20)).sum() / max(n150, 1),
                         "n_w05_lt20": n20, "frac_lt20_to_gt150_vs_w05": ((e0 < 20) & (e > 150)).sum() / max(n20, 1)})
        print(seed, " | ".join(f"w{w}: aug {E[(w, 'synthetic_val_aug')].mean():.2f} clean {E[(w, 'synthetic_val')].mean():.2f} lb {E[(w, 'lightbox')].mean():.2f}" for w in W), flush=True)
    per = pd.DataFrame(rows); run.write_metrics(rows, "per_seed")
    cols = [c for c in per.columns if c not in ("seed", "w_dino", "domain")]
    agg = per.groupby(["domain", "w_dino"])[cols].mean().join(per.groupby(["domain", "w_dino"])[["mean_deg"]].std().rename(columns={"mean_deg": "mean_deg_std"})).reset_index()
    run.write_metrics(agg.to_dict("records"), "metrics"); (run.sub("tables") / "metrics.md").write_text(md_table(agg.to_dict("records")))
    a = agg[agg.domain == "synthetic_val_aug"].set_index("w_dino"); wstar = float(a.mean_deg.idxmin())
    run.done(w_star_valaug=wstar); print("w* (val-aug) =", wstar)
    print(md_table(agg.to_dict("records"), ["domain", "w_dino", "mean_deg", "mean_deg_std", "median_deg", "frac_gt150"])); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
