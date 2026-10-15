"""EXP-109 predictor consistency check (DESCRIPTIVE ONLY since the 2026-10-15 review; the label convention is derived
from geometry in exp109_convention_geometry.py): 512 synthetic-val crops extracted unrotated and rotated by 90 and 37 deg
(compact caches gate512_r0 / _r90 / _r37); frozen EXP-036 PoE trees (seed 0) predict on each; the rotated predictions
are compared with the GT rotated by each candidate label convention. Also checks compact (r0) == full-cache features.
Writes the chosen convention to outputs/experiments/exp109_convention.txt (or NONE)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import data as D, featsets, inplane as IP, phase3 as P, tesseract as T  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

SPECS = {"dinov3_vitl16:tokens8": "64x1024", "moge2_vitl:normals16~and": "256x4"}
run = Run("EXP-109", "real_domain_aug_tokens16_orbit", {"stage": "convention_gate"}, seed=0)
try:
    dev = torch.device("cuda")
    full = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv"); tr = ((full.domain == "synthetic") & (full.split == "train")).to_numpy()
    g = pd.read_csv(REPO / "outputs/data_manifests/exp109_gate512.csv")
    pos = pd.Series(np.arange(len(full)), index=full.image_relpath)[g.image_relpath].to_numpy()
    CAM = D.load_camera(Path(yaml.safe_load(open(REPO / "configs/paths.yaml"))["speedplus_root"])); K = CAM.K
    d = IP.ray_undistorted(K, CAM.dist, g.crop_cx.to_numpy(), g.crop_cy.to_numpy())
    qg = g[[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    feats, eq = {}, []
    for sp in SPECS:
        _, xf, _ = featsets.load(sp, "full"); st = featsets.train_stats(xf, tr)
        for tag in ("gate512_r0", "gate512_r90", "gate512_r37"):
            names, x, _ = featsets.load(sp, tag); assert names == g.image_relpath.tolist()
            feats[(sp, tag)] = featsets.standardize_to_tensor(x, None, "cpu", torch.float32, stats=st)
            if tag == "gate512_r0":
                ref = featsets.standardize_to_tensor(xf[pos], None, "cpu", torch.float32, stats=st)
                dif = (feats[(sp, tag)] - ref).abs()
                eq.append({"spec": sp, "max_abs_diff_std": float(dif.max()), "mean_abs_diff_std": float(dif.mean()),
                           "rel_l2": float((feats[(sp, tag)] - ref).norm() / ref.norm())})
        del xf
    run.write_metrics(eq, "compact_equivalence"); print(md_table(eq))
    trees = []
    for sp, tsh in SPECS.items():
        for dd in sorted((REPO / "outputs/experiments/EXP-036_transformer_soft_targets").glob("RUN-*-seed0")):
            cfg = yaml.safe_load((dd / "config.yaml").read_text())
            if cfg.get("features") == sp and (dd / "checkpoints/best.pt").exists():
                t = P.HierTransformer(tuple(int(v) for v in tsh.split("x")), 5, dropout=cfg["dropout"]).to(dev)
                t.load_state_dict(torch.load(dd / "checkpoints/best.pt", map_location=dev)["model"]); t.eval(); trees.append(t); break

    def predict(tag):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            c, p = fused_beam(trees, [feats[(sp, tag)].to(dev) for sp in SPECS], "poe", 1)
        return T.decode(c.cpu().numpy(), p.cpu().numpy())

    geo = lambda a, b: np.degrees(2 * np.arccos(np.clip(np.abs((a * b).sum(1)), 0, 1)))
    rows = [{"view": "r0", "mode": "-", "mean_deg": geo(predict("gate512_r0"), qg).mean()}]
    for tag, th in (("gate512_r90", 90.0), ("gate512_r37", 37.0)):
        qp = predict(tag)
        for m in IP.MODES:
            rows.append({"view": tag[-3:], "mode": m, "mean_deg": geo(qp, IP.rotate_label(qg, th, d, m)).mean(),
                         "median_deg": float(np.median(geo(qp, IP.rotate_label(qg, th, d, m))))})
    run.write_metrics(rows, "gate"); print(md_table(rows))
    # Descriptive only (review 2026-10-15): the convention is DERIVED from geometry (exp109_convention_geometry.py) and read
    # from exp109_convention.txt; this predictor check never selects it.
    conv = (REPO / "outputs/experiments/exp109_convention.txt").read_text().strip()
    run.done(derived_convention=conv, r0_mean=float(rows[0]["mean_deg"]))
    print("DESCRIPTIVE predictor check; derived convention =", conv); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
