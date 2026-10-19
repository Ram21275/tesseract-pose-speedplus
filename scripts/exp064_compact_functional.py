"""EXP-064 compact-gate follow-up (2026-10-19, after the strict gate FAILED; see report). Functional check: EXP-109 arm-A
trees (seed 0) on the 512 SPEED+ gate crops, fed (a) compact features and (b) full-cache features of the same images,
both standardized with the same statistics (synthetic-train rows of the full cache, 20k-row subsample, rng 0; also used
for the stat-standardized difference row). Reported: per-image angle between the two greedy predictions, fraction of
changed greedy leaves. Reading fixed before running: mean angle < 0.1 deg and changed leaves <= 1% => no impact."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import featsets, phase3 as P, tesseract as T  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

run = Run("EXP-064", "broader_object_validation", {"stage": "compact_functional"}, seed=0)
SD, SM = "dinov3_vitl16:tokens8", "moge2_vitl:normals16~and"; SHAPE = {SD: (64, 1024), SM: (256, 4)}
try:
    torch.set_num_threads(8); dev = torch.device("cpu")
    full = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv")
    tr = np.flatnonzero(((full.domain == "synthetic") & (full.split == "train")).to_numpy())
    sub = np.sort(np.random.default_rng(0).choice(tr, 20000, replace=False))
    runs = {}
    for d in sorted((REPO / "outputs/experiments/EXP-109_real_domain_aug_tokens16_orbit").glob("RUN-*-seed0")):
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        if (d / "checkpoints/best.pt").exists() and cfg.get("features") in SHAPE and (cfg.get("aug_tags") or "") == "exp109_train_aug1,exp109_train_aug2":
            runs[cfg["features"]] = d
    assert set(runs) == {SD, SM}, runs
    X, rows = {}, []
    for sp in (SD, SM):
        bb = sp.split(":")[0]
        with h5py.File(featsets.find_cache(bb, "gate512_r0") / "features.h5", "r") as f:
            gn = [s.decode() for s in f["image_relpath"][:]]
        pos = pd.Series(np.arange(len(full)), index=full.image_relpath)[gn].to_numpy(); o = np.argsort(pos)
        _, xs = featsets.load_rows(sp, "full", sub); st = featsets.train_stats(xs, np.ones(len(xs), bool)); del xs
        _, xg = featsets.load_rows(sp, "gate512_r0", np.arange(len(gn))); xg = xg[o]
        _, xf = featsets.load_rows(sp, "full", pos[o])
        d = np.abs(xg.astype(np.float32) - xf.astype(np.float32)) / np.asarray(st[1], np.float32).reshape(1, -1)
        rows.append({"check": f"{sp} diff / train-stat sd", "max": float(d.max()), "p99": float(np.percentile(d, 99)), "n_gt_1e-2": int((d > 1e-2).sum()), "n_elem": d.size})
        X[(sp, "compact")] = featsets.standardize_to_tensor(xg, None, "cpu", torch.float32, stats=st)
        X[(sp, "full")] = featsets.standardize_to_tensor(xf, None, "cpu", torch.float32, stats=st)
    MS = {}
    for sp in (SD, SM):
        m = P.HierTransformer(SHAPE[sp], 5); m.load_state_dict(torch.load(runs[sp] / "checkpoints/best.pt", map_location=dev)["model"]); m.eval(); MS[sp] = m
    geo = lambda a, b: np.degrees(2 * np.arccos(np.clip(np.abs((a * b).sum(-1)), 0, 1)))
    with torch.no_grad():
        for name, keys, how in (("DINO greedy", (SD,), "single"), ("MoGe-2 greedy", (SM,), "single"), ("PoE greedy", (SD, SM), "poe")):
            pr = {}
            for src in ("compact", "full"):
                c, p = fused_beam([MS[k] for k in keys], [X[(k, src)] for k in keys], how, 1)
                pr[src] = (c.numpy(), p.numpy(), T.decode(c.numpy(), p.numpy()))
            a = geo(pr["compact"][2], pr["full"][2]); ch = ((pr["compact"][0] != pr["full"][0]) | (pr["compact"][1] != pr["full"][1]).any(1)).mean()
            rows.append({"check": name, "mean_angle_deg": float(a.mean()), "p99_angle_deg": float(np.percentile(a, 99)), "max_angle_deg": float(a.max()), "frac_leaf_changed": float(ch)})
            print(rows[-1], flush=True)
    ok = all(r["mean_angle_deg"] < 0.1 and r["frac_leaf_changed"] <= 0.01 for r in rows if "mean_angle_deg" in r)
    run.write_metrics(rows, "compact_functional"); (run.sub("tables") / "compact_functional.md").write_text(md_table(rows))
    run.done(no_impact=ok); print(md_table(rows)); print("NO_IMPACT" if ok else "IMPACT"); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
