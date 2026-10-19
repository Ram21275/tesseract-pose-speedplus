"""EXP-064 pre-registered compact-feature gate (run 2026-10-19, after the T-LESS results; recorded as such):
on the 512 SPEED+ crops (subset gate512_r0, compact extractor) the compact tokens8 / normals16~and features must match
the full SPEED+ caches (subset full) for the same images: max abs difference <= 1e-2 after standardization (full-cache
train-row statistics over the matched rows), consensus-mask agreement >= 0.99."""
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import featsets  # noqa: E402
from tfpose.cachepaths import CACHE  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

run = Run("EXP-064", "broader_object_validation", {"stage": "compact_gate"}, seed=0)
try:
    def names(bb, sub):
        with h5py.File(featsets.find_cache(bb, sub) / "features.h5", "r") as f:
            return [s.decode() for s in f["image_relpath"][:]]
    rows_out = []
    for spec in ("dinov3_vitl16:tokens8", "moge2_vitl:normals16~and"):
        bb = spec.split(":")[0]
        gn = names(bb, "gate512_r0"); fn = {n: i for i, n in enumerate(names(bb, "full"))}
        fi = np.array([fn[n] for n in gn]); order = np.argsort(fi)
        _, Xg = featsets.load_rows(spec, "gate512_r0", np.arange(len(gn))); Xg = Xg[order]
        _, Xf = featsets.load_rows(spec, "full", fi[order])
        Xg, Xf = Xg.astype(np.float32), Xf.astype(np.float32)
        sd = Xf.std(0) + 1e-6; d = np.abs(Xg - Xf) / sd
        rows_out.append({"featset": spec, "n": len(gn), "max_abs_std_diff": float(d.max()), "p99_abs_std_diff": float(np.percentile(d, 99)),
                         "mean_abs_std_diff": float(d.mean()), "corr": float(np.corrcoef(Xg.ravel(), Xf.ravel())[0, 1])})
        print(rows_out[-1], flush=True)
    with h5py.File(CACHE / "consensus_masks__subset_gate512_r0/masks.h5", "r") as a, h5py.File(CACHE / "consensus_masks__subset_full/masks.h5", "r") as b:
        an = [s.decode() for s in a["image_relpath"][:]]; bn = {s.decode(): i for i, s in enumerate(b["image_relpath"][:])}
        bi = np.array([bn[n] for n in an]); o = np.argsort(bi)
        ma = a["and_mask"][:][o]; mb = b["and_mask"][np.sort(bi)]
        agree = float((ma.astype(bool) == mb.astype(bool)).mean())
    rows_out.append({"featset": "consensus and_mask", "n": len(an), "pixel_agreement": agree})
    print(rows_out[-1])
    ok = all(r.get("max_abs_std_diff", 0) <= 1e-2 for r in rows_out) and agree >= 0.99
    run.write_metrics(rows_out, "compact_gate"); (run.sub("tables") / "compact_gate.md").write_text(md_table(rows_out))
    run.done(passed=ok); print("PASS" if ok else "FAIL"); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
