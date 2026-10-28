"""EXP-127 quantization floor on identical rotations (no learning, CPU): the synthetic-val GT rotations of the evaluation set
(exp109_test3 synthetic rows) and the synthetic val-aug rotations (exp109_syn_val rotated by the EXP-109 val-aug angles).
Per grid: Tesseract L5 encode->decode (the tree's own leaf and the flat same-grid hard anchor), Tesseract L5 nearest centre,
balanced Hopf 1771x74 nearest, SPACE-HOP Hopf 10923x12 nearest (EXP-113 grid, descriptive). Mean / median / p95 / max deg."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import data as D, featsets, flatgrid as FG, inplane as IP, tesseract as T  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

run = Run("EXP-127", "hierarchy_vs_flat_same_grid", {"stage": "quantization"}, seed=0)
try:
    conv = (REPO / "outputs/experiments/exp109_convention.txt").read_text().strip()
    cam = D.load_camera(Path(yaml.safe_load(open(REPO / "configs/paths.yaml"))["speedplus_root"]))
    t3 = pd.read_csv(REPO / "outputs/data_manifests/exp109_test3.csv"); t3 = t3[t3.domain == "synthetic"]
    va = pd.read_csv(REPO / "outputs/data_manifests/exp109_syn_val.csv")
    qv = va[[f"q_can_{k}" for k in "wxyz"]].to_numpy(); dv = IP.ray_undistorted(cam.K, cam.dist, va.crop_cx.to_numpy(), va.crop_cy.to_numpy())
    ang = pd.read_csv(featsets.find_cache("dinov3_vitl16", "exp109_valaug") / "aug.csv").rot_deg.to_numpy()
    SETS = {"synthetic_val": t3[[f"q_can_{k}" for k in "wxyz"]].to_numpy(), "synthetic_val_aug": IP.rotate_label(qv, ang, dv, conv)}
    geo = lambda a, b: np.degrees(2 * np.arccos(np.clip(np.abs((a * b).sum(-1)), 0, 1)))

    def nearest(q, G):
        best = np.zeros(len(q))
        for s in range(0, len(G), 8192):
            best = np.maximum(best, np.abs(q @ G[s:s + 8192].T).max(1))
        return np.degrees(2 * np.arccos(np.clip(best, 0, 1)))

    rows = []
    for sname, q in SETS.items():
        q = q / np.linalg.norm(q, axis=1, keepdims=True); c, p = T.encode(q, 5)
        E = {"tesseract_L5 encode-decode (tree leaf / flat same-grid hard anchor)": geo(q, T.decode(c, p)),
             "tesseract_L5 nearest centre": nearest(q, FG.make_grid("tesseract_L5")),
             "hopf1771x74 nearest (balanced Hopf)": nearest(q, FG.make_grid("hopf1771x74")),
             "hopf10923x12 nearest (EXP-113 grid, descriptive)": nearest(q, FG.make_grid("hopf10923x12"))}
        for g, e in E.items():
            rows.append({"set": sname, "n": len(q), "grid": g, "mean_deg": e.mean(), "median_deg": np.median(e), "p95_deg": np.percentile(e, 95), "max_deg": e.max()})
            print(rows[-1], flush=True)
    run.write_metrics(rows, "quantization"); (run.sub("tables") / "quantization.md").write_text(md_table(rows))
    run.done(); print(md_table(rows)); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
