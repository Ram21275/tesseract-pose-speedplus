"""EXP-124 detection on the EXP-109 test3 rows (synthetic/validation, lightbox, sunlamp): top-1 box of the EXP-124 detector
on the 960 x 600 image, scaled to full resolution, turned into the square crop with the GT-crop rule (centre of the box,
side = max(w, h) x (1 + 2 x 0.15), min 32 px). No detection -> whole-image crop (counted). Writes
outputs/data_manifests/exp124_det_test3.csv (same rows and order as exp109_test3.csv; crop_* replaced; det_conf, det_iou
= IoU with the GT keypoint box) and per-domain detection statistics."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose.runlog import Run  # noqa: E402

Y = Path("/media/kavinder/hdd2/RAM/speedplus_yolo")
run = Run("EXP-124", "speedplus_detected_official_score", {"stage": "detect"}, seed=0)
try:
    from ultralytics import YOLO
    det = sorted(d for d in (REPO / "outputs/experiments/EXP-124_speedplus_detected_official_score").glob("RUN-*")
                 if yaml.safe_load((d / "config.yaml").read_text()).get("stage") == "detector" and (d / "checkpoints/best.pt").exists())[-1]
    (run.dir / "source_runs.txt").write_text(f"detector {det.name}\n")
    t3 = pd.read_csv(REPO / "outputs/data_manifests/exp109_test3.csv"); bx = pd.read_csv(Y / "boxes.csv").set_index("image_relpath").loc[t3.image_relpath]
    files = [str(Y / "images" / s / f) for s, f in zip(bx.split, bx.file)]
    m = YOLO(str(det / "checkpoints/best.pt")); B = np.full((len(t3), 4), np.nan); C = np.zeros(len(t3))
    for s in range(0, len(files), 64):
        for i, r in enumerate(m.predict(files[s:s + 64], imgsz=960, conf=0.001, max_det=1, verbose=False), s):
            if len(r.boxes):
                B[i] = r.boxes.xyxy[0].cpu().numpy() * 2.0; C[i] = float(r.boxes.conf[0])
    miss = np.isnan(B[:, 0]); B[miss] = [0, 0, 1920, 1200]
    w, h = B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]; g = bx[["x0", "y0", "x1", "y1"]].to_numpy()
    inter = np.clip(np.minimum(B[:, 2:], g[:, 2:]) - np.maximum(B[:, :2], g[:, :2]), 0, None).prod(1)
    iou = inter / (w * h + (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]) - inter)
    out = t3.copy(); out["crop_cx"], out["crop_cy"] = (B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2
    out["crop_side"] = np.maximum(np.maximum(w, h), 32.0) * 1.3; out["det_conf"], out["det_iou"], out["det_miss"] = C, iou, miss
    out.to_csv(REPO / "outputs/data_manifests/exp124_det_test3.csv", index=False, float_format="%.9g")
    rows = [{"domain": d, "n": int(k.sum()), "miss": int(miss[k].sum()), "iou_mean": float(iou[k].mean()), "iou_median": float(np.median(iou[k])),
             "iou_lt_0.5": float((iou[k] < 0.5).mean()), "iou_lt_0.75": float((iou[k] < 0.75).mean()),
             "centre_err_px_mean": float(np.hypot(out.crop_cx[k] - t3.crop_cx[k], out.crop_cy[k] - t3.crop_cy[k]).mean()),
             "side_ratio_median": float(np.median(out.crop_side[k] / t3.crop_side[k]))}
            for d in ("synthetic", "lightbox", "sunlamp") for k in [(t3.domain == d).to_numpy()]]
    run.write_metrics(rows, "detection"); run.done(); print(pd.DataFrame(rows).round(3).to_string(index=False)); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
