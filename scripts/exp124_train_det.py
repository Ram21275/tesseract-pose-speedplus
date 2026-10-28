"""EXP-124 detector: Ultralytics YOLO11s (COCO-pretrained), one class (Tango keypoint box), trained on SPEED+ synthetic/train
only; checkpoint = best synthetic/validation fitness (Ultralytics default 0.1 mAP50 + 0.9 mAP50-95). Settings fixed in the
pre-registration: imgsz 960 (images stored at 960 x 600), 20 epochs, batch 32, seed 0, flips lr/ud 0.5, no rotation /
shear, mosaic with close_mosaic 5, brightness jitter hsv_v 0.6 (images are grey: hsv_h = hsv_s = 0); Ultralytics'
default albumentations (blur, median blur, CLAHE, to-grey, p = 0.01 each). Outputs on hdd2/RAM/exp124_det."""
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose.runlog import Run  # noqa: E402

CFG = dict(data="/media/kavinder/hdd2/RAM/speedplus_yolo/data.yaml", imgsz=960, epochs=20, batch=32, workers=6, seed=0, deterministic=True,
           fliplr=0.5, flipud=0.5, degrees=0.0, shear=0.0, mosaic=1.0, close_mosaic=5, hsv_h=0.0, hsv_s=0.0, hsv_v=0.6,
           project="/media/kavinder/hdd2/RAM/exp124_det", name="yolo11s_960", exist_ok=True, plots=False)
run = Run("EXP-124", "speedplus_detected_official_score", {"stage": "detector", **CFG}, seed=0)
try:
    from ultralytics import YOLO
    m = YOLO("/media/kavinder/hdd2/RAM/weights/yolo11s.pt"); res = m.train(**CFG)
    best = Path(CFG["project"]) / CFG["name"] / "weights/best.pt"; shutil.copy(best, run.sub("checkpoints") / "best.pt")
    shutil.copy(Path(CFG["project"]) / CFG["name"] / "results.csv", run.sub("logs") / "results.csv")
    run.done(map50_95=float(res.box.map), map50=float(res.box.map50)); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
