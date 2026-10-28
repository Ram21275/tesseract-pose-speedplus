"""EXP-124 data stage (CPU): SPEED+ detection dataset for the box detector.
Label = tight box of the 11 projected Tango keypoints (the box our GT square crops are built from: data.gt_square_box),
clipped to the image; one class. Images are written once at half resolution (960 x 600, grayscale JPEG q95) to
hdd2/RAM/speedplus_yolo/images/{train,val,lightbox,sunlamp}. Only train (synthetic/train) and val (synthetic/validation)
enter detector training and selection; lightbox / sunlamp labels exist only to measure detection IoU (test-only, DEC-006)."""
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import data as D  # noqa: E402

PATHS = yaml.safe_load(open(REPO / "configs/paths.yaml")); ROOT = Path(PATHS["speedplus_root"])
OUT = Path("/media/kavinder/hdd2/RAM/speedplus_yolo"); CAM = D.load_camera(ROOT); PTS = D.load_tango_points(PATHS["tango_points"])
SPLIT = {("synthetic", "train"): "train", ("synthetic", "validation"): "val", ("lightbox", "test"): "lightbox", ("sunlamp", "test"): "sunlamp"}


def one(r):
    sub = SPLIT[(r["domain"], r["split"])]; stem = r["image_relpath"].replace("/", "__").rsplit(".", 1)[0]
    uv = D.project_points(PTS, np.array([r[f"q_can_{k}"] for k in "wxyz"]), np.array([r["t_x"], r["t_y"], r["t_z"]]), CAM)
    x0, y0 = np.clip(uv.min(0), 0, [CAM.width, CAM.height]); x1, y1 = np.clip(uv.max(0), 0, [CAM.width, CAM.height])
    img = OUT / "images" / sub / f"{stem}.jpg"
    if not img.exists():
        g = D.read_gray(ROOT, r["image_relpath"]); cv2.imwrite(str(img), cv2.resize(g, (CAM.width // 2, CAM.height // 2), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95])
    (OUT / "labels" / sub / f"{stem}.txt").write_text(f"0 {(x0 + x1) / 2 / CAM.width:.6f} {(y0 + y1) / 2 / CAM.height:.6f} {(x1 - x0) / CAM.width:.6f} {(y1 - y0) / CAM.height:.6f}\n")
    return r["image_relpath"], sub, f"{stem}.jpg", x0, y0, x1, y1


if __name__ == "__main__":
    df = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv")
    for s in SPLIT.values():
        (OUT / "images" / s).mkdir(parents=True, exist_ok=True); (OUT / "labels" / s).mkdir(parents=True, exist_ok=True)
    with Pool(6) as p:
        rows = p.map(one, df.to_dict("records"), chunksize=256)
    pd.DataFrame(rows, columns=["image_relpath", "split", "file", "x0", "y0", "x1", "y1"]).to_csv(OUT / "boxes.csv", index=False)
    (OUT / "data.yaml").write_text(f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n  0: tango\n")
    print(pd.DataFrame(rows, columns=["a", "split", "f", "x0", "y0", "x1", "y1"]).groupby("split").size())
