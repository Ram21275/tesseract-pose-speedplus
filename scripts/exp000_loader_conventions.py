"""EXP-000: immutable loader and coordinate conventions.

Decides the rotation convention by reprojecting the 11 Tango keypoints with
both R and R^T and checking (a) brightness at projected keypoints (spacecraft
is bright against dark space), (b) agreement with an independent bbox table.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfpose import data as D  # noqa: E402
from tfpose import rotations as rot  # noqa: E402
from tfpose.runlog import Run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n-per-domain", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

paths = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs/paths.yaml"))
root = Path(paths["speedplus_root"])
run = Run("EXP-000", "loader_conventions", {"paths": paths, **vars(args)}, seed=args.seed)
try:
    rng = np.random.default_rng(args.seed)
    cam = D.load_camera(root)
    pts = D.load_tango_points(paths["tango_points"])
    split_rows = []
    all_recs = {}
    for (dom, sp) in D.SPLITS:
        recs = D.load_split(root, dom, sp)
        all_recs[(dom, sp)] = recs
        qs = np.stack([r["q_source"] for r in recs])
        n_missing = sum(not (root / r["image_relpath"]).exists() for r in recs)
        split_rows.append({"domain": dom, "split": sp, "n": len(recs), "missing_images": n_missing,
                           "q_norm_min": float(np.linalg.norm(qs, axis=1).min()),
                           "q_norm_max": float(np.linalg.norm(qs, axis=1).max()),
                           "frac_w_negative": float(np.mean(qs[:, 0] < 0)),
                           "tz_min_m": float(min(r["r"][2] for r in recs)),
                           "tz_max_m": float(max(r["r"][2] for r in recs))})
    names = [r["image_relpath"] for v in all_recs.values() for r in v]
    assert len(set(names)) == len(names), "duplicate image paths across splits"
    run.write_metrics(split_rows, "splits")

    # cross-check bbox table (derived elsewhere; used only for agreement)
    cc = pd.read_csv(paths["crosscheck_train_csv"], header=None, skipinitialspace=True)
    cc_box = {row[0]: (row[1], row[2], row[3], row[4]) for row in cc.itertuples(index=False)}

    conv_rows = []
    fig_dir = run.sub("figures")
    for dom, sp in [("synthetic", "train"), ("lightbox", "test"), ("sunlamp", "test")]:
        recs = all_recs[(dom, sp)]
        idx = rng.choice(len(recs), size=min(args.n_per_domain, len(recs)), replace=False)
        stats = {c: {"bright": [], "inside": [], "iou": []} for c in ["R", "RT"]}
        for j, i in enumerate(idx):
            r = recs[i]
            img = D.read_gray(root, r["image_relpath"])
            assert img.shape == (cam.height, cam.width), img.shape
            bg = float(np.median(img))
            for conv in ["R", "RT"]:
                uv = D.project_points(pts, r["q"], r["r"], cam, conv)
                ins = (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
                stats[conv]["inside"].append(ins.mean())
                if ins.any():
                    blur = cv2.GaussianBlur(img, (9, 9), 0)
                    v = blur[uv[ins, 1].astype(int), uv[ins, 0].astype(int)].astype(float)
                    stats[conv]["bright"].append(float(np.mean(v > bg + 20)))
                key = Path(r["image_relpath"]).name
                full = f"synthetic/images/{key}"
                if dom == "synthetic" and full in cc_box:
                    x0, x1, y0, y1 = cc_box[full]
                    a = (uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max())
                    b = (x0, y0, x1, y1)
                    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
                    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
                    inter = iw * ih
                    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
                    stats[conv]["iou"].append(inter / union if union > 0 else 0.0)
            if j < 4:  # overlays
                vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                for conv, col in [("R", (0, 255, 0)), ("RT", (0, 0, 255))]:
                    for p in D.project_points(pts, r["q"], r["r"], cam, conv):
                        cv2.circle(vis, (int(p[0]), int(p[1])), 8, col, 3)
                # body axes for the selected convention
                ax = D.project_points(np.array([[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]),
                                      r["q"], r["r"], cam, "R")
                for k, col in zip(range(1, 4), [(255, 0, 0), (0, 255, 255), (255, 0, 255)]):
                    cv2.line(vis, tuple(map(int, ax[0])), tuple(map(int, ax[k])), col, 3)
                cv2.imwrite(str(fig_dir / f"overlay_{dom}_{Path(r['image_relpath']).stem}.jpg"),
                            cv2.resize(vis, (960, 600)))
        for conv in ["R", "RT"]:
            s = stats[conv]
            conv_rows.append({"domain": dom, "convention": conv, "n_images": len(idx),
                              "kpt_inside_frac": float(np.mean(s["inside"])),
                              "kpt_on_bright_frac": float(np.mean(s["bright"])) if s["bright"] else float("nan"),
                              "bbox_iou_vs_crosscheck_mean": float(np.mean(s["iou"])) if s["iou"] else float("nan"),
                              "bbox_iou_vs_crosscheck_min": float(np.min(s["iou"])) if s["iou"] else float("nan")})
    run.write_metrics(conv_rows, "convention")

    # numerical convention tests (also covered by tests/test_geometry.py)
    q = rot.random_quats(10000, rng)
    tests = {
        "identity_ok": bool(np.allclose(rot.quat_to_matrix(np.array([1.0, 0, 0, 0])), np.eye(3))),
        "roundtrip_max_err_deg": float(np.degrees(np.max(rot.geodesic_distance(q, rot.matrix_to_quat(rot.quat_to_matrix(q)))))),
        "rodrigues_agree_max_abs": float(max(np.abs(rot.quat_to_matrix(rot.axis_angle_to_quat(a, t)) - cv2.Rodrigues(np.array(a, float) * t)[0]).max()
                                         for a in np.eye(3) for t in [np.pi / 2, np.pi, 2.0])),
    }
    run.write_metrics(tests, "metrics")
    run.done()
    print(pd.DataFrame(split_rows).to_string())
    print(pd.DataFrame(conv_rows).to_string())
    print(tests)
    print("RUN", run.rel())
except Exception as e:
    import traceback
    run.fail(traceback.format_exc())
    raise
