"""EXP-109 amendment (review point 4): derive the in-plane-rotation label transform from geometry, not from a predictor.

Derivation (pose convention from EXP-000: p_cam = R(q) p_body + r; pixel u = K * distort(p_cam / z); y axis down):
  the crop warp rotates image content by theta about the crop centre c (cv2.getRotationMatrix2D, theta > 0 = counter-
  clockwise on screen). A camera-frame rotation R_s about the unit ray d through c maps c to itself and, for fx = fy and
  near c, rotates the image about c. With y down, R_s = Rot(d, phi) appears on screen as a clockwise rotation by phi,
  so matching the warp needs phi = -theta:  p_cam' = Rot(d, -theta) p_cam  =>  R' = Rot(d, -theta) R,  r' = Rot(d, -theta) r,
  i.e. q' = Exp(-theta d) (x) q  (inplane.MODES "pre-").
Verification (no model involved): project the 11 Tango points with (q, r) and the real distorted camera, push them through
the exact crop+rotation affine; project with (q', r') and push through the plain crop affine; compare in crop pixels, for
all 4 candidate transforms. Also: round trip q -> q' -> q, and the ray computed from the undistorted crop centre.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import data as D, inplane as IP  # noqa: E402
from tfpose.rotations import quat_to_matrix  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

run = Run("EXP-109", "real_domain_aug_tokens16_orbit", {"stage": "convention_geometry"}, seed=0)
try:
    paths = yaml.safe_load(open(REPO / "configs/paths.yaml")); root = Path(paths["speedplus_root"])
    cam = D.load_camera(root); pts = D.load_tango_points(paths["tango_points"])
    g = pd.read_csv(REPO / "outputs/data_manifests/exp109_gate512.csv")
    OUT = 224
    rng = np.random.default_rng(0)
    rows = []
    for i, r in g.iterrows():
        q = np.array([r.q_can_w, r.q_can_x, r.q_can_y, r.q_can_z]); t = np.array([r.t_x, r.t_y, r.t_z])
        d = IP.ray_undistorted(cam.K, cam.dist, r.crop_cx, r.crop_cy)[0]
        for th in (90.0, 37.0, -120.0, float(rng.uniform(-180, 180))):
            s = OUT / r.crop_side
            C = np.array([[s, 0, OUT / 2 - s * r.crop_cx], [0, s, OUT / 2 - s * r.crop_cy]])
            Rr = cv2.getRotationMatrix2D((OUT / 2, OUT / 2), th, 1.0)
            Mrot = np.concatenate([Rr[:, :2] @ C[:, :2], (Rr[:, :2] @ C[:, 2] + Rr[:, 2])[:, None]], 1)
            uv = D.project_points(pts, q, t, cam)
            target = uv @ Mrot[:, :2].T + Mrot[:, 2]                                   # warped image positions (crop px)
            for m in IP.MODES:
                q2 = IP.rotate_label(q[None], th, d[None], m)[0]
                sgn = -1.0 if m in ("pre-", "post-") else 1.0
                Rs = cv2.Rodrigues(sgn * np.radians(th) * d)[0]
                t2 = Rs @ t if m.startswith("pre") else t                           # pre: whole camera frame rotates
                uv2 = D.project_points(pts, q2, t2, cam) @ C[:, :2].T + C[:, 2]
                rows.append({"row": i, "theta": th, "mode": m, "err_px": float(np.linalg.norm(uv2 - target, axis=1).max())})
            # exact label error of the derived transform: PnP pose that reproduces the warped points (in the plain-crop image)
            Cinv = np.linalg.inv(np.vstack([C, [0, 0, 1]]))[:2]
            img_pts = (target @ Cinv[:, :2].T + Cinv[:, 2]).reshape(-1, 1, 2)
            q2 = IP.rotate_label(q[None], th, d[None], "pre-")[0]
            Rs = cv2.Rodrigues(-np.radians(th) * d)[0]
            ok_, rv, tv = cv2.solvePnP(pts.reshape(-1, 1, 3), img_pts, cam.K, cam.dist, rvec=cv2.Rodrigues(quat_to_matrix(q2))[0],
                                       tvec=(Rs @ t).reshape(3, 1), useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            Rp = cv2.Rodrigues(rv)[0]; Rd = quat_to_matrix(q2)
            ang = np.degrees(np.arccos(np.clip((np.trace(Rp.T @ Rd) - 1) / 2, -1, 1)))
            rows.append({"row": i, "theta": th, "mode": "label_err_deg(pre-, PnP)", "err_px": float(ang)})
            back = IP.rotate_label(IP.rotate_label(q[None], th, d[None], "pre-"), -th, d[None], "pre-")[0]
            rows.append({"row": i, "theta": th, "mode": "roundtrip(pre-)", "err_px": float(np.degrees(2 * np.arccos(min(1.0, abs(back @ q)))))})
    df = pd.DataFrame(rows)
    summ = df.groupby("mode").err_px.agg(["mean", "median", "max", lambda x: np.percentile(x, 95)]).reset_index()
    summ.columns = ["mode", "mean", "median", "max", "p95"]
    run.write_metrics(summ.to_dict("records"), "geometry_check"); (run.sub("tables") / "geometry_check.md").write_text(md_table(summ.to_dict("records")))
    print(md_table(summ.to_dict("records")))
    print("(err_px = max over the 11 Tango points of the crop-pixel distance at 224 px; roundtrip row = degrees)")
    sm = summ.set_index("mode")
    # pass rule (amendment 2026-10-15, fixed before this measurement): label error from the affine-warp approximation
    # p95 < 0.5 deg (well below the 2.5 deg leaf quantization), round trip < 0.01 deg, and pre- clearly best of the 4.
    ok = (sm.loc["label_err_deg(pre-, PnP)", "p95"] < 0.5 and sm.loc["roundtrip(pre-)", "max"] < 0.01
          and sm.loc["pre-", "median"] < 0.1 * min(sm.loc[m, "median"] for m in ("pre+", "post+", "post-")))
    (REPO / "outputs/experiments/exp109_convention.txt").write_text(("pre-" if ok else "NONE") + "\n")
    run.done(derived="pre-", verified=bool(ok)); print("DERIVED pre- VERIFIED", ok); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
