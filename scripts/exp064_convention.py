"""EXP-064 (DEC-018 amendment): derive and verify the in-plane label transform for each dataset before any training.
For 300 validation rows per dataset and theta in {90, 37, -120, random}: project the 3-D model points with the GT pose and
the real camera, push them through the image warp the extractor applies (SYMSOL-T: rotation about the principal point; T-LESS / YCB-V: the exact roll homography); project the
rolled pose (xobj.roll_pose) and push through the plain crop affine; compare in crop pixels (224 px). Exact label error by
PnP. Also the target-convention round trip (allocentric datasets). Pass: label error p95 < 0.5 deg, round trip < 0.01 deg."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import xobj as XO  # noqa: E402
from tfpose.rotations import quat_to_matrix  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

run = Run("EXP-064", "broader_object_validation", {"stage": "convention"}, seed=0)
OUT = 224
try:
    rows_out, verdict = [], {}
    for ds in ("tless", "ycbv", "symsolt"):
        df = XO.manifest(ds); sel = df[df.split == "validation"].sample(300, random_state=0)
        K = XO.intrinsics(ds, sel); q, t = XO.pose_ego(ds, sel); d = XO.crop_ray(K, sel)
        objc = XO.DATASETS[ds]["obj_col"]; pts_cache = {}
        rng = np.random.default_rng(0); errs, lab, rt = [], [], []
        for i, (_, r) in enumerate(sel.iterrows()):
            key = r[objc] if ds != "symsolt" else (r[objc], r.image_relpath.split("/")[0])
            if key not in pts_cache:
                pts_cache[key] = XO.model_points(ds, r[objc], r)
            P = pts_cache[key]
            s = OUT / r.crop_side
            C = np.array([[s, 0, OUT / 2 - s * r.crop_cx], [0, s, OUT / 2 - s * r.crop_cy]])
            for th in (90.0, 37.0, -120.0, float(rng.uniform(-180, 180))):
                R = quat_to_matrix(q[i]); uv, _ = cv2.projectPoints(P.reshape(-1, 1, 3), cv2.Rodrigues(R)[0], t[i].reshape(3, 1), K[i], np.zeros(5))
                if ds == "symsolt":                          # affine rotation about the principal point (exact here)
                    Rr = cv2.getRotationMatrix2D((OUT / 2, OUT / 2), th, 1.0)
                    M = np.concatenate([Rr[:, :2] @ C[:, :2], (Rr[:, :2] @ C[:, 2] + Rr[:, 2])[:, None]], 1)
                    tgt = uv.reshape(-1, 2) @ M[:, :2].T + M[:, 2]
                else:                                        # the extractor's exact roll homography (inplane.crop_rolled_homography)
                    dd = d[i]; thr = -np.radians(th); Kx = np.array([[0, -dd[2], dd[1]], [dd[2], 0, -dd[0]], [-dd[1], dd[0], 0]])
                    Rs = np.eye(3) + np.sin(thr) * Kx + (1 - np.cos(thr)) * Kx @ Kx
                    Mh = np.vstack([C, [0, 0, 1]]) @ K[i] @ Rs @ np.linalg.inv(K[i])
                    hp = np.concatenate([uv.reshape(-1, 2), np.ones((len(P), 1))], 1) @ Mh.T; tgt = hp[:, :2] / hp[:, 2:]
                q2, t2 = XO.roll_pose(q[i][None], t[i][None], th, d[i][None])
                uv2, _ = cv2.projectPoints(P.reshape(-1, 1, 3), cv2.Rodrigues(quat_to_matrix(q2[0]))[0], t2[0].reshape(3, 1), K[i], np.zeros(5))
                errs.append(float(np.linalg.norm(uv2.reshape(-1, 2) @ C[:, :2].T + C[:, 2] - tgt, axis=1).max()))
                Cinv = np.linalg.inv(np.vstack([C, [0, 0, 1]]))[:2]
                img = (tgt @ Cinv[:, :2].T + Cinv[:, 2]).reshape(-1, 1, 2)
                ok, rv, tv = cv2.solvePnP(P.reshape(-1, 1, 3), img, K[i], np.zeros(5), rvec=cv2.Rodrigues(quat_to_matrix(q2[0]))[0],
                                          tvec=t2[0].reshape(3, 1), useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
                Rp = cv2.Rodrigues(rv)[0]
                lab.append(float(np.degrees(np.arccos(np.clip((np.trace(Rp.T @ quat_to_matrix(q2[0])) - 1) / 2, -1, 1)))))
                qt = XO.roll_target(ds, q[i][None], t[i][None], th, d[i][None]); back = XO.from_target(ds, qt, t2)
                rt.append(float(np.degrees(2 * np.arccos(min(1.0, abs(float(back[0] @ q2[0])))))))
        e, l, r_ = np.array(errs), np.array(lab), np.array(rt)
        ok = bool(np.percentile(l, 95) < 0.5 and r_.max() < 0.01)
        verdict[ds] = ok
        rows_out.append({"dataset": ds, "n_checks": len(e), "px_err_mean": e.mean(), "px_err_p95": np.percentile(e, 95), "px_err_max": e.max(),
                         "label_err_deg_mean": l.mean(), "label_err_deg_p95": np.percentile(l, 95), "label_err_deg_max": l.max(),
                         "target_roundtrip_deg_max": r_.max(), "verified": ok})
        print(rows_out[-1], flush=True)
    run.write_metrics(rows_out, "convention"); (run.sub("tables") / "convention.md").write_text(md_table(rows_out))
    (REPO / "outputs/experiments/exp064_convention.json").write_text(json.dumps(verdict, indent=2))
    run.done(**{f"verified_{k}": v for k, v in verdict.items()}); print(md_table(rows_out)); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
