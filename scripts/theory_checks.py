"""CPU checks of the theory supplement on saved EXP-131 predictions (depth 5, DINOv3; no GPU, no training).
(a) per-view quality: error of each rolled view's top beam candidate (mapped back), view 0 vs views 1-3 (orbit remark);
(b) level at which wrong-mode predictions leave the GT path (selection is decided at the top of the tree);
(c) canonical-label boundary: fraction of GT poses whose two closest-to-identity symmetric copies are within 5 deg;
(d) viewing-ray angle between predicted and GT translation (allocentric-conversion bound).
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tfpose import tesseract as T  # noqa: E402
from tfpose.rotations import matrix_to_quat, quat_multiply, normalize  # noqa: E402

RUNS = {"tlessall": "outputs/experiments/EXP-131_all_objects/RUN-20261119-200618-seed0/predictions",
        "ycbvall": "outputs/experiments/EXP-131_all_objects/RUN-20261119-221129-seed0/predictions"}
L = 5


def ang(a, b):
    return np.degrees(2 * np.arccos(np.clip(np.abs((a * b).sum(-1)), 0, 1)))


def nearest_copy(qh, qg, sq):
    qs = np.stack([quat_multiply(qg, np.broadcast_to(s, qg.shape)) for s in sq], 1)   # (N,S,4)
    d = ang(qh[:, None], qs); i = d.argmin(1)
    return qs[np.arange(len(qg)), i], d.min(1)


out = {}
for ds, pdir in RUNS.items():
    R = {"view_err": {s: [[] for _ in range(4)] for s in ("validation", "test")}, "diverge": [], "canon": [], "ray": []}
    for f in sorted(glob.glob(f"{pdir}/{ds}_*_seed*_DINO.npz")):
        z = np.load(f, allow_pickle=True); sq = normalize(np.stack([matrix_to_quat(s) for s in z["sym"]]))
        qg = z["q_tgt_gt"]
        for s in ("validation", "test"):
            m = z["split"] == s
            for k in range(4):
                _, d = nearest_copy(z["cand"][m, 8 * k], qg[m], sq); R["view_err"][s][k].append(d)
        m = z["split"] == "test"; qh = z["q_orbit_product" if ds == "tlessall" else "q_greedy"][m]
        qn, d = nearest_copy(qh, qg[m], sq); w = d > 30
        if w.any():
            c1, p1 = T.encode(qh[w], L); c2, p2 = T.encode(qn[w], L)
            same = np.concatenate([(c1 == c2)[:, None], p1 == p2], 1)            # level 0 = chart
            R["diverge"].append(np.argmin(np.cumprod(same, 1), 1))               # first differing level
        if len(sq) > 1 and "_seed0_" in f:
            qs = np.stack([quat_multiply(qg[m], np.broadcast_to(s, qg[m].shape)) for s in sq], 1)
            di = np.sort(np.degrees(2 * np.arccos(np.clip(np.abs(qs[..., 0]), 0, 1))), 1)
            R["canon"].append(di[:, 1] - di[:, 0] < 5.0)
        if "_seed0_" in f:
            a, b = z["t_used"][m], z["t_gt"][m]
            R["ray"].append(np.degrees(np.arccos(np.clip((a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1), -1, 1))))
    r = {}
    for s in ("validation", "test"):
        for k in range(4):
            e = np.concatenate(R["view_err"][s][k]); r[f"{s}_view{k}_mean"] = float(e.mean()); r[f"{s}_view{k}_R10"] = float((e <= 10).mean())
    dv = np.concatenate(R["diverge"]); r["wrong_n"] = int(len(dv))
    r["wrong_first_diverge_level_frac"] = {int(l): float((dv == l).mean()) for l in range(L + 1)}
    if R["canon"]:
        c = np.concatenate(R["canon"]); r["canon_boundary_frac_sym_objects"] = float(c.mean())
    ray = np.concatenate(R["ray"]); r["ray_angle_deg"] = {"median": float(np.median(ray)), "p95": float(np.percentile(ray, 95)), "max": float(ray.max())}
    out[ds] = r
    print(ds, json.dumps(r, indent=1), flush=True)
os.makedirs("outputs/theory", exist_ok=True)
json.dump(out, open("outputs/theory/theory_checks.json", "w"), indent=1)
