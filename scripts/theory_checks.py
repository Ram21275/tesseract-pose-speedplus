"""CPU checks of the theory supplement on saved EXP-131 predictions (depth 5, DINOv3; no GPU, no training).
(a) per-view quality: error of each rolled view's top beam candidate (mapped back), view 0 vs views 1-3 (orbit remark);
(b) level at which wrong-mode predictions leave the GT path (selection is decided at the top of the tree);
(c) canonical-label boundary: fraction of GT poses whose two closest-to-identity symmetric copies are within 5 deg;
(d) viewing-ray angle and ray-rotation distance between predicted and GT translation (allocentric-conversion bound);
(e) geometric null for (b): the GT rotated by the same error angle about a random axis;
(f) greedy truncation: errors of the level-l prefix of the depth-5 greedy leaf (exactly the greedy level-l decision), l = 1..5;
(g) exact largest centre-to-corner cell radius r_L, L = 1..7; (h) canonical-label non-equivariance under a 90 deg roll.
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tfpose import tesseract as T  # noqa: E402
from tfpose.rotations import matrix_to_quat, quat_multiply, normalize, random_quats, axis_angle_to_quat  # noqa: E402
from tfpose.allo import ray_rot  # noqa: E402

RUNS = {"tlessall": "outputs/experiments/EXP-131_all_objects/RUN-20261119-200618-seed0/predictions",
        "ycbvall": "outputs/experiments/EXP-131_all_objects/RUN-20261119-221129-seed0/predictions"}
L = 5


def ang(a, b):
    return np.degrees(2 * np.arccos(np.clip(np.abs((a * b).sum(-1)), 0, 1)))


def nearest_copy(qh, qg, sq):
    qs = np.stack([quat_multiply(qg, np.broadcast_to(s, qg.shape)) for s in sq], 1)   # (N,S,4)
    d = ang(qh[:, None], qs); i = d.argmin(1)
    return qs[np.arange(len(qg)), i], d.min(1)


def first_div(qa, qb):
    c1, p1 = T.encode(qa, L); c2, p2 = T.encode(qb, L)
    same = np.concatenate([(c1 == c2)[:, None], p1 == p2], 1)                    # level 0 = chart
    return np.where(same.all(1), L + 1, np.argmin(same, 1))


def expv(axis, deg):
    a = np.deg2rad(deg)[:, None]
    return np.concatenate([np.cos(a / 2), np.sin(a / 2) * axis], 1)


rng = np.random.default_rng(0)
out = {}
for ds, pdir in RUNS.items():
    R = {"view_err": {s: [[] for _ in range(4)] for s in ("validation", "test")}, "diverge": [], "null": [], "canon": [], "ray": [],
         "raydist": [], "trunc": {l: [] for l in range(1, L + 1)}}
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
            R["diverge"].append(first_div(qh[w], qn[w]))
            ax = rng.normal(size=(w.sum(), 3)); ax /= np.linalg.norm(ax, axis=1, keepdims=True)
            R["null"].append(first_div(normalize(quat_multiply(qn[w], expv(ax, d[w]))), qn[w]))
        cg, pg = T.encode(z["q_greedy"][m], L)
        for l in range(1, L + 1):
            _, dl = nearest_copy(T.decode(cg, pg[:, :l]), qg[m], sq); R["trunc"][l].append(dl)
        if len(sq) > 1 and "_seed0_" in f:
            qs = np.stack([quat_multiply(qg[m], np.broadcast_to(s, qg[m].shape)) for s in sq], 1)
            di = np.sort(np.degrees(2 * np.arccos(np.clip(np.abs(qs[..., 0]), 0, 1))), 1)
            R["canon"].append(di[:, 1] - di[:, 0] < 5.0)
        if "_seed0_" in f:
            a, b = z["t_used"][m], z["t_gt"][m]
            R["ray"].append(np.degrees(np.arccos(np.clip((a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1), -1, 1))))
            M = np.einsum("nij,nkj->nik", ray_rot(a), ray_rot(b))
            R["raydist"].append(np.degrees(np.arccos(np.clip((np.trace(M, axis1=1, axis2=2) - 1) / 2, -1, 1))))
    r = {}
    for s in ("validation", "test"):
        for k in range(4):
            e = np.concatenate(R["view_err"][s][k]); r[f"{s}_view{k}_mean"] = float(e.mean()); r[f"{s}_view{k}_R10"] = float((e <= 10).mean())
    dv = np.concatenate(R["diverge"]); r["wrong_n"] = int(len(dv))
    r["wrong_first_diverge_level_frac"] = {int(l): float((dv == l).mean()) for l in range(L + 1)}
    nl = np.concatenate(R["null"]); r["null_first_diverge_level_frac"] = {int(l): float((nl == l).mean()) for l in range(L + 2)}
    r["greedy_truncation_test"] = {l: {"mean": float(np.concatenate(R["trunc"][l]).mean()),
                                       **{f"R@{t}": float((np.concatenate(R["trunc"][l]) <= t).mean()) for t in (2, 5, 10)}} for l in range(1, L + 1)}
    rd = np.concatenate(R["raydist"]); r["ray_rot_dist_deg"] = {"mean": float(rd.mean()), "median": float(np.median(rd)),
                                                               "p95": float(np.percentile(rd, 95)), "max": float(rd.max())}
    if R["canon"]:
        c = np.concatenate(R["canon"]); r["canon_boundary_frac_sym_objects"] = float(c.mean())
    ray = np.concatenate(R["ray"]); r["ray_angle_deg"] = {"median": float(np.median(ray)), "p95": float(np.percentile(ray, 95)), "max": float(ray.max())}
    out[ds] = r
    print(ds, json.dumps(r, indent=1), flush=True)
# (g) exact cell radii
rad = {}
for Ld in range(1, 8):
    n = 4 * 8 ** Ld; mx = 0.0
    for s0 in range(0, n, 1 << 20):
        c, p = T.leaf_to_path(np.arange(s0, min(n, s0 + (1 << 20))), Ld)
        mx = max(mx, float(ang(T.cell_corners(c, p), T.decode(c, p)[:, None]).max()))
    rad[Ld] = mx
out["cell_radius_exact_deg"] = rad; print("r_L", rad, flush=True)
# (h) canonical labels under a 90 deg roll
q = random_quats(200000, rng); ax = np.array([0.1, 0.2, 1.0]); ax /= np.linalg.norm(ax)
r90 = normalize(axis_angle_to_quat(ax, np.pi / 2)[None])[0]
def kappa(q, S):
    qs = np.stack([quat_multiply(q, np.broadcast_to(s, q.shape)) for s in S], 1)
    return qs[np.arange(len(q)), np.abs(qs[..., 0]).argmax(1)]
ne = {}
for nm, k in (("C2", 2), ("C4", 4)):
    S = [normalize(axis_angle_to_quat(np.array([0, 0, 1.0]), 2 * np.pi * i / k)[None])[0] for i in range(k)]
    a = kappa(quat_multiply(np.broadcast_to(r90, q.shape), q), S); b = quat_multiply(np.broadcast_to(r90, q.shape), kappa(q, S))
    ne[nm] = float((np.abs((a * b).sum(1)) < 1 - 1e-9).mean())
out["canonical_nonequivariant_frac_roll90"] = ne; print("canonical", ne, flush=True)
os.makedirs("outputs/theory", exist_ok=True)
json.dump(out, open("outputs/theory/theory_checks.json", "w"), indent=1)
