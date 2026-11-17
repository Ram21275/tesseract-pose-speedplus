"""EXP-131 data stage (CPU only): YCB-V symmetry sets for the 21 objects, in the frame of the original YCB-Video models
(points.xyz, metres), taken from the BOP YCB-V models_eval/models_info.json (BOP frame, mm).

The BOP models are the original models shifted to their bounding-box centre (axes unchanged), so p_orig = p_bop + c (mm).
c is estimated per object on the dense original mesh vertices (textured.obj; points.xyz is too sparse to resolve a few degrees):
bbox-centre start, then translation-only nearest-neighbour refinement. Gate 1 (frame): a full rigid ICP (rotation free) from the
same start finds a residual rotation < 0.5 deg and the shift-only fit is within 1.5x the cloud's own sampling floor (NN distance
between two halves of the vertex set). A BOP symmetry (R, t) becomes (R, t + c - R c).
Gate 2: every symmetry maps the original vertices onto themselves within max(2x sampling floor, 1% of the diameter).
Writes outputs/data_manifests/ycbv_exp131_symmetries.npz:
  <obj>          (S,3,3) rotations for the labels (identity first; continuous symmetries in 36 steps, as T-LESS / EXP-106),
  <obj>__T4_mm   (S',4,4) full transforms in the original frame (mm; continuous in 315 steps, the BOP toolkit rule) for MSSD/MSPD.
Also writes ycbv_exp131_symmetries_checks.json."""
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import xobj as XO  # noqa: E402

PATHS = yaml.safe_load(open(REPO / "configs/paths.yaml"))
Y = Path(PATHS["ycbv_root"]); B = Path(PATHS["tless_root"]).parent / "ycbv"
CLASSES = (Y / "image_sets/classes.txt").read_text().split()
info = json.load(open(B / "models_eval/models_info.json"))


def obj_vertices(path):
    return np.array([l.split()[1:4] for l in open(path) if l.startswith("v ")], float)


def rodrigues(a, th):
    a = np.asarray(a, float); a = a / np.linalg.norm(a); K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def transforms(m, steps):
    """BOP toolkit get_symmetry_transformations: discrete x continuous, 4x4 in the BOP frame (mm)."""
    disc = [np.eye(4)] + [np.array(s, float).reshape(4, 4) for s in m.get("symmetries_discrete", [])]
    cont = [np.eye(4)]
    for c in m.get("symmetries_continuous", []):
        o = np.array(c["offset"], float); cont = []
        for k in range(steps):
            R = rodrigues(c["axis"], 2 * np.pi * k / steps); T = np.eye(4); T[:3, :3] = R; T[:3, 3] = o - R @ o; cont.append(T)
    return [d @ c for d in disc for c in cont]


out, checks, ok = {}, {}, True
for i, o in enumerate(CLASSES, 1):
    P = obj_vertices(Y / "models" / o / "textured.obj") * 1000.0
    Q = XO._ply_vertices(B / f"models_eval/obj_{i:06d}.ply")
    h = np.random.default_rng(0).permutation(len(P)); floor = float(cKDTree(P[h[::2]]).query(P[h[1::2]])[0].mean())
    c = (P.min(0) + P.max(0)) / 2 - (Q.min(0) + Q.max(0)) / 2; tP = cKDTree(P)
    for _ in range(15):                                                       # translation-only refinement
        d, j = tP.query(Q + c); c = c + np.mean(P[j] - (Q + c), 0)
    fit = float(tP.query(Q + c)[0].mean()); dia = info[str(i)]["diameter"]
    Rr, tr = np.eye(3), c.copy()                                              # rigid ICP (rotation free), Kabsch per step
    for _ in range(30):
        X = Q @ Rr.T + tr; _, j = tP.query(X); A, Bm = Q, P[j]; ma, mb = A.mean(0), Bm.mean(0)
        U, _, Vt = np.linalg.svd((A - ma).T @ (Bm - mb)); Dm = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        Rr = Vt.T @ Dm @ U.T; tr = mb - Rr @ ma
    rot_res = float(np.degrees(np.arccos(np.clip((np.trace(Rr) - 1) / 2, -1, 1))))
    m = info[str(i)]; out_sym = []
    for steps, key in ((36, o), (315, f"{o}__T4_mm")):
        Ts = []
        for T in transforms(m, steps):
            T2 = T.copy(); T2[:3, 3] = T[:3, 3] + c - T[:3, :3] @ c; Ts.append(T2)
        Ts = np.stack(Ts); out[key] = Ts[:, :3, :3] if steps == 36 else Ts
        if steps == 315:
            Ps = P[np.random.default_rng(0).choice(len(P), min(5000, len(P)), replace=False)]
            out_sym = [float(tP.query(Ps @ T[:3, :3].T + T[:3, 3])[0].mean()) for T in Ts[1:]]
    worst = max(out_sym) if out_sym else 0.0
    g1, g2 = (rot_res < 0.5) and (fit < 1.5 * floor), worst < max(2 * floor, 0.01 * dia)
    ok &= g1 and g2
    checks[o] = dict(bop_id=i, offset_mm=c.round(3).tolist(), shift_fit_mm=round(fit, 3), sampling_floor_mm=round(floor, 3), icp_residual_rot_deg=round(rot_res, 3), n_sym_labels=len(out[o]), n_sym_metric=len(out[f"{o}__T4_mm"]),
                     worst_sym_nn_mm=round(worst, 3), diameter_mm=dia, gate_shift=bool(g1), gate_sym=bool(g2))
    print(o, checks[o], flush=True)
np.savez(REPO / "outputs/data_manifests/ycbv_exp131_symmetries.npz", **out)
(REPO / "outputs/data_manifests/ycbv_exp131_symmetries_checks.json").write_text(json.dumps(dict(all_pass=bool(ok), objects=checks), indent=1))
print("ALL GATES PASS" if ok else "GATE FAILED"); sys.exit(0 if ok else 1)
