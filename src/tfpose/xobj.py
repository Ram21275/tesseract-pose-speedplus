"""EXP-064 dataset adapters (T-LESS, YCB-Video, SYMSOL-T) for the locked recipe (DEC-018).

Label conventions for the in-plane (camera-roll) augmentation and orbit voting, derived like EXP-109:
  the crop warp rotates image content by theta about the crop centre c; this equals a camera rotation Rot(d, -theta) about
  the unit ray d through c (pinhole, no distortion in these datasets), so p_cam' = Rot(d, -theta) p_cam:
    egocentric: R' = Rot(d, -theta) R,  t' = Rot(d, -theta) t
    allocentric (T-LESS, YCB-V targets): R_allo' = R_ray(t')^T R'  (exact, recomputed from t')
  SYMSOL-T uses the full 224 frame (crop centre = principal point, d = optical axis) and egocentric targets.
Verified by projecting each dataset's 3-D model points (scripts/exp064_convention.py)."""
from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .allo import mat_to_quat, ray_rot
from .rotations import quat_to_matrix

REPO = Path(__file__).resolve().parents[2]
PATHS = yaml.safe_load(open(REPO / "configs/paths.yaml"))
SYM_Z = 5.25 / np.tan(np.radians(30.0))                 # SYMSOL-T camera distance (LiePose synth.py, hfov 60, x 5.25)
SYM_F = (224 - 112) * SYM_Z / 5.25

DATASETS = {
    "tless": dict(manifest="outputs/data_manifests/tless_exp106.csv", root_key="tless_root", key_cols="obj_id,inst_id",
                  obj_col="obj_id", label="allo", sym="outputs/data_manifests/tless_exp106_symmetries.npz"),
    "ycbv": dict(manifest="outputs/data_manifests/ycbv_exp064.csv", root_key="ycbv_root", key_cols="obj,inst_id",
                 obj_col="obj", label="allo", sym=None),
    "symsolt": dict(manifest="outputs/data_manifests/symsolt_exp064.csv", root_key="symsolt_root", key_cols="",
                    obj_col="shape", label="ego", sym="outputs/data_manifests/symsolt_exp064_symmetries.npz"),
}


def manifest(ds: str) -> pd.DataFrame:
    return pd.read_csv(REPO / DATASETS[ds]["manifest"])


def intrinsics(ds: str, df: pd.DataFrame) -> np.ndarray:
    """(N,3,3) camera matrices."""
    N = len(df); K = np.zeros((N, 3, 3)); K[:, 2, 2] = 1
    if ds == "symsolt":
        K[:, 0, 0] = K[:, 1, 1] = SYM_F; K[:, 0, 2] = K[:, 1, 2] = 112.0
    else:
        K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2] = df.cam_fx, df.cam_fy, df.cam_cx, df.cam_cy
    return K


def pose_ego(ds: str, df: pd.DataFrame):
    """Egocentric (q wxyz, t camera frame) per row."""
    if ds == "symsolt":
        q = df[[f"q_{k}" for k in "wxyz"]].to_numpy()
        t = df[[f"t_{k}" for k in "xyz"]].to_numpy() + np.array([0, 0, SYM_Z])
    else:
        q = df[[f"q_ego_{k}" for k in "wxyz"]].to_numpy(); t = df[[f"t_{k}" for k in "xyz"]].to_numpy()
    return q / np.linalg.norm(q, axis=1, keepdims=True), t


def target(ds: str, df: pd.DataFrame) -> np.ndarray:
    """The rotation the tree predicts: allocentric for T-LESS / YCB-V, egocentric for SYMSOL-T."""
    if DATASETS[ds]["label"] == "allo":
        return df[[f"q_allo_{k}" for k in "wxyz"]].to_numpy()
    return pose_ego(ds, df)[0]


def crop_ray(K: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    c = np.stack([df.crop_cx.to_numpy(), df.crop_cy.to_numpy(), np.ones(len(df))], 1)
    d = np.einsum("nij,nj->ni", np.linalg.inv(K), c)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _rotvec_mat(axis: np.ndarray, ang: np.ndarray) -> np.ndarray:
    a = axis / np.linalg.norm(axis, axis=1, keepdims=True); c, s = np.cos(ang), np.sin(ang)
    Kx = np.zeros((len(a), 3, 3)); Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 2] = -a[:, 2], a[:, 1], -a[:, 0]
    Kx[:, 1, 0], Kx[:, 2, 0], Kx[:, 2, 1] = a[:, 2], -a[:, 1], a[:, 0]
    return np.eye(3)[None] + s[:, None, None] * Kx + (1 - c)[:, None, None] * Kx @ Kx


def roll_pose(q_ego, t, theta_deg, d):
    """Egocentric pose after an in-plane crop rotation by theta (cv2 sign) about the crop centre ray d."""
    Rs = _rotvec_mat(d, -np.radians(np.broadcast_to(theta_deg, (len(q_ego),))))
    R2 = Rs @ quat_to_matrix(q_ego); t2 = np.einsum("nij,nj->ni", Rs, t)
    return mat_to_quat(R2), t2


def roll_target(ds: str, q_ego, t, theta_deg, d) -> np.ndarray:
    """Training / orbit label after an in-plane rotation by theta, in the dataset's target convention."""
    q2, t2 = roll_pose(q_ego, t, theta_deg, d)
    if DATASETS[ds]["label"] == "allo":
        return mat_to_quat(np.transpose(ray_rot(t2), (0, 2, 1)) @ quat_to_matrix(q2))
    return q2


def from_target(ds: str, q_tgt, t) -> np.ndarray:
    """Target rotation -> egocentric rotation (needs t for allocentric datasets)."""
    if DATASETS[ds]["label"] == "allo":
        return mat_to_quat(ray_rot(t) @ quat_to_matrix(q_tgt))
    return q_tgt


def symmetry_set(ds: str, obj) -> np.ndarray:
    """Object-frame symmetry rotations as (S,3,3) matrices (identity first); YCB-V: identity only."""
    p = DATASETS[ds]["sym"]
    if p is None:
        return np.eye(3)[None]
    z = np.load(REPO / p)
    key = f"obj{int(obj)}" if ds == "tless" else str(obj)
    S = z[key]
    if S.ndim == 2 and S.shape[1] == 4:                  # SYMSOL-T groups stored as quaternions (wxyz)
        S = quat_to_matrix(S)
    return S


def _ply_vertices(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        n, fmt, props, inv = 0, None, [], False
        while True:
            l = f.readline().decode(errors="ignore").strip()
            if l.startswith("format"):
                fmt = l.split()[1]
            if l.startswith("element vertex"):
                n = int(l.split()[-1]); inv = True
            elif l.startswith("element"):
                inv = False
            if l.startswith("property") and inv:
                props.append(l.split()[1:])
            if l == "end_header":
                break
        if fmt == "ascii":
            return np.loadtxt(f, max_rows=n)[:, :3]
        dt = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8", "uchar": "u1", "uint8": "u1",
              "int": "i4", "int32": "i4", "ushort": "u2", "short": "i2", "char": "i1"}
        rec = np.dtype([(pp[-1], ("<" if "little" in fmt else ">") + dt[pp[0]]) for pp in props if pp[0] != "list"])
        a = np.frombuffer(f.read(rec.itemsize * n), dtype=rec, count=n)
        return np.stack([a["x"], a["y"], a["z"]], 1).astype(float)


def model_points(ds: str, obj, df_row=None, max_points: int = 2000) -> np.ndarray:
    root = Path(PATHS[DATASETS[ds]["root_key"]])
    if ds == "tless":
        P = _ply_vertices(root / f"models_cad/obj_{int(obj):06d}.ply")
    elif ds == "ycbv":
        P = np.loadtxt(root / "models" / str(obj) / "points.xyz")
    else:
        split_dir = df_row.image_relpath.split("/")[0]
        P = _ply_vertices(root / split_dir / str(obj) / "points.ply")
    if len(P) > max_points:
        P = P[np.random.default_rng(0).choice(len(P), max_points, replace=False)]
    return P
