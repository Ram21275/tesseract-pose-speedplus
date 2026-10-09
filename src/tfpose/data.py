"""Read-only SPEED+ loader.

The dataset directory is never written to. Derived manifests go to
``outputs/data_manifests/`` and reference images by relative path.

SPEED+ conventions (verified by reprojection in EXP-000, see
outputs/reports/EXP-000_loader_conventions.md):
- ``q_vbs2tango_true`` is scalar-first ``[w, x, y, z]``.
- With R = quat_to_matrix(q) (active rotation, see rotations.py) and
  r = ``r_Vo2To_vbs_true`` (metres, camera frame), a Tango body point p maps
  to the camera frame as ``p_cam = R @ p + r`` (object-to-camera transform).
- Camera frame: x right, y down, z forward (OpenCV); images are distorted, so
  projection uses ``cv2.projectPoints`` with ``distCoeffs``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import scipy.io

from .rotations import normalize, quat_to_matrix

SPLITS = {
    ("synthetic", "train"): "synthetic/train.json",
    ("synthetic", "validation"): "synthetic/validation.json",
    ("lightbox", "test"): "lightbox/test.json",
    ("sunlamp", "test"): "sunlamp/test.json",
}

# Set by EXP-000. "R" means p_cam = quat_to_matrix(q) @ p + r.
ROTATION_CONVENTION = "R"


@dataclass
class Camera:
    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int


def load_camera(root: Path) -> Camera:
    p = Path(root) / "camera.json"
    if not p.exists():
        raise FileNotFoundError(p)
    c = json.loads(p.read_text())
    return Camera(np.array(c["cameraMatrix"], dtype=np.float64),
                  np.array(c["distCoeffs"], dtype=np.float64), int(c["Nu"]), int(c["Nv"]))


def load_tango_points(path: Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return scipy.io.loadmat(str(path))["tango3Dpoints"].T.astype(np.float64)  # (11, 3)


def load_split(root: Path, domain: str, split: str) -> list[dict]:
    key = (domain, split)
    if key not in SPLITS:
        raise KeyError(f"unknown split {key}")
    p = Path(root) / SPLITS[key]
    if not p.exists():
        raise FileNotFoundError(p)
    recs = []
    for e in json.loads(p.read_text()):
        q = np.array(e["q_vbs2tango_true"], dtype=np.float64)
        recs.append({
            "image_relpath": f"{domain}/images/{e['filename']}",
            "domain": domain,
            "split": split,
            "q_source": q,
            "q": normalize(q),
            "r": np.array(e["r_Vo2To_vbs_true"], dtype=np.float64),
        })
    if not recs:
        raise ValueError(f"empty split {key}")
    return recs


def rotation_cam_from_body(q: np.ndarray, convention: str = ROTATION_CONVENTION) -> np.ndarray:
    R = quat_to_matrix(q)
    if convention == "R":
        return R
    if convention == "RT":
        return np.swapaxes(R, -1, -2)
    raise ValueError(f"unknown convention {convention}")


def project_points(pts: np.ndarray, q: np.ndarray, r: np.ndarray, cam: Camera,
                   convention: str = ROTATION_CONVENTION, distort: bool = True) -> np.ndarray:
    R = rotation_cam_from_body(q, convention)
    rvec, _ = cv2.Rodrigues(R)
    uv, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), rvec, r.reshape(3, 1), cam.K,
                              cam.dist if distort else np.zeros(5))
    return uv.reshape(-1, 2)


def gt_square_box(uv: np.ndarray, cam: Camera, margin: float = 0.15, min_size: float = 32.0):
    """Square box around projected keypoints, enlarged by ``margin`` per side.

    The box may extend beyond the image; cropping pads with zeros.
    Returns (cx, cy, side).
    """
    x0, y0 = uv.min(0)
    x1, y1 = uv.max(0)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    side = max(x1 - x0, y1 - y0, min_size) * (1 + 2 * margin)
    return float(cx), float(cy), float(side)


def crop_square(img: np.ndarray, cx: float, cy: float, side: float, out: int) -> np.ndarray:
    """Affine crop+resize (bilinear, zero padding outside the image)."""
    s = out / side
    M = np.array([[s, 0, out / 2 - s * cx], [0, s, out / 2 - s * cy]], dtype=np.float64)
    return cv2.warpAffine(img, M, (out, out), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def read_gray(root: Path, relpath: str) -> np.ndarray:
    p = Path(root) / relpath
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(p)
    return img


def manifest_hash(rows: list[str]) -> str:
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()[:16]


# DEC-005 (WITHDRAWN by DEC-006; kept for reproducibility of descriptive rows): fixed real-domain validation split = seeded 20% of lightbox (rng 0), used for selection and gates
# from Phase 4 on; the remaining lightbox images ("lightbox_test") and all of sunlamp stay test-only.
REAL_VAL_FRAC, REAL_VAL_SEED = 0.2, 0


def real_val_names(lightbox_names: list[str]) -> list[str]:
    names = sorted(lightbox_names)
    k = int(round(REAL_VAL_FRAC * len(names)))
    return sorted(np.random.default_rng(REAL_VAL_SEED).choice(names, k, replace=False).tolist())


def eval_groups(df) -> dict:
    """Boolean masks for the DEC-005 evaluation groups over a subset manifest."""
    lb = (df.domain == "lightbox").to_numpy()
    val = df.image_relpath.isin(set(real_val_names(df.image_relpath[lb].tolist()))).to_numpy() & lb
    return {"synthetic_val": ((df.domain == "synthetic") & (df.split == "validation")).to_numpy(),
            "lightbox_val": val, "lightbox_test": lb & ~val, "lightbox": lb,
            "sunlamp": (df.domain == "sunlamp").to_numpy()}
