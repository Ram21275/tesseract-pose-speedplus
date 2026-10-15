"""EXP-064 data stage, SYMSOL-T (CPU only): manifest + per-shape symmetry groups, following the LiePose protocol.

LiePose (src/liepose/data/symsolt, exp/symsolt/run.py): train on train-45k (45k per shape), test on test-5k (5k per shape),
image = the non-colour render ("image", color_mode none), full 224x224 frame, rotation = quaternion wxyz (egocentric pose
of the render), error = min over rotations_equivalent. We keep exactly that; the "crop" is the whole frame.
Validation (selection) = 2,000 per shape held out from train-45k (rng 0); the rest is train. A per-shape random rank lets a
pre-registered train cap select the first K. Symmetry groups S (object frame, R_eq = R S) are recovered per shape and checked
to be constant across samples.
"""
import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import data as D  # noqa: E402

ROOT = Path(yaml.safe_load(open(REPO / "configs/paths.yaml"))["symsolt_root"])
SHAPES = ["tet", "cube", "icosa", "cone", "cyl"]
OUT = REPO / "outputs/data_manifests/symsolt_exp064"
N_VAL = 2000


def qmul(a, b):
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0); w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def conj(q):
    return q * np.array([1, -1, -1, -1])


rows, groups = [], {}
for split_dir, split in (("train-45k", "train"), ("test-5k", "test")):
    ann = pickle.load(gzip.open(ROOT / split_dir / "annotations.gz"))
    for e in ann:
        rows.append({"image_relpath": f"{split_dir}/{e['image']}", "split": split, "shape": SHAPES[e["label_shape"]],
                     "label_shape": e["label_shape"], "num_equivalents": e["num_equivalents"],
                     **{f"q_{k}": v for k, v in zip("wxyz", e["rotation"])}, **{f"t_{k}": v for k, v in zip("xyz", e["translation"])}})
        s = SHAPES[e["label_shape"]]
        if s not in groups:                                           # object-frame group S = R^-1 R_eq
            q = np.asarray(e["rotation"]); qe = np.asarray(e["rotations_equivalent"]).reshape(-1, 4)
            groups[s] = qmul(conj(q)[None], qe)
    # constancy check on 200 samples per shape
    for s_id, s in enumerate(SHAPES):
        sub = [e for e in ann if e["label_shape"] == s_id][:200]
        G = groups[s]
        for e in sub:
            q = np.asarray(e["rotation"]); qe = np.asarray(e["rotations_equivalent"]).reshape(-1, 4)
            pred = qmul(np.broadcast_to(q, G.shape), G)
            d = np.abs(pred @ qe.T).max(1)                            # every R S matches some stored equivalent
            assert d.min() > 1 - 1e-4, (split_dir, s, d.min())
df = pd.DataFrame(rows)
rng = np.random.default_rng(0)
df["train_rank"] = -1
for s in SHAPES:
    ii = np.flatnonzero(((df.split == "train") & (df["shape"] == s)).to_numpy())
    perm = rng.permutation(ii)
    df.loc[perm[:N_VAL], "split"] = "validation"
    df.loc[perm[N_VAL:], "train_rank"] = np.arange(len(perm) - N_VAL)
df["domain"] = "symsolt"
df["crop_cx"], df["crop_cy"], df["crop_side"], df["crop_margin"] = 112.0, 112.0, 224.0, 0.0
assert all((ROOT / p).exists() for p in df.image_relpath.sample(500, random_state=0))
df.to_csv(f"{OUT}.csv", index=False, float_format="%.9g")
Path(f"{OUT}.sha256.txt").write_text(D.manifest_hash(df.image_relpath.tolist()) + "\n")
np.savez(f"{OUT}_symmetries.npz", **{s: groups[s] for s in SHAPES})
print(df.groupby(["split", "shape"]).size().unstack(0).to_string())
print("group sizes", {s: len(groups[s]) for s in SHAPES}, "| total", len(df))
