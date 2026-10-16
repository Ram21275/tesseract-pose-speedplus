"""EXP-109 arm B: standardized DINOv3 16x16 tokens (256 x 1024, fp16) for subset_full as an on-disk memmap (SSD),
statistics from synthetic train only; avoids holding 36 GB in host memory during training."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose.cachepaths import CACHE  # noqa: E402
from tfpose import featsets  # noqa: E402

df = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv")
tr = np.flatnonzero(((df.domain == "synthetic") & (df.split == "train")).to_numpy())
path = featsets.find_cache("dinov3_vitl16", "full")
out = CACHE / "tokens16_full_std.npy"
STATS_ONLY = "--stats-only" in sys.argv
with h5py.File(path / "features.h5", "r") as f:
    assert [s.decode() for s in f["image_relpath"][:]] == df.image_relpath.tolist()
    ds = f["last_grid"]; N = ds.shape[0]; D = int(np.prod(ds.shape[1:]))
    s1 = np.zeros(D); s2 = np.zeros(D); trs = set(tr.tolist())
    for s0 in range(0, N, 1024):
        c = np.asarray(ds[s0:s0 + 1024], dtype=np.float64).reshape(-1, D)
        m = np.array([i in trs for i in range(s0, s0 + len(c))])
        s1 += c[m].sum(0); s2 += (c[m] ** 2).sum(0)
        print("stats", s0, flush=True)
    mu = s1 / len(tr); sd = np.sqrt(np.maximum(s2 / len(tr) - mu ** 2, 0)) + 1e-6
    np.savez(out.with_name("tokens16_full_stats.npz"), mu=mu.astype(np.float32), sd=sd.astype(np.float32))
    if STATS_ONLY:
        print("stats saved"); sys.exit(0)
    mm = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(N, D))
    for s0 in range(0, N, 1024):
        mm[s0:s0 + 1024] = ((np.asarray(ds[s0:s0 + 1024], dtype=np.float32).reshape(-1, D) - mu) / sd).astype(np.float16)
        print("write", s0, flush=True)
    mm.flush()
(out.with_suffix(".json")).write_text(json.dumps({"source_cache": path.name, "n": N, "dim": D, "token_shape": "256x1024",
                                                  "stats": "synthetic train", "order": "subset_full manifest"}, indent=2))
print("done", out)
