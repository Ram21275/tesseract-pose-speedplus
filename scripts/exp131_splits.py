"""EXP-131 data stage (CPU only): the per-split manifests the EXP-064 tools read, from the all-object crop manifests.
exp064_<ds>_all.csv = the manifest as is; _train = rows with split == train; _eval = the other rows (validation + test), order kept.
This is the relation of the committed EXP-064 files (checked: exp064_tless_* and exp064_ycbv_* equal these derivations exactly).
  python scripts/exp131_splits.py tlessall ycbvall"""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import xobj as XO  # noqa: E402

for ds in sys.argv[1:]:
    df = XO.manifest(ds); out = REPO / "outputs/data_manifests" / f"exp064_{ds}"
    df.to_csv(f"{out}_all.csv", index=False, float_format="%.9g")
    df[df.split == "train"].to_csv(f"{out}_train.csv", index=False, float_format="%.9g")
    df[df.split != "train"].to_csv(f"{out}_eval.csv", index=False, float_format="%.9g")
    print(ds, df.groupby([XO.DATASETS[ds]["obj_col"], "split"]).size().unstack(1).fillna(0).astype(int).sum().to_dict(), "objects", df[XO.DATASETS[ds]["obj_col"]].nunique())
