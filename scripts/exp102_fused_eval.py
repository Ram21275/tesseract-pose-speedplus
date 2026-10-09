"""EXP-102: evaluate adapted branches alone and fused (equal-weight PoE) from the cached adapted features."""
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import tesseract as T  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.metrics import rotation_metrics  # noqa: E402
from tfpose.predictor import HierMLP  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

EXP = "EXP-102_lora_adapters"
run = Run("EXP-102", "lora_adapters", {"stage": "fused_eval"}, seed=0)
try:
    dev = torch.device("cuda")
    df = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv")
    q = df[[f"q_can_{k}" for k in "wxyz"]].to_numpy()

    def load(branch, seed):
        for d in sorted((REPO / "outputs/experiments" / EXP).glob(f"RUN-*-seed{seed}")):
            cfg = yaml.safe_load((d / "config.yaml").read_text()); st = yaml.safe_load((d / "status.json").read_text())
            if cfg.get("branch") == branch and not cfg.get("max_steps") and st.get("state") == "completed":
                ck = torch.load(d / "checkpoints/best.pt", map_location=dev)
                cache = REPO / st["adapted_cache"]
                with h5py.File(cache / "features.h5") as f:
                    feats = {g: (torch.as_tensor(f[f"{g}/feat"][:], device=dev).float(), f[f"{g}/index"][:]) for g in f.keys()}
                d_in = next(iter(feats.values()))[0].shape[1]
                m = HierMLP(d_in, 5).to(dev); m.load_state_dict(ck["head"]); m.eval()
                return m, feats, d.name
        raise FileNotFoundError(branch, seed)

    rows, used = [], []
    for seed in (0, 1, 2):
        try:
            mA, fA, nA = load("dino", seed); mB, fB, nB = load("moge", seed)
        except FileNotFoundError as e:
            print("skip seed", seed, e); continue
        used += [nA, nB]
        for g in ["synthetic_val", "lightbox", "sunlamp"]:
            (xa, ia), (xb, ib) = fA[g], fB[g]; assert np.array_equal(ia, ib)
            for name, ms, xs, rule in [("A_dino_adapted", [mA], [xa], "single"), ("B_moge_adapted", [mB], [xb], "single"),
                                       ("PoE(A,B)_adapted", [mA, mB], [xa, xb], "poe")]:
                c, p = fused_beam(ms, xs, rule, 1)
                rows.append({"seed": seed, "method": name, "domain": g, **rotation_metrics(T.decode(c.cpu().numpy(), p.cpu().numpy()), q[ia])})
    per = pd.DataFrame(rows); run.write_metrics(rows, "per_seed")
    gb = per.groupby(["method", "domain"])
    agg = gb[["mean_deg", "median_deg", "acc@10", "acc@20"]].mean().join(gb[["mean_deg"]].std().rename(columns={"mean_deg": "mean_deg_std"})).join(gb.size().rename("n_seeds")).reset_index()
    run.write_metrics(agg.to_dict("records"), "metrics"); (run.dir / "source_runs.txt").write_text("\n".join(used) + "\n"); run.done()
    print(md_table(agg.to_dict("records"))); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
