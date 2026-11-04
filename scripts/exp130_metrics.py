"""EXP-130 (pre-registered): ADD(-S) / AUC on YCB-V and AR_MSSD / AR_MSPD on T-LESS from per-image predictions (exp064_eval --save-preds)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import xobj as XO  # noqa: E402
from tfpose.rotations import quat_to_matrix  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

PATHS = yaml.safe_load(open(REPO / "configs/paths.yaml")); XD = REPO / "outputs/experiments/EXP-130_literature_matched_metrics"
run = Run("EXP-130", "literature_matched_metrics", {"stage": "metrics"}, seed=0)


def src_run(ds):
    out = [d for d in sorted(XD.glob("RUN-*"), reverse=True) if list((d / "predictions").glob(f"{ds}_*_seed2_DINO.npz")) and '"completed"' in (d / "status.json").read_text()]
    return out[0]


def ycbv_rows():
    rows = []
    for f in sorted((src_run("ycbv") / "predictions").glob("ycbv_*.npz")):
        z = dict(np.load(f)); stem = f.stem; obj = stem[len("ycbv_"):].rsplit("_seed", 1)[0]; seed = int(stem.rsplit("_seed", 1)[1].split("_")[0]); var = stem.rsplit("_", 1)[1]
        P = np.loadtxt(Path(PATHS["ycbv_root"]) / "models" / obj / "points.xyz"); tree = None
        step = max(1, len(P) // 1500); Q = P[::step]; diam = np.sqrt(((Q[:, None] - Q[None]) ** 2).sum(-1)).max()
        m = z["split"] == "test"; Rg = quat_to_matrix(z["q_ego_gt"][m]); tg = z["t_gt"][m]; tp = z["t_used"][m]
        Xg = np.einsum("nij,kj->nki", Rg, P) + tg[:, None]
        for mode in ("greedy", "orbit_product"):
            Xp = np.einsum("nij,kj->nki", quat_to_matrix(z[f"q_ego_{mode}"][m]), P) + tp[:, None]
            add = np.linalg.norm(Xp - Xg, axis=2).mean(1); adds = np.array([cKDTree(a).query(b)[0].mean() for a, b in zip(Xg, Xp)])
            auc = lambda e: float(np.mean([(e < t).mean() for t in np.linspace(0, 0.1, 1001)[1:]]))
            rows.append(dict(obj=obj, seed=seed, variant=var, mode=mode, n=int(m.sum()), diameter_m=diam, ADD_01d=float((add < 0.1 * diam).mean()), ADDS_01d=float((adds < 0.1 * diam).mean()),
                             AUC_ADD=auc(add), AUC_ADDS=auc(adds), ADD_2cm=float((add < 0.02).mean()), sym_mean=float(z[f"err_{mode}"][m].mean())))
    return pd.DataFrame(rows)


def sym_set(info):
    """BOP toolkit get_symmetry_transformations: 4x4 list; continuous steps = ceil(pi*d/(0.01*d)) = 315."""
    disc = [np.eye(4)] + [np.array(s).reshape(4, 4) for s in info.get("symmetries_discrete", [])]
    cont = [np.eye(4)]
    for c in info.get("symmetries_continuous", []):
        a = np.array(c["axis"], float); o = np.array(c["offset"], float); new = []
        for k in range(1, 316):
            th = 2 * np.pi * k / 315; K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]]); R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = o - R @ o; new.append(T)
        cont = [np.eye(4)] + new
    return np.array([d @ c for d in disc for c in cont])


def tless_rows():
    root = Path(PATHS["tless_root"]) / "models_eval"; info = json.load(open(root / "models_info.json")); w = XO.DATASETS["tless"]["img_wh"][0]; rows = []
    TAU_S = np.arange(0.05, 0.51, 0.05); TAU_P = np.arange(5, 51, 5) * w / 640.0
    for f in sorted((src_run("tless") / "predictions").glob("tless_*.npz")):
        z = dict(np.load(f)); stem = f.stem; obj = stem.split("_")[1]; seed = int(stem.rsplit("_seed", 1)[1].split("_")[0]); var = stem.rsplit("_", 1)[1]
        P = XO._ply_vertices(root / f"obj_{int(obj):06d}.ply"); i = info[obj]; S = sym_set(i); dia = i["diameter"]
        PS = np.einsum("sij,kj->ski", S[:, :3, :3], P) + S[:, None, :3, 3]                   # (sym, k, 3), object frame
        m = np.flatnonzero(z["split"] == "test")
        for mode in ("greedy", "orbit_product"):
            ms, mp = [], []
            for n in m:
                Rg, tg = quat_to_matrix(z["q_ego_gt"][n]), z["t_gt"][n]; Rp, tp = quat_to_matrix(z[f"q_ego_{mode}"][n]), z["t_used"][n]; K = z["K"][n]
                Xp = P @ Rp.T + tp; Xg = PS @ Rg.T + tg; ms.append(np.linalg.norm(Xp[None] - Xg, axis=2).max(1).min())
                up = Xp @ K.T; up = up[:, :2] / up[:, 2:]; ug = Xg @ K.T; ug = ug[..., :2] / ug[..., 2:]; mp.append(np.linalg.norm(up[None] - ug, axis=2).max(1).min())
            ms, mp = np.array(ms), np.array(mp)
            rows.append(dict(obj=int(obj), seed=seed, variant=var, mode=mode, n=len(m), AR_MSSD=float(np.mean([(ms < t * dia).mean() for t in TAU_S])), AR_MSPD=float(np.mean([(mp < t).mean() for t in TAU_P])),
                             sym_mean=float(z[f"err_{mode}"][m].mean()), n_sym=len(S)))
    return pd.DataFrame(rows)


def agg(df, cols):
    w = df.n
    ob = df.groupby(["variant", "mode", "seed"])[cols].mean(); inst = df.groupby(["variant", "mode", "seed"]).apply(lambda g: pd.Series({c: np.average(g[c], weights=g.n) for c in cols}), include_groups=False)
    out = []
    for nm, t in (("object_balanced", ob), ("instance_pooled", inst)):
        a = t.groupby(["variant", "mode"]).agg(["mean", "std"]); a.columns = [f"{x}_{y}" for x, y in a.columns]; a = a.reset_index(); a.insert(0, "avg", nm); out.append(a)
    return pd.concat(out)


try:
    gate_ref = {"ycbv": "RUN-20261103-165739-seed0", "tless": "RUN-20261103-162450-seed0"}   # the EXP-129 arms with the same predicted translation
    for ds, fn, cols in (("ycbv", ycbv_rows, ["ADD_01d", "ADDS_01d", "AUC_ADD", "AUC_ADDS", "ADD_2cm"]), ("tless", tless_rows, ["AR_MSSD", "AR_MSPD"])):
        df = fn(); run.write_metrics(df.to_dict("records"), f"{ds}_per_object_seed")
        ref = pd.read_csv(REPO / "outputs/experiments/EXP-129_rotation_with_predicted_translation" / gate_ref[ds] / "metrics/per_object_seed.csv"); ref = ref[ref.split == "test"]; ref["obj"] = ref.obj.astype(str)
        g = df.assign(obj=df.obj.astype(str)).merge(ref, on=["obj", "seed", "variant", "mode"], suffixes=("", "_r")); d = (g.sym_mean - g.sym_mean_r).abs().max()
        print(ds, "gate: rows", len(g), "max |d sym_mean| vs EXP-129", d, flush=True); assert len(g) == len(df) and d < 0.01, "gate failed"
        a = agg(df, cols); run.write_metrics(a.to_dict("records"), f"{ds}_summary"); (run.sub("tables") / f"{ds}_summary.md").write_text(md_table(a.to_dict("records"))); print(md_table(a.to_dict("records")))
        po = df.groupby(["variant", "mode", "obj"])[cols].mean().reset_index(); run.write_metrics(po.to_dict("records"), f"{ds}_per_object"); print(md_table(po.to_dict("records")))
    run.done(); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
