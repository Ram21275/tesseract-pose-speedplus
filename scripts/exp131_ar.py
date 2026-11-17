"""EXP-131 (pre-registered): full BOP AR = mean(AR_VSD, AR_MSSD, AR_MSPD) from per-image predictions (exp064_eval --save-preds).

  python scripts/exp131_ar.py --dataset tlessall --src-exp EXP-131_all_objects --variant PoE --mode orbit_product
  python scripts/exp131_ar.py --dataset ycbvall  --src-exp EXP-131_all_objects --variant DINO --mode greedy
  regression (must reproduce EXP-130 AR_MSSD / AR_MSPD within 0.2):  --dataset tless|ycbv --src-exp EXP-130_literature_matched_metrics

MSSD / MSPD: BOP toolkit definitions (max surface / projected distance, min over the BOP symmetry set; continuous symmetries in
315 steps; MSSD thresholds 0.05:0.05:0.5 x diameter on models_eval vertices (T-LESS) / textured.obj vertices (YCB-V, original
frame, BOP symmetries converted by exp131_ycbv_symmetries.py); MSPD thresholds 5:5:50 x (image width / 640) px).
VSD: BOP19 definition (pose_error.vsd, visib_mode bop19, delta 15 mm, tau 0.05:0.05:0.5 x diameter, distance images, correctness
thresholds 0.05:0.05:0.5). No OpenGL renderer is installed, so depth is rendered by an area-weighted point-splat z-buffer of the
CAD mesh (300k surface samples, 3x3 hole / back-surface fix). Gate: the GT pose renders to AR_VSD >= 99 on every 25th instance
(it is scored against the real depth image, so this also checks units, intrinsics and the depth scale).
Aggregation: instance-pooled (the BOP toolkit's definition) and object-balanced, per seed, mean +- seed std.
YCB-V is also reported on the BOP19 test-target subset (4,123 instances, all among the original keyframes)."""
import argparse
import json
import sys
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import minimum_filter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import xobj as XO  # noqa: E402
from tfpose.rotations import quat_to_matrix  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, help="tlessall / ycbvall (EXP-131), tless / ycbv (EXP-130 regression), *smoke")
ap.add_argument("--src-exp", required=True, help="experiment folder whose newest completed RUN-*/predictions holds the npz files")
ap.add_argument("--variant", default="PoE")
ap.add_argument("--mode", default="orbit_product")
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--no-vsd", action="store_true", help="MSSD / MSPD only (regression against EXP-130)")
args = ap.parse_args()
PATHS = yaml.safe_load(open(REPO / "configs/paths.yaml")); BASE = XO.base(args.dataset)
TL = Path(PATHS["tless_root"]); YC = Path(PATHS["ycbv_root"]); BY = TL.parent / "ycbv"
TAUS = np.arange(0.05, 0.51, 0.05); DELTA = 15.0; NPTS = 300_000
CLASSES = (YC / "image_sets/classes.txt").read_text().split()


def sym_tless(info):
    disc = [np.eye(4)] + [np.array(s).reshape(4, 4) for s in info.get("symmetries_discrete", [])]
    cont = [np.eye(4)]
    for c in info.get("symmetries_continuous", []):
        a = np.array(c["axis"], float); o = np.array(c["offset"], float); cont = [np.eye(4)]
        for k in range(1, 316):
            th = 2 * np.pi * k / 315; K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]]); R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = o - R @ o; cont.append(T)
    return np.array([d @ c for d in disc for c in cont])


def obj_vertices(path):
    return np.array([l.split()[1:4] for l in open(path) if l.startswith("v ")], float)


def load_obj(obj):
    """(eval vertices mm, symmetry 4x4 mm (S,4,4), diameter mm, mesh vertices mm, faces, pose scale to mm)."""
    if BASE == "tless":
        info = json.load(open(TL / "models_eval/models_info.json"))[str(int(obj))]
        P = XO._ply_vertices(TL / f"models_eval/obj_{int(obj):06d}.ply"); S = sym_tless(info)
        lines = open(TL / f"models_cad/obj_{int(obj):06d}.ply").read().split("\n")
        nv = int(next(l for l in lines if l.startswith("element vertex")).split()[-1]); nf = int(next(l for l in lines if l.startswith("element face")).split()[-1])
        h = lines.index("end_header") + 1
        V = np.array([l.split()[:3] for l in lines[h:h + nv]], float); F = np.array([l.split()[1:4] for l in lines[h + nv:h + nv + nf]], int)
        return P, S, info["diameter"], V, F, 1.0
    V, F = [], []
    for l in open(YC / f"models/{obj}/textured.obj"):
        if l.startswith("v "):
            V.append(l.split()[1:4])
        elif l.startswith("f "):
            F.append([int(t.split("/")[0]) - 1 for t in l.split()[1:4]])
    V = np.array(V, float) * 1000.0; F = np.array(F, int)
    S = np.load(REPO / "outputs/data_manifests/ycbv_exp131_symmetries.npz")[f"{obj}__T4_mm"]
    dia = json.load(open(BY / "models_eval/models_info.json"))[str(CLASSES.index(obj) + 1)]["diameter"]
    if args.dataset == "ycbv":                                                  # EXP-130 regression: its own conventions
        S = np.eye(4)[None]; Pp = np.loadtxt(YC / "models" / obj / "points.xyz") * 1000.0
        Q = Pp[:: max(1, len(Pp) // 1500)]; dia = float(np.sqrt(((Q[:, None] - Q[None]) ** 2).sum(-1)).max())
        return Pp, S, dia, V, F, 1000.0
    return V, S, dia, V, F, 1000.0


def sample(V, F, n, rng):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]; ar = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    i = rng.choice(len(F), n, p=ar / ar.sum()); u, v = rng.random(n), rng.random(n); s = u + v > 1; u[s], v[s] = 1 - u[s], 1 - v[s]
    return a[i] + u[:, None] * (b[i] - a[i]) + v[:, None] * (c[i] - a[i])


def render(X, K, W, H):
    z = X[:, 2]; u = np.round(X[:, 0] / z * K[0, 0] + K[0, 2]).astype(int); v = np.round(X[:, 1] / z * K[1, 1] + K[1, 2]).astype(int)
    ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H); D = np.full(H * W, np.inf); np.minimum.at(D, v[ok] * W + u[ok], z[ok]); D = D.reshape(H, W)
    M = minimum_filter(D, 3, mode="constant", cval=np.inf)
    nb = cv2.filter2D(np.isfinite(D).astype(np.float32), -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT)
    fix = (np.isinf(D) & (nb >= 5)) | (np.isfinite(D) & (D > M + 5.0)); D[fix] = M[fix]; D[np.isinf(D)] = 0.0
    return D


@lru_cache(maxsize=8)
def rays(W, H, fx, fy, cx, cy):
    u, v = np.meshgrid(np.arange(W), np.arange(H)); return np.sqrt(1 + ((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2)


def vsd(d_est, d_gt, d_test, ray, dia):
    e, g, t = d_est * ray, d_gt * ray, d_test * ray
    vis = lambda dm: (dm > 0) & ((((dm - t) <= DELTA) & (t > 0)) | (t == 0))
    vg = vis(g); ve = vis(e) | (vg & (e > 0)); inter = vg & ve; nu = (vg | ve).sum()
    if nu == 0:
        return np.ones(len(TAUS))
    diff = np.abs(e[inter] - g[inter]); ni = inter.sum()
    return np.array([((diff >= tau * dia).sum() + (nu - ni)) / nu for tau in TAUS])


@lru_cache(maxsize=64)
def tl_cam(sc):
    return json.load(open(TL / f"test_primesense/{sc}/scene_camera.json"))


def depth(rel):
    rel = rel.split("#")[0]
    if BASE == "tless":
        sc, im = rel.split("/")[1], rel.split("/")[-1]
        return cv2.imread(str(TL / f"test_primesense/{sc}/depth/{im}"), cv2.IMREAD_UNCHANGED).astype(np.float64) * tl_cam(sc)[str(int(im[:-4]))]["depth_scale"]
    return cv2.imread(str(YC / rel.replace("-color.jpg", "-depth.png").replace("-color.png", "-depth.png")), cv2.IMREAD_UNCHANGED).astype(np.float64) / 10.0   # factor 10000 -> mm


def one(f):
    f = Path(f); z = dict(np.load(f)); stem = f.stem
    obj = stem[len(args.dataset) + 1:].rsplit("_seed", 1)[0]
    seed = int(stem.rsplit("_seed", 1)[1].split("_")[0])
    P, S, dia, V, F, sc = load_obj(obj); W, H = XO.DATASETS[args.dataset]["img_wh"]; TAU_P = np.arange(5, 51, 5) * W / 640.0
    m = np.flatnonzero(z["split"] == "test"); PS = np.einsum("sij,kj->ski", S[:, :3, :3], P) + S[:, None, :3, 3]
    Xs = None if args.no_vsd else sample(V, F, NPTS, np.random.default_rng(0)); rows = []
    for c, n in enumerate(m):
        Rg, tg = quat_to_matrix(z["q_ego_gt"][n]), z["t_gt"][n] * sc; Rp, tp = quat_to_matrix(z[f"q_ego_{args.mode}"][n]), z["t_used"][n] * sc; K = z["K"][n]
        Xp = P @ Rp.T + tp; Xg = PS @ Rg.T + tg; mssd = np.linalg.norm(Xp[None] - Xg, axis=2).max(1).min()
        up = Xp @ K.T; up = up[:, :2] / up[:, 2:]; ug = Xg @ K.T; ug = ug[..., :2] / ug[..., 2:]; mspd = np.linalg.norm(up[None] - ug, axis=2).max(1).min()
        r = dict(obj=str(obj), seed=seed, key=str(z["image_relpath"][n]), MSSD=float(np.mean([mssd < t * dia for t in TAUS])), MSPD=float(np.mean([mspd < t for t in TAU_P])))
        if not args.no_vsd:
            dt = depth(str(z["image_relpath"][n])); ray = rays(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2]); dg = render(Xs @ Rg.T + tg, K, W, H)
            r["VSD"] = float(np.mean([(vsd(render(Xs @ Rp.T + tp, K, W, H), dg, dt, ray, dia) < th).mean() for th in TAUS]))
            if c % 25 == 0:
                r["VSD_gt"] = float(np.mean([(vsd(dg, dg, dt, ray, dia) < th).mean() for th in TAUS]))
                vm = (dg > 0) & (dt > 0); r["gt_depth_agree_mm"] = float(np.median(np.abs(dg[vm] - dt[vm]))) if vm.any() else np.nan
                e = dt[vm] - dg[vm]; un = e > -DELTA; r["gt_depth_agree_unocc_mm"] = float(np.median(np.abs(e[un]))) if un.any() else np.nan
        rows.append(r)
    print(stem, len(rows), flush=True)
    return rows


run = Run("EXP-131", "all_objects_ar", vars(args) | {"stage": "ar"}, seed=0)
try:
    XD = REPO / "outputs/experiments" / args.src_exp
    src = [d for d in sorted(XD.glob("RUN-*"), reverse=True) if list((d / "predictions").glob(f"{args.dataset}_*_{args.variant}.npz")) and '"completed"' in (d / "status.json").read_text()][0]
    files = sorted(str(p) for p in (src / "predictions").glob(f"{args.dataset}_*_{args.variant}.npz"))
    (run.dir / "source_runs.txt").write_text(f"{args.src_exp}/{src.name}\n")
    with Pool(args.workers) as pool:
        df = pd.DataFrame([r for rs in pool.imap_unordered(one, files) for r in rs])
    cols = ["MSSD", "MSPD"] + ([] if args.no_vsd else ["VSD"])
    df["AR"] = df[cols].mean(1) if not args.no_vsd else np.nan
    subsets = {"all": np.ones(len(df), bool)}
    if BASE == "ycbv" and args.dataset != "ycbv":                            # BOP19 test targets (scene_id, im_id, obj_id)
        tg = json.load(open(BY / "test_targets_bop19.json")); keyset = {(x["scene_id"], x["im_id"], x["obj_id"]) for x in tg}
        sid = df.key.str.split("/").str[1].astype(int); iid = df.key.str.split("/").str[2].str[:6].astype(int); oid = df.obj.map(lambda o: CLASSES.index(o) + 1)
        subsets["bop19"] = np.array([(a, b, c) in keyset for a, b, c in zip(sid, iid, oid)])
    out = []
    for sname, msk in subsets.items():
        d = df[msk]; mc = cols + (["AR"] if not args.no_vsd else [])
        inst = d.groupby("seed")[mc].mean(); obal = d.groupby(["seed", "obj"])[mc].mean().groupby("seed").mean()
        for nm, t in (("instance_pooled", inst), ("object_balanced", obal)):
            out.append(dict(subset=sname, avg=nm, n=int(msk.sum() / max(1, d.seed.nunique())), **{f"{c}_mean": 100 * t[c].mean() for c in mc}, **{f"{c}_std": 100 * t[c].std() for c in mc}))
    if not args.no_vsd:
        g = df.dropna(subset=["VSD_gt"]); gate = dict(gt_vsd_min=float(g.VSD_gt.min() * 100), gt_vsd_mean=float(g.VSD_gt.mean() * 100),
                                                     gt_depth_agree_median_mm=float(g.gt_depth_agree_mm.median()), gt_depth_agree_unocc_median_mm=float(g.gt_depth_agree_unocc_mm.median()), pass_=bool(g.VSD_gt.mean() >= 0.99))
        run.write_metrics(gate, "vsd_gate"); print("VSD gate", gate)
    run.write_metrics(df.to_dict("records"), "per_instance"); run.write_metrics(out, "summary")
    po = df.groupby(["obj"])[cols + (["AR"] if not args.no_vsd else [])].mean().reset_index(); run.write_metrics(po.to_dict("records"), "per_object")
    (run.sub("tables") / "summary.md").write_text(md_table(out)); print(md_table(out)); run.done(); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
