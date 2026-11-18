"""EXP-131 (pre-registered): log-likelihood of the trees' predicted rotation density at the ground truth (no retraining).
The tree is a piecewise-constant density on SO(3): p(R) = P(leaf(R)) / vol(leaf(R)). vol is in the quaternion metric, in which
the 4 charts tile SO(3) with total volume pi^2 (checked by quadrature), so a uniform density scores -log pi^2 = -2.29 (the IPDF
convention). Per test image: the mean over the object's symmetric GT copies of log p (IPDF protocol; groups larger than 60
are subsampled evenly to 60). Single view (view 0), the selected variant, per-level PoE of the branches' log-softmaxes, fp32.

  python scripts/exp131_ll.py --dataset tlessall --runs-dir EXP-131_all_objects --filter trans=frozen --variant PoE
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import featsets, phase3 as P, tesseract as T, xobj as XO  # noqa: E402
from tfpose.rotations import matrix_to_quat, quat_to_matrix  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=list(XO.DATASETS))
ap.add_argument("--runs-dir", required=True)
ap.add_argument("--filter", default="", help="k=v,... on the train-run config")
ap.add_argument("--variant", default="PoE", choices=["PoE", "DINO"])
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--depth", type=int, default=5, help="EXP-132: tree depth L (runs without the key are depth 5)")
ap.add_argument("--out", default="EXP-131:all_objects_ll", help="<EXP>:<name> of the output run")
ap.add_argument("--max-copies", type=int, default=60)
args = ap.parse_args()
ds = args.dataset; SD, SM = "dinov3_vitl16:tokens8", "moge2_vitl:normals16~and"; SHAPE = {SD: (64, 1024), SM: (256, 4)}
KEYS = (SD, SM) if args.variant == "PoE" else (SD,); L = args.depth
DEFAULTS = {"label": "raw", "extra_train": "None", "teacher_variant": "None", "trans": "none", "depth": "5"}
FILT = dict(kv.split("=", 1) for kv in args.filter.split(",") if kv); FILT.setdefault("depth", str(L))
run = Run(*args.out.split(":", 1), vars(args) | {"stage": "ll"}, seed=0)


def leaf_logvol(chart, path):
    lo, hi = T._cell_bounds(chart, path); c = 0.5 * (lo + hi); side = (hi - lo)[:, 0]
    return np.log(side ** 3) - 2 * np.log1p((c ** 2).sum(1))                      # midpoint rule of (1+|u|^2)^-2 d^3u


try:
    dev = torch.device("cuda")
    ALL = pd.read_csv(REPO / f"outputs/data_manifests/exp064_{ds}_all.csv"); EV = pd.read_csv(REPO / f"outputs/data_manifests/exp064_{ds}_eval.csv")
    oc = XO.DATASETS[ds]["obj_col"]; runs = {}
    for d in sorted((REPO / "outputs/experiments" / args.runs_dir).glob("RUN-*")):
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        if any(str(cfg.get(k, DEFAULTS.get(k))) != v for k, v in FILT.items()):
            continue
        if cfg.get("stage") == "train" and cfg.get("dataset") == ds and (d / "checkpoints/best.pt").exists():
            runs[(str(cfg["obj"]), cfg["seed"], cfg["features"])] = d
    (run.dir / "source_runs.txt").write_text("".join(f"{k}\t{args.runs_dir}/{d.name}\n" for k, d in sorted(runs.items())))

    def tree_lp(ms, xs, q):
        c, p = T.encode(q, L); ct, pt = torch.as_tensor(c, device=dev), torch.as_tensor(p, device=dev)
        outs = [m.logits_tf(m.encode(x), ct, pt) for m, x in zip(ms, xs)]
        root = F.log_softmax(sum(F.log_softmax(o[0].float(), -1) for o in outs), -1)
        child = F.log_softmax(sum(F.log_softmax(o[1].float(), -1) for o in outs), -1)
        lp = (root.gather(1, ct[:, None])[:, 0] + child.gather(2, pt[:, :, None])[..., 0].sum(1)).double().cpu().numpy()
        return lp - leaf_logvol(c, p)

    rows = []
    for obj in sorted({k[0] for k in runs}):
        objv = obj if isinstance(ALL[oc].iloc[0], str) else type(ALL[oc].iloc[0])(obj)
        a_ev = np.flatnonzero(((ALL.split != "train") & (ALL[oc] == objv)).to_numpy()); de = EV[EV[oc] == objv]; test = (de.split == "test").to_numpy()
        qgt = XO.target(ds, de); S = XO.symmetry_set(ds, objv)
        if len(S) > args.max_copies:
            S = S[np.linspace(0, len(S) - 1, args.max_copies).round().astype(int)]
        Rq = quat_to_matrix(qgt); bs = max(1, 1024 // len(S))
        for seed in (int(s) for s in args.seeds.split(",")):
            if any((obj, seed, k) not in runs for k in KEYS):
                continue
            ms, xs = [], []
            for k in KEYS:
                ck = torch.load(runs[(obj, seed, k)] / "checkpoints/best.pt", map_location=dev, weights_only=False)
                m = P.HierTransformer(SHAPE[k], L).to(dev); m.load_state_dict(ck["model"]); m.eval(); ms.append(m)
                _, x0 = featsets.load_rows(k, f"exp064_{ds}_all", a_ev)
                xs.append(featsets.standardize_to_tensor(x0, None, "cpu", torch.float32, stats=(ck["mu"], ck["sd"])))
            idx = np.flatnonzero(test); ll = []
            with torch.no_grad():
                for s0 in range(0, len(idx), bs):
                    r = idx[s0:s0 + bs]; B = len(r); C = len(S)
                    qc = matrix_to_quat(np.einsum("nij,sjk->nsik", Rq[r], S).reshape(-1, 3, 3))
                    x = [xx[torch.as_tensor(r)].to(dev).repeat_interleave(C, 0) for xx in xs]
                    ll.append(tree_lp(ms, x, qc).reshape(B, C).mean(1))
            ll = np.concatenate(ll)
            rows.append(dict(obj=obj, seed=seed, n=len(ll), LL=float(ll.mean()), LL_median=float(np.median(ll)), n_sym=len(S)))
            print(rows[-1], flush=True); del ms, xs; torch.cuda.empty_cache()
    df = pd.DataFrame(rows); ps = df.groupby("seed").LL.mean()
    summ = [dict(dataset=ds, variant=args.variant, objects=df.obj.nunique(), LL_object_balanced=float(ps.mean()), LL_seed_std=float(ps.std()),
                 LL_instance_pooled=float(df.groupby("seed").apply(lambda g: np.average(g.LL, weights=g.n), include_groups=False).mean()))]
    run.write_metrics(rows, "per_object_seed"); run.write_metrics(summ, "summary")
    (run.sub("tables") / "summary.md").write_text(md_table(summ)); print(md_table(summ)); run.done(); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
