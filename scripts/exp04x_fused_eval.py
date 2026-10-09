"""Phase-4 fused evaluation (EXP-040–042): PoE greedy leaf (DEC-003) + both branches' continuous heads.

Fixed fusion rules (pre-registered, no fitted parameters):
- residual: q = q_leaf (x) Exp(mean of the two branches' deltas at the PoE leaf)       (EXP-035 rule)
- flow:     velocity average  omega = (omega_A + omega_B) / 2 at the same state (a heuristic, NOT an exact PoE);
            local base centred on the PoE leaf; point estimate = KDE mode of the M endpoint samples.
Groups: synthetic val (selection) and lightbox/sunlamp (test-only, DEC-000/DEC-006). Rows are reported for A, B (each with its own greedy leaf) and the fused system.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import featsets, phase3 as P, so3flow as S, tesseract as T  # noqa: E402
from tfpose.data import eval_groups  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.metrics import rotation_metrics  # noqa: E402
from tfpose.predictor import decode_torch  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True, help="experiment holding the Phase-4 head runs")
ap.add_argument("--a", default="dinov3_vitl16:tokens8")
ap.add_argument("--b", default="moge2_vitl:normals16~and")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--m", type=int, default=32)
ap.add_argument("--nfe", type=int, default=10)
ap.add_argument("--mode", default=None, help="only load head runs with this mode (default: any)")
ap.add_argument("--base", default=None, help="only load flow runs with this base (default: any)")
ap.add_argument("--save-samples", action="store_true", help="save fused endpoint samples (EXP-043 input)")
args = ap.parse_args()
run = Run(args.exp.split("_", 1)[0], args.exp.split("_", 1)[1], dict(vars(args), stage="fused_eval"), seed=0)
try:
    dev = torch.device("cuda")
    df = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv")
    q = df[[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    tr = ((df.domain == "synthetic") & (df.split == "train")).to_numpy()
    groups = {k: v for k, v in eval_groups(df).items() if k in ("synthetic_val", "lightbox", "sunlamp")}   # DEC-006: real domains test-only
    Qt = torch.as_tensor(q, dtype=torch.float32, device=dev)
    feats = {}

    def feat(spec):
        if spec not in feats:
            names, X, _ = featsets.load(spec, "full"); assert names == df.image_relpath.tolist()
            feats[spec] = featsets.standardize_to_tensor(X, tr, "cpu" if X.nbytes > 12e9 else dev, torch.float16, pin=False)
            del X
        return feats[spec]

    def load(spec, seed):
        """(tree, head, head_args, x, run name) for the Phase-4 run of this branch and seed."""
        for d in sorted((REPO / "outputs/experiments" / args.exp).glob(f"RUN-*-seed{seed}")):
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            if cfg.get("features") != spec or not (d / "checkpoints/best.pt").exists():
                continue
            if (args.mode and cfg["mode"] != args.mode) or (args.base and cfg["base"] != args.base):
                continue
            src = (d / "source_runs.txt").read_text().split()[0]
            scfg = yaml.safe_load((REPO / "outputs/experiments" / cfg["src_exp"] / src / "config.yaml").read_text())
            tree = P.HierTransformer(tuple(int(v) for v in scfg["token_shape"].split("x")), scfg["depth"], dropout=scfg["dropout"]).to(dev)
            tree.load_state_dict(torch.load(REPO / "outputs/experiments" / cfg["src_exp"] / src / "checkpoints/best.pt", map_location=dev)["model"])
            tree.eval()
            if cfg["mode"] == "residual":
                head = S.ResidualHead040(layers=cfg["layers"]).to(dev)
            else:
                head = S.FlowHead(layers=cfg["layers"], cell=cfg["base"] == "local").to(dev)
            head.load_state_dict(torch.load(d / "checkpoints/best.pt", map_location=dev)["head"]); head.eval()
            return tree, head, cfg, feat(spec), d.name
        raise FileNotFoundError(spec, seed)

    rows, used, fpreds = [], [], []
    for seed in [int(s) for s in args.seeds.split(",")]:
        A = load(args.a, seed); B = load(args.b, seed); used += [A[4], B[4]]
        mode, base = A[2]["mode"], A[2]["base"]; assert (mode, base) == (B[2]["mode"], B[2]["base"])
        sig_b = np.radians(A[2]["sigma_b_deg"])
        for name, members in [("A", [A]), ("B", [B]), ("fused", [A, B])]:
            for g, msk in groups.items():
                idx = np.flatnonzero(msk); pts, qcs, smps = [], [], []; t0 = time.time()
                bs = 512 if mode == "residual" else max(8, 4096 // args.m)
                with torch.no_grad():
                    for s in range(0, len(idx), bs):
                        b = torch.as_tensor(idx[s:s + bs], device=dev)
                        xs = [(m[3][b] if m[3].is_cuda else m[3][b.cpu()].to(dev, non_blocking=True)).float() for m in members]
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            c, p, hs = fused_beam([m[0] for m in members], xs, "poe" if len(members) > 1 else "single", 1, return_h=True)
                        mems = [h.float() for h in hs]; qc = decode_torch(c, p).float(); qcs.append(qc)
                        if mode == "residual":
                            delta = torch.stack([m[1](mem, qc) for m, mem in zip(members, mems)]).mean(0)
                            pts.append(S.qcanon(P.qmul(qc, P.rotvec_to_quat(delta)))); continue
                        M = args.m; qck = qc.repeat_interleave(M, 0); memk = [mem.repeat_interleave(M, 0) for mem in mems]
                        q0 = S.uniform_quats(len(qck), dev) if base == "uniform" else S.local_quats(qck, sig_b)
                        cond = qck if base == "local" else None
                        vel = lambda qq, tt: torch.stack([m[1](mk, qq, tt, cond) for m, mk in zip(members, memk)]).mean(0)  # noqa: E731
                        smp = S.integrate(vel, q0, args.nfe).view(len(b), M, 4)
                        smps.append(smp); pts.append(S.kde_mode(smp))
                torch.cuda.synchronize(); dt = (time.time() - t0) / len(idx)
                pt, qc = torch.cat(pts), torch.cat(qcs); qg = Qt[torch.as_tensor(idx, device=dev)]
                ept = S.geo_deg(pt, qg).cpu().numpy(); ec = S.geo_deg(qc, qg).cpu().numpy()
                row = {"seed": seed, "method": name, "domain": g, "latency_ms_per_image": 1000 * dt,
                       **rotation_metrics(pt.cpu().numpy(), q[idx]), "leaf_mean_deg": float(ec.mean()), "leaf_median_deg": float(np.median(ec))}
                if smps:
                    smp = torch.cat(smps); es = S.geo_deg(smp, qg[:, None]).cpu().numpy(); dmode = S.geo_deg(smp, pt[:, None]).cpu().numpy()
                    row.update({"best_of_M_mean_deg": float(es.min(1).mean()), "sample_mean_deg": float(es.mean()),
                                **{f"any_sample_within_{t}": float((es.min(1) <= t).mean()) for t in (5, 10, 20)},
                                "spread_deg": float(dmode.mean()),
                                **{f"coverage@{a}": float((ept <= np.quantile(dmode, a, axis=1)).mean()) for a in (0.5, 0.8, 0.9, 0.95)}})
                    if args.save_samples and name == "fused" and seed == 0:
                        np.save(run.sub("samples") / f"{g}_fused_seed0_fp16.npy", smp.half().cpu().numpy())
                        np.save(run.sub("samples") / f"{g}_index.npy", idx)
                rows.append(row)
                if name == "fused" and seed == 0:              # per-image predictions for figures / later analysis
                    fpreds.append(pd.DataFrame({"image_relpath": df.image_relpath.values[idx], "group": g, "err_deg": ept, "leaf_err_deg": ec,
                                                **{f"q_pred_{k}": pt[:, i].cpu().numpy() for i, k in enumerate("wxyz")}}))
                print(name, seed, g, f"point {ept.mean():.2f}/{np.median(ept):.2f}  leaf {ec.mean():.2f}", flush=True)
    if fpreds:
        pd.concat(fpreds).to_csv(run.sub("predictions") / "fused_seed0.csv", index=False, float_format="%.6g")
    per = pd.DataFrame(rows); run.write_metrics(rows, "per_seed")
    num = [c for c in per.columns if c not in ("seed", "method", "domain")]
    agg = per.groupby(["method", "domain"])[num].mean().join(per.groupby(["method", "domain"])[["mean_deg"]].std().rename(columns={"mean_deg": "mean_deg_std"})).reset_index()
    run.write_metrics(agg.to_dict("records"), "metrics"); (run.sub("tables") / "metrics.md").write_text(md_table(agg.to_dict("records")))
    (run.dir / "source_runs.txt").write_text("\n".join(used) + "\n"); run.done()
    print(md_table(agg.to_dict("records"), ["method", "domain", "mean_deg", "mean_deg_std", "median_deg", "acc@10", "leaf_mean_deg"]))
    print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
