"""Supplementary flow figures: per-example data behind EXP-107 F1/F2/F3 (not saved by that run), recomputed and stored in
predictions/flow_examples.npz. Same models (EXP-036 clean PoE trees + EXP-042 fused uniform-base SO(3) flow, seed 0), same pre-registered
examples (EXP-107 tables/examples.csv), same order and RNG seed as exp107_figures.py, so the flow samples match the EXP-107 figures."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import featsets, phase3 as P, so3flow as S, tesseract as T  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.runlog import Run  # noqa: E402

run = Run("FIG-flow", "supplementary_flow_examples", {"stage": "figures"}, seed=0)
SPECS = ["dinov3_vitl16:tokens8", "moge2_vitl:normals16~and"]; NFE = 10; FLOW = "EXP-042_image_conditioned_so3_posterior"
E107 = REPO / "outputs/experiments/EXP-107_paper_visuals_metrics_efficiency/RUN-20261016-024840-seed0"
try:
    dev = torch.device("cuda")
    df = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv"); q = df[[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    tr = ((df.domain == "synthetic") & (df.split == "train")).to_numpy()
    ex = pd.read_csv(E107 / "tables/examples.csv"); rows = ex.row.to_numpy(); X = {}
    for sp in SPECS:                                                             # keep only the example rows, with the train statistics
        names, x, _ = featsets.load(sp, "full"); assert names == df.image_relpath.tolist()
        X[sp] = featsets.standardize_to_tensor(x[rows], None, "cpu", torch.float16, stats=featsets.train_stats(x, tr)); del x
    trees, heads = [], []
    for spec in SPECS:
        for d in sorted((REPO / "outputs/experiments" / FLOW).glob("RUN-*-seed0")):
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            if cfg.get("features") != spec or cfg.get("mode") != "flow" or cfg.get("base") != "uniform" or not (d / "checkpoints/best.pt").exists():
                continue
            sd = REPO / "outputs/experiments" / cfg["src_exp"] / (d / "source_runs.txt").read_text().split()[0]
            scfg = yaml.safe_load((sd / "config.yaml").read_text())
            t = P.HierTransformer(tuple(int(v) for v in scfg["token_shape"].split("x")), scfg["depth"], dropout=scfg["dropout"]).to(dev)
            t.load_state_dict(torch.load(sd / "checkpoints/best.pt", map_location=dev)["model"]); t.eval()
            h = S.FlowHead(layers=cfg["layers"]).to(dev); h.load_state_dict(torch.load(d / "checkpoints/best.pt", map_location=dev)["head"]); h.eval()
            trees.append(t); heads.append(h); break
        else:
            raise FileNotFoundError(spec)
    out = {}; torch.manual_seed(0)
    for i, (g, rule, row, val) in enumerate(ex.itertuples(index=False)):
        xs = [X[sp][torch.as_tensor([i])].to(dev).float() for sp in SPECS]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            c, p, sc, hs = fused_beam(trees, xs, "poe", 512, return_all=True)
        c, p = c[0].cpu().numpy(), p[0].cpu().numpy(); pr = torch.softmax(sc[0].float(), 0).cpu().numpy()
        with torch.no_grad():
            mk = [m.float().repeat_interleave(256, 0) for m in hs]
            vel = lambda qq, tt: torch.stack([hh(m, qq, tt, None) for hh, m in zip(heads, mk)]).mean(0)
            qq = S.uniform_quats(256, dev); traj = [qq.cpu().numpy()]
            for k in range(NFE):
                tt = torch.full((256,), k / NFE, device=dev)
                qq = S.qcanon(P.qmul(qq, P.rotvec_to_quat(vel(qq, tt) / NFE))); qq = qq / qq.norm(dim=-1, keepdim=True); traj.append(qq.cpu().numpy())
        lev = {}
        for l in range(1, 6):                                                    # coarse-to-fine mass of the top-512 leaves
            key = c * 8 ** l + T.leaf_id(np.zeros_like(c), p[:, :l]); uk, inv = np.unique(key, return_inverse=True); mass = np.bincount(inv, weights=pr)
            first = np.array([np.flatnonzero(inv == j)[0] for j in range(len(uk))]); lev[f"lev{l}_q"] = T.decode(c[first], p[first, :l]); lev[f"lev{l}_mass"] = mass
        tag = f"{g}_{rule}"
        out.update({f"{tag}__traj": np.stack(traj), f"{tag}__leaf_q": T.decode(c, p), f"{tag}__leaf_p": pr, f"{tag}__gt": q[row], **{f"{tag}__{k}": v for k, v in lev.items()}})
        print(tag, flush=True)
    np.savez_compressed(run.sub("predictions") / "flow_examples.npz", **out); run.done(); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
