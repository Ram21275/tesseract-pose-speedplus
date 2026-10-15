"""EXP-111 arm "verifier" (EXP-045 design, single DINOv3 branch): candidates = beam-8 leaves of the EXP-109 arm-A DINOv3
tree x {I, 180 deg about body x, y, z} + the 8 corners of each leaf cell (96, = EXP-105 C3); score = re-noise / return
with the EXP-111 flow head of the same seed (tau 0.8, 16 draws, NFE 10): -mean geodesic distance of the returned draws to
the candidate; prediction = argmax. 3 seeds; flip transitions vs the same tree's greedy leaf (with denominators)."""
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
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

E111, TAU, D, NFE, CH = "EXP-111_dino_aug_flow", 0.8, 16, 10, 24
run = Run("EXP-111", "dino_aug_flow", {"stage": "verifier", "tau": TAU, "draws": D, "nfe": NFE}, seed=0)
try:
    dev = torch.device("cuda"); torch.manual_seed(0)
    df = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv"); q = df[[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    tr = ((df.domain == "synthetic") & (df.split == "train")).to_numpy()
    groups = {"synthetic_val": ((df.domain == "synthetic") & (df.split == "validation")).to_numpy(),
              "lightbox": (df.domain == "lightbox").to_numpy(), "sunlamp": (df.domain == "sunlamp").to_numpy()}
    names, x, _ = featsets.load("dinov3_vitl16:tokens8", "full"); assert names == df.image_relpath.tolist()
    X = featsets.standardize_to_tensor(x, tr, "cpu", torch.float16); del x
    FL = torch.eye(4, device=dev)
    geo = lambda a, b: np.degrees(2 * np.arccos(np.clip(np.abs((a * b).sum(-1)), 0, 1)))
    rows = []
    for seed in (0, 1, 2):
        d = [d for d in sorted((REPO / "outputs/experiments" / E111).glob(f"RUN-*-seed{seed}"))
             if yaml.safe_load((d / "config.yaml").read_text()).get("mode") == "flow" and (d / "checkpoints/best.pt").exists()][-1]
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        sd = REPO / "outputs/experiments" / cfg["src_exp"] / (d / "source_runs.txt").read_text().split()[0]
        tree = P.HierTransformer((64, 1024), 5).to(dev); tree.load_state_dict(torch.load(sd / "checkpoints/best.pt", map_location=dev)["model"]); tree.eval()
        head = S.FlowHead(layers=cfg["layers"]).to(dev); head.load_state_dict(torch.load(d / "checkpoints/best.pt", map_location=dev)["head"]); head.eval()

        @torch.no_grad()
        def renoise(mem, cand):                                         # mem (B,T,d), cand (B,K,4) -> (B,K)
            B, K, _ = cand.shape; mk = mem.repeat_interleave(K * D, 0)
            qc = cand.reshape(-1, 4).repeat_interleave(D, 0); q0 = S.uniform_quats(len(qc), dev)
            qq = P.qmul(q0, P.rotvec_to_quat(TAU * S.relative_rotvec(q0, qc))); dt = 1.0 / NFE
            for i in range(int(round(TAU * NFE)), NFE):
                tt = torch.full((len(qq),), i * dt, device=dev)
                qq = S.qcanon(P.qmul(qq, P.rotvec_to_quat(dt * head(mk, qq, tt, None)))); qq = qq / qq.norm(dim=-1, keepdim=True)
            return -S.geo_deg(qq, qc).view(B, K, D).mean(-1)

        for g, msk in groups.items():
            idx = np.flatnonzero(msk); eg, ev, t_all = [], [], 0.0
            for s in range(0, len(idx), 16):
                b = idx[s:s + 16]; xb = X[torch.as_tensor(b)].to(dev).float(); B = len(b)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    c8, p8, _, hs = fused_beam([tree], [xb], "poe", 8, return_all=True)
                mem = hs[0].float(); c8n, p8n = c8.cpu().numpy().reshape(-1), p8.cpu().numpy().reshape(B * 8, 5)
                cen = torch.as_tensor(T.decode(c8n, p8n), dtype=torch.float32, device=dev).view(B, 8, 4)
                cor = torch.as_tensor(T.cell_corners(c8n, p8n), dtype=torch.float32, device=dev).view(B, 64, 4)
                fl = P.qmul(cen[:, :, None].expand(B, 8, 4, 4).reshape(-1, 4), FL.repeat(B * 8, 1)).view(B, 32, 4)
                cand = S.qcanon(torch.cat([fl, cor], 1))                                   # (B, 96, 4)
                sc = torch.cat([renoise(mem, cand[:, k:k + CH]) for k in range(0, 96, CH)], 1)
                pick = cand[torch.arange(B), sc.argmax(1)].cpu().numpy()
                torch.cuda.synchronize(); t_all += time.perf_counter() - t0
                eg.append(geo(cen[:, 0].cpu().numpy(), q[b])); ev.append(geo(pick, q[b]))
            e0, e = np.concatenate(eg), np.concatenate(ev); n150, n20 = int((e0 > 150).sum()), int((e0 < 20).sum())
            for name, ee in (("tree beam-8 top-1", e0), ("tree + flow verifier (EXP-045 design)", e)):
                rows.append({"seed": seed, "domain": g, "system": name, "mean_deg": ee.mean(), "median_deg": np.median(ee), "acc@10": float((ee <= 10).mean()),
                             "frac_gt90": float((ee > 90).mean()), "frac_gt150": float((ee > 150).mean()),
                             "n_tree_gt150": n150, "frac_gt150_to_lt20": float(((e0 > 150) & (ee < 20)).sum() / max(n150, 1)),
                             "n_tree_lt20": n20, "frac_lt20_to_gt150": float(((e0 < 20) & (ee > 150)).sum() / max(n20, 1)),
                             "ms_per_image_incl_tree": 1000 * t_all / len(idx)})
            print(seed, g, f"tree {e0.mean():.2f} verifier {e.mean():.2f}", flush=True)
    per = pd.DataFrame(rows); run.write_metrics(rows, "verifier_per_seed")
    cols = [c for c in per.columns if c not in ("seed", "domain", "system")]
    agg = per.groupby(["domain", "system"])[cols].mean().join(per.groupby(["domain", "system"])[["mean_deg"]].std().rename(columns={"mean_deg": "mean_deg_std"})).reset_index()
    run.write_metrics(agg.to_dict("records"), "verifier"); (run.sub("tables") / "verifier.md").write_text(md_table(agg.to_dict("records")))
    run.done(); print(md_table(agg.to_dict("records"), ["domain", "system", "mean_deg", "mean_deg_std", "median_deg", "frac_gt150", "n_tree_gt150", "frac_gt150_to_lt20", "frac_lt20_to_gt150", "ms_per_image_incl_tree"]))
    print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
