"""EXP-105: candidate expansion (180 deg body-axis flips, leaf-cell corners) + best-fit selection on SPEED+.

All models frozen: EXP-036 PoE trees and EXP-042 fused flow heads (same seed). Per image, one union of candidates is
scored once and every candidate arm is a subset of it:
  U = 8 beam centres x {I, Rx(180), Ry(180), Rz(180)} (32, body frame q (x) f)  +  8 x 8 leaf-cell corners (64)
  G = GT leaf centre x {I, X, Y, Z} (4; flip-discrimination metric only, never used for prediction)
Arms: C0 centres (8) | C1 flips (32) | C2 centres + corners (72) | C3 all (96).
Scorers: tree (PoE leaf log-prob) | flow_logp (exact CNF log-density, Euler backwards, autograd divergence) |
  renoise_t{tau} (SDEdit return: -mean geodesic distance of re-noised-and-integrated draws to the candidate) |
  tmpl_dino / tmpl_normals (mean per-token cosine to the k nearest-rotation synthetic-TRAIN images).
DEC-006: synthetic val selects; lightbox / sunlamp descriptive.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import featsets, phase3 as P, so3flow as S, tesseract as T  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--flow-exp", default="EXP-042_image_conditioned_so3_posterior")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--groups", default="synthetic_val,lightbox,sunlamp")
ap.add_argument("--nfe", type=int, default=10)
ap.add_argument("--renoise-draws", type=int, default=16)
ap.add_argument("--taus", default="0.5,0.8")
ap.add_argument("--k-tmpl", type=int, default=8)
ap.add_argument("--batch", type=int, default=16, help="images per batch (x 100 candidates)")
ap.add_argument("--chunk", type=int, default=25, help="candidates per flow-scorer call (GPU memory)")
ap.add_argument("--limit", type=int, default=0, help="debug: first n images per group")
args = ap.parse_args()
run = Run("EXP-105", "candidate_expansion_best_fit", vars(args), seed=0)
L = 5
SPECS = ["dinov3_vitl16:tokens8", "moge2_vitl:normals16~and"]
TAUS = [float(t) for t in args.taus.split(",")]
try:
    dev = torch.device("cuda")
    torch.manual_seed(0)
    df = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv")
    q = df[[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    chart, path = T.encode(q, L)
    tr = ((df.domain == "synthetic") & (df.split == "train")).to_numpy(); tr_idx = np.flatnonzero(tr)
    allg = {"synthetic_val": ((df.domain == "synthetic") & (df.split == "validation")).to_numpy(),
            "lightbox": (df.domain == "lightbox").to_numpy(), "sunlamp": (df.domain == "sunlamp").to_numpy()}
    groups = {g: allg[g] for g in args.groups.split(",")}
    X = {}
    for sp in SPECS:
        names, x, _ = featsets.load(sp, "full"); assert names == df.image_relpath.tolist()
        if sp == SPECS[1]:
            nrm = torch.as_tensor(x.reshape(len(x), 256, 4)[:, :, :3].astype(np.float32))      # mask-weighted normals (raw)
        X[sp] = featsets.standardize_to_tensor(x, tr, "cpu", torch.float16); del x
    # template banks (synthetic train only): unit-normalised DINO tokens on the GPU, unit normals on the CPU
    tok = X[SPECS[0]].view(len(df), 64, 1024)
    TD = torch.empty(len(tr_idx), 64, 1024, dtype=torch.float16, device=dev)
    for i in range(0, len(tr_idx), 2048):
        TD[i:i + 2048] = F.normalize(tok[torch.as_tensor(tr_idx[i:i + 2048])].to(dev).float(), dim=-1).half()
    NU = F.normalize(nrm, dim=-1)                      # zero vectors (background) stay zero
    NW = (nrm.norm(dim=-1) > 1e-3).float()             # foreground tokens
    QTR = torch.as_tensor(q[tr_idx], dtype=torch.float32, device=dev)
    FLIPS = torch.tensor([[1., 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], device=dev)

    def load_seed(seed):
        """(trees [A, B], flow heads [A, B]) from the EXP-042 runs of this seed (trees = their source EXP-036 runs)."""
        trees, heads, used = [], [], []
        for spec in SPECS:
            for d in sorted((REPO / "outputs/experiments" / args.flow_exp).glob(f"RUN-*-seed{seed}")):
                cfg = yaml.safe_load((d / "config.yaml").read_text())
                if cfg.get("features") != spec or not (d / "checkpoints/best.pt").exists() or cfg.get("mode") != "flow":
                    continue
                src = (d / "source_runs.txt").read_text().split()[0]
                sd = REPO / "outputs/experiments" / cfg["src_exp"] / src
                scfg = yaml.safe_load((sd / "config.yaml").read_text())
                t = P.HierTransformer(tuple(int(v) for v in scfg["token_shape"].split("x")), scfg["depth"], dropout=scfg["dropout"]).to(dev)
                t.load_state_dict(torch.load(sd / "checkpoints/best.pt", map_location=dev)["model"]); t.eval()
                h = S.FlowHead(layers=cfg["layers"], cell=cfg["base"] == "local").to(dev)
                h.load_state_dict(torch.load(d / "checkpoints/best.pt", map_location=dev)["head"]); h.eval()
                assert cfg["base"] == "uniform"
                trees.append(t); heads.append(h); used += [f"{cfg['src_exp']}/{src}", f"{args.flow_exp}/{d.name}"]
                break
            else:
                raise FileNotFoundError(spec, seed)
        return trees, heads, used

    def tree_logp(trees, xs, cand):
        """PoE log-prob of each candidate's leaf. xs: per-branch (B, D); cand (B, K, 4) -> (B, K)."""
        B, K, _ = cand.shape
        c, p = T.encode(cand.reshape(-1, 4).double().cpu().numpy(), L)
        c, p = torch.as_tensor(c, device=dev), torch.as_tensor(p, device=dev)
        outs = [m.logits_tf(m.encode(x).repeat_interleave(K, 0), c, p) for m, x in zip(trees, xs)]
        root = F.log_softmax(sum(F.log_softmax(o[0].float(), -1) for o in outs), -1)
        child = F.log_softmax(sum(F.log_softmax(o[1].float(), -1) for o in outs), -1)
        return (root.gather(1, c[:, None])[:, 0] + child.gather(2, p[:, :, None])[..., 0].sum(1)).view(B, K)

    def make_vel(heads, mems, rep):
        mk = [m.float().repeat_interleave(rep, 0) for m in mems]
        return lambda qq, tt: torch.stack([h(m, qq, tt, None) for h, m in zip(heads, mk)]).mean(0)

    def flow_logp(heads, mems, cand):
        """log p_1(cand) up to the constant Haar log-density: -sum_i dt * div omega, integrating backwards (Euler)."""
        B, K, _ = cand.shape
        vel = make_vel(heads, mems, K); qq = cand.reshape(-1, 4).float(); dt = 1.0 / args.nfe
        lp = torch.zeros(len(qq), device=dev)
        for i in reversed(range(args.nfe)):
            tt = torch.full((len(qq),), i * dt, device=dev)
            eps = torch.zeros(len(qq), 3, device=dev, requires_grad=True)
            with torch.enable_grad():
                qe = P.qmul(qq, torch.cat([torch.ones(len(qq), 1, device=dev), 0.5 * eps], -1))   # first order q (x) Exp(eps)
                w = vel(qe, tt)
                div = sum(torch.autograd.grad(w[:, j].sum(), eps, retain_graph=j < 2)[0][:, j] for j in range(3))
            lp -= dt * div.detach()
            qq = S.qcanon(P.qmul(qq, P.rotvec_to_quat(-dt * w.detach())))
            qq = qq / qq.norm(dim=-1, keepdim=True)
        return lp.view(B, K)

    @torch.no_grad()
    def renoise(heads, mems, cand, tau):
        """-mean geodesic distance (deg) back to the candidate after re-noising to tau and integrating to 1."""
        B, K, _ = cand.shape; D = args.renoise_draws
        vel = make_vel(heads, mems, K * D)
        qc = cand.reshape(-1, 4).float().repeat_interleave(D, 0)
        q0 = S.uniform_quats(len(qc), dev)
        qq = P.qmul(q0, P.rotvec_to_quat(tau * S.relative_rotvec(q0, qc)))
        dt = 1.0 / args.nfe
        for i in range(int(round(tau * args.nfe)), args.nfe):
            tt = torch.full((len(qq),), i * dt, device=dev)
            qq = S.qcanon(P.qmul(qq, P.rotvec_to_quat(dt * vel(qq, tt)))); qq = qq / qq.norm(dim=-1, keepdim=True)
        return -S.geo_deg(qq, qc).view(B, K, D).mean(-1)

    @torch.no_grad()
    def templates(b, cand):
        """Mean per-token cosine to the k nearest-rotation synthetic-train images: (dino, normals), each (B, K)."""
        B, K, _ = cand.shape
        nn_ = (cand.reshape(-1, 4).float() @ QTR.T).abs().topk(args.k_tmpl, -1).indices          # (B*K, k) rows of tr_idx
        a = F.normalize(tok[b.cpu()].to(dev).float(), dim=-1).half()                               # (B, 64, 1024)
        sd = torch.empty(B, K, device=dev)
        for i in range(B):
            g = TD[nn_[i * K:(i + 1) * K].reshape(-1)]                                             # (K*k, 64, 1024)
            sd[i] = torch.einsum("td,ntd->n", a[i].float(), g.float()).view(K, -1).mean(-1) / 64
        rows = torch.as_tensor(tr_idx, device=dev)[nn_].cpu()                                     # (B*K, k) image ids
        na, wa = NU[b.cpu()], NW[b.cpu()]                                                          # (B, 256, 3), (B, 256)
        nt, wt = NU[rows], NW[rows]                                                                # (B*K, k, 256, ...)
        na = na.repeat_interleave(K, 0)[:, None]; wa = wa.repeat_interleave(K, 0)[:, None]
        w = wa * wt
        sn = ((na * nt).sum(-1) * w).sum(-1) / w.sum(-1).clamp_min(1.0)
        return sd, sn.mean(-1).view(B, K).to(dev)

    def geo(qa, qb):
        return S.geo_deg(qa, qb)

    ARMS = {"C0_centres": list(range(0, 32, 4)), "C1_flips": list(range(32)),
            "C2_centres+corners": list(range(0, 32, 4)) + list(range(32, 96)), "C3_all": list(range(96))}
    SCORERS = ["tree", "flow_logp"] + [f"renoise_t{t:g}" for t in TAUS] + ["tmpl_dino", "tmpl_normals"]
    rows, disc_rows, used = [], [], []
    store = {}
    for seed in [int(s) for s in args.seeds.split(",")]:
        trees, heads, u = load_seed(seed); used += u
        for g, msk in groups.items():
            idx = np.flatnonzero(msk)[: args.limit or None]
            sc = {s: [] for s in SCORERS}; errs, gerrs, cds, times = [], [], [], {s: 0.0 for s in SCORERS}
            for s0 in range(0, len(idx), args.batch):
                bi = idx[s0:s0 + args.batch]; b = torch.as_tensor(bi, device=dev); B = len(b)
                xs = [X[sp][b.cpu()].to(dev).float() for sp in SPECS]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    c8, p8, _, hs = fused_beam(trees, xs, "poe", 8, return_all=True)
                c8n, p8n = c8.cpu().numpy().reshape(-1), p8.cpu().numpy().reshape(B * 8, L)
                cen = torch.as_tensor(T.decode(c8n, p8n), dtype=torch.float32, device=dev).view(B, 8, 4)
                cor = torch.as_tensor(T.cell_corners(c8n, p8n), dtype=torch.float32, device=dev).view(B, 64, 4)
                fl = P.qmul(cen[:, :, None].expand(B, 8, 4, 4).reshape(-1, 4), FLIPS.repeat(B * 8, 1)).view(B, 32, 4)
                gc = torch.as_tensor(T.decode(chart[bi], path[bi]), dtype=torch.float32, device=dev)
                gfl = P.qmul(gc[:, None].expand(B, 4, 4).reshape(-1, 4), FLIPS.repeat(B, 1)).view(B, 4, 4)
                cand = S.qcanon(torch.cat([fl, cor, gfl], 1))                                       # (B, 100, 4)
                qg = torch.as_tensor(q[bi], dtype=torch.float32, device=dev)
                cds.append(geo(cen[:, :, None].expand(-1, -1, 8, -1).reshape(-1, 4), cor.reshape(-1, 4)).cpu())
                errs.append(geo(cand[:, :96], qg[:, None]).cpu()); gerrs.append(geo(gfl, qg[:, None]).cpu())
                mems = [h.float() for h in hs]
                for s in SCORERS:
                    torch.cuda.synchronize(); t0 = time.time()
                    if s == "tree":
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            v = tree_logp(trees, xs, cand)
                    elif s == "flow_logp":
                        v = torch.cat([flow_logp(heads, mems, cand[:, k:k + args.chunk]) for k in range(0, cand.shape[1], args.chunk)], 1)
                    elif s.startswith("renoise"):
                        v = torch.cat([renoise(heads, mems, cand[:, k:k + args.chunk], float(s.split("_t")[1]))
                                       for k in range(0, cand.shape[1], args.chunk)], 1)
                    elif s == "tmpl_dino":
                        v, vn = templates(b, cand)
                    else:
                        v = vn
                    torch.cuda.synchronize(); times[s] += time.time() - t0
                    sc[s].append(v.float().cpu())
                if s0 // args.batch % 50 == 0:
                    print(f"seed {seed} {g} {s0}/{len(idx)}", flush=True)
            E = torch.cat(errs).numpy(); GE = torch.cat(gerrs).numpy()
            for s in SCORERS:
                V = torch.cat(sc[s]).numpy(); store[f"s{seed}_{g}_{s}"] = V.astype(np.float32)
                disc_rows.append({"seed": seed, "domain": g, "scorer": s, "flip_disc_acc": float((V[:, 96:].argmax(1) == 0).mean()),
                                  "ms_per_image": 1000 * times[s] / len(idx)})
                for arm, ii in ARMS.items():
                    pick = np.asarray(ii)[V[:, ii].argmax(1)]; e = E[np.arange(len(E)), pick]
                    rows.append({"seed": seed, "domain": g, "arm": arm, "scorer": s, "mean_deg": e.mean(), "median_deg": np.median(e),
                                 "p95_deg": float(np.percentile(e, 95)), "acc@10": float((e <= 10).mean()),
                                 "frac_gt90": float((e > 90).mean()), "frac_gt150": float((e > 150).mean())})
            for arm, ii in ARMS.items():
                e = E[:, ii].min(1)
                rows.append({"seed": seed, "domain": g, "arm": arm, "scorer": "ORACLE", "mean_deg": e.mean(), "median_deg": np.median(e),
                             "p95_deg": float(np.percentile(e, 95)), "acc@10": float((e <= 10).mean()),
                             "frac_gt90": float((e > 90).mean()), "frac_gt150": float((e > 150).mean())})
            e = E[:, 0]
            rows.append({"seed": seed, "domain": g, "arm": "beam8_top1", "scorer": "PoE (EXP-036)", "mean_deg": e.mean(), "median_deg": np.median(e),
                         "p95_deg": float(np.percentile(e, 95)), "acc@10": float((e <= 10).mean()),
                         "frac_gt90": float((e > 90).mean()), "frac_gt150": float((e > 150).mean())})
            store[f"s{seed}_{g}_err"] = E.astype(np.float32); store[f"s{seed}_{g}_index"] = idx
            cd = torch.cat(cds)
            print(seed, g, "beam8_top1", f"{E[:, 0].mean():.2f}", "| centre-corner deg", f"{cd.mean().item():.2f}", flush=True)
        if seed == 0:
            run.write_metrics({"leaf_centre_to_corner_deg_mean": float(cd.mean()), "leaf_centre_to_corner_deg_max": float(cd.max())},
                              "cell_geometry")
        del trees, heads; torch.cuda.empty_cache()
    per = pd.DataFrame(rows); run.write_metrics(rows, "per_seed")
    agg = per.groupby(["domain", "arm", "scorer"])[["mean_deg", "median_deg", "p95_deg", "acc@10", "frac_gt90", "frac_gt150"]].mean().join(
        per.groupby(["domain", "arm", "scorer"])[["mean_deg"]].std().rename(columns={"mean_deg": "mean_deg_std"})).reset_index()
    run.write_metrics(agg.to_dict("records"), "metrics"); (run.sub("tables") / "metrics.md").write_text(md_table(agg.to_dict("records")))
    dd = pd.DataFrame(disc_rows); run.write_metrics(disc_rows, "flip_discrimination_per_seed")
    dagg = dd.groupby(["domain", "scorer"])[["flip_disc_acc", "ms_per_image"]].mean().join(
        dd.groupby(["domain", "scorer"])[["flip_disc_acc"]].std().rename(columns={"flip_disc_acc": "flip_disc_std"})).reset_index()
    run.write_metrics(dagg.to_dict("records"), "flip_discrimination"); (run.sub("tables") / "flip_discrimination.md").write_text(md_table(dagg.to_dict("records")))
    np.savez_compressed(run.sub("predictions") / "scores.npz", **store)
    (run.dir / "source_runs.txt").write_text("\n".join(used) + "\n"); run.done()
    print(md_table(dagg.to_dict("records")))
    print(md_table(agg[agg.domain == "synthetic_val"].to_dict("records"), ["arm", "scorer", "mean_deg", "mean_deg_std", "frac_gt90"]))
    print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
