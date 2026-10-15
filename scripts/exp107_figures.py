"""EXP-107: paper figures + comparison-ready metrics (pre-registered in outputs/reports/EXP-107_*.md).

GPU-light: frozen EXP-036 trees + EXP-042 fused flow heads.
  - log-likelihood of GT, 3 seeds: tree = log(P_PoE(GT leaf) / vol(leaf)), flow = exact CNF log-density
    (vol / density w.r.t. the Haar measure normalised to total volume pi^2, so uniform = -log(pi^2) = -2.29, as Implicit-PDF)
  - example images by the pre-registered rule (seed 0): per domain the 10/50/90th-percentile PoE greedy error,
    plus the lightbox image with the most flow-sample mass within 30 deg of a 180-deg body-axis flip of GT
  - F1 SO(3) plots (flow 256 samples | tree top-64 leaves), F2 trajectories (+ mp4), F3 coarse-to-fine levels
  - F4 speed-accuracy, F5 error histograms, F6 calibration (from the sweep runs and their per-image predictions)
"""
import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import featsets, phase3 as P, so3flow as S, tesseract as T, vis_so3 as V  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--flow-exp", default="EXP-042_image_conditioned_so3_posterior")
ap.add_argument("--sweep-exp", default="EXP-107_paper_visuals_metrics_efficiency")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--nfe", type=int, default=10)
ap.add_argument("--chunk", type=int, default=2048)
args = ap.parse_args()
run = Run("EXP-107", "paper_visuals_metrics_efficiency", {**vars(args), "stage": "figures"}, seed=0)
L = 5
SPECS = ["dinov3_vitl16:tokens8", "moge2_vitl:normals16~and"]
DOM = ["synthetic_val", "lightbox", "sunlamp"]
FIG = run.sub("figures")
try:
    dev = torch.device("cuda"); torch.manual_seed(0)
    df = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv")
    q = df[[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    chart, path = T.encode(q, L)
    tr = ((df.domain == "synthetic") & (df.split == "train")).to_numpy()
    groups = {"synthetic_val": ((df.domain == "synthetic") & (df.split == "validation")).to_numpy(),
              "lightbox": (df.domain == "lightbox").to_numpy(), "sunlamp": (df.domain == "sunlamp").to_numpy()}
    X = {}
    for sp in SPECS:
        names, x, _ = featsets.load(sp, "full"); assert names == df.image_relpath.tolist()
        X[sp] = featsets.standardize_to_tensor(x, tr, "cpu", torch.float16); del x

    # ---- leaf-cell volumes (Haar, total pi^2): integral of (1+|u|^2)^-2 over the cell, 4^3-point Gauss-Legendre ----
    gx, gw = np.polynomial.legendre.leggauss(4)
    def cell_volume(c, p):
        lo, hi = T._cell_bounds(c, p)                                           # (N,3)
        u = lo[:, None, :] + (hi - lo)[:, None, :] * (np.stack(np.meshgrid(gx, gx, gx, indexing="ij"), -1).reshape(-1, 3)[None] + 1) / 2
        w = np.prod(np.stack(np.meshgrid(gw, gw, gw, indexing="ij"), -1).reshape(-1, 3), 1)[None] / 8
        return (w * (1 + (u ** 2).sum(-1)) ** -2).sum(1) * np.prod(hi - lo, 1)
    ca, pa = T.all_cells(3); tot = cell_volume(ca, pa).sum()
    assert abs(tot - np.pi ** 2) < 1e-3, tot                                     # sanity: the 4 charts tile SO(3)

    def load_seed(seed):
        trees, heads = [], []
        for spec in SPECS:
            for d in sorted((REPO / "outputs/experiments" / args.flow_exp).glob(f"RUN-*-seed{seed}")):
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
                raise FileNotFoundError(spec, seed)
        return trees, heads

    def vel_fn(heads, mems, rep):
        mk = [m.float().repeat_interleave(rep, 0) for m in mems]
        return lambda qq, tt: torch.stack([h(m, qq, tt, None) for h, m in zip(heads, mk)]).mean(0)

    def flow_logp(heads, mems, cand):                                            # (B,4) -> (B,), Haar density, total pi^2
        vel = vel_fn(heads, mems, 1); qq = cand.float(); dt = 1.0 / args.nfe; lp = torch.zeros(len(qq), device=dev)
        for i in reversed(range(args.nfe)):
            tt = torch.full((len(qq),), i * dt, device=dev); eps = torch.zeros(len(qq), 3, device=dev, requires_grad=True)
            with torch.enable_grad():
                w = vel(P.qmul(qq, torch.cat([torch.ones(len(qq), 1, device=dev), 0.5 * eps], -1)), tt)
                div = sum(torch.autograd.grad(w[:, j].sum(), eps, retain_graph=j < 2)[0][:, j] for j in range(3))
            lp -= dt * div.detach()
            qq = S.qcanon(P.qmul(qq, P.rotvec_to_quat(-dt * w.detach()))); qq = qq / qq.norm(dim=-1, keepdim=True)
        return lp - np.log(np.pi ** 2)

    def tree_logp_gt(trees, xs, b):
        c = torch.as_tensor(chart[b], device=dev); p = torch.as_tensor(path[b], device=dev)
        outs = [m.logits_tf(m.encode(x), c, p) for m, x in zip(trees, xs)]
        root = torch.log_softmax(sum(torch.log_softmax(o[0].float(), -1) for o in outs), -1)
        child = torch.log_softmax(sum(torch.log_softmax(o[1].float(), -1) for o in outs), -1)
        return (root.gather(1, c[:, None])[:, 0] + child.gather(2, p[:, :, None])[..., 0].sum(1)).cpu().numpy()

    # ---------------- log-likelihood of GT (3 seeds) + seed-0 greedy errors ----------------
    ll_rows, greedy_err, mems0 = [], {}, None
    for seed in [int(s) for s in args.seeds.split(",")]:
        trees, heads = load_seed(seed)
        for g in DOM:
            idx = np.flatnonzero(groups[g]); lt, lf, ge = [], [], []
            for s0 in range(0, len(idx), args.chunk):
                b = idx[s0:s0 + args.chunk]; xs = [X[sp][torch.as_tensor(b)].to(dev).float() for sp in SPECS]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    lt.append(tree_logp_gt(trees, xs, b))
                    c1, p1, hs = fused_beam(trees, xs, "poe", 1, return_h=True)
                ge.append(np.degrees(2 * np.arccos(np.clip(np.abs((T.decode(c1.cpu().numpy(), p1.cpu().numpy()) * q[b]).sum(1)), 0, 1))))
                lf.append(flow_logp(heads, [h.float() for h in hs], torch.as_tensor(q[b], dtype=torch.float32, device=dev)).cpu().numpy())
            lt = np.concatenate(lt) - np.log(cell_volume(chart[idx], path[idx])); lf = np.concatenate(lf)
            ll_rows.append({"seed": seed, "domain": g, "tree_ll_gt": float(lt.mean()), "flow_ll_gt": float(lf.mean()),
                            "tree_ll_gt_median": float(np.median(lt)), "flow_ll_gt_median": float(np.median(lf))})
            if seed == 0:
                greedy_err[g] = np.concatenate(ge)
            print(seed, g, f"LL tree {lt.mean():.3f} flow {lf.mean():.3f}", flush=True)
        if seed != 0:
            del trees, heads; torch.cuda.empty_cache()
        else:
            T0, H0 = trees, heads
    llp = pd.DataFrame(ll_rows); run.write_metrics(ll_rows, "loglik_per_seed")
    lla = llp.groupby("domain").agg(["mean", "std"]); lla.columns = [f"{a}_{b}" for a, b in lla.columns]; lla = lla.reset_index()
    run.write_metrics(lla.to_dict("records"), "loglik"); (run.sub("tables") / "loglik.md").write_text(md_table(lla.to_dict("records")))

    # ---------------- examples by the pre-registered rule ----------------
    FL = torch.eye(4, device=dev)[1:]                                            # 180 deg about body x, y, z
    sm = np.load(REPO / "outputs/experiments/EXP-043_flow_induced_tesseract_masses/RUN-20261013-004254-seed0/samples/lightbox_fused_seed0_fp16.npy").astype(np.float32)
    sidx = np.load(REPO / "outputs/experiments/EXP-043_flow_induced_tesseract_masses/RUN-20261013-004254-seed0/samples/lightbox_index.npy")
    qg = torch.as_tensor(q[sidx], dtype=torch.float32)
    flips = P.qmul(qg[:, None].expand(-1, 3, 4).reshape(-1, 4), FL.cpu().repeat(len(qg), 1)).view(-1, 3, 4)
    smt = torch.as_tensor(sm); smt = smt / smt.norm(dim=-1, keepdim=True)
    dflip = S.geo_deg(smt[:, :, None].expand(-1, -1, 3, -1).reshape(-1, 4), flips[:, None].expand(-1, sm.shape[1], -1, -1).reshape(-1, 4)).view(len(qg), -1, 3).min(-1).values
    flipmass = (dflip <= 30).float().mean(1).numpy()
    ex = []
    for g in DOM:
        idx = np.flatnonzero(groups[g]); e = greedy_err[g]
        for pc in (10, 50, 90):
            k = int(np.argmin(np.abs(e - np.percentile(e, pc)))); ex.append((g, f"p{pc}", int(idx[k]), float(e[k])))
    k = int(np.argmax(flipmass)); ex.append(("lightbox", "flipmode", int(sidx[k]), float(flipmass[k])))
    pd.DataFrame(ex, columns=["domain", "rule", "row", "value"]).to_csv(run.sub("tables") / "examples.csv", index=False)

    # ---------------- F1 / F2 / F3 per example ----------------
    torch.manual_seed(0)
    for g, rule, row, val in ex:
        b = torch.as_tensor([row]); xs = [X[sp][b].to(dev).float() for sp in SPECS]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            c64, p64, sc64, hs = fused_beam(T0, xs, "poe", 512, return_all=True)
        c64, p64 = c64[0].cpu().numpy(), p64[0].cpu().numpy(); pr = torch.softmax(sc64[0].float(), 0).cpu().numpy()
        cen = T.decode(c64, p64)
        with torch.no_grad():
            vel = vel_fn(H0, [h.float() for h in hs], 256); qq = S.uniform_quats(256, dev); traj = [qq.cpu().numpy()]
            for i in range(args.nfe):
                tt = torch.full((256,), i / args.nfe, device=dev)
                qq = S.qcanon(P.qmul(qq, P.rotvec_to_quat(vel(qq, tt) / args.nfe))); qq = qq / qq.norm(dim=-1, keepdim=True); traj.append(qq.cpu().numpy())
        traj = np.stack(traj)                                                    # (NFE+1, 256, 4)
        gt = q[row][None]
        fig = plt.figure(figsize=(12, 3.6))
        a1 = V.so3_axes(fig, (0.0, 0.05, 0.47, 0.85)); V.plot_so3(a1, traj[-1], q_gt=gt); a1.set_title("SO(3) flow: 256 samples", fontsize=10)
        a2 = V.so3_axes(fig, (0.48, 0.05, 0.47, 0.85)); V.plot_so3(a2, cen[:64], prob=pr[:64] / pr[:64].max() * 0.25, q_gt=gt)
        a2.set_title("Tesseract tree (PoE): top-64 leaves", fontsize=10); V.tilt_wheel(fig, (0.9, 0.02, 0.08, 0.2))
        fig.suptitle(f"{g} {rule} (row {row}, value {val:.2f})", fontsize=9); fig.savefig(FIG / f"F1_{g}_{rule}.png", dpi=200); fig.savefig(FIG / f"F1_{g}_{rule}.pdf"); plt.close(fig)
        fig = plt.figure(figsize=(6.5, 3.6)); a = V.so3_axes(fig, (0.0, 0.05, 1.0, 0.85))
        lon = []; lat = []
        for t in range(traj.shape[0]):
            R = V._q2R(traj[t, :16]); lo, la, _ = V._coords(R); lon.append(lo); lat.append(la)
        lon, lat = np.array(lon), np.array(lat)
        for j in range(16):
            seg = np.abs(np.diff(lon[:, j])) < np.pi                              # do not draw across the map seam
            for t in range(traj.shape[0] - 1):
                if seg[t]:
                    a.plot(lon[t:t + 2, j], lat[t:t + 2, j], color=plt.cm.viridis(t / args.nfe), linewidth=1)
        V.plot_so3(a, traj[-1, :16], q_gt=gt, const_size=14); a.set_title(f"flow trajectories τ: 0 → 1 ({g} {rule})", fontsize=9)
        fig.savefig(FIG / f"F2_{g}_{rule}.png", dpi=200); fig.savefig(FIG / f"F2_{g}_{rule}.pdf"); plt.close(fig)
        frames = FIG / f"F2_{g}_{rule}_frames"; frames.mkdir(exist_ok=True)
        for t in range(traj.shape[0]):
            fig = plt.figure(figsize=(6.5, 3.6)); a = V.so3_axes(fig, (0.0, 0.05, 1.0, 0.85)); V.plot_so3(a, traj[t], q_gt=gt)
            a.set_title(f"τ = {t / args.nfe:.1f}", fontsize=10); fig.savefig(frames / f"{t:03d}.png", dpi=120); plt.close(fig)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "4", "-i", str(frames / "%03d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(FIG / f"F2_{g}_{rule}.mp4")], check=False)
        fig = plt.figure(figsize=(15, 3.2))
        for li, l in enumerate([1, 2, 3, 4, 5]):
            key = c64 * 8 ** l + T.leaf_id(np.zeros_like(c64), p64[:, :l]) if l else c64
            uk, inv = np.unique(key, return_inverse=True); mass = np.bincount(inv, weights=pr)
            first = np.array([np.flatnonzero(inv == j)[0] for j in range(len(uk))])
            cc = T.decode(c64[first], p64[first, :l]); top = np.argsort(mass)[::-1][:64]
            a = V.so3_axes(fig, (li * 0.2, 0.05, 0.2, 0.8)); V.plot_so3(a, cc[top], prob=mass[top] * 0.25, q_gt=gt, gt_size=300)
            a.set_title(f"level {l}: {len(uk)} cells", fontsize=9)
        fig.suptitle(f"coarse-to-fine PoE mass (top-512 leaves aggregated), {g} {rule}", fontsize=9)
        fig.savefig(FIG / f"F3_{g}_{rule}.png", dpi=200); fig.savefig(FIG / f"F3_{g}_{rule}.pdf"); plt.close(fig)

    # ---------------- F4 / F5 / F6 from the sweep runs ----------------
    sw = []
    for d in sorted((REPO / "outputs/experiments" / args.sweep_exp).glob("RUN-*-seed0")):
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        if "nfe" in cfg and (d / "metrics/metrics.csv").exists():
            m = pd.read_csv(d / "metrics/metrics.csv"); m["nfe"], m["m"], m["run"] = cfg["nfe"], cfg["m"], d.name; sw.append(m)
    sw = pd.concat(sw); sw.to_csv(run.sub("tables") / "sweep.csv", index=False)
    tree_lat = 0.37
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.5))
    for j, g in enumerate(DOM):
        a = sw[(sw.domain == g) & (sw.m == 32)].sort_values("nfe"); b = sw[(sw.domain == g) & (sw.nfe == 10)].sort_values("m")
        ax[0, j].plot(a.nfe, a.mean_deg, "o-", label="KDE mode"); ax[0, j].plot(a.nfe, a.best_of_M_mean_deg, "s--", label="best of M (oracle)")
        ax[0, j].axhline(a.leaf_mean_deg.iloc[0], color="k", ls=":", label="tree (PoE greedy)"); ax[0, j].set_xscale("log"); ax[0, j].set_xlabel("NFE (M = 32)")
        ax[0, j].set_ylabel("mean error (°)"); ax[0, j].set_title(g); ax2 = ax[0, j].twinx(); ax2.plot(a.nfe, a.latency_ms_per_image, "r^:", alpha=0.6); ax2.set_ylabel("ms / img", color="r")
        ax[1, j].plot(b.m, b.mean_deg, "o-"); ax[1, j].plot(b.m, b.best_of_M_mean_deg, "s--"); ax[1, j].axhline(b.leaf_mean_deg.iloc[0], color="k", ls=":")
        ax[1, j].set_xscale("log", base=2); ax[1, j].set_xlabel("samples M (NFE = 10)"); ax[1, j].set_ylabel("mean error (°)")
        ax3 = ax[1, j].twinx(); ax3.plot(b.m, b.latency_ms_per_image, "r^:", alpha=0.6); ax3.set_ylabel("ms / img", color="r")
    ax[0, 0].legend(fontsize=8); fig.suptitle(f"speed–accuracy, fused flow, seed 0 (tree ≈ {tree_lat} ms / img batched)"); fig.tight_layout()
    fig.savefig(FIG / "F4_speed_accuracy.png", dpi=200); fig.savefig(FIG / "F4_speed_accuracy.pdf"); plt.close(fig)
    ref = sw[(sw.nfe == 10) & (sw.m == 32)].run.iloc[0]
    pf = pd.read_csv(REPO / "outputs/experiments" / args.sweep_exp / ref / "predictions/fused_seed0.csv")
    ecol = "err_deg" if "err_deg" in pf.columns else [c for c in pf.columns if "err" in c][0]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for j, g in enumerate(DOM):
        bins = np.linspace(0, 180, 46)
        ax[j].hist(greedy_err[g], bins=bins, alpha=0.6, label="tree (PoE greedy)"); ax[j].hist(pf[pf.group == g][ecol], bins=bins, alpha=0.6, label="flow (KDE mode)")
        ax[j].set_yscale("log"); ax[j].set_xlabel("rotation error (°)"); ax[j].set_title(g)
    ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "F5_error_hist.png", dpi=200); fig.savefig(FIG / "F5_error_hist.pdf"); plt.close(fig)
    fig, a = plt.subplots(figsize=(4.2, 4))
    lv = [0.5, 0.8, 0.9, 0.95]
    for g in DOM:
        r = sw[(sw.domain == g) & (sw.nfe == 10) & (sw.m == 32)].iloc[0]; a.plot(lv, [r[f"coverage@{v}"] for v in lv], "o-", label=g)
    a.plot([0.45, 1], [0.45, 1], "k:"); a.set_xlabel("nominal coverage"); a.set_ylabel("empirical coverage"); a.legend(fontsize=8); fig.tight_layout()
    fig.savefig(FIG / "F6_calibration.png", dpi=200); fig.savefig(FIG / "F6_calibration.pdf"); plt.close(fig)

    # ---------------- comparison-ready metrics table (seed 0, NFE 10, M 32; tree = PoE greedy) ----------------
    rows = []
    for g in DOM:
        for name, e in (("tree (PoE greedy)", greedy_err[g]), ("flow (KDE mode)", pf[pf.group == g][ecol].to_numpy())):
            rows.append({"domain": g, "method": name, "E_R_mean": e.mean(), "E_R_median": np.median(e), "E_R_p95": np.percentile(e, 95),
                         **{f"acc@{t}": float((e <= t).mean()) for t in (2, 5, 10, 15, 30)}, "frac_gt90": float((e > 90).mean()), "frac_gt150": float((e > 150).mean())})
        r = sw[(sw.domain == g) & (sw.nfe == 10) & (sw.m == 32)].iloc[0]
        rows[-1].update({"spread_deg": r.sample_mean_deg, "coverage@0.9": r["coverage@0.9"], "best_of_32": r.best_of_M_mean_deg})
    run.write_metrics(rows, "comparison_table"); (run.sub("tables") / "comparison_table.md").write_text(md_table(rows))
    run.done(); print(md_table(rows)); print(md_table(lla.to_dict("records"))); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
