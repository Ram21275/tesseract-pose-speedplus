"""EXP-109 evaluation (3 seeds): EXP-036 vs A (augmentation-trained trees) vs B (DINOv3 tokens16 tree + EXP-036 MoGe-2 tree),
and C = in-plane orbit PoE voting on top of EXP-036 and A. Domains: synthetic val (clean), synthetic val-aug (seed-1000
augmentation, selection proxy), lightbox, sunlamp (descriptive). All PoE greedy except C."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import data as D, featsets, inplane as IP, phase3 as P, tesseract as T  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

E109, E036 = "EXP-109_real_domain_aug_tokens16_orbit", "EXP-036_transformer_soft_targets"
SD, SM = "dinov3_vitl16:tokens8", "moge2_vitl:normals16~and"
run = Run("EXP-109", "real_domain_aug_tokens16_orbit", {"stage": "eval"}, seed=0)
try:
    dev = torch.device("cuda")
    conv = (REPO / "outputs/experiments/exp109_convention.txt").read_text().strip()
    full = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv"); tr = ((full.domain == "synthetic") & (full.split == "train")).to_numpy()
    t3 = pd.read_csv(REPO / "outputs/data_manifests/exp109_test3.csv"); va = pd.read_csv(REPO / "outputs/data_manifests/exp109_syn_val.csv")
    pos3 = pd.Series(np.arange(len(full)), index=full.image_relpath)[t3.image_relpath].to_numpy()
    K = D.load_camera(Path(yaml.safe_load(open(REPO / "configs/paths.yaml"))["speedplus_root"])).K
    q3 = t3[[f"q_can_{k}" for k in "wxyz"]].to_numpy(); d3 = IP.ray(K, t3.crop_cx.to_numpy(), t3.crop_cy.to_numpy())
    dom3 = np.where(t3.domain == "synthetic", "synthetic_val", t3.domain)
    qv = va[[f"q_can_{k}" for k in "wxyz"]].to_numpy(); dv = IP.ray(K, va.crop_cx.to_numpy(), va.crop_cy.to_numpy())

    def tag_dir(bb, tag):
        return featsets.find_cache(bb, tag)
    ang_va = pd.read_csv(tag_dir("dinov3_vitl16", "exp109_valaug") / "aug.csv").rot_deg.to_numpy()
    qva = IP.rotate_label(qv, ang_va, dv, conv)
    # ---------- features: dict[(spec, view)] -> standardized tensor (cpu fp16); views over the "test" set and the "valaug" set
    F_ = {}
    for sp in (SD, SM):
        _, xf, _ = featsets.load(sp, "full"); st = featsets.train_stats(xf, tr)
        F_[(sp, "test", 0)] = featsets.standardize_to_tensor(xf[pos3], None, "cpu", torch.float16, stats=st); del xf
        for k in (90, 180, 270):
            n, x, _ = featsets.load(sp, f"exp109_test3_r{k}"); assert n == t3.image_relpath.tolist()
            F_[(sp, "test", k)] = featsets.standardize_to_tensor(x, None, "cpu", torch.float16, stats=st)
        for k in (0, 90, 180, 270):
            n, x, _ = featsets.load(sp, "exp109_valaug" if k == 0 else f"exp109_valaug_r{k}"); assert n == va.image_relpath.tolist()
            F_[(sp, "valaug", k)] = featsets.standardize_to_tensor(x, None, "cpu", torch.float16, stats=st)
    mm = np.load(REPO / "outputs/shared_cache/tokens16_full_std.npy", mmap_mode="r")
    s16 = np.load(REPO / "outputs/shared_cache/tokens16_full_stats.npz")
    import h5py
    h16 = h5py.File(tag_dir("dinov3_vitl16", "exp109_valaug") / "features.h5", "r")
    F_[("tok16", "test", 0)] = lambda r: torch.from_numpy(np.asarray(mm[pos3[r]]))          # streamed (memory guard)
    F_[("tok16", "valaug", 0)] = lambda r: torch.from_numpy(((np.asarray(h16["last_grid16"][r[0]:r[-1] + 1], dtype=np.float32).reshape(len(r), -1)
                                                              - s16["mu"]) / s16["sd"]).astype(np.float16))

    def load_tree(d, tsh):
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        m = P.HierTransformer(tuple(int(v) for v in tsh.split("x")), cfg["depth"], dropout=cfg["dropout"]).to(dev)
        m.load_state_dict(torch.load(d / "checkpoints/best.pt", map_location=dev)["model"]); m.eval(); return m

    def find(exp, seed, pred):
        for d in sorted((REPO / "outputs/experiments" / exp).glob(f"RUN-*-seed{seed}")):
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            if (d / "checkpoints/best.pt").exists() and pred(cfg):
                return d
        raise FileNotFoundError(exp, seed)

    def tree_lp(ms, xs, cand):                                         # PoE leaf log-prob of candidates (B,C,4)
        B, C, _ = cand.shape
        c, p = T.encode(cand.reshape(-1, 4), 5); c, p = torch.as_tensor(c, device=dev), torch.as_tensor(p, device=dev)
        outs = [m.logits_tf(m.encode(x).repeat_interleave(C, 0), c, p) for m, x in zip(ms, xs)]
        root = F.log_softmax(sum(F.log_softmax(o[0].float(), -1) for o in outs), -1)
        child = F.log_softmax(sum(F.log_softmax(o[1].float(), -1) for o in outs), -1)
        return (root.gather(1, c[:, None])[:, 0] + child.gather(2, p[:, :, None])[..., 0].sum(1)).view(B, C).cpu().numpy()

    def predict(ms, keys, setname, rows, orbit, base_ang, dray):
        out = []
        for s in range(0, len(rows), 512):
            r = rows[s:s + 512]
            get = lambda k, v: (F_[(k, setname, v)](r) if callable(F_[(k, setname, v)]) else F_[(k, setname, v)][r])
            xs = [get(k, 0).to(dev).float() for k in keys]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                if not orbit:
                    c, p = fused_beam(ms, xs, "poe", 1); out.append(T.decode(c.cpu().numpy(), p.cpu().numpy())); continue
                cands, views = [], []
                for k in (0, 90, 180, 270):
                    xk = [get(kk, k).to(dev).float() for kk in keys]; views.append((k, xk))
                    c8, p8, _, _ = fused_beam(ms, xk, "poe", 8, return_all=True)
                    qk = T.decode(c8.cpu().numpy().reshape(-1), p8.cpu().numpy().reshape(-1, 5)).reshape(len(r), 8, 4)
                    th = np.repeat(base_ang[r] + k, 8) if k else np.repeat(base_ang[r], 8)
                    back = IP.rotate_label(qk.reshape(-1, 4), -(th - base_ang[np.repeat(r, 8)]), np.repeat(dray[r], 8, 0), conv) if k else qk.reshape(-1, 4)
                    cands.append(back.reshape(len(r), 8, 4))
                cand = np.concatenate(cands, 1)                                     # (b, 32, 4) in the original (view-0) frame
                score = np.zeros(cand.shape[:2])
                for k, xk in views:
                    mk = IP.rotate_label(cand.reshape(-1, 4), np.full(cand.shape[0] * 32, float(k)), np.repeat(dray[r], 32, 0), conv) if k else cand.reshape(-1, 4)
                    score += tree_lp(ms, xk, mk.reshape(len(r), 32, 4))
                out.append(cand[np.arange(len(r)), score.argmax(1)])
        return np.concatenate(out)

    geo = lambda a, b: np.degrees(2 * np.arccos(np.clip(np.abs((a * b).sum(1)), 0, 1)))
    rows_out, used = [], []
    for seed in (0, 1, 2):
        t036 = [load_tree(find(E036, seed, lambda c, s=sp: c.get("features") == s), tsh) for sp, tsh in ((SD, "64x1024"), (SM, "256x4"))]
        tA = [load_tree(find(E109, seed, lambda c, s=sp: c.get("features") == s and bool(c.get("aug_tags"))), tsh) for sp, tsh in ((SD, "64x1024"), (SM, "256x4"))]
        tB = load_tree(find(E109, seed, lambda c: bool(c.get("memmap"))), "256x1024")
        systems = {"EXP-036 PoE greedy": (t036, (SD, SM), False), "A PoE greedy": (tA, (SD, SM), False),
                   "B tokens16 + EXP-036 MoGe PoE greedy": ([tB, t036[1]], ("tok16", SM), False),
                   "C orbit on EXP-036": (t036, (SD, SM), True), "C orbit on A": (tA, (SD, SM), True)}
        for name, (ms, keys, orbit) in systems.items():
            p3 = predict(ms, keys, "test", np.arange(len(t3)), orbit, np.zeros(len(t3)), d3)
            pv = predict(ms, keys, "valaug", np.arange(len(va)), orbit, ang_va, dv)
            for dn, e in [(g, geo(p3[dom3 == g], q3[dom3 == g])) for g in ("synthetic_val", "lightbox", "sunlamp")] + [("synthetic_val_aug", geo(pv, qva))]:
                rows_out.append({"seed": seed, "system": name, "domain": dn, "mean_deg": e.mean(), "median_deg": np.median(e), "p95_deg": np.percentile(e, 95),
                                 "acc@5": float((e <= 5).mean()), "acc@10": float((e <= 10).mean()), "acc@20": float((e <= 20).mean()),
                                 "frac_gt90": float((e > 90).mean()), "frac_gt150": float((e > 150).mean())})
            print(seed, name, " | ".join(f"{r['domain']} {r['mean_deg']:.2f}" for r in rows_out[-4:]), flush=True)
    per = pd.DataFrame(rows_out); run.write_metrics(rows_out, "per_seed")
    cols = [c for c in per.columns if c not in ("seed", "system", "domain")]
    agg = per.groupby(["domain", "system"])[cols].mean().join(per.groupby(["domain", "system"])[["mean_deg"]].std().rename(columns={"mean_deg": "mean_deg_std"})).reset_index()
    run.write_metrics(agg.to_dict("records"), "metrics"); (run.sub("tables") / "metrics.md").write_text(md_table(agg.to_dict("records")))
    run.done(convention=conv); print(md_table(agg.to_dict("records"), ["domain", "system", "mean_deg", "mean_deg_std", "median_deg", "acc@10", "frac_gt90"])); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
