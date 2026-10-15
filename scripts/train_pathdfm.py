"""EXP-108 (DEC-015): discrete flow matching on the Tesseract path, both branches trained jointly with one shared source
sample per image: x0 ~ PoE tree (frozen EXP-036, temperature 1), x1 = GT path, x_t = factorized mixture with kappa(t) = t.
Loss = sum over branches and tokens of CE(p_theta(x1_i | x_t, t, image), x1_i). Checkpoint = lowest synthetic-val mean of
the fused CTMC sampler (NFE 4, one sample per image, fixed generator seed)."""
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
from tfpose import featsets, pathflow as PF, phase3 as P, tesseract as T  # noqa: E402
from tfpose.runlog import Run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default="EXP-108_path_flow_revision")
ap.add_argument("--base-exp", default="EXP-036_transformer_soft_targets")
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--nfe-val", type=int, default=4)
ap.add_argument("--eval-every", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
exp_id, short = args.exp.split("_", 1)
run = Run(exp_id, short, {**vars(args), "method": "dfm"}, seed=args.seed)
L = 5
SPECS = {"dinov3_vitl16:tokens8": "64x1024", "moge2_vitl:normals16~and": "256x4"}
try:
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed); dev = torch.device("cuda")
    df = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv")
    q = df[[f"q_can_{k}" for k in "wxyz"]].to_numpy(); chart, path = T.encode(q, L)
    tr = ((df.domain == "synthetic") & (df.split == "train")).to_numpy(); va = ((df.domain == "synthetic") & (df.split == "validation")).to_numpy()
    X, trees, used = {}, [], []
    for sp, tsh in SPECS.items():
        names, x, _ = featsets.load(sp, "full"); assert names == df.image_relpath.tolist()
        X[sp] = featsets.standardize_to_tensor(x, tr, "cpu", torch.float16); del x
        for d in sorted((REPO / "outputs/experiments" / args.base_exp).glob(f"RUN-*-seed{args.seed}")):
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            if cfg.get("features") == sp and (d / "checkpoints/best.pt").exists():
                t = P.HierTransformer(tuple(int(v) for v in tsh.split("x")), 5, dropout=cfg["dropout"]).to(dev)
                t.load_state_dict(torch.load(d / "checkpoints/best.pt", map_location=dev)["model"]); t.eval(); t.requires_grad_(False)
                trees.append(t); used.append(f"{args.base_exp}/{d.name}"); break
    (run.dir / "source_runs.txt").write_text("\n".join(used) + "\n")
    models = [PF.PathDFM(L).to(dev) for _ in SPECS]
    params = [p for m in models for p in m.parameters()]; n_params = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    tr_idx = np.flatnonzero(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * int(np.ceil(len(tr_idx) / args.batch)), pct_start=0.05)
    Ct, Pt = torch.as_tensor(chart, device=dev), torch.as_tensor(path, device=dev)

    def mems_of(b):
        xs = [X[sp][torch.as_tensor(b)].to(dev).float() for sp in SPECS]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return [m.encode(x) for m, x in zip(trees, xs)]

    @torch.no_grad()
    def evaluate(mask):
        for m in models:
            m.eval()
        idx = np.flatnonzero(mask); e = []; g = torch.Generator(device=dev).manual_seed(1234)
        for s in range(0, len(idx), 1024):
            b = idx[s:s + 1024]; mm = mems_of(b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                c0, p0 = PF.sample_tree_paths(trees, mm, 1.0, g)
                c1, p1 = PF.dfm_sample(models, mm, c0, p0, args.nfe_val, "poe", g)
            e.append(np.degrees(2 * np.arccos(np.clip(np.abs((T.decode(c1.cpu().numpy(), p1.cpu().numpy()) * q[b]).sum(1)), 0, 1))))
        return float(np.concatenate(e).mean())

    log = open(run.sub("logs") / "train.log", "w"); best = (np.inf, None, 0)
    for ep in range(1, args.epochs + 1):
        for m in models:
            m.train()
        perm = rng.permutation(tr_idx); tot = 0.0
        for s in range(0, len(perm), args.batch):
            b = perm[s:s + args.batch]; bt = torch.as_tensor(b, device=dev); mm = mems_of(b)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                c0, p0 = PF.sample_tree_paths(trees, mm, 1.0)
            t = torch.rand(len(b), device=dev)
            ct, pt = PF.mix_tokens(c0, p0, Ct[bt], Pt[bt], t)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = 0.0
                for m, mem in zip(models, mm):
                    rl, cl = m(mem.float(), ct, pt, t)
                    loss = loss + F.cross_entropy(rl.float(), Ct[bt]) + sum(F.cross_entropy(cl[:, l].float(), Pt[bt][:, l]) for l in range(L))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(b)
        if ep % args.eval_every == 0 or ep == args.epochs:
            vm = evaluate(va); print(f"ep {ep} loss {tot / len(perm):.4f} val_mean_nfe{args.nfe_val} {vm:.4f}", file=log, flush=True)
            if vm < best[0]:
                best = (vm, [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in models], ep)
    torch.save({"models": best[1], "specs": list(SPECS)}, run.sub("checkpoints") / "best.pt")
    run.done(best_epoch=best[2], best_val_mean=best[0], n_params=n_params, trainable_params=n_params)
    print(f"seed {args.seed} best epoch {best[2]} val mean {best[0]:.3f}"); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
