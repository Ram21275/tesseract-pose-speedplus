"""EXP-108 trainer: path reviser for one branch on top of its frozen EXP-036 tree (discrete flow matching on the path).
Checkpoint = lowest synthetic-val mean after K=4 revisions of the branch's own greedy path."""
import argparse
import json
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
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.runlog import Run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default="EXP-108_path_flow_revision")
ap.add_argument("--base-exp", default="EXP-036_transformer_soft_targets")
ap.add_argument("--features", required=True)
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--eval-every", type=int, default=5)
ap.add_argument("--no-geo", action="store_true", help="matched control: geometry token features zeroed (same modules / budget)")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
if args.exp != "EXP-000_never":                     # DEC-015: the denoiser arms of EXP-108 were dropped before any run
    sys.exit("train_pathflow.py (iterative path denoiser) is superseded by train_pathdfm.py (DEC-015); not run.")
exp_id, short = args.exp.split("_", 1)
run = Run(exp_id, short, vars(args), seed=args.seed)
L = 5
try:
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed); dev = torch.device("cuda")
    df = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv")
    q = df[[f"q_can_{k}" for k in "wxyz"]].to_numpy()
    chart, path = T.encode(q, L)
    tr = ((df.domain == "synthetic") & (df.split == "train")).to_numpy(); va = ((df.domain == "synthetic") & (df.split == "validation")).to_numpy()
    names, x, info = featsets.load(args.features, "full"); assert names == df.image_relpath.tolist()
    X = featsets.standardize_to_tensor(x, tr, "cpu", torch.float16); del x
    tree = None
    for d in sorted((REPO / "outputs/experiments" / args.base_exp).glob(f"RUN-*-seed{args.seed}")):
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        if cfg.get("features") == args.features and (d / "checkpoints/best.pt").exists():
            tree = P.HierTransformer(tuple(int(v) for v in cfg["token_shape"].split("x")), cfg["depth"], dropout=cfg["dropout"]).to(dev)
            tree.load_state_dict(torch.load(d / "checkpoints/best.pt", map_location=dev)["model"]); tree.eval(); tree.requires_grad_(False)
            (run.dir / "source_runs.txt").write_text(f"{args.base_exp}/{d.name}\n"); break
    assert tree is not None
    rev = PF.PathReviser(L, geo=not args.no_geo).to(dev)
    n_params = sum(p.numel() for p in rev.parameters())
    opt = torch.optim.AdamW(rev.parameters(), lr=args.lr, weight_decay=args.wd)
    tr_idx = np.flatnonzero(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * int(np.ceil(len(tr_idx) / args.batch)), pct_start=0.05)

    @torch.no_grad()
    def evaluate(mask, K):
        rev.eval(); idx = np.flatnonzero(mask); e0, eK, ch = [], [], []
        for s in range(0, len(idx), 1024):
            b = idx[s:s + 1024]; xb = X[torch.as_tensor(b)].to(dev).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                c, p, hs = fused_beam([tree], [xb], "single", 1, return_h=True)
                c2, p2, st = PF.revise([rev], [hs[0]], c, p, K, "single")
            for cc, pp, out in ((c, p, e0), (c2, p2, eK)):
                out.append(np.degrees(2 * np.arccos(np.clip(np.abs((T.decode(cc.cpu().numpy(), pp.cpu().numpy()) * q[b]).sum(1)), 0, 1))))
            ch.append(((c2 != c) | (p2 != p).any(1)).cpu().numpy())
        e0, eK, ch = np.concatenate(e0), np.concatenate(eK), np.concatenate(ch)
        return {"greedy_mean": float(e0.mean()), f"rev{K}_mean": float(eK.mean()), "frac_changed": float(ch.mean())}

    log = open(run.sub("logs") / "train.log", "w"); best = (np.inf, None, 0)
    for ep in range(1, args.epochs + 1):
        rev.train(); perm = rng.permutation(tr_idx); tot = 0.0
        for s in range(0, len(perm), args.batch):
            b = perm[s:s + args.batch]; t = rng.random(len(b))
            cc, pp = PF.corrupt(q[b], chart[b], path[b], t, rng)
            cc, pp = torch.as_tensor(cc, device=dev), torch.as_tensor(pp, device=dev)
            gc, gp = torch.as_tensor(chart[b], device=dev), torch.as_tensor(path[b], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    mem = tree.encode(X[torch.as_tensor(b)].to(dev).float())
                rl, cl = rev(mem, cc, pp)
                loss = F.cross_entropy(rl.float(), gc) + sum(F.cross_entropy(cl[:, l].float(), gp[:, l]) for l in range(L))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(b)
        if ep % args.eval_every == 0 or ep == args.epochs:
            vm = evaluate(va, args.k)
            print(f"ep {ep} loss {tot / len(perm):.4f} " + " ".join(f"{k} {v:.4f}" for k, v in vm.items()), file=log, flush=True)
            if vm[f"rev{args.k}_mean"] < best[0]:
                best = (vm[f"rev{args.k}_mean"], {k: v.detach().clone() for k, v in rev.state_dict().items()}, ep)
    rev.load_state_dict(best[1]); torch.save({"reviser": best[1]}, run.sub("checkpoints") / "best.pt")
    rows = []
    for g, m in (("synthetic_val", va), ("lightbox", (df.domain == "lightbox").to_numpy()), ("sunlamp", (df.domain == "sunlamp").to_numpy())):
        rows.append({"features": args.features, "seed": args.seed, "domain": g, **evaluate(m, args.k)})
    run.write_metrics(rows, "metrics"); run.done(best_epoch=best[2], n_params=n_params, trainable_params=n_params)
    print(args.features, "seed", args.seed, " | ".join(f"{r['domain']}: greedy {r['greedy_mean']:.2f} -> rev {r[f'rev{args.k}_mean']:.2f} (changed {r['frac_changed']:.3f})" for r in rows))
    print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
