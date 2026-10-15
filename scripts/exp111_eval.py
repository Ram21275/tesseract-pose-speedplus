"""EXP-111 summary + timing: DINOv3-only augmented tree (EXP-109 A) + flow head (EXP-042 recipe, trained with the same
augmented views). From the trainer's per-image predictions (3 seeds): flow KDE mode vs the same tree's greedy leaf, per
domain, with flip transitions (denominators). Timing (batch 1 / 64, median of 5, cuda-synchronized): DINOv3 features,
DINOv3 tree greedy, flow NFE 10 x M 32, KDE."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import features as Fx, phase3 as P, so3flow as S  # noqa: E402
from tfpose.fusion import fused_beam  # noqa: E402
from tfpose.runlog import Run, md_table  # noqa: E402

E111 = "EXP-111_dino_aug_flow"
run = Run("EXP-111", "dino_aug_flow", {"stage": "summary_timing"}, seed=0)
try:
    rows, heads, trees = [], {}, {}
    for d in sorted((REPO / "outputs/experiments" / E111).glob("RUN-*")):
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        if cfg.get("mode") != "flow" or not (d / "predictions/predictions.csv").exists():
            continue
        pr = pd.read_csv(d / "predictions/predictions.csv")
        pr["domain"] = pr.group.str.replace("lightbox_val", "lightbox").str.replace("lightbox_test", "lightbox")
        for g, x in pr.groupby("domain"):
            if g not in ("synthetic_val", "lightbox", "sunlamp"):
                continue
            e, e0 = x.err_deg.to_numpy(), x.leaf_err_deg.to_numpy(); n150, n20 = int((e0 > 150).sum()), int((e0 < 20).sum())
            for name, ee in (("tree greedy (EXP-109 A DINOv3)", e0), ("tree + flow KDE (EXP-111)", e)):
                rows.append({"seed": cfg["seed"], "domain": g, "system": name, "mean_deg": ee.mean(), "median_deg": np.median(ee),
                             "acc@10": float((ee <= 10).mean()), "frac_gt90": float((ee > 90).mean()), "frac_gt150": float((ee > 150).mean()),
                             "n_tree_gt150": n150, "frac_gt150_to_lt20": float(((e0 > 150) & (ee < 20)).sum() / max(n150, 1)),
                             "n_tree_lt20": n20, "frac_lt20_to_gt150": float(((e0 < 20) & (ee > 150)).sum() / max(n20, 1))})
        if cfg["seed"] == 0:
            heads[0] = (d, cfg)
    per = pd.DataFrame(rows); run.write_metrics(rows, "per_seed")
    cols = [c for c in per.columns if c not in ("seed", "domain", "system")]
    agg = per.groupby(["domain", "system"])[cols].mean().join(per.groupby(["domain", "system"])[["mean_deg"]].std().rename(columns={"mean_deg": "mean_deg_std"})).reset_index()
    run.write_metrics(agg.to_dict("records"), "metrics"); (run.sub("tables") / "metrics.md").write_text(md_table(agg.to_dict("records")))
    print(md_table(agg.to_dict("records"), ["domain", "system", "mean_deg", "mean_deg_std", "median_deg", "frac_gt150", "n_tree_gt150", "frac_gt150_to_lt20", "frac_lt20_to_gt150"]))

    # ---------------- timing (seed 0 tree + head) ----------------
    dev = torch.device("cuda"); d, cfg = heads[0]
    src = (d / "source_runs.txt").read_text().split()[0]; sd = REPO / "outputs/experiments" / cfg["src_exp"] / src
    tree = P.HierTransformer((64, 1024), 5).to(dev); tree.load_state_dict(torch.load(sd / "checkpoints/best.pt", map_location=dev)["model"]); tree.eval()
    head = S.FlowHead(layers=cfg["layers"]).to(dev); head.load_state_dict(torch.load(d / "checkpoints/best.pt", map_location=dev)["head"]); head.eval()
    dino = Fx.DinoV3("vit_large_patch16_dinov3.lvd1689m", dev)
    df = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv"); rows64 = list(df[df.domain == "lightbox"].iloc[:64].itertuples())
    root = Path(yaml.safe_load(open(REPO / "configs/paths.yaml"))["speedplus_root"])

    def timeit(fn, warm=2, rep=5):
        for _ in range(warm):
            fn()
        ts = []
        for _ in range(rep):
            torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        return float(np.median(ts)) * 1000
    tim = []
    for bs in (1, 64):
        xd = Fx.load_crops(root, rows64[:bs], dino.size)
        x = torch.randn(bs, 64 * 1024, device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            t_bb = timeit(lambda: dino(xd))
            t_tree = timeit(lambda: fused_beam([tree], [x], "single", 1))
            mem = tree.encode(x).float()
        memk = mem.repeat_interleave(32, 0); q0 = S.uniform_quats(bs * 32, dev)
        with torch.no_grad():
            t_flow = timeit(lambda: S.integrate(lambda qq, tt: head(memk, qq, tt, None), q0, 10))
            smp = S.integrate(lambda qq, tt: head(memk, qq, tt, None), q0, 10).view(bs, 32, 4)
            t_kde = timeit(lambda: S.kde_mode(smp))
        for comp, ms in (("DINOv3-L features", t_bb), ("DINOv3 tree greedy", t_tree), ("flow NFE10 M32 (1 head)", t_flow), ("KDE", t_kde),
                         ("TOTAL tree only", t_bb + t_tree), ("TOTAL tree + flow", t_bb + t_tree + t_flow + t_kde)):
            tim.append({"batch": bs, "component": comp, "ms_per_image": ms / bs, "fps_or_img_per_s": 1000 * bs / ms})
    run.write_metrics(tim, "timing"); (run.sub("tables") / "timing.md").write_text(md_table(tim))
    gpu = __import__("subprocess").run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    run.done(gpu_state=gpu); print(md_table(tim)); print("GPU", gpu); print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc()); raise
