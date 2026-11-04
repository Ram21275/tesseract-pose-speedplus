"""Supplementary figures (paper/figures/supplementary). Flow figures: EXP-036 clean trees + EXP-042 SO(3) flow, the pre-registered EXP-107 examples
(FIG-flow per-example data, tables/sweep.csv); these predate the final recipe and are a negative result. Others: FIG-qualitative predictions (seed 0),
EXP-129 / EXP-126 / EXP-124 per-seed metrics and EXP-127 per-image predictions (3 seeds)."""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import data as D, vis_so3 as V  # noqa: E402

OUT = REPO / "paper/figures/supplementary"; OUT.mkdir(parents=True, exist_ok=True)
E = REPO / "outputs/experiments"; PATHS = yaml.safe_load(open(REPO / "configs/paths.yaml"))
E107 = E / "EXP-107_paper_visuals_metrics_efficiency/RUN-20261016-024840-seed0"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
GREY, MAG, BLUE, ORG = "#8a8a8a", "#c02080", "#3060c0", "#e08020"
DL = {"synthetic_val": "synthetic validation", "lightbox": "lightbox (real)", "sunlamp": "sunlamp (real)"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight"); plt.close(fig); print("wrote", name)


# ------------------------------------------------------------- flow: SO(3) posterior, trajectories, tree levels
fl = sorted((E / "FIG-flow_supplementary_flow_examples").glob("RUN-*/predictions/flow_examples.npz"))[-1]; Z = dict(np.load(fl))
EX = pd.read_csv(E107 / "tables/examples.csv"); MAN = pd.read_csv(REPO / "outputs/data_manifests/subset_full.csv")
TITLE = {"synthetic_val_p50": "synthetic, median tree error", "lightbox_p50": "lightbox, median tree error", "lightbox_flipmode": "lightbox, flow mass on a flip"}
GT_H = Line2D([], [], color="k", marker="o", ms=8, mfc="none", ls="", label="ground truth")


def crop(tag):
    r = EX[(EX.domain + "_" + EX.rule) == tag].iloc[0]; m = MAN.iloc[int(r.row)]
    img = D.crop_square(D.read_gray(Path(PATHS["speedplus_root"]), m.image_relpath), m.crop_cx, m.crop_cy, m.crop_side, 224)
    lo, hi = np.percentile(img, (1, 99.7)); return np.clip((img.astype(float) - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)


fig = plt.figure(figsize=(7.2, 5.4))
for i, tag in enumerate(TITLE):
    y = 0.68 - i * 0.33; a = fig.add_axes((0.0, y, 0.2, 0.29)); a.imshow(crop(tag), cmap="gray"); a.set_xticks([]); a.set_yticks([]); a.set_ylabel(TITLE[tag], fontsize=7)
    b = V.so3_axes(fig, (0.22, y, 0.38, 0.29)); V.plot_so3(b, Z[f"{tag}__traj"][-1], q_gt=Z[f"{tag}__gt"][None], const_size=3, gt_size=110)
    p = Z[f"{tag}__leaf_p"][:64]; c = V.so3_axes(fig, (0.61, y, 0.38, 0.29)); V.plot_so3(c, Z[f"{tag}__leaf_q"][:64], prob=p / p.max() * 0.12, q_gt=Z[f"{tag}__gt"][None], gt_size=110)
    if i == 0:
        a.set_title("input crop", fontsize=8); b.set_title("SO(3) flow: 256 samples", fontsize=8); c.set_title("tesseract tree: top-64 leaves", fontsize=8)
fig.legend(handles=[Line2D([], [], color=BLUE, marker="o", ms=3, ls="", label="flow sample"), Line2D([], [], color=BLUE, marker="o", ms=7, ls="", label="tree leaf (area = probability)"), GT_H],
           loc="upper left", bbox_to_anchor=(0.03, 0.03), ncol=3, fontsize=7, edgecolor="#bbbbbb", title="position: where the object's x axis points; colour: rotation about it (wheel)", title_fontsize=7)
V.tilt_wheel(fig, (0.86, -0.11, 0.09, 0.13)); save(fig, "flow_vs_tree_posterior")

fig = plt.figure(figsize=(7.2, 2.4))
for j, tag in enumerate(("synthetic_val_p50", "lightbox_flipmode")):
    a = V.so3_axes(fig, (j * 0.45, 0.02, 0.44, 0.84)); tr = Z[f"{tag}__traj"]; nfe = len(tr) - 1
    ll = np.array([V._coords(V._q2R(tr[t, :16]))[:2] for t in range(nfe + 1)])              # (T,2,16)
    for k in range(16):
        for t in range(nfe):
            if abs(ll[t + 1, 0, k] - ll[t, 0, k]) < np.pi:                                  # do not draw across the map seam
                a.plot(ll[t:t + 2, 0, k], ll[t:t + 2, 1, k], color=plt.cm.viridis(t / nfe), lw=1)
    V.plot_so3(a, tr[-1, :16], q_gt=Z[f"{tag}__gt"][None], const_size=12, gt_size=110); a.set_title(TITLE[tag], fontsize=8)
sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1)); cb = fig.colorbar(sm, cax=fig.add_axes((0.915, 0.15, 0.015, 0.6))); cb.set_label("flow time τ (noise → sample)", fontsize=7); cb.ax.tick_params(labelsize=6)
fig.legend(handles=[Line2D([], [], color=plt.cm.viridis(0.5), lw=1.2, label="16 sample trajectories"), Line2D([], [], color=BLUE, marker="o", ms=4, ls="", label="final sample"), GT_H],
           loc="upper center", bbox_to_anchor=(0.45, 0.02), ncol=3, fontsize=7, edgecolor="#bbbbbb")
save(fig, "flow_trajectories")

fig = plt.figure(figsize=(7.4, 2.5))
for i, tag in enumerate(("synthetic_val_p50", "lightbox_flipmode")):
    for l in range(1, 6):
        q, m = Z[f"{tag}__lev{l}_q"], Z[f"{tag}__lev{l}_mass"]; top = np.argsort(m)[::-1][:64]
        a = V.so3_axes(fig, ((l - 1) * 0.2, 0.52 - i * 0.48, 0.195, 0.42)); V.plot_so3(a, q[top], prob=m[top] * 0.06, q_gt=Z[f"{tag}__gt"][None], size_scale=2500, gt_size=70)
        if i == 0:
            a.set_title(f"level {l}" + (" (4 × 8 cells)" if l == 1 else f" (4 × 8$^{l}$ cells)"), fontsize=7)
    fig.text(0.0, 0.73 - i * 0.48, {"synthetic_val_p50": "synthetic,\nmedian error", "lightbox_flipmode": "lightbox,\nflip"}[tag], rotation=90, fontsize=6.5, va="center", ha="right")
fig.legend(handles=[Line2D([], [], color=BLUE, marker="o", ms=6, ls="", label="tree cell (area = probability mass)"), GT_H], loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=2, fontsize=7, edgecolor="#bbbbbb")
save(fig, "tree_coarse_to_fine")

# ------------------------------------------------------------- flow: sweep and calibration (seed 0, clean trees)
sw = pd.read_csv(E107 / "tables/sweep.csv")
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.1))
for a, g in zip(axs, DL):
    r = sw[(sw.domain == g) & (sw.nfe == 10)].sort_values("m"); a.plot(r.m, r.mean_deg, "o-", ms=3, color=BLUE); a.plot(r.m, r.best_of_M_mean_deg, "s--", ms=3, color=ORG)
    a.axhline(r.leaf_mean_deg.iloc[0], color="k", ls=":", lw=1); a.set_xscale("log", base=2); a.set_title(DL[g], fontsize=8); a.set_xlabel("flow samples M (10 steps)"); a.set_ylim(0); a.tick_params(labelsize=6.5)
axs[0].set_ylabel("mean rotation error (°)"); fig.tight_layout()
fig.legend(handles=[Line2D([], [], color=BLUE, marker="o", ms=3, label="flow, KDE mode (deployable)"), Line2D([], [], color=ORG, marker="s", ms=3, ls="--", label="best of M samples (oracle, uses GT)"), Line2D([], [], color="k", ls=":", label="tree, 1 view")],
           loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=7, edgecolor="#bbbbbb")
save(fig, "flow_samples_vs_error")

fig, a = plt.subplots(figsize=(2.6, 2.4)); lv = [0.5, 0.8, 0.9, 0.95]
for g, col in zip(DL, (GREY, BLUE, ORG)):
    r = sw[(sw.domain == g) & (sw.nfe == 10) & (sw.m == 32)].iloc[0]; a.plot(lv, [r[f"coverage@{v}"] for v in lv], "o-", ms=3, color=col, label=DL[g])
a.plot([0.45, 1], [0.45, 1], "k:", lw=1, label="perfect calibration"); a.set_xlabel("nominal coverage"); a.set_ylabel("empirical coverage"); a.legend(fontsize=6, frameon=False); fig.tight_layout(); save(fig, "flow_calibration")

# ------------------------------------------------------------- flow and refiners vs orbit voting
pi = pd.read_csv(E / "EXP-127_hierarchy_vs_flat_same_grid/RUN-20261103-143538-seed0/predictions/per_image.csv.gz")
ms = pi[(pi.variant == "PoE") & (pi["mode"] == "orbit_product")].groupby(["family", "seed", "domain"]).err_deg.mean().groupby(["family", "domain"]).mean()
d117 = pd.read_csv(E / "EXP-117_orbit_distillation/RUN-20261020-114253-seed0/metrics/metrics.csv"); d117 = d117[(d117.family == "tree:teacher_variant=PoE") & (d117.variant == "PoE") & (d117["mode"] == "orbit_product")].set_index("domain").mean_deg
DOMS = list(DL)
ROWS = [   # label, (synth, lightbox, sunlamp), kind. Sources: tab:negative in sec/X_suppl.tex (EXP-036/042/045/105/108/044), EXP-111, EXP-127, EXP-117
    ("Clean PoE tree (reference)", (5.54, 26.9, 36.7), "ref"), ("+ SO(3) flow, KDE mode", (5.22, 29.8, 37.6), "flow"), ("+ flow re-noise verifier", (4.78, 26.5, 32.7), "flow"),
    ("Tree distilled from flow (1 seed)", (5.62, 26.2, 37.7), "flow"), ("Discrete flow on tree paths", (7.22, 29.8, 39.8), "flow"), ("Best of 32 flow samples (oracle)", (2.60, 9.7, 10.6), "oracle"),
    ("Aug. DINOv3 tree (reference)", (4.84, 22.6, 22.1), "ref"), ("+ SO(3) flow, KDE mode", (4.10, 25.2, 24.4), "flow"), ("+ flow verifier", (4.42, 22.93, 21.98), "flow"),
    ("Aug. PoE tree + orbit voting", tuple(ms["tree"][d] for d in DOMS), "orbit"), ("Distilled PoE + orbit voting", tuple(d117[d] for d in DOMS), "orbit")]
ST = {"ref": dict(color=GREY, marker="s"), "flow": dict(color=BLUE, marker="o"), "oracle": dict(mfc="none", color=ORG, marker="D"), "orbit": dict(color=MAG, marker="*", ms=8)}
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True); ys = np.arange(len(ROWS))[::-1]
for j, (a, g) in enumerate(zip(axs, DOMS)):
    for y, (lab, v, kind) in zip(ys, ROWS):
        a.plot([v[j]], [y], ls="", **{"ms": 5, **ST[kind]}); a.text(v[j] + (axs[j].get_xlim()[1] * 0), y + 0.28, f"{v[j]:.1f}", fontsize=5, ha="center", color="#444444")
    a.set_title(DL[g], fontsize=8); a.set_xlabel("mean rotation error (°)"); a.set_xlim(0); a.grid(axis="x", alpha=0.3, lw=0.4); a.axhline(4.5, color="#cccccc", lw=0.6); a.axhline(1.5, color="#cccccc", lw=0.6); a.tick_params(labelsize=6.5)
axs[0].set_yticks(ys); axs[0].set_yticklabels([r[0] for r in ROWS], fontsize=6.5); fig.tight_layout()
fig.legend(handles=[Line2D([], [], ls="", label=l, **{"ms": 5, **ST[k]}) for k, l in (("ref", "tree alone (reference)"), ("flow", "flow-based approach"), ("oracle", "oracle (uses GT, not deployable)"), ("orbit", "orbit voting"))],
           loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=7, edgecolor="#bbbbbb")
save(fig, "flow_and_refiners_vs_orbit")

# ------------------------------------------------------------- per-image error structure (seed 0, final configurations)
F = E / "FIG-qualitative_qualitative_figures"; ERR = {}
for ds, var in (("tless", "PoE"), ("ycbv", "DINO"), ("symsolt", "DINO")):
    run = [d for d in sorted(F.glob("RUN-*"), reverse=True) if list((d / "predictions").glob(f"{ds}_*_seed0_{var}.npz"))][0]; g, o = [], []
    for f in sorted((run / "predictions").glob(f"{ds}_*_seed0_{var}.npz")):
        z = np.load(f); m = z["split"] == "test"; g.append(z["egoerr_greedy"][m]); o.append(z["egoerr_orbit_product"][m])
    ERR[ds] = np.concatenate(g), np.concatenate(o)
spr = lambda c: [d for d in sorted(F.glob("RUN-*"), reverse=True) if (d / f"predictions/speedplus_{c}_S1_seed0.npz").exists()][0] / f"predictions/speedplus_{c}_S1_seed0.npz"
zg, zd = np.load(spr("gt")), np.load(spr("det")); ms_ = np.isin(zg["domain"], ["lightbox", "sunlamp"]); ERR["speedplus"] = zg["err_greedy"][ms_], zg["err_orbit_product"][ms_]
NM = {"tless": "T-LESS", "ycbv": "YCB-V", "symsolt": "SYMSOL-T", "speedplus": "SPEED+ (LB+SL)"}
fig, axs = plt.subplots(1, 4, figsize=(7.4, 1.9)); B = np.linspace(0, 180, 37)
for a, ds in zip(axs, NM):
    g, o = ERR[ds]; a.hist(g, bins=B, histtype="step", color=GREY, lw=1.1); a.hist(o, bins=B, histtype="step", color=MAG, lw=1.1); a.set_yscale("log"); a.set_title(NM[ds], fontsize=8); a.set_xlabel("rotation error (°)"); a.set_xticks([0, 90, 180]); a.tick_params(labelsize=6.5)
axs[0].set_ylabel("test images (log)"); fig.tight_layout()
fig.legend(handles=[Line2D([], [], color=GREY, label="1 view"), Line2D([], [], color=MAG, label="orbit voting")], loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=7, edgecolor="#bbbbbb")
save(fig, "error_histograms_full_range")

assert (zg["image_relpath"] == zd["image_relpath"]).all()
fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.5), sharey=True)
for a, g in zip(axs, ("lightbox", "sunlamp")):
    m = zg["domain"] == g; a.scatter(zg["err_orbit_product"][m], zd["err_orbit_product"][m], s=1, alpha=0.2, color=BLUE, rasterized=True); a.plot([0.5, 180], [0.5, 180], "k--", lw=0.7)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(0.5, 200); a.set_ylim(0.5, 200); a.set_title(g, fontsize=8); a.set_xlabel("error with GT crop (°)"); a.tick_params(labelsize=6.5)
axs[0].set_ylabel("error with detector crop (°)"); fig.tight_layout()
fig.legend(handles=[Line2D([], [], color=BLUE, marker="o", ms=3, ls="", label="one test image (same model)"), Line2D([], [], color="k", ls="--", lw=0.7, label="no change")], loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=7, edgecolor="#bbbbbb")
save(fig, "detector_vs_gt_crop")

# ------------------------------------------------------------- per-object means with seed spread (3 seeds)
SRC = [("T-LESS (cm-free: rotation only)", E / "EXP-129_rotation_with_predicted_translation/RUN-20261103-162450-seed0", "PoE", "orbit_product", "ego_mean"),
       ("YCB-V", E / "EXP-129_rotation_with_predicted_translation/RUN-20261103-165739-seed0", "DINO", "greedy", "ego_mean"),
       ("SYMSOL-T", E / "EXP-126_translation_xobj/RUN-20261030-141752-seed0", "DINO", "orbit_product", "sym_mean")]
fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.2), gridspec_kw=dict(width_ratios=[7, 6, 5, 2]))
for a, (nm, d, v, mo, col) in zip(axs, SRC):
    t = pd.read_csv(d / "metrics/per_object_seed.csv"); t = t[(t.split == "test") & (t.variant == v) & (t["mode"] == mo)]; t["obj"] = t.obj.astype(str)
    order = sorted(t.obj.unique(), key=lambda s: (len(s), s))
    for i, o in enumerate(order):
        y = t[t.obj == o][col].to_numpy(); a.bar(i, y.mean(), color="#d8d8e8", width=0.7); a.scatter([i] * len(y), y, s=7, color=BLUE, zorder=3)
    a.set_xticks(range(len(order))); a.set_xticklabels([o.split("_", 1)[-1].replace("_", " ") for o in order], rotation=60, ha="right", fontsize=6); a.set_title(nm.split(" (")[0], fontsize=8); a.tick_params(axis="y", labelsize=6.5)
t = pd.read_csv(E / "EXP-124_speedplus_detected_official_score/RUN-20261028-231916-seed0/metrics/per_seed.csv"); t = t[t.system.str.startswith("S1") & (t.rot_mode == "orbit_product") & (t.trans_mode == "orbit_mean")]
for i, o in enumerate(("lightbox", "sunlamp")):
    y = t[t.domain == o].E_R_mean_deg.to_numpy(); axs[3].bar(i, y.mean(), color="#d8d8e8", width=0.7); axs[3].scatter([i] * len(y), y, s=7, color=BLUE, zorder=3)
axs[3].set_xticks([0, 1]); axs[3].set_xticklabels(["lightbox", "sunlamp"], rotation=60, ha="right", fontsize=6); axs[3].set_title("SPEED+", fontsize=8); axs[3].tick_params(axis="y", labelsize=6.5); axs[0].set_ylabel("mean rotation error (°)"); fig.tight_layout()
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#d8d8e8", label="mean over 3 seeds"), Line2D([], [], color=BLUE, marker="o", ms=3, ls="", label="one seed")], loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=7, edgecolor="#bbbbbb")
save(fig, "per_object_errors")

# ------------------------------------------------------------- tree vs flat over the same leaves vs balanced Hopf (EXP-127)
FAM = (("tree", "tesseract tree (2.4 M)", MAG), ("flat:grid=tesseract_L5", "flat, same 131,072 leaves (36.1 M)", BLUE), ("flat:grid=hopf1771x74", "flat, balanced Hopf 1771×74 (36.1 M)", ORG))
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.1)); x = np.linspace(0, 30, 200); pp = pi[(pi.variant == "PoE") & (pi["mode"] == "orbit_product")]
for a, g in zip(axs, DL):
    for fam, lab, col in FAM:
        e = pp[(pp.family == fam) & (pp.domain == g)].err_deg.to_numpy(); a.plot(x, np.searchsorted(np.sort(e), x) / len(e), color=col, lw=1.2)
    a.set_title(DL[g], fontsize=8); a.set_xlabel("rotation error (°)"); a.set_ylim(0, 1); a.grid(alpha=0.3, lw=0.4); a.tick_params(labelsize=6.5)
axs[0].set_ylabel("fraction of images"); fig.tight_layout()
fig.legend(handles=[Line2D([], [], color=c, lw=1.4, label=l) for _, l, c in FAM], loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=7, edgecolor="#bbbbbb")
save(fig, "tree_vs_flat_cdf")
print("EXP-127 check (3-seed PoE orbit, tree / same / Hopf):", {g: [round(ms[f][g], 2) for f, _, _ in FAM] for g in DL}); print("EXP-117 student PoE orbit:", {g: round(d117[g], 2) for g in DOMS})
