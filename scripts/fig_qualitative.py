"""Paper figures from the finalised configurations (seed 0), using the per-image predictions saved by scripts/run_queue18.sh
(FIG-qualitative runs; their metrics reproduce the seed-0 rows of EXP-129 / EXP-126 / EXP-124).

Configurations (all numbers in the paper):
  T-LESS   : EXP-126 checkpoints = EXP-106 A2 trees (before distillation), PoE + orbit_product; DINOv3 translation, orbit mean
             (the EXP-129 configuration of the Table A 6D cell); rotation converted with the predicted translation.
  YCB-V    : EXP-122 synthetic-trained DINOv3 tree (EXP-126 checkpoint), 1 view; DINOv3 translation, view 0; predicted-t conversion.
  SYMSOL-T : EXP-106 A2 DINOv3 tree (EXP-126 checkpoint), orbit_product; DINOv3 translation, orbit mean (scene units).
  SPEED+   : S1 = EXP-117 distilled PoE students, orbit_product + EXP-118 joint DINOv3 translation, orbit mean; GT crops
             (and the EXP-124 detector crops in speedplus_detected).

Example selection (fixed rule, no hand-picking): test split, seed 0; for each object / shape / domain the images whose final
rotation error is closest to the 10th, 50th and 90th percentile, plus the worst image. The orbit-voting example is the
SPEED+ lightbox image, among those where greedy is wrong by >150 deg and orbit voting is within 10 deg, whose orbit error is
closest to the median of that group. The SO(3) examples are the median-error image of each shape / object.
Every selected image and its errors are written to selection.csv.

Drawing: the object's 3D bounding box (from its model points) and body axes, projected with the full camera model.
Green = ground truth, magenta = prediction (rotation AND predicted translation). For symmetric objects the ground truth is
drawn as the symmetry-equivalent pose closest to the prediction (the same physical pose), so the boxes are comparable.
Crosses mark the projected object origin (GT green, predicted magenta)."""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import data as D, vis_so3 as V, xobj as XO  # noqa: E402
from tfpose.rotations import quat_to_matrix  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(REPO / "paper/figures/qualitative"))
ap.add_argument("--size", type=int, default=224)
args = ap.parse_args()
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
PATHS = yaml.safe_load(open(REPO / "configs/paths.yaml"))
FIGD = REPO / "outputs/experiments/FIG-qualitative_qualitative_figures"
GT_C, PR_C = "#22dd44", "#ff2bd6"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
NAMES = {"tless": "T-LESS", "ycbv": "YCB-V", "symsolt": "SYMSOL-T"}
CFG = {"tless": ("PoE", "orbit_product"), "ycbv": ("DINO", "greedy"), "symsolt": ("DINO", "orbit_product")}
SEL = []
SRC = {}


def latest_run(pattern):
    for d in sorted(FIGD.glob("RUN-*"), reverse=True):
        if (d / "status.json").exists() and '"completed"' in (d / "status.json").read_text() and list((d / "predictions").glob(pattern)):
            return d
    raise FileNotFoundError(pattern)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print("wrote", name)


def load_general(ds):
    """One row per test image, final configuration, seed 0."""
    var, mode = CFG[ds]; run = latest_run(f"{ds}_*_seed0_{var}.npz"); SRC[ds] = run.relative_to(REPO)
    rows = []
    for f in sorted((run / "predictions").glob(f"{ds}_*_seed0_{var}.npz")):
        z = dict(np.load(f)); obj = f.name[len(ds) + 1:].rsplit("_seed0", 1)[0]
        tpred = z["t_used"] if ds != "symsolt" else z["t_views"].mean(0)          # SYMSOL-T: rotation needs no t; draw orbit-mean t
        for i in np.flatnonzero(z["split"] == "test"):
            rows.append(dict(ds=ds, obj=obj, image=str(z["image_relpath"][i]), K=z["K"][i], crop=z["crop"][i], sym=z["sym"],
                             q_gt=z["q_ego_gt"][i], t_gt=z["t_gt"][i], q_pred=z[f"q_ego_{mode}"][i], t_pred=tpred[i],
                             q_greedy=z["q_ego_greedy"][i], err=float(z[f"egoerr_{mode}"][i]), err_greedy=float(z["egoerr_greedy"][i]), err_orbit=float(z["egoerr_orbit_product"][i]),
                             cand=z["cand"][i] if "cand" in z else None, score=z["score"][i] if "score" in z else None,
                             q_tgt_gt=z["q_tgt_gt"][i], t_views=z["t_views"][:, i] if z["t_views"].size else None))
    return pd.DataFrame(rows)


def load_spp(crops):
    run = latest_run(f"speedplus_{crops}_S1_seed0.npz"); SRC[f"speedplus_{crops}"] = run.relative_to(REPO)
    z = dict(np.load(run / f"predictions/speedplus_{crops}_S1_seed0.npz"))   # decompress once
    m = np.isin(z["domain"], ["lightbox", "sunlamp"])
    return pd.DataFrame([dict(ds="speedplus", obj=str(z["domain"][i]), image=str(z["image_relpath"][i]), crop=z["crop"][i], q_gt=z["q_gt"][i],
                              t_gt=z["t_gt"][i], q_pred=z["q_orbit_product"][i], t_pred=z["t_orbit_mean"][i], q_greedy=z["q_greedy"][i],
                              err=float(z["err_orbit_product"][i]), err_greedy=float(z["err_greedy"][i]), err_orbit=float(z["err_orbit_product"][i]), cand=z["q_cand"][i], score=z["q_score"][i])
                         for i in np.flatnonzero(m)])


# ---------------------------------------------------------------- geometry and drawing
CAM = D.load_camera(Path(PATHS["speedplus_root"])); KP = D.load_tango_points(PATHS["tango_points"]); _PTS = {}


def model_pts(r):
    if r.ds == "speedplus":
        return KP
    key = (r.ds, r.obj)
    if key not in _PTS:
        ev = pd.read_csv(REPO / f"outputs/data_manifests/exp064_{r.ds}_eval.csv"); row = ev[ev.image_relpath == r.image].iloc[0]
        objv = row[XO.DATASETS[r.ds]["obj_col"]]
        _PTS[key] = XO.model_points(r.ds, objv, row, max_points=4000)
    return _PTS[key]


def project(r, pts, q, t):
    if r.ds == "speedplus":
        return D.project_points(pts, q, t, CAM)
    X = pts @ quat_to_matrix(q).T + t; uv = X @ r.K.T; return uv[:, :2] / uv[:, 2:3]


def image(r):
    if r.ds == "speedplus":
        return cv2.cvtColor(D.read_gray(Path(PATHS["speedplus_root"]), r.image), cv2.COLOR_GRAY2RGB)
    return cv2.imread(str(Path(PATHS[XO.DATASETS[r.ds]["root_key"]]) / r.image))[..., ::-1]


def gt_equiv(r, q_pred=None):
    """GT rotation (egocentric) replaced by the symmetry-equivalent closest to the prediction."""
    q_pred = r.q_pred if q_pred is None else q_pred
    if r.ds == "speedplus" or r.sym is None or len(r.sym) <= 1:
        return r.q_gt
    Rg = quat_to_matrix(r.q_gt) @ r.sym; k = np.argmax(np.einsum("ij,sij->s", quat_to_matrix(q_pred), Rg))
    return np.roll(Rotation.from_matrix(Rg[k]).as_quat(), 1)


def draw_pose(ax, r, q, t, col, lw, s):
    pts = model_pts(r); lo, hi = pts.min(0), pts.max(0)
    C = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    L = 0.55 * (hi - lo).max(); A = np.array([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L]])
    cx, cy, side = r.crop; sc = s / side
    tc = lambda uv: np.stack([sc * (uv[:, 0] - cx) + s / 2, sc * (uv[:, 1] - cy) + s / 2], 1)
    c = tc(project(r, C, q, t))
    for a, b in [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]:
        ax.plot(*c[[a, b]].T, color=col, lw=lw, alpha=0.95, solid_capstyle="round")
    p = tc(project(r, A, q, t))
    for k, ac in zip(range(1, 4), ("#ff4040", "#40ff40", "#4080ff")):
        ax.plot(*p[[0, k]].T, color=ac, lw=lw * 1.4, ls="-" if col == GT_C else (0, (2, 1)))
    ax.plot(*p[0], marker="+", ms=9, mew=1.8, color=col)


def crop_img(r, s, roll=0.0):
    cx, cy, side = r.crop; img = D.crop_square(image(r), cx, cy, side, s)
    if r.ds == "speedplus":                                                   # display only: percentile contrast stretch of dark real images
        lo, hi = np.percentile(img, (1, 99.7)); img = np.clip((img.astype(float) - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)
    if roll:
        img = cv2.warpAffine(img, cv2.getRotationMatrix2D((s / 2 - 0.5, s / 2 - 0.5), roll, 1.0), (s, s))
    return img


def et_text(r):
    e = np.linalg.norm(r.t_pred - r.t_gt)
    if r.ds == "speedplus":
        return f"{e:.2f} m"
    if r.ds == "symsolt":
        return f"{e:.3f} u"
    return f"{e * XO.DATASETS[r.ds]['t_to_cm']:.1f} cm"


def panel(ax, r, s, label=None, q_pred=None, title=True):
    q_pred = r.q_pred if q_pred is None else q_pred
    ax.imshow(crop_img(r, s)); draw_pose(ax, r, gt_equiv(r, q_pred), r.t_gt, GT_C, 1.3, s); draw_pose(ax, r, q_pred, r.t_pred, PR_C, 1.1, s)
    ax.set_xlim(0, s); ax.set_ylim(s, 0); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(f"{label + '  ' if label else ''}$E_R$ {r.err:.1f}°  $E_T$ {et_text(r)}", fontsize=7, pad=2)


def pick(g, q):
    v = g.err.to_numpy(); target = v.max() if q == "max" else np.percentile(v, q)
    return g.iloc[int(np.argmin(np.abs(v - target)))]


QS = (10, 50, 90, "max"); QL = {10: "10th pct", 50: "median", 90: "90th pct", "max": "worst"}


def quantile_grid(df, name, title, cols, col_label):
    fig, axs = plt.subplots(len(QS), len(cols), figsize=(1.55 * len(cols), 1.62 * len(QS)), squeeze=False)
    for j, c in enumerate(cols):
        g = df[df.obj == c]
        for i, q in enumerate(QS):
            r = pick(g, q); panel(axs[i, j], r, args.size)
            SEL.append(dict(figure=name, column=c, row=QL[q], image=r.image, E_R_deg=round(r.err, 2), E_R_greedy_deg=round(r.err_greedy, 2), E_T=et_text(r)))
            if i == 0:
                axs[i, j].text(0.5, 1.22, col_label(c), transform=axs[i, j].transAxes, ha="center", fontsize=8, weight="bold")
            if j == 0:
                axs[i, j].set_ylabel(QL[q], fontsize=8)
    fig.suptitle(title, fontsize=9, y=1.0); fig.tight_layout(h_pad=0.6, w_pad=0.3); save(fig, name)


# ---------------------------------------------------------------- figures
DATA = {ds: load_general(ds) for ds in ("tless", "ycbv", "symsolt")}
DATA["speedplus"] = load_spp("gt"); SPD = load_spp("det")
for ds, df in DATA.items():
    print(ds, len(df), "test images; mean E_R", round(df.err.mean(), 3), "| greedy", round(df.err_greedy.mean(), 3))
    for o, g in df.groupby("obj"):
        print("   ", o, len(g), round(g.err.mean(), 3))

quantile_grid(DATA["tless"], "tless_quantiles", "T-LESS (real test): multi-path PoE trees + orbit voting, DINOv3 translation (orbit mean); "
              "rotation converted with the predicted translation", sorted(DATA["tless"].obj.unique(), key=int), lambda c: f"obj {int(c)}")
quantile_grid(DATA["ycbv"], "ycbv_quantiles", "YCB-V (real test): synthetic-trained DINOv3 tree, 1 view, DINOv3 translation; predicted-translation conversion",
              sorted(DATA["ycbv"].obj.unique()), lambda c: c.split("_", 1)[-1].replace("_", " ")[:16])
quantile_grid(DATA["symsolt"], "symsolt_quantiles", "SYMSOL-T (test): multi-path DINOv3 tree + orbit voting, DINOv3 translation (orbit mean)",
              ["tet", "cube", "icosa", "cone", "cyl"], lambda c: c)
quantile_grid(DATA["speedplus"], "speedplus_quantiles", "SPEED+ real domains, GT crops: distilled PoE students + orbit voting, joint DINOv3 translation",
              ["lightbox", "sunlamp"], lambda c: c)
quantile_grid(SPD, "speedplus_detected_quantiles", "SPEED+ real domains, crops from the synthetic-only detector (same model)", ["lightbox", "sunlamp"], lambda c: c)

# combined overview: the median example of every object / domain, one row per application
fig = plt.figure(figsize=(7.2, 4.6)); gs = fig.add_gridspec(4, 7, hspace=0.22, wspace=0.05)
ROWS = [("tless", sorted(DATA["tless"].obj.unique(), key=int)), ("ycbv", sorted(DATA["ycbv"].obj.unique())),
        ("symsolt", ["tet", "cube", "icosa", "cone", "cyl"]), ("speedplus", ["lightbox", "sunlamp"])]
for i, (ds, cols) in enumerate(ROWS):
    for j, c in enumerate(cols):
        ax = fig.add_subplot(gs[i, j]); r = pick(DATA[ds][DATA[ds].obj == c], 50); panel(ax, r, args.size, title=False)
        ax.set_title(f"{r.err:.1f}° · {et_text(r)}", fontsize=6.5, pad=1.5)
        SEL.append(dict(figure="overview_medians", column=c, row=ds, image=r.image, E_R_deg=round(r.err, 2), E_R_greedy_deg=round(r.err_greedy, 2), E_T=et_text(r)))
        if j == 0:
            ax.set_ylabel({"tless": "T-LESS", "ycbv": "YCB-V", "symsolt": "SYMSOL-T", "speedplus": "SPEED+"}[ds], fontsize=8, weight="bold")
fig.suptitle("Median-error test image per object (seed 0, final configurations). Green: ground truth; magenta: prediction "
             "(rotation and predicted translation)", fontsize=7, y=0.95)
save(fig, "overview_medians")

# orbit voting repairing a flip (SPEED+ lightbox, GT crops) and on T-LESS
def orbit_figure(df, ds_name, name, dom=None):
    g = df if dom is None else df[df.obj == dom]
    rep = g[(g.err_greedy > 150) & (g.err < 10)]
    if rep.empty:
        print("no repaired flip for", name); return
    r = rep.iloc[int(np.argmin(np.abs(rep.err - rep.err.median())))]
    s = args.size; fig, axs = plt.subplots(2, 5, figsize=(7.4, 3.3))
    conv = (lambda q: q) if r.ds in ("speedplus", "symsolt") else (lambda q: XO.from_target(r.ds, q[None], r.t_pred[None])[0])   # candidates are in the target frame
    sc = r.score - r.score.max(); p = np.exp(sc) / np.exp(sc).sum()
    for k in range(4):
        axs[0, k].imshow(crop_img(r, s, roll=90 * k)); axs[0, k].set_title(f"input view rolled {90 * k}°", fontsize=7, pad=2)
        qk = conv(r.cand[8 * k]); e = sym_err_one(r, qk)
        rr = r.copy(); rr["err"] = e; panel(axs[1, k], rr, s, q_pred=qk, title=False)
        axs[1, k].set_title(f"view {90 * k}° top-1: {e:.0f}°", fontsize=7, pad=2)
    win = int(np.argmax(r.score)); axs[0, 4].bar(np.arange(32), p, color=[plt.cm.tab10(i // 8) for i in range(32)], width=0.8)
    axs[0, 4].annotate(f"selected: view {90 * (win // 8)}°,\nbeam rank {win % 8 + 1}", (win, p[win]), xytext=(2, 0.55 * p.max()), fontsize=6,
                       arrowprops=dict(arrowstyle="->", lw=0.6))
    axs[0, 4].set_title("orbit score of the 32 candidates", fontsize=7, pad=2); axs[0, 4].set_xticks([3.5, 11.5, 19.5, 27.5]); axs[0, 4].set_xticklabels(["0°", "90°", "180°", "270°"], fontsize=6)
    axs[0, 4].set_yticks([]); axs[0, 4].set_xlabel("candidates by source view", fontsize=6)
    panel(axs[1, 4], r, s, title=False); axs[1, 4].set_title(f"orbit vote {r.err:.1f}° (1 view {r.err_greedy:.0f}°)", fontsize=7, pad=2, weight="bold")
    for a in axs[0, :4]:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{ds_name}: each rolled view proposes 8 hypotheses (beam), mapped back to view 0; all 32 are scored in all four views", fontsize=7.5)
    fig.tight_layout(); save(fig, name)
    SEL.append(dict(figure=name, column="", row="repaired flip", image=r.image, E_R_deg=round(r.err, 2), E_R_greedy_deg=round(r.err_greedy, 2), E_T=et_text(r)))


def sym_err_one(r, q):
    Rg = quat_to_matrix(r.q_gt)[None] if (r.ds == "speedplus" or r.sym is None) else quat_to_matrix(r.q_gt) @ r.sym
    c = np.einsum("ij,sij->s", quat_to_matrix(q), Rg).max(); return float(np.degrees(np.arccos(np.clip((c - 1) / 2, -1, 1))))


orbit_figure(DATA["speedplus"], "SPEED+ lightbox (real, GT crop)", "orbit_voting_speedplus", dom="lightbox")
# T-LESS: no orbit figure. Under the same rule only 2 of 2,345 seed-0 test images are repaired flips (and 4 are broken):
# on T-LESS orbit voting improves the error distribution rather than repairing ~180 deg flips (see rotation_error_cdf).


# SO(3) view of the 32 orbit candidates against the symmetry set (I-PDF / LiePose style), median-error image per shape
def so3_figure(df, cols, name, title, label):
    fig = plt.figure(figsize=(2.3 * len(cols), 2.9))
    for j, c in enumerate(cols):
        r = pick(df[df.obj == c], 50); w = 1.0 / len(cols)
        a0 = fig.add_axes((j * w + 0.01, 0.56, w - 0.02, 0.36)); a0.imshow(crop_img(r, 160)); a0.set_xticks([]); a0.set_yticks([])
        a0.set_title(f"{label(c)}: $E_R$ {r.err:.1f}°", fontsize=7, pad=2)
        ax = V.so3_axes(fig, (j * w + 0.005, 0.04, w - 0.01, 0.5))
        sc = r.score - r.score.max(); p = np.exp(sc) / np.exp(sc).sum()
        eq = np.array([np.roll(Rotation.from_matrix(quat_to_matrix(r.q_tgt_gt) @ S).as_quat(), 1) for S in r.sym])
        V.plot_so3(ax, r.cand, prob=p, q_gt=eq[:: max(1, len(eq) // 72)], size_scale=500, const_size=4, gt_size=45)
        SEL.append(dict(figure=name, column=c, row="median", image=r.image, E_R_deg=round(r.err, 2), E_R_greedy_deg=round(r.err_greedy, 2), E_T=et_text(r)))
    fig.suptitle(title, fontsize=7.5, y=1.0); save(fig, name)


so3_figure(DATA["symsolt"], ["tet", "cube", "icosa", "cone", "cyl"], "so3_symmetry_symsolt",
           "SYMSOL-T: the 32 orbit candidates (dot area = orbit score) and every symmetry-equivalent ground truth (rings); "
           "continuous symmetries (cone, cylinder) shown by 72 of their sampled equivalents", lambda c: c)
tsym = [o for o in sorted(DATA["tless"].obj.unique(), key=int) if len(DATA["tless"][DATA["tless"].obj == o].iloc[0].sym) > 1][:4]
so3_figure(DATA["tless"], tsym, "so3_symmetry_tless", "T-LESS (allocentric frame): orbit candidates and the symmetry-equivalent ground truths",
           lambda c: f"obj {int(c)}")

# translation: predicted vs GT depth and lateral error per application (all test images, seed 0)
fig, axs = plt.subplots(2, 4, figsize=(7.4, 3.8))
UNIT = {"tless": (0.1, "cm"), "ycbv": (100.0, "cm"), "symsolt": (1.0, "units"), "speedplus": (1.0, "m")}
for j, ds in enumerate(("tless", "ycbv", "symsolt", "speedplus")):
    df = DATA[ds]; f, u = UNIT[ds]; tg = np.stack(df.t_gt) * f; tp = np.stack(df.t_pred) * f; e = np.linalg.norm(tp - tg, axis=1)
    a = axs[0, j]; a.scatter(tg[:, 2], tp[:, 2], s=1.5, alpha=0.35, color="#3060c0", rasterized=True)
    lo, hi = np.percentile(tg[:, 2], [0.5, 99.5]); a.plot([lo, hi], [lo, hi], "k--", lw=0.7); a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    a.set_title(f"{NAMES.get(ds, 'SPEED+ (LB+SL)')}", fontsize=8); a.set_xlabel(f"GT depth ({u})", fontsize=7); a.tick_params(labelsize=6)
    if j == 0:
        a.set_ylabel("predicted depth", fontsize=7)
    lat = np.linalg.norm(tp[:, :2] - tg[:, :2], axis=1); dep = np.abs(tp[:, 2] - tg[:, 2])
    b = axs[1, j]; mx = np.percentile(e, 98)
    b.hist([lat, dep], bins=np.linspace(0, mx, 30), color=["#e08020", "#3060c0"], label=["lateral (x,y)", "depth (z)"], histtype="step", lw=1.2)
    b.axvline(e.mean(), color="k", lw=0.8, ls=":"); b.set_xlabel(f"error ({u}); mean |Δt| {e.mean():.3g}", fontsize=7); b.tick_params(labelsize=6); b.set_yticks([])
    if j == 0:
        b.legend(fontsize=6, frameon=False)
fig.suptitle("Translation predicted from the frozen tokens (DINOv3 block): predicted vs GT depth (top) and error split into depth and lateral parts (bottom)", fontsize=8); fig.tight_layout(); save(fig, "translation_depth_lateral")

# rotation error CDFs: 1 view (greedy) vs the selected configuration
fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.0))
for a, ds in zip(axs, ("tless", "ycbv", "symsolt", "speedplus")):
    df = DATA[ds]; x = np.linspace(0, 45, 300)
    sel = "1 view" if ds == "ycbv" else "orbit"
    for v, lab, col in ((df.err_greedy.to_numpy(), "1 view", "#999999"), (df.err_orbit.to_numpy(), "orbit", "#c02080")):
        lab = lab + (" (selected)" if lab == sel else "")
        a.plot(x, [(v <= t).mean() for t in x], color=col, lw=1.3, label=f"{lab} (>45°: {100 * (v > 45).mean():.1f}%)")
    a.set_title(NAMES.get(ds, "SPEED+ (LB+SL)"), fontsize=8); a.set_xlabel("rotation error (°)", fontsize=7); a.set_ylim(0, 1); a.tick_params(labelsize=6)
    a.legend(fontsize=5.5, frameon=False, loc="lower right"); a.grid(alpha=0.3, lw=0.4)
axs[0].set_ylabel("fraction of test images", fontsize=7)
fig.tight_layout(); save(fig, "rotation_error_cdf")

pd.DataFrame(SEL).to_csv(OUT / "selection.csv", index=False)
(OUT / "sources.txt").write_text("".join(f"{k}\t{v}\n" for k, v in SRC.items()))
print("done", OUT)
