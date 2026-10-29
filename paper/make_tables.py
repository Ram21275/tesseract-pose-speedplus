"""Generate the paper's LaTeX table fragments (paper/tables/*.tex) from the run artifacts.

Every number in a results table is read from a named evaluation run (see RUNS and paper/SOURCES.md); nothing is typed by
hand except (i) the external numbers of other papers, labelled as such, and (ii) the post-hoc constant-prediction pair
control, recomputed here by `constant_pair()` with the same pair sampling as scripts/exp064_zeroshot.py.
Run from the repo root:  .venv/bin/python paper/make_tables.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "paper/tables"; OUT.mkdir(exist_ok=True)
E = REPO / "outputs/experiments"
RUNS = {
    "timing_eval": E / "EXP-112_locked_recipe_eval_timing/RUN-20261016-225940-seed0",   # EXP-036 baseline row, flip stats
    "timing": E / "EXP-112_locked_recipe_eval_timing/RUN-20261016-231146-seed0",
    "flat": E / "EXP-113_tesseract_vs_flat_hopf/RUN-20261017-042606-seed0",
    "scoring": E / "EXP-114_orbit_scoring_rules/RUN-20261016-231825-seed0",
    "augment": E / "EXP-115_augmentation_ablation/RUN-20261017-121554-seed0",
    "consistency": E / "EXP-116_roll_consistent_tree/RUN-20261018-194430-seed0",
    "distill_dino": E / "EXP-117_orbit_distillation/RUN-20261020-113318-seed0",
    "distill_poe": E / "EXP-117_orbit_distillation/RUN-20261020-114253-seed0",
    "sixdof": E / "EXP-118_translation_6dof/RUN-20261020-031510-seed0",
    "tless": E / "EXP-064_broader_object_validation/RUN-20261019-063238-seed0",
    "symsolt": E / "EXP-064_broader_object_validation/RUN-20261019-233710-seed0",
    "ycbv": E / "EXP-064_broader_object_validation/RUN-20261019-224920-seed0",
    "zs_tless": E / "EXP-064_broader_object_validation/RUN-20261019-142923-seed0",
    "zs_symsolt": E / "EXP-064_broader_object_validation/RUN-20261019-173435-seed0",
    "zs_ycbv": E / "EXP-064_broader_object_validation/RUN-20261020-003420-seed0",
    # 2026-10-28 additions (DEC-024 / 025). A missing run (still queued) leaves its cells blank.
    "ycbv_syn": E / "EXP-122_ycbv_synthetic_training/RUN-20261027-200852-seed0",
    "tless_A0": E / "EXP-106_tless_multipath_tree/RUN-20261027-201558-seed0",
    "tless_A1": E / "EXP-106_tless_multipath_tree/RUN-20261027-232339-seed0",
    "tless_A2": E / "EXP-106_tless_multipath_tree/RUN-20261028-221625-seed0",
    "tless_A2D": None,                                                                 # EXP-119 T-LESS: queued
    "symsolt_A0": E / "EXP-106_tless_multipath_tree/RUN-20261028-014408-seed0",
    "symsolt_A1": E / "EXP-106_tless_multipath_tree/RUN-20261028-041345-seed0",
    "symsolt_A2": E / "EXP-106_tless_multipath_tree/RUN-20261028-120353-seed0",
    "symsolt_A2D": None,                                                               # EXP-119 SYMSOL-T: queued
    "spp_gt": E / "EXP-124_speedplus_detected_official_score/RUN-20261028-231916-seed0",  # EXP-124 GT-crop reference (S1)
    "spp_det": E / "EXP-124_speedplus_detected_official_score/RUN-20261029-232454-seed0",  # EXP-124 detected crops (S1)
    "quant127": E / "EXP-127_hierarchy_vs_flat_same_grid/RUN-20261029-003312-seed0",      # EXP-127 quantization floor (CPU)
    "flat127": None,                                                                   # EXP-127 same-grid / balanced-Hopf flat: queued
    "trans_tless": None, "trans_symsolt": None,                                        # EXP-126 translation: running
    "trans_ycbv": E / "EXP-126_translation_xobj/RUN-20261030-044920-seed0",
    "deploy128": None,                                                                 # EXP-128 head deployment cost: queued
    "quant128": None,                                                                  # EXP-128 accuracy per precision: queued (dict of runs)
}
DOM = ["synthetic_val", "lightbox", "sunlamp"]


def csv(key, name="metrics"):
    return pd.read_csv(RUNS[key] / "metrics" / f"{name}.csv")


def have(key):
    return RUNS.get(key) is not None and (RUNS[key] / "metrics").exists()


def pm(m, s, d=2):
    return f"{m:.{d}f}" + (f"\\,{{\\scriptsize$\\pm${s:.{d}f}}}" if s == s and s is not None else "")


def write(name, body):
    (OUT / f"{name}.tex").write_text(body)
    print("wrote", name)


def orbit_row(m, fam, variant, mode, aug=None):
    d = m[(m.family == fam) & (m.variant == variant) & (m["mode"] == mode)]
    if aug is not None:
        d = d[d.train_aug.astype(str) == aug]
    return {r.domain: r for r in d.itertuples()}


# ---------------------------------------------------------------- main SPEED+ table (rotation)
def main_table():
    base = csv("timing_eval"); base = {r.domain: r for r in base[base.system == "EXP-036 PoE greedy"].itertuples()}
    tim = csv("timing", "timing"); tms = dict(zip(tim[tim.batch == 1].system, tim[tim.batch == 1].ms_per_img_total))
    rows = []
    rows.append(("Clean tree (no augmentation)", "PoE", "1 view", base, tms["A-PoE greedy"]))
    for variant, key, tag in (("DINO", "distill_dino", "DINOv3"), ("PoE", "distill_poe", "PoE")):
        m = csv(key)
        stud = [f for f in m.family.unique() if "teacher" in f][0]
        for fam, name in (("tree", "Augmented tree (base recipe)"), (stud, "+ orbit distillation")):
            for mode in ("greedy", "orbit_product"):
                ms = tms[("A-DINO" if variant == "DINO" else "A-PoE") + (" greedy" if mode == "greedy" else " + orbit")]
                rows.append((name, tag, "1 view" if mode == "greedy" else "orbit (4)", orbit_row(m, fam, variant, mode), ms))
    L = [r"\begin{tabular}{@{}llc ccc cc r@{}}", r"\toprule",
         r"Training & Feat. & Inference & Synth.\ val & Lightbox & Sunlamp & \multicolumn{2}{c}{${>}150^\circ$ (\%)} & ms \\",
         r"\cmidrule(lr){7-8} & & & \multicolumn{3}{c}{mean $E_R$ ($^\circ$), 3 seeds} & LB & SL & \\", r"\midrule"]
    for i, (name, feat, inf, d, ms) in enumerate(rows):
        if i in (1, 5):
            L.append(r"\midrule")
        cells = [pm(d[g].mean_deg, getattr(d[g], "mean_deg_std", float("nan"))) for g in DOM]
        best = name.startswith("+") and feat == "PoE" and inf.startswith("orbit")
        if best:
            cells = [r"\textbf{" + c.split("\\,")[0] + "}" + ("\\," + c.split("\\,", 1)[1] if "\\," in c else "") for c in cells]
        fl = [f"{100 * d[g].frac_gt150:.1f}" for g in ("lightbox", "sunlamp")]
        L.append(f"{name} & {feat} & {inf} & " + " & ".join(cells) + " & " + " & ".join(fl) + f" & {ms:.0f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("main", "\n".join(L))
    # median / acc@10 companion numbers for the text
    m = csv("distill_poe"); stud = [f for f in m.family.unique() if "teacher" in f][0]
    d = orbit_row(m, stud, "PoE", "orbit_product"); a = orbit_row(m, "tree", "PoE", "orbit_product")
    lines = []
    for g in DOM + ["synthetic_val_aug"]:
        lines.append(f"% {g}: student+orbit mean {d[g].mean_deg:.2f} median {d[g].median_deg:.2f} | base+orbit mean {a[g].mean_deg:.2f} median {a[g].median_deg:.2f}")
    (OUT / "main_notes.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- flip transitions (base recipe vs clean tree)
def flips_table():
    m = csv("timing_eval")
    L = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
         r"System & \multicolumn{2}{c}{Lightbox} & \multicolumn{2}{c}{Sunlamp} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r" & fixed & broken & fixed & broken \\", r"\midrule"]
    for sys_, name in (("A-PoE greedy", "Aug.\\ tree, 1 view"), ("A-PoE + orbit", "Aug.\\ tree + orbit"), ("A-DINO + orbit", "Aug.\\ DINOv3 + orbit")):
        c = []
        for g in ("lightbox", "sunlamp"):
            r = m[(m.system == sys_) & (m.domain == g)].iloc[0]
            c += [f"{100 * r.frac_corrected_gt150_to_lt20:.0f}\\% of {r.n_base_gt150:.0f}", f"{100 * r.frac_broken_lt20_to_gt150:.1f}\\%"]
        L.append(f"{name} & " + " & ".join(c) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("flips", "\n".join(L))


# ---------------------------------------------------------------- 6-DoF
def sixdof_table():
    m = csv("sixdof")
    def get(arm, var, rm, tm):
        d = m[(m.arm == arm) & (m.variant == var) & (m.rot_mode == rm) & (m.trans_mode == tm)]
        return {r.domain: r for r in d.itertuples()}
    L = [r"\begin{tabular}{@{}lll ccc ccc@{}}", r"\toprule",
         r"Arm & Feat. & $\hat t$ & \multicolumn{3}{c}{$E_T$ (m)} & \multicolumn{3}{c}{$E_R$ ($^\circ$), orbit} \\",
         r"\cmidrule(lr){4-6}\cmidrule(lr){7-9} & & & Syn. & LB & SL & Syn. & LB & SL \\", r"\midrule"]
    b = get("B_box_only", "-", "-", "box")
    L.append(r"Box only & -- & -- & " + " & ".join(f"{b[g].E_T_mean_m:.3f}" for g in DOM) + r" & -- & -- & -- \\")
    L.append(r"\midrule")
    stds = []
    for arm, an in (("F_frozen_tree", "Frozen"), ("J_joint", "Joint")):
        for var, vn in (("DINO", "DINOv3"), ("PoE", "PoE")):
            for tm, tn in (("greedy", "1 view"), ("orbit_mean", "orbit")):
                d = get(arm, var, "orbit_product", tm); stds += [d[g].E_T_mean_m_std for g in DOM]
                et = [f"{d[g].E_T_mean_m:.3f}" for g in DOM]
                if arm == "J_joint" and var == "DINO" and tm == "orbit_mean":
                    et = [r"\textbf{" + et[0] + "}", et[1], r"\textbf{" + et[2] + "}"]
                if arm == "J_joint" and var == "PoE" and tm == "orbit_mean":
                    et = [et[0], r"\textbf{" + et[1] + "}", et[2]]
                er = [f"{d[g].E_R_mean_deg:.2f}" for g in DOM] if tm == "greedy" else ["", "", ""]
                L.append(f"{an if tm == 'greedy' else ''} & {vn if tm == 'greedy' else ''} & {tn} & " + " & ".join(et) + " & " + " & ".join(er) + r" \\")
        if arm == "F_frozen_tree":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("sixdof", "\n".join(L))
    (OUT / "sixdof_maxstd.tex").write_text(f"{max(stds):.3f}")
    d = get("J_joint", "PoE", "orbit_product", "orbit_mean"); e = get("J_joint", "DINO", "orbit_product", "orbit_mean")
    (OUT / "sixdof_notes.tex").write_text("".join(f"% {g}: J PoE esa {d[g].esa_score:.3f}, J DINO esa {e[g].esa_score:.3f}, J PoE e_t_norm {d[g].e_t_norm_mean:.4f}\n" for g in DOM))


# ---------------------------------------------------------------- cross-dataset
def xds_table():
    L = [r"\begin{tabular}{@{}ll cc cc cc@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{T-LESS (7 obj.)} & \multicolumn{2}{c}{SYMSOL-T (5)} & \multicolumn{2}{c}{YCB-V (6)} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8} Feat. & Inference & val & test & val & test & val & test \\", r"\midrule"]
    tabs = {k: csv(k, "object_balanced") for k in ("tless", "symsolt", "ycbv")}
    for var, vn in (("DINO", "DINOv3"), ("PoE", "PoE")):
        for mode, mn in (("greedy", "1 view"), ("orbit_product", "orbit (4)")):
            c = []
            for k in ("tless", "symsolt", "ycbv"):
                t = tabs[k]
                for sp in ("validation", "test"):
                    r = t[(t.variant == var) & (t["mode"] == mode) & (t.split == sp)].iloc[0]
                    c.append(pm(r.sym_mean, r.sym_mean_std, 2 if k == "symsolt" else 1))
            L.append(f"{vn} & {mn} & " + " & ".join(c) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("xds", "\n".join(L))
    # accuracy-at-threshold (test) for T-LESS / SYMSOL-T vs LiePose context
    out = []
    for k in ("tless", "symsolt"):
        p = csv(k, "per_object_seed"); p = p[p.split == "test"]
        for var, mode in (("PoE", "orbit_product"), ("DINO", "orbit_product")):
            d = p[(p.variant == var) & (p["mode"] == mode)].groupby("obj")[["sym_acc@2", "sym_acc@5", "sym_acc@10", "sym_median"]].mean().mean()
            out.append(f"% {k} {var} {mode}: R@2 {100*d['sym_acc@2']:.1f} R@5 {100*d['sym_acc@5']:.1f} R@10 {100*d['sym_acc@10']:.1f} median {d['sym_median']:.2f}")
    (OUT / "xds_notes.tex").write_text("\n".join(out) + "\n")
    # per-object / per-shape test table (supplement)
    L = [r"\begin{tabular}{@{}l cc@{}}", r"\toprule", r"Object & PoE+orbit & DINOv3+orbit \\", r"\midrule"]
    for k, title in (("tless", "T-LESS"), ("symsolt", "SYMSOL-T"), ("ycbv", "YCB-V")):
        p = csv(k, "per_object"); p = p[p.split == "test"]
        L.append(r"\multicolumn{3}{@{}l}{\emph{" + title + r"}} \\")
        for obj in sorted(p.obj.astype(str).unique(), key=lambda o: (0, int(o)) if o.isdigit() else (1, o)):
            v = [p[(p.obj.astype(str) == obj) & (p.variant == var) & (p["mode"] == "orbit_product")].sym_mean.iloc[0] for var in ("PoE", "DINO")]
            L.append(f"\\quad {obj.replace('_', r'\_')} & {v[0]:.2f} & {v[1]:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("xds_perobj", "\n".join(L))


# ---------------------------------------------------------------- ablations
def flat_table():
    """Tree vs flat heads (PoE). EXP-113: flat over SPACE-HOP's 10923x12 Hopf grid. EXP-127 (queued, blank until run): flat
    over the tree's own 131,072 leaves and over a balanced Hopf grid. Floor = mean nearest-anchor (tree: leaf) error on the
    synthetic-val GT rotations (EXP-127 quantization run)."""
    m = csv("flat"); q = csv("quant127", "quantization"); q = q[q.set == "synthetic_val"]
    floor = lambda key: q[q.grid.str.startswith(key)].mean_deg.iloc[0]
    m127 = csv("flat127") if have("flat127") else None
    rows = [("Tesseract tree (2.4\\,M)", "131,072 leaves", floor("tesseract_L5 encode"), m, "tree"),
            ("Flat (36.1\\,M)", "Hopf 10,923$\\times$12", floor("hopf10923x12"), m, "flat"),
            ("Flat, same grid", "131,072 leaves", floor("tesseract_L5 encode"), m127, "flat:grid=tesseract_L5"),
            ("Flat, balanced Hopf", "Hopf 1,771$\\times$74", floor("hopf1771x74"), m127, "flat:grid=hopf1771x74")]
    L = [r"\begin{tabular}{@{}llc l ccc@{}}", r"\toprule", r"Head (params/branch) & Grid & Floor & Inference & Synth. & LB & SL \\", r"\midrule"]
    for fn, grid, fl, mm, fam in rows:
        for k, (mode, mn) in enumerate((("greedy", "1 view"), ("orbit_product", "orbit"))):
            lead = f"{fn} & {grid} & {fl:.2f}" if k == 0 else " & & "
            if mm is None:
                L.append(f"{lead} & {mn} & & & \\\\"); continue
            d = orbit_row(mm, fam, "PoE", mode)
            L.append(f"{lead} & {mn} & " + " & ".join(pm(d[g].mean_deg, d[g].mean_deg_std) for g in DOM) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("flat", "\n".join(L))


def scoring_table():
    m = csv("scoring")
    names = {"greedy": "1 view (greedy)", "orbit_maxconf": "max-confidence", "orbit_vote1": "top-1 vote",
             "orbit_vote32": "weighted vote (32)", "orbit_product": "orbit product"}
    L = [r"\begin{tabular}{@{}l ccc@{}}", r"\toprule", r"Selection over 32 candidates & Synth. & LB & SL \\", r"\midrule"]
    for mode in ("greedy", "orbit_maxconf", "orbit_vote1", "orbit_vote32", "orbit_product"):
        d = orbit_row(m, "tree", "PoE", mode)
        L.append(f"{names[mode]} & " + " & ".join(pm(d[g].mean_deg, d[g].mean_deg_std) for g in DOM) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("scoring", "\n".join(L))


def augment_table():
    m = csv("augment")
    arms = (("nan", "none"), ("exp115_train_photo1,exp115_train_photo2", "photometric"),
            ("exp115_train_rot1,exp115_train_rot2", "in-plane roll"), ("exp109_train_aug1,exp109_train_aug2", "both"))
    L = [r"\begin{tabular}{@{}l ccc ccc@{}}", r"\toprule",
         r"Augmentation & \multicolumn{3}{c}{1 view} & \multicolumn{3}{c}{orbit (4)} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7} & Syn. & LB & SL & Syn. & LB & SL \\", r"\midrule"]
    for aug, an in arms:
        c = []
        for mode in ("greedy", "orbit_product"):
            d = orbit_row(m, "tree", "PoE", mode, aug)
            c += [pm(d[g].mean_deg, d[g].mean_deg_std, 2) if g == "synthetic_val" else f"{d[g].mean_deg:.2f}" for g in DOM]
        L.append(f"{an} & " + " & ".join(c) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("augment", "\n".join(L))


def consistency_table():
    m = csv("consistency"); dd = csv("distill_dino"); dp = csv("distill_poe")
    L = [r"\begin{tabular}{@{}l cc ccc@{}}", r"\toprule",
         r"Training objective & \multicolumn{2}{c}{1 view, DINOv3} & \multicolumn{3}{c}{PoE + orbit} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-6} & Syn. & Syn.-rot. & Syn. & LB & SL \\", r"\midrule"]
    specs = [("Base recipe (GT loss)", m, "tree"), ("+ paired views", m, "tree:mode=C1"),
             ("+ leaf-level transported consistency", m, "tree:mode=C2"), ("+ hierarchical transported consistency", m, "tree:mode=T")]
    for name, mm, fam in specs:
        g1 = orbit_row(mm, fam, "DINO", "greedy"); o = orbit_row(mm, fam, "PoE", "orbit_product")
        L.append(f"{name} & {pm(g1['synthetic_val'].mean_deg, g1['synthetic_val'].mean_deg_std)} & {pm(g1['synthetic_val_aug'].mean_deg, g1['synthetic_val_aug'].mean_deg_std)} & "
                 + " & ".join(f"{o[g].mean_deg:.2f}" for g in DOM) + r" \\")
    sd = [f for f in dd.family.unique() if "teacher" in f][0]; sp_ = [f for f in dp.family.unique() if "teacher" in f][0]
    g1 = orbit_row(dd, sd, "DINO", "greedy"); o = orbit_row(dp, sp_, "PoE", "orbit_product")
    L.append(r"\midrule")
    L.append(f"+ orbit distillation & {pm(g1['synthetic_val'].mean_deg, g1['synthetic_val'].mean_deg_std)} & {pm(g1['synthetic_val_aug'].mean_deg, g1['synthetic_val_aug'].mean_deg_std)} & "
             + " & ".join(f"\\textbf{{{o[g].mean_deg:.2f}}}" for g in DOM) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("consistency", "\n".join(L))
    # retention
    out = []
    for key, var in (("distill_dino", "DINO"), ("distill_poe", "PoE")):
        mm = csv(key); st = [f for f in mm.family.unique() if "teacher" in f][0]
        ag, ao, sg = orbit_row(mm, "tree", var, "greedy"), orbit_row(mm, "tree", var, "orbit_product"), orbit_row(mm, st, var, "greedy")
        ret = {g: (ag[g].mean_deg - sg[g].mean_deg) / (ag[g].mean_deg - ao[g].mean_deg) for g in DOM + ["synthetic_val_aug"]}
        out.append(f"% retention {var}: " + ", ".join(f"{g} {100 * v:.0f}%" for g, v in ret.items()))
    (OUT / "distill_notes.tex").write_text("\n".join(out) + "\n")


def timing_table():
    t = csv("timing", "timing")
    names = {"A-DINO greedy": "DINOv3, 1 view", "A-DINO + orbit": "DINOv3 + orbit", "A-PoE greedy": "PoE, 1 view", "A-PoE + orbit": "PoE + orbit"}
    L = [r"\begin{tabular}{@{}l rr rr@{}}", r"\toprule", r"System & \multicolumn{2}{c}{batch 1} & \multicolumn{2}{c}{batch 64} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5} & ms & FPS & ms/img & img/s \\", r"\midrule"]
    for s, n in names.items():
        a = t[(t.system == s) & (t.batch == 1)].iloc[0]; b = t[(t.system == s) & (t.batch == 64)].iloc[0]
        L.append(f"{n} & {a.ms_per_img_total:.1f} & {a.fps_or_img_per_s:.1f} & {b.ms_per_img_total:.1f} & {b.fps_or_img_per_s:.0f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("timing", "\n".join(L))


# ---------------------------------------------------------------- zero-shot (supplement)
def constant_pair(ds):
    """Post-hoc control: pair error of a constant prediction, same pair sampling as exp064_zeroshot.py."""
    from tfpose import xobj as XO
    from tfpose.rotations import quat_to_matrix
    ALL = pd.read_csv(REPO / f"outputs/data_manifests/exp064_{ds}_all.csv"); EV = pd.read_csv(REPO / f"outputs/data_manifests/exp064_{ds}_eval.csv")
    oc = XO.DATASETS[ds]["obj_col"]; rng = np.random.default_rng(0); res = []
    for obj in sorted(ALL[oc].unique().tolist(), key=str):
        de = EV.iloc[np.flatnonzero((EV[oc] == obj).to_numpy())]
        isv = (de.split == "validation").to_numpy(); vi = np.flatnonzero(isv); ti = np.flatnonzero(~isv)
        if len(vi) > 1000:
            vi = np.sort(rng.choice(vi, 1000, replace=False))
        de = de.iloc[np.sort(np.concatenate([vi, ti]))]; isv = (de.split == "validation").to_numpy()
        q, _ = XO.pose_ego(ds, de); Rg = quat_to_matrix(q); S = XO.symmetry_set(ds, obj)
        tst = np.flatnonzero(~isv); pairs = tst[rng.integers(0, len(tst), (2000, 2))]; pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        e = []
        for s0 in range(0, len(pairs), 300):
            i, j = pairs[s0:s0 + 300, 0], pairs[s0:s0 + 300, 1]
            c = np.einsum("nsii->ns", np.einsum("nij,sjk,nlk->nsil", Rg[i], S, Rg[j]))
            e.append(np.degrees(np.arccos(np.clip((c.max(1) - 1) / 2, -1, 1))))
        res.append(np.concatenate(e).mean())
    return float(np.mean(res))


def zeroshot_table():
    L = [r"\begin{tabular}{@{}l cc ccc@{}}", r"\toprule",
         r"Dataset & \multicolumn{2}{c}{Raw $E_R$} & \multicolumn{3}{c}{Frame-free pair error} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-6} & frozen & chance & frozen & chance & constant \\", r"\midrule"]
    for key, ds, name in (("zs_tless", "tless", "T-LESS"), ("zs_symsolt", "symsolt", "SYMSOL-T"), ("zs_ycbv", "ycbv", "YCB-V")):
        a = csv(key, "object_balanced"); r = a[(a.stats == "speedplus") & (a.split == "test") & (a.variant == "PoE") & (a["mode"] == "orbit_product")].iloc[0]
        const = r.const_pair_mean if "const_pair_mean" in a.columns and r.const_pair_mean == r.const_pair_mean else constant_pair(ds)
        L.append(f"{name} & {r.sym_mean:.1f} & {r.chance_sym_mean:.1f} & {pm(r.pair_mean, r.pair_mean_std, 1)} & {r.chance_pair_mean:.1f} & {const:.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("zeroshot", "\n".join(L))


# ---------------------------------------------------------------- symmetry-aware training (EXP-106) + distillation (EXP-119)
def ob(key, var, mode, split):
    t = csv(key, "object_balanced"); r = t[(t.variant == var) & (t["mode"] == mode) & (t.split == split)]
    return r.iloc[0] if len(r) else None


def sym_table():
    L = [r"\begin{tabular}{@{}l c cc cc@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{T-LESS (7 obj.)} & \multicolumn{2}{c}{SYMSOL-T (5)} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6} Training target & Inference & val & test & val & test \\", r"\midrule"]
    arms = (("A0", "single annotated label"), ("A1", "canonical label"), ("A2", "all equivalents (ours)"), ("A2D", "\;+ orbit distillation"))
    for a, name in arms:
        for mode, mn in (("greedy", "1 view"), ("orbit_product", "orbit (4)")):
            c = []
            for ds, d in (("tless", 1), ("symsolt", 2)):
                k = f"{ds}_{a}"
                for sp in ("validation", "test"):
                    r = ob(k, "PoE", mode, sp) if have(k) else None
                    c.append(pm(r.sym_mean, r.sym_mean_std, d) if r is not None else "")
            L.append(f"{name if mode == 'greedy' else ''} & {mn} & " + " & ".join(c) + r" \\")
        if a != "A2D":
            L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("sym", "\n".join(L))


# ---------------------------------------------------------------- T-LESS vs LiePose (recall, test)
LIEPOSE_TLESS = {1: (38.22, 78.20, 89.10), 4: (44.28, 86.36, 93.43), 5: (49.47, 91.58, 96.84), 11: (25.14, 77.71, 91.43),
                 18: (16.44, 59.59, 78.77), 21: (37.71, 72.57, 77.71), 27: (33.33, 81.25, 94.79)}   # LiePose Tab. 8


def recall(key, var, mode):
    p = csv(key, "per_object_seed"); p = p[(p.split == "test") & (p.variant == var) & (p["mode"] == mode)]
    d = p.groupby("obj")[["sym_acc@2", "sym_acc@5", "sym_acc@10", "sym_mean"]].mean().mean()
    return [100 * d["sym_acc@2"], 100 * d["sym_acc@5"], 100 * d["sym_acc@10"], d["sym_mean"]]


def tless_sota_table():
    L = [r"\begin{tabular}{@{}l l cccc@{}}", r"\toprule", r"Method & Inference & R@2 & R@5 & R@10 & mean ($^\circ$) \\", r"\midrule"]
    lp = np.array(list(LIEPOSE_TLESS.values())).mean(0)
    L.append(f"LiePose~\\cite{{hsiao2024liepose}} & 1 sample & {lp[0]:.1f} & {lp[1]:.1f} & {lp[2]:.1f} & -- \\\\")
    L.append(r"\midrule")
    for k, name in (("tless_A0", "Ours, A0"), ("tless_A1", "Ours, A1"), ("tless_A2", "Ours, A2"), ("tless_A2D", "Ours, A2 + distill.")):
        for mode, mn in (("greedy", "1 view"), ("orbit_product", "orbit (4)")):
            v = recall(k, "PoE", mode) if have(k) else None
            cells = [f"{x:.1f}" for x in v] if v else ["", "", "", ""]
            L.append(f"{name if mode == 'greedy' else ''} & {mn} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tless_sota", "\n".join(L))


# ---------------------------------------------------------------- SYMSOL-T vs LiePose (per shape, test)
LIEPOSE_SYMSOLT = {"tet": 0.59, "cube": 0.58, "icosa": 0.64, "cone": 0.54, "cyl": 0.41}             # LiePose Tab. 3 (SE(3) model)


def symsol_sota_table():
    shapes = list(LIEPOSE_SYMSOLT)
    L = [r"\begin{tabular}{@{}l " + "c" * (len(shapes) + 1) + r"@{}}", r"\toprule", "Method & " + " & ".join(shapes) + r" & avg. \\", r"\midrule"]
    L.append("LiePose~\\cite{hsiao2024liepose} & " + " & ".join(f"{v:.2f}" for v in LIEPOSE_SYMSOLT.values()) + f" & {np.mean(list(LIEPOSE_SYMSOLT.values())):.2f} \\\\")
    L.append(r"\midrule")
    for k, name in (("symsolt_A0", "Ours, A0 + orbit"), ("symsolt_A2", "Ours, A2 + orbit"), ("symsolt_A2D", "Ours, A2 + distill.\\ + orbit")):
        if have(k):
            p = csv(k, "per_object"); p = p[(p.split == "test") & (p.variant == "DINO") & (p["mode"] == "orbit_product")].set_index("obj").sym_mean
            L.append(f"{name} & " + " & ".join(f"{p[s_]:.2f}" for s_ in shapes) + f" & {p[shapes].mean():.2f} \\\\")
        else:
            L.append(f"{name} & " + " & ".join("" for _ in shapes) + r" &  \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("symsol_sota", "\n".join(L))


# ---------------------------------------------------------------- YCB-V: real-only vs + synthetic composites (EXP-064 vs EXP-122)
def ycbv_table():
    L = [r"\begin{tabular}{@{}l l c cc c@{}}", r"\toprule", r"Training data & Feat. & Inference & val & test & test acc@10 \\", r"\midrule"]
    for k, name in (("ycbv", "real videos"), ("ycbv_syn", "+ synthetic (ours)")):
        for var, vn in (("DINO", "DINOv3"), ("PoE", "PoE")):
            for mode, mn in (("greedy", "1 view"), ("orbit_product", "orbit (4)")):
                v, t = ob(k, var, mode, "validation"), ob(k, var, mode, "test")
                L.append(f"{name if (var, mode) == ('DINO', 'greedy') else ''} & {vn} & {mn} & {pm(v.sym_mean, v.sym_mean_std, 1)} & {pm(t.sym_mean, t.sym_mean_std, 1)} & {100 * t.acc10:.0f}\\% \\\\")
        if k == "ycbv":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("ycbv", "\n".join(L))
    p = csv("ycbv_syn", "per_object"); p = p[(p.split == "test") & (p["mode"] == "greedy")]
    q = csv("ycbv", "per_object"); q = q[(q.split == "test") & (q["mode"] == "greedy")]
    L = [r"\begin{tabular}{@{}l cc cc@{}}", r"\toprule", r" & \multicolumn{2}{c}{real videos} & \multicolumn{2}{c}{+ synthetic} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5} Object & DINOv3 & PoE & DINOv3 & PoE \\", r"\midrule"]
    for obj in sorted(p.obj.unique()):
        g = lambda d, var: d[(d.obj == obj) & (d.variant == var)].sym_mean.iloc[0]
        L.append(f"{obj.replace('_', r'\_')} & {g(q, 'DINO'):.1f} & {g(q, 'PoE'):.1f} & {g(p, 'DINO'):.1f} & {g(p, 'PoE'):.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("ycbv_perobj", "\n".join(L))


# ---------------------------------------------------------------- SPEED+ against published results (official score)
SPP_PUBLISHED = [   # method, setting, E_R LB, E_R SL, score LB, score SL (as published; see SOURCES.md)
    (r"TangoUnchained$^*$~\cite{park2023spec}", "detector + keypoints/PnP, real images", 3.19, 4.30, 0.073, 0.090),
    (r"lava1302$^*$~\cite{park2023spec}", "detector + keypoints/PnP, self-training", 6.66, 2.73, 0.165, 0.059),
    (r"SPNv2~\cite{park2024spnv2}", "full image, no adaptation", 6.48, 11.07, 0.142, 0.230),
    (r"SPNv2 + ODR$^*$~\cite{park2024spnv2}", "+ test-time adaptation", 5.58, 9.79, 0.122, 0.197),
    (r"EagerNet~\cite{ulmer2023eagernet}", "detector + dense correspondences", 1.75, 2.66, 0.039, 0.059),
    (r"SPNv3-B$^\dagger$~\cite{park2026spnv3}", "GT crop, keypoints/PnP, fine-tuned", 2.03, 3.40, 0.047, 0.074),
    (r"Church \& Nikolian$^\dagger$~\cite{church2026vfm}", "GT crop, DINOv3-H+ LoRA, 12 passes", 1.17, 1.56, 0.0273, 0.0383),
    (r"\quad frozen DINOv3-L + head~\cite{church2026vfm}", "GT crop, frozen, 1 pass", 15.47, 24.49, None, None),
    (r"FastPose-ViT~\cite{ancey2026fastpose}", "detector, direct regression", 12.26, 18.28, None, None),
    (r"PAID-ViT~\cite{xie2026paidvit}", "crop, direct regression", 12.29, 11.76, 0.237, 0.230),
]


def trans_xobj_table():
    """EXP-126: translation on T-LESS / SYMSOL-T / YCB-V, real test sets, DINOv3 block on the frozen selected tree.
    Rows: box only; view 0 (validation choice where it wins); orbit mean (pre-registered primary). E_T in cm (T-LESS, YCB-V)
    or scene units (SYMSOL-T). LiePose T-LESS average from Tab. 8 of LiePose over our 7 objects (EXP-126 report)."""
    TO_CM = {"tless": 0.1, "ycbv": 100.0, "symsolt": 1.0}
    L = [r"\begin{tabular}{@{}ll cccc@{}}", r"\toprule", r"Dataset & Method & T@2 & T@5 & T@10 & mean $E_T$ \\", r"\midrule"]
    L.append(r"T-LESS & LiePose~\cite{hsiao2024liepose} & 68.0 & 88.4 & 95.2 & -- \\")
    for k, (ds, name) in enumerate((("tless", "T-LESS"), ("ycbv", "YCB-V"), ("symsolt", "SYMSOL-T"))):
        rows = (("box_only", "box only"), ("greedy", "ours, 1 view"), ("orbit_mean", "ours, orbit mean"))
        if have(f"trans_{ds}"):
            t = csv(f"trans_{ds}", "trans_object_balanced"); t = t[(t.split == "test") & (t.variant == "DINO")]
            for n, (mode, lab) in enumerate(rows):
                r = t[t.trans_mode == mode].iloc[0]; e = r.E_T_mean * TO_CM[ds]
                rec = " & ".join(f"{100 * r[c]:.1f}" for c in ("T@2cm", "T@5cm", "T@10cm")) if ds != "symsolt" else "-- & -- & --"
                L.append(f"{name if (n == 0 and ds != 'tless') else ''} & {lab} & {rec} & {e:.{3 if ds == 'symsolt' else 2}f} \\\\")
        else:
            for n, (mode, lab) in enumerate(rows):
                L.append(f"{name if (n == 0 and ds != 'tless') else ''} & {lab} & & & & \\\\")
        if ds != "symsolt":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("trans_xobj", "\n".join(L))


def deploy_tables():
    """EXP-128 (queued): downstream-head cost (size, latency, memory per precision) and accuracy per precision. Blank until run."""
    assert not have("deploy128"), "fill deploy_tables from the EXP-128 runs before enabling them"
    L = [r"\begin{tabular}{@{}l r ccc@{}}", r"\toprule", r"Head module & Params & fp32 (MB) & fp16 (MB) & int8 (MB) \\", r"\midrule",
         r"Tree, DINOv3 branch & 2,391,564 & & & \\", r"Tree, MoGe-2 branch & 2,128,404 & & & \\", r"Translation block & 331,011 & & & \\",
         r"\bottomrule", r"\end{tabular}"]
    write("deploy_size", "\n".join(L))
    L = [r"\begin{tabular}{@{}ll cccc cc@{}}", r"\toprule",
         r" & & \multicolumn{4}{c}{GPU (ms/img)} & \multicolumn{2}{c}{CPU, 1 thread (ms/img)} \\", r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}",
         r"Head & Inference & fp32 & bf16 & fp16 & peak MB & fp32 & int8 \\", r"\midrule"]
    for v in ("DINOv3", "PoE"):
        for inf in ("1 view", "orbit (4)"):
            L.append(f"{v if inf == '1 view' else ''} & {inf} & & & & & & \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("deploy_latency", "\n".join(L))
    L = [r"\begin{tabular}{@{}ll cccc@{}}", r"\toprule", r"Application & Inference & fp32 & bf16 & fp16 & int8 (w) \\", r"\midrule"]
    for app in ("T-LESS (A2, PoE)", "SYMSOL-T (A2, DINOv3)", "YCB-V (syn., DINOv3)", "SPEED+ (distilled, PoE)"):
        for inf in ("1 view", "orbit (4)"):
            L.append(f"{app if inf == '1 view' else ''} & {inf} & & & & \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("deploy_quant", "\n".join(L))


def spp_sota_table():
    f = lambda v, d=2: "" if v is None else f"{v:.{d}f}"
    L = [r"\begin{tabular}{@{}l l cc cc@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{$E_R$ ($^\circ$)} & \multicolumn{2}{c}{score} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6} Method & Setting & LB & SL & LB & SL \\", r"\midrule"]
    for m, st, a, b, c, d in SPP_PUBLISHED:
        L.append(f"{m} & {st} & {f(a)} & {f(b)} & {f(c, 3)} & {f(d, 3)} \\\\")
    L.append(r"\midrule")
    t = csv("distill_poe"); r = {x.domain: x for x in t[(t.family == "tree:teacher_variant=PoE") & (t.variant == "PoE") & (t["mode"] == "orbit_product")].itertuples()}
    six = csv("sixdof"); s2 = {x.domain: x for x in six[(six.arm == "J_joint") & (six.variant == "PoE") & (six.rot_mode == "orbit_product") & (six.trans_mode == "orbit_mean")].itertuples()}
    g = csv("spp_gt"); g = {x.domain: x for x in g[(g.crops == "gt") & g.system.str.startswith("S1") & (g.rot_mode == "orbit_product") & (g.trans_mode == "orbit_mean")].itertuples()}
    L.append(f"Ours, distilled + orbit & GT crop, frozen, 4 views & {r['lightbox'].mean_deg:.2f} & {r['sunlamp'].mean_deg:.2f} & {g['lightbox'].score:.3f} & {g['sunlamp'].score:.3f} \\\\")
    L.append(f"Ours, 6-DoF (joint PoE) + orbit & GT crop, frozen, 4 views & {s2['lightbox'].E_R_mean_deg:.2f} & {s2['sunlamp'].E_R_mean_deg:.2f} & {s2['lightbox'].esa_score:.3f} & {s2['sunlamp'].esa_score:.3f} \\\\")
    if have("spp_det"):
        dd = csv("spp_det"); dd = {x.domain: x for x in dd[(dd.crops == "det") & dd.system.str.startswith("S1") & (dd.rot_mode == "orbit_product") & (dd.trans_mode == "orbit_mean")].itertuples()}
        L.append(f"Ours, distilled + orbit & synthetic-only detector, frozen & {dd['lightbox'].E_R_mean_deg:.2f} & {dd['sunlamp'].E_R_mean_deg:.2f} & {dd['lightbox'].score:.3f} & {dd['sunlamp'].score:.3f} \\\\")
    else:
        L.append(r"Ours, distilled + orbit & synthetic-only detector, frozen &  &  &  &  \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("spp_sota", "\n".join(L))


if __name__ == "__main__":
    trans_xobj_table(); deploy_tables()
    main_table(); flips_table(); sixdof_table(); xds_table(); flat_table(); scoring_table(); augment_table()
    consistency_table(); timing_table(); zeroshot_table()
    sym_table(); tless_sota_table(); symsol_sota_table(); ycbv_table(); spp_sota_table()
