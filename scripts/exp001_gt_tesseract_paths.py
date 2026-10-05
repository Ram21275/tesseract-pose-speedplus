"""EXP-001: SPEED+ ground-truth quaternion -> Tesseract path (all domains).

Writes the derived label manifest to outputs/data_manifests/ (references images
by relative path; dataset untouched) and checks the geometry invariants on the
real label distribution.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfpose import data as D  # noqa: E402
from tfpose import rotations as rot  # noqa: E402
from tfpose import tesseract as T  # noqa: E402
from tfpose.runlog import Run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--depth", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
paths = yaml.safe_load(open(REPO / "configs/paths.yaml"))
root = Path(paths["speedplus_root"])
run = Run("EXP-001", "gt_tesseract_paths", {"paths": paths, **vars(args)}, seed=args.seed)
try:
    rng = np.random.default_rng(args.seed)
    L = args.depth
    recs = [r for k in D.SPLITS for r in D.load_split(root, *k)]
    q = np.stack([r["q"] for r in recs])
    qc, chart = T.canonicalize(q)
    c, path = T.encode(q, L)
    assert np.array_equal(c, chart)
    centre = T.decode(c, path)
    lid = T.leaf_id(c, path)
    qerr = np.degrees(rot.geodesic_distance(q, centre))

    checks = {}
    c2, p2 = T.encode(-q, L)
    checks["path_q_eq_path_negq"] = bool(np.array_equal(c, c2) and np.array_equal(path, p2))
    c3, p3 = T.encode(centre, L)
    checks["encode_decode_roundtrip"] = bool(np.array_equal(c, c3) and np.array_equal(path, p3))
    checks["all_finite"] = bool(np.all(np.isfinite(centre)) and np.all(np.isfinite(qc)))
    checks["canonical_chart_positive"] = bool(np.all(qc[np.arange(len(qc)), chart] > 0))
    for l in range(1, L + 1):
        cl, pl = T.encode(q, l)
        checks[f"prefix_consistent_L{l}"] = bool(np.array_equal(pl, path[:, :l]))
    # ties in the real labels
    a = np.sort(np.abs(q), axis=1)
    checks["n_exact_chart_ties"] = int(np.sum(a[:, -1] == a[:, -2]))
    checks["min_chart_margin"] = float(np.min(a[:, -1] - a[:, -2]))

    # seam-neighbour consistency: perturb GT rotations by tiny rotations and
    # check that crossing a chart/cell seam never yields a centre farther than
    # the cell bound, i.e. the decoded centre is continuous up to cell size.
    eps = np.radians(1e-3)
    axes = rot.normalize(rng.standard_normal((len(q), 3)))
    dq = rot.axis_angle_to_quat(axes, eps)
    qp = rot.quat_multiply(q, dq)
    cp, pp = T.encode(qp, L)
    moved = (cp != c) | np.any(pp != path, axis=1)
    chart_moved = cp != c
    corners = T.cell_corners(c, path)
    bound = np.max(2 * np.arccos(np.clip(np.abs(np.einsum("nd,nkd->nk", centre, corners)), 0, 1)), axis=1)
    d_centres = rot.geodesic_distance(centre, T.decode(cp, pp))
    seam = {"seam_eps_deg": 1e-3, "n_leaf_changed": int(moved.sum()), "n_chart_changed": int(chart_moved.sum()),
            "max_centre_jump_over_2x_bound": float(np.max(d_centres / (2 * bound + 1e-12))) if moved.any() else 0.0}
    # synthetic seam probes: points exactly on chart seams |q_i| == |q_j|
    probe = rot.random_quats(200000, rng)
    i0 = np.argmax(np.abs(probe), 1)
    tmp = np.abs(probe).copy()
    tmp[np.arange(len(tmp)), i0] = -1
    j0 = np.argmax(tmp, 1)
    probe[np.arange(len(probe)), j0] = np.sign(probe[np.arange(len(probe)), j0]) * np.abs(probe[np.arange(len(probe)), i0])
    probe = rot.normalize(probe)
    pc, ppth = T.encode(probe, L)
    nc, npth = T.encode(-probe, L)
    seam["seam_probe_antipodal_consistent"] = bool(np.array_equal(pc, nc) and np.array_equal(ppth, npth))
    seam["seam_probe_chart_is_first_max"] = bool(np.all(pc == np.minimum(i0, j0)))
    seam["seam_probe_err_le_bound"] = bool(np.all(rot.geodesic_distance(probe, T.decode(pc, ppth)) <= np.max(bound) + 1e-9))
    checks.update(seam)

    # per-domain label stats
    dom = np.array([r["domain"] for r in recs])
    split = np.array([r["split"] for r in recs])
    rows = []
    for d_, s_ in D.SPLITS:
        m = (dom == d_) & (split == s_)
        cc = np.bincount(chart[m], minlength=4) / m.sum()
        row = {"domain": d_, "split": s_, "n": int(m.sum())}
        row.update({f"chart{k}_frac": float(cc[k]) for k in range(4)})
        for l in range(1, L + 1):
            row[f"occupied_leaves_L{l}"] = int(len(np.unique(T.leaf_id(c[m], path[m, :l]))))
        row.update({"qerr_L5_mean_deg": float(qerr[m].mean()), "qerr_L5_max_deg": float(qerr[m].max())})
        rows.append(row)
    run.write_metrics(checks, "metrics")
    run.write_metrics(rows, "label_stats")

    # derived manifest
    man_dir = REPO / "outputs/data_manifests"
    man_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "image_relpath": [r["image_relpath"] for r in recs], "domain": dom, "split": split,
        **{f"q_src_{k}": [r["q_source"][i] for r in recs] for i, k in enumerate("wxyz")},
        **{f"q_can_{k}": qc[:, i] for i, k in enumerate("wxyz")},
        "root_chart": c, "child_path": ["".join(map(str, p)) for p in path], f"leaf_id_L{L}": lid,
        **{f"leaf_centre_{k}": centre[:, i] for i, k in enumerate("wxyz")},
        f"quant_err_L{L}_deg": qerr,
        **{f"t_{k}": [r["r"][i] for r in recs] for i, k in enumerate("xyz")},
        "source_annotation": [D.SPLITS[(r["domain"], r["split"])] for r in recs],
    })
    out = man_dir / f"speedplus_tesseract_L{L}.csv"
    df.to_csv(out, index=False, float_format="%.9g")
    df.to_csv(out.with_suffix(".tsv"), index=False, sep="\t", float_format="%.9g")
    h = D.manifest_hash(df["image_relpath"].tolist() + df["child_path"].tolist())
    (man_dir / f"speedplus_tesseract_L{L}.sha256.txt").write_text(f"{h}  rows={len(df)}  code=EXP-001\n")
    run.done(manifest=str(out.relative_to(REPO)), manifest_hash=h)
    print(checks)
    print(pd.DataFrame(rows).T.to_string())
    print("RUN", run.rel())
except Exception:
    import traceback
    run.fail(traceback.format_exc())
    raise
