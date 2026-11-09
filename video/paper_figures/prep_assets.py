"""Data assets for the manim paper figures (run with the project venv, from the repo root):
    .venv/bin/python video/paper_figures/prep_assets.py
Re-uses the drawing code of scripts/fig_qualitative.py (its function definitions are executed, not its figure code), so
crops, overlays and colours are exactly those of the qualitative figures. Writes PNG crops and assets.json."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent / "assets"; OUT.mkdir(exist_ok=True)
src = (REPO / "scripts/fig_qualitative.py").read_text(); cut = src.index("DATA = {ds")
sys.argv = ["fig_qualitative.py", "--out", str(OUT / "_unused")]
NS = {"__name__": "figq", "__file__": str(REPO / "scripts/fig_qualitative.py")}; exec(compile(src[:cut], "fig_qualitative.py", "exec"), NS)
np, pd, plt, cv2, quat_to_matrix = NS["np"], NS["pd"], NS["plt"], NS["cv2"], NS["quat_to_matrix"]
S = 320


def sym_err_one(r, q):
    Rg = quat_to_matrix(r.q_gt)[None] if (r.ds == "speedplus" or r.sym is None) else quat_to_matrix(r.q_gt) @ r.sym
    c = np.einsum("ij,sij->s", quat_to_matrix(q), Rg).max(); return float(np.degrees(np.arccos(np.clip((c - 1) / 2, -1, 1))))


def save_panel(r, name, q_pred=None, overlay=True, roll=0.0, gt=True):
    fig = plt.figure(figsize=(2, 2), dpi=S / 2); ax = fig.add_axes([0, 0, 1, 1])
    if overlay:
        q_pred = r.q_pred if q_pred is None else q_pred
        ax.imshow(NS["crop_img"](r, S))
        if gt:
            NS["draw_pose"](ax, r, NS["gt_equiv"](r, q_pred), r.t_gt, NS["GT_C"], 2.0, S)
        NS["draw_pose"](ax, r, q_pred, r.t_pred, NS["PR_C"], 1.7, S)
        ax.set_xlim(0, S); ax.set_ylim(S, 0)
    else:
        ax.imshow(NS["crop_img"](r, S, roll=roll))
    ax.axis("off"); fig.savefig(OUT / f"{name}.png"); plt.close(fig)


A = {}
# ---- orbit-voting example: the same fixed-rule image as paper/figures/qualitative/orbit_voting_speedplus
spp = NS["load_spp"]("gt"); g = spp[spp.obj == "lightbox"]; rep = g[(g.err_greedy > 150) & (g.err < 10)]
r = rep.iloc[int(np.argmin(np.abs(rep.err - rep.err.median())))]
cand_err = [sym_err_one(r, r.cand[i]) for i in range(32)]
sc = r.score - r.score.max(); p = np.exp(sc) / np.exp(sc).sum()
for k in range(4):
    save_panel(r, f"orbit_view{k}", overlay=False, roll=90 * k)
    save_panel(r, f"orbit_top{k}", q_pred=r.cand[8 * k])
save_panel(r, "orbit_final"); save_panel(r, "orbit_greedy", q_pred=r.q_greedy); save_panel(r, "orbit_raw", overlay=False)
A["orbit"] = dict(image=r.image, err_final=r.err, err_greedy=r.err_greedy, cand_err=cand_err, score=r.score.tolist(), prob=p.tolist(),
                  top_err=[cand_err[8 * k] for k in range(4)], winner=int(np.argmax(r.score)))
# ---- one median-error example per application (first object in the fixed overview rule), for the teaser
sel = pd.read_csv(REPO / "paper/figures/qualitative/selection.csv"); ov = sel[sel.figure == "overview_medians"]
A["teaser"] = {}
for ds, obj in (("tless", "5"), ("ycbv", "035_power_drill"), ("symsolt", "cube")):
    row = ov[(ov.row == ds) & (ov.column.astype(str) == obj)].iloc[0]; df = NS["load_general"](ds); rr = df[(df.obj == obj) & (df.image == row.image)].iloc[0]
    save_panel(rr, f"teaser_{ds}"); A["teaser"][ds] = dict(obj=obj, image=row.image, err=rr.err)
A["teaser"]["speedplus"] = dict(obj="lightbox", image=r.image, err=r.err)
(OUT / "assets.json").write_text(json.dumps(A, indent=1, default=float))
print(json.dumps({k: (v if k != "orbit" else {x: v[x] for x in ("err_final", "err_greedy", "top_err", "winner")}) for k, v in A.items()}, indent=1, default=float))

# ---- multi-path example: the T-LESS teaser image (object 5, two-fold symmetry); training-frame (allocentric) label and its
#      symmetry-equivalent members, with their tesseract paths from the project encoder
from scipy.spatial.transform import Rotation as _R  # noqa: E402
from tfpose import tesseract as TS  # noqa: E402
df = NS["load_general"]("tless"); rr = df[(df.obj == "5") & (df.image == A["teaser"]["tless"]["image"])].iloc[0]
Rq = quat_to_matrix(rr.q_tgt_gt); eq = [np.roll(_R.from_matrix(Rq @ S_).as_quat(), 1) for S_ in rr.sym]
c, pth = TS.encode(np.array(eq), 5)
A["multipath"] = dict(obj="5", n_sym=len(rr.sym), q=[list(map(float, q)) for q in eq], chart=c.tolist(), path=pth.tolist())
(OUT / "assets.json").write_text(json.dumps(A, indent=1, default=float))
print(A["multipath"])
