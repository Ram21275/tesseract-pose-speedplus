"""Redraw a manim figure dump (pfcommon.dump_vector) as a vector PDF: text (LaTeX glyph outlines), equations, boxes, lines
and arrows stay vector paths; only the photographs/feature maps are embedded rasters (at their source resolution).
Usage (project venv):  python vec2pdf.py <dump.pkl> <out.pdf> [width_in=7.0]"""
import pickle
import sys

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from PIL import Image  # noqa: E402

matplotlib.rcParams["pdf.compression"] = 9


def to_path(subs):
    verts, codes = [], []
    for sp in subs:
        verts.append(sp[0]); codes.append(Path.MOVETO)
        for i in range(1, len(sp) - 1, 2):                 # quadratic Bezier: handle, anchor
            verts += [sp[i], sp[i + 1]]; codes += [Path.CURVE3, Path.CURVE3]
    return Path(np.array(verts), codes)


def main(src, out, width_in=7.0):
    D = pickle.load(open(src, "rb")); x0, y0, x1, y1 = D["box"]
    W, H = x1 - x0, y1 - y0; height_in = width_in * H / W
    pt_per_unit = 72.0 * width_in / W
    fig = plt.figure(figsize=(width_in, height_in)); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.axis("off")
    z = 0
    for p in D["prims"]:
        z += 1
        if p["kind"] == "image":
            im = np.asarray(Image.open(p["path"]).convert("RGBA"))
            (ux, uy), (dx, dy) = p["ul"], p["dr"]
            ax.imshow(im, extent=(ux, dx, dy, uy), interpolation="none", zorder=z, alpha=p["opacity"])
            continue
        path = to_path(p["subs"])
        f, s, lw = p["fill"], p["stroke"], p["sw"] * pt_per_unit
        parts = []
        if f[3] > 0:
            parts.append(PathPatch(path, facecolor=f[:3], alpha=f[3], edgecolor="none", lw=0))
        if s[3] > 0 and lw > 0:
            parts.append(PathPatch(path, facecolor="none", edgecolor=s[:3], alpha=s[3], lw=lw, joinstyle="round", capstyle="butt"))
        if p.get("behind"):
            parts.reverse()
        for q in parts:
            q.set_zorder(z); ax.add_patch(q)
    fig.savefig(out); plt.close(fig)
    print(out, f"{width_in:.2f}x{height_in:.2f} in, {len(D['prims'])} primitives")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(float(a) for a in sys.argv[3:]))
