"""Paper figure: the tesseract partition of SO(3) and the root-to-leaf path of one rotation.
Everything is computed with the project's encoder/decoder (src/tfpose/tesseract.py via ../common.py): chart choice,
sign normalisation, projection, nested cell boxes, path digits and the error of each level's cell centre.
Render:  python render.py paper_figures/fig_tesseract.py FigTesseract -s -w -r 3600x1350 --video_dir <dir>"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pfcommon import *  # noqa: F401,F403,E402

Q = quat([0.62, -0.35, 0.70], 2.25)            # the example rotation of the explainer video (fixed)
CH, QS, U = free_coords(Q)
CHART, PATH = encode(Q, 5)
assert CHART == CH
NAMES = ["w", "x", "y", "z"]
MEAN_ERR = [39.27, 19.95, 10.01, 5.01, 2.51]  # mean own-cell-centre error per depth over 2e6 Haar rotations (paper, Sec. 3.3)
ERR = [geodesic_deg(Q, centre(CHART, PATH[:l])) for l in range(6)]


class FigTesseract(PaperScene):
    def construct(self):
        # ---------------------------------------------------------- (a) chart choice
        bars = VGroup()
        for i in range(4):
            v = QS[i]
            r = Rectangle(width=0.55, height=max(abs(v) * 2.4, 0.04)).set_stroke(width=0)
            r.set_fill(CH4[i] if i == CH else "#C9CED6", 1)
            r.move_to(RIGHT * 0.8 * i, aligned_edge=DOWN if v >= 0 else UP)
            bars.add(r)
        base = Line(LEFT * 0.45, RIGHT * 2.85, stroke_width=2).set_color(INK)
        labs = VGroup(*[tx(f"q_{NAMES[i]}", 26).next_to(bars[i], DOWN if QS[i] >= 0 else UP, buff=0.12) for i in range(4)])
        for i in range(4):
            labs[i].set_y(-0.32)
        vals = VGroup(*[tx(f"{QS[i]:+.2f}", 22, MUTED).next_to(bars[i], UP if QS[i] >= 0 else DOWN, buff=0.08) for i in range(4)])
        pa = VGroup(bars, base, labs, vals)
        eqa = VGroup(tx(r"c=\arg\max_i |q_i| = " + NAMES[CH], 30), tx(r"q \leftarrow \mathrm{sign}(q_c)\,q", 28),
                     tx(r"u = \big(q_j/q_c\big)_{j\neq c}\in[-1,1]^3", 28)).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        eqa[0][-1].set_color(CH4[CH])
        eqa.next_to(pa, DOWN, buff=0.45, aligned_edge=LEFT)
        ta = tt("chart: largest quaternion coordinate", 26, bold=True).next_to(pa, UP, buff=0.4, aligned_edge=LEFT)
        A = VGroup(ta, pa, eqa)
        la = letter("a").next_to(ta, LEFT, buff=0.15)

        # ---------------------------------------------------------- (b) nested cells in the projected cube
        S = 2.0; O = RIGHT * 7.2 + DOWN * 0.2
        kw = dict(s=S, o=O)
        cube = cube_lines(-np.ones(3), np.ones(3), CH4[CH], sw=3, **kw)
        mids = VGroup()
        for k in range(3):                                         # the first split: three mid-planes, drawn faint
            for a, b in [((-1, -1), (1, -1)), ((1, -1), (1, 1)), ((1, 1), (-1, 1)), ((-1, 1), (-1, -1))]:
                p0 = np.insert(np.array(a, float), k, 0.0); p1 = np.insert(np.array(b, float), k, 0.0)
                mids.add(DashedLine(iso(p0, **kw), iso(p1, **kw), dash_length=0.06, stroke_width=1.2).set_color("#B8BEC8"))
        nest = VGroup()
        for l in (1, 2, 3):
            lo, hi = cell_box(PATH[:l])
            nest.add(cube_faces(lo, hi, LV[l], 0.18 if l < 3 else 0.35, **kw), cube_lines(lo, hi, LV[l], sw=3.2, **kw))
        dot = Dot(iso(U, **kw), radius=0.06).set_fill(GT, 1).set_stroke(WHITE, 1.5)
        corner = -np.ones(3) * 1.0
        axl = VGroup()
        for k in range(3):
            e = np.eye(3)[k] * 0.55; st = corner + np.array([-0.12, -0.12, -0.12]) * (np.arange(3) != k)
            axl.add(Arrow(iso(st, **kw), iso(st + e, **kw), buff=0, stroke_width=2.5).set_color(MUTED))
            axl.add(tx(f"u_{k}", 24, MUTED).next_to(iso(st + e, **kw), [LEFT + DOWN * 0.2, DOWN, LEFT][k], buff=0.08))
        # zoom inset: the level-3 cell, magnified, with levels 4 and 5 inside
        lo3, hi3 = cell_box(PATH[:3]); Z = RIGHT * 11.2 + DOWN * 0.2; s3 = 2.0 / (hi3[0] - lo3[0]) * 0.55 * S
        zmap = lambda p: (np.asarray(p) - (lo3 + hi3) / 2) * s3 / S
        zkw = dict(s=S, o=Z)
        zin = VGroup(cube_lines(zmap(lo3), zmap(hi3), LV[3], sw=3, **zkw))
        for l in (4, 5):
            lo, hi = cell_box(PATH[:l])
            zin.add(cube_faces(zmap(lo), zmap(hi), LV[l], 0.22 if l == 4 else 0.45, **zkw), cube_lines(zmap(lo), zmap(hi), LV[l], sw=3, **zkw))
        zdot = Dot(iso(zmap(U), **zkw), radius=0.055).set_fill(GT, 1).set_stroke(WHITE, 1.5)
        c5 = (np.array(cell_box(PATH)[0]) + np.array(cell_box(PATH)[1])) / 2
        zc5 = Dot(iso(zmap(c5), **zkw), radius=0.045).set_fill(PRED, 1)
        link = VGroup(*[DashedLine(iso(c, **kw), iso(zmap(c), **zkw), dash_length=0.08, stroke_width=1.2).set_color(MUTED)
                        for c in (np.array([hi3[0], lo3[1], hi3[2]]), np.array([hi3[0], lo3[1], lo3[2]]))])
        tb = tt(r"cube of chart " + NAMES[CH] + r": 8 children per level", 26, bold=True)
        tb.next_to(cube, UP, buff=0.55).set_x((O[0] + Z[0]) / 2)
        zl = tt(r"level 3 cell, magnified", 21, MUTED).next_to(zin, DOWN, buff=0.25)
        key = VGroup(*[VGroup(Square(0.22).set_fill(LV[l], 0.6).set_stroke(LV[l], 2), tt(rf"$\ell={l}$", 21)).arrange(RIGHT, buff=0.1)
                       for l in range(1, 6)]).arrange(RIGHT, buff=0.3)
        mk = VGroup(VGroup(Dot(radius=0.06).set_fill(GT, 1), tt("rotation $q$", 21)).arrange(RIGHT, buff=0.1),
                    VGroup(Dot(radius=0.05).set_fill(PRED, 1), tt("leaf centre $\\hat q$", 21)).arrange(RIGHT, buff=0.1)).arrange(RIGHT, buff=0.35)
        leg = VGroup(key, mk).arrange(RIGHT, buff=0.5).next_to(VGroup(cube, zin), DOWN, buff=0.55).set_x((O[0] + Z[0]) / 2)
        B = VGroup(tb, mids, cube, nest, dot, axl, link, zin, zdot, zc5, zl, leg)
        lb = letter("b").next_to(tb, LEFT, buff=0.15)

        # ---------------------------------------------------------- (c) the path as digits, cells and error per level
        rows = [(r"$\ell$", "digit", "cells", r"$\delta(q,\hat q_\ell)$", "mean")]
        rows.append(("0", NAMES[CH], "4", f"{ERR[0]:.1f}" + r"$^\circ$", "--"))
        for l in range(1, 6):
            rows.append((str(l), str(PATH[l - 1]), f"{4 * 8 ** l:,}", f"{ERR[l]:.1f}" + r"$^\circ$", f"{MEAN_ERR[l - 1]:.2f}" + r"$^\circ$"))
        cells = [[tt(c, 23, bold=(i == 0)) for c in row] for i, row in enumerate(rows)]
        for i, row in enumerate(cells[1:], 1):
            row[1].set_color(LV[i - 1] if i > 1 else CH4[CH])
            row[1].scale(1.15)
        W = [0.55, 0.9, 1.35, 1.35, 1.05]
        tab = VGroup()
        for i, row in enumerate(cells):
            x = 0
            for j, c in enumerate(row):
                c.move_to(RIGHT * (x + W[j] / 2) + DOWN * 0.5 * i); x += W[j]
                tab.add(c)
        rule = VGroup(Line(LEFT * 0.05, RIGHT * (sum(W) + 0.05), stroke_width=1.6).set_color(INK).move_to(tab, UP).shift(DOWN * 0.3))
        rule.set_x(tab.get_x())
        path_t = tx(r"\mathrm{path}(q)=(" + NAMES[CH] + r";\," + ",".join(str(k) for k in PATH) + ")", 30)
        tc = tt("one rotation = one root-to-leaf path", 26, bold=True)
        foot = tt(r"$4\cdot 8^5=131{,}072$ leaves; mean error halves per level", 21, MUTED)
        C = VGroup(tc, path_t, VGroup(tab, rule), foot).arrange(DOWN, buff=0.32, aligned_edge=LEFT)
        C.move_to(RIGHT * 17.0 + UP * 0.15)
        lc = letter("c").next_to(tc, LEFT, buff=0.15)

        A.move_to(LEFT * 0.4 + UP * 0.1)
        la.next_to(ta, LEFT, buff=0.15)
        y = max(ta.get_y(), tb.get_y(), tc.get_y())
        for t_, l_ in ((ta, la), (tb, lb), (tc, lc)):
            t_.set_y(y); l_.next_to(t_, LEFT, buff=0.15)
        allm = [A, la, B, lb, C, lc]
        self.add(*allm)
        fit(self, allm, pad=0.3)
