"""S12_Flat: tree vs flat classifier. Numbers from paper/tables/flat.tex (flat over the SPACE-HOP Hopf grid
10,923 x 12 = 131,076 anchors; output layer 256 x 131,076 = 33.6 M; tree 2.4 M; synthetic PoE + orbit 3.87 vs 7.53 deg;
floors 2.51 / 6.06 deg). The Hopf grid is src/tfpose/grids.py::spacehop_hopf_grid(10923, 12) and the tree leaves are
src/tfpose/tesseract.decode of all 131,072 cells; the roll dials show the exact anchors within 4 deg of one direction."""
import os, sys, types; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401,E702
from _c_helpers import *  # noqa: F401,F403
from PIL import Image

if "torch" not in sys.modules:          # grids.py imports torch only for other functions; the Hopf grid is pure NumPy
    _t = types.ModuleType("torch"); _t.no_grad = lambda *a, **k: (lambda f: f); sys.modules["torch"] = _t
    from tfpose.grids import spacehop_hopf_grid  # noqa: E402
    del sys.modules["torch"]            # scipy probes sys.modules["torch"]; do not leave the stub behind
else:
    from tfpose.grids import spacehop_hopf_grid  # noqa: E402

N_HOPF = 10923 * 12
QH = spacehop_hopf_grid(10923, 12); assert len(QH) == N_HOPF == 131076
RH = ROT.quat_to_matrix(QH)
_ch = np.repeat(np.arange(4), 8 ** 5); _idx = np.tile(np.arange(8 ** 5), 4)
_paths = np.stack([(_idx >> (3 * (4 - l))) & 7 for l in range(5)], 1).astype(np.int64)
RT = ROT.quat_to_matrix(T.decode(_ch, _paths))
DIR = np.array([0.3, 0.5, 0.8]); DIR /= np.linalg.norm(DIR)
_r1 = np.cross([0, 0, 1.0], DIR); _r1 /= np.linalg.norm(_r1); _r2 = np.cross(DIR, _r1)


def rolls(R, cone=4.0):
    ang = np.degrees(np.arccos(np.clip(R[:, :, 2] @ DIR, -1, 1))); x = R[ang < cone][:, :, 0]
    return np.arctan2(x @ _r2, x @ _r1)


ROLL_H, ROLL_T = rolls(RH), rolls(RT)
HOPF_DIRS = RH[::12, :, 2]                                   # the 10,923 Fibonacci directions
Q_EX = quat([0.2, -0.7, 0.4], 2.1); CH_EX, PATH_EX = encode(Q_EX)

SHEET = ASSETS / "s12_sheet.png"
if not SHEET.exists():
    rng = np.random.default_rng(0)
    a = (rng.normal(0, 1, (64, 1600)) * 38 + 70).clip(0, 255).astype(np.uint8)
    Image.fromarray(np.stack([a * 0.75, a * 0.8, a * 0.9], -1).astype(np.uint8)).save(SHEET)


class S12_Flat(Scene):
    default_camera_config = dict(background_color=BG)

    def construct(self):
        setup_scene(self)
        rng = np.random.default_rng(1)
        # ------------------------------------------------------------------ b1: the flat output layer vs the whole tree
        b = Beat(self, "S12_Flat_b1")
        feat = VGroup(*[Square(0.22).set_fill(C_DINO, 0.25 + 0.6 * rng.random()).set_stroke(C_DINO, 1)
                        for _ in range(14)]).arrange(DOWN, buff=0.03).move_to([-6.45, -0.2, 0])
        fl = Tex("256", font_size=30, color=C_DINO).next_to(feat, UP, 0.1)
        b.play(LaggedStart(*[FadeIn(f) for f in feat], lag_ratio=0.05), FadeIn(fl), rt=0.9)
        sheet = ImageMobject(str(SHEET)).stretch_to_fit_width(12.2).stretch_to_fit_height(2.6).move_to([0.15, -0.2, 0])
        frame_s = Rectangle(12.2, 2.6).move_to(sheet).set_stroke(GREY_B, 1.5)
        sl = Tex(r"W\in\mathbb{R}^{256\times 131{,}076}", font_size=36).next_to(frame_s, DOWN, 0.25)
        b.play(FadeIn(sheet, shift=0.3 * RIGHT), ShowCreation(frame_s), rt=1.3)
        b.play(Write(sl), rt=0.9)
        # the logit strip: one score per anchor (subsampled 1 in ~218)
        nb = 600; xs = np.linspace(-5.95, 6.25, nb); hts = 0.12 + 0.5 * rng.random(nb) ** 3
        strip = VGroup(*[Line([x, 1.45, 0], [x, 1.45 + h, 0]).set_stroke(C_BASE, 1.6) for x, h in zip(xs, hts)])
        stl = lbl("131,076 logits, one per anchor  (drawn 1 in 218)", 22, C_BASE).move_to([0.15, 2.65, 0])
        b.play(LaggedStart(*[ShowCreation(s) for s in strip], lag_ratio=0.002), FadeIn(stl), rt=1.6)
        pl = Tex(r"256\times131{,}076 = 33.6\,\mathrm{M}\ \text{parameters}", font_size=40, color=C_BASE).move_to([0.15, 3.45, 0])
        b.play(Write(pl), rt=1.0)
        # collapse to a parameter bar (1 M = 0.36 units)
        u = 0.36; x0 = -6.2
        fbar = hbar(x0, -1.0, 33.6 * u, 0.55, C_BASE, 0.85)
        fbl = lbl("flat output layer: 33.6 M", 24, C_BASE).next_to(fbar, UP, 0.12).align_to(fbar, LEFT)
        b.play(FadeOut(strip), FadeOut(stl), FadeOut(feat), FadeOut(fl), FadeOut(sl),
               Transform(frame_s, fbar), sheet.animate.replace(fbar, stretch=True).set_opacity(0), pl.animate.set_opacity(0),
               rt=1.3)
        self.remove(sheet, pl); self.add(fbl)
        # the whole tree
        tbar = hbar(x0, -2.4, 2.4 * u, 0.55, C_PRED, 0.95)
        tbl = lbl("whole tree: 2.4 M", 24, C_PRED).next_to(tbar, RIGHT, 0.25)
        b.play(grow_right(tbar), FadeIn(tbl), rt=1.0)
        copies = VGroup(*[tbar.copy().set_stroke(BG, 1.5) for _ in range(14)])
        targets = [hbar(x0 + k * 2.4 * u, -1.0, 2.4 * u, 0.55, C_PRED, 0.55).set_stroke(BG, 1.5) for k in range(14)]
        cnt = lbl("more than ten times the whole tree", 28, C_PRED).move_to([2.2, 0.35, 0])
        b.play(LaggedStart(*[Transform(c, t) for c, t in zip(copies, targets)], lag_ratio=0.12), rt=2.6)
        b.play(FadeIn(cnt, scale=1.4), rt=0.6)
        b.play(Indicate(tbar, color=WHITE), Indicate(tbl, color=WHITE), rt=0.6)
        b.wait(max(b.left - 0.3, 0))
        b.done()

        # ------------------------------------------------------------------ b2: 44 visited nodes vs all anchors; error
        b = Beat(self, "S12_Flat_b2")
        b.play(*[FadeOut(m) for m in self.mobjects], rt=0.5)
        # flat strip, all lit at once
        nb = 700; xs = np.linspace(-6.6, 6.6, nb); hts = 0.08 + 0.35 * rng.random(nb) ** 3
        strip = VGroup(*[Line([x, 3.0, 0], [x, 3.0 + h, 0]).set_stroke(C_BASE, 1.4) for x, h in zip(xs, hts)])
        sc = lbl("flat: all 131,076 anchors scored", 22, C_BASE).move_to([0, 2.65, 0])
        b.play(ShowCreation(strip, lag_ratio=0), FadeIn(sc), rt=0.8)
        # tree rows: chart (4) then 5 levels of 8, children of the decoded path of a real rotation (encode)
        cx = -3.4; ys = [1.85 - 0.82 * l for l in range(6)]
        rows = []
        for l in range(6):
            n = 4 if l == 0 else 8; sp = 0.62 if l == 0 else 0.5
            rows.append(VGroup(*[Dot([cx + (k - (n - 1) / 2) * sp, ys[l], 0], radius=0.09).set_fill(GREY_D, 1)
                                 .set_stroke(GREY_B, 1) for k in range(n)]))
        choice = [CH_EX] + PATH_EX
        counter = ValueTracker(0)
        ctxt = always_redraw(lambda: VGroup(lbl("tree nodes scored:", 22, C_PRED),
                                            Integer(int(counter.get_value()), font_size=34, color=C_PRED))
                             .arrange(RIGHT, buff=0.15).move_to([cx, -3.35, 0]))
        self.add(ctxt)
        lv = VGroup(*[Tex(r"\ell=" + str(l), font_size=22, color=LEVEL_COLORS[l]).move_to([cx - 2.75, ys[l], 0]) for l in range(6)])
        edges = VGroup()
        for l in range(6):
            col = LEVEL_COLORS[l]
            anims = [FadeIn(rows[l]), rows[l].animate.set_fill(col, 0.55), FadeIn(lv[l]), counter.animate.increment_value(len(rows[l]))]
            if l > 0:
                src = rows[l - 1][choice[l - 1]].get_center()
                e = VGroup(*[Line(src, d.get_center(), buff=0.09).set_stroke(GREY_B, 1, 0.6) for d in rows[l]])
                edges.add(e); anims.append(ShowCreation(e))
            b.play(*anims, rt=0.62)
            b.play(rows[l][choice[l]].animate.set_fill(col, 1).scale(1.5).set_stroke(WHITE, 2), rt=0.28)
        tot = Tex(r"4+8\cdot5=44", font_size=40, color=C_PRED).move_to([cx + 3.55, -0.6, 0])
        b.play(Write(tot), rt=0.8)
        # one level finer: flat x8, tree +8
        ghost = VGroup(*[Dot([cx + (k - 3.5) * 0.5, ys[5] - 0.62, 0], radius=0.07).set_stroke(GREY_B, 1).set_fill(opacity=0)
                         for k in range(8)])
        gl = lbl("finer: +8 nodes", 20, GREY_B).next_to(ghost, RIGHT, 0.25)
        fl8 = lbl("finer: 8x anchors, 8x output layer", 20, C_BASE).move_to([3.4, 2.25, 0])
        b.play(FadeIn(ghost), FadeIn(gl), strip.animate.stretch(8, 0, about_point=[-6.6, 3.0, 0]), FadeIn(fl8), rt=1.2)
        b.play(FadeOut(ghost), FadeOut(gl), FadeOut(tot), strip.animate.stretch(1 / 8, 0, about_point=[-6.6, 3.0, 0]), FadeOut(fl8),
               rt=0.6)
        # error bars (synthetic validation, PoE + orbit, trained the same way)
        y0, s = -2.7, 0.47; bx = [2.2, 4.6]
        axl = Line([1.2, y0, 0], [5.8, y0, 0]).set_stroke(GREY_B, 2)
        cap = VGroup(lbl("synthetic, PoE + orbit,", 20, GREY_B), lbl("same training recipe", 20, GREY_B)).arrange(DOWN, buff=0.06)
        cap.move_to([3.4, 2.05, 0])
        b.play(ShowCreation(axl), FadeIn(cap), rt=0.5)
        bt = vbar(bx[0], y0, 3.87 * s, 1.2, C_PRED); bf = vbar(bx[1], y0, 7.53 * s, 1.2, C_BASE)
        nt = lbl("tree", 22, C_PRED).next_to([bx[0], y0, 0], DOWN, 0.15)
        nf = lbl("flat (Hopf grid)", 22, C_BASE).next_to([bx[1], y0, 0], DOWN, 0.15)
        vt = Tex(r"3.87^\circ", font_size=38, color=C_PRED).next_to(bt, UP, 0.1)
        vf = Tex(r"7.53^\circ", font_size=38, color=C_BASE).next_to(bf, UP, 0.1)
        b.play(grow_up(bt), grow_up(bf), FadeIn(nt), FadeIn(nf), rt=1.6)
        b.play(FadeIn(vt, shift=0.2 * UP), FadeIn(vf, shift=0.2 * UP), rt=0.6)
        # decode once more: a token runs root -> leaf along the 6 chosen nodes, the flat head sweeps every anchor
        runner = Dot(rows[0][choice[0]].get_center(), radius=0.12).set_fill(WHITE, 1)
        path_pts = [rows[l][choice[l]].get_center() for l in range(6)]
        trail = VMobject().set_points_as_corners(path_pts).set_stroke(WHITE, 4)
        sweep = Rectangle(0.5, 0.55).move_to([-6.6, 3.2, 0]).set_fill(C_BASE, 0.5).set_stroke(width=0)
        self.add(runner, sweep)
        b.play(MoveAlongPath(runner, trail), ShowCreation(trail), sweep.animate.move_to([6.6, 3.2, 0]), rt=1.8)
        b.play(FadeOut(sweep), Flash(runner, color=WHITE), rt=0.6)
        b.wait(max(b.left - 0.3, 0))
        b.done()

        # ------------------------------------------------------------------ b3: grid floors, 12 rolls; pending control
        b = Beat(self, "S12_Flat_b3")
        self.remove(ctxt)
        b.play(*[FadeOut(m) for m in self.mobjects], rt=0.5)
        sc_c = np.array([-3.7, -0.15, 0]); Rr = 2.55
        th = ValueTracker(0.0)

        def sphere_pts():
            Rz = rotm([0, 0, 1], th.get_value()); V = VIEW @ Rz
            p = HOPF_DIRS @ V.T; p = p[p[:, 2] > 0]
            return sc_c + Rr * np.c_[p[:, 0], p[:, 1], 0 * p[:, 0]]

        outline = Circle(radius=Rr).move_to(sc_c).set_stroke(GREY_B, 1.5)
        cloud = DotCloud(sphere_pts(), radius=0.022, color="#C8D1DA").set_opacity(0.85)
        cloud.add_updater(lambda m: m.set_points(sphere_pts()))
        dmark = always_redraw(lambda: Dot(sc_c + Rr * ((VIEW @ rotm([0, 0, 1], th.get_value()) @ DIR) * [1, 1, 0]),
                                          radius=0.11).set_fill(C_GT, 1).set_stroke(BLACK, 1.5))
        hl = VGroup(lbl("SPACE-HOP Hopf grid", 24, C_BASE), lbl("10,923 directions x 12 rolls", 22, C_BASE)).arrange(DOWN, buff=0.08)
        hl.move_to([sc_c[0], 3.35, 0])
        b.play(ShowCreation(outline), FadeIn(cloud), FadeIn(hl), th.animate.set_value(0.35), rt=1.2)
        self.add(dmark)
        dl = lbl("one direction d", 20, C_GT).move_to(sc_c + [0, -Rr - 0.35, 0])
        b.play(FadeIn(dmark, scale=2), FadeIn(dl), th.animate.set_value(0.6), rt=0.8)

        def dial(cen, angles, col, title, n_lbl):
            c = Circle(radius=1.15).move_to(cen).set_stroke(GREY_D, 2)
            sp = VGroup(*[Line(cen, cen + 1.15 * np.array([np.cos(a), np.sin(a), 0])).set_stroke(col, 1.4, 0.75)
                          for a in angles])
            t = VGroup(lbl(title, 21, col), lbl(n_lbl, 18, GREY_B)).arrange(DOWN, buff=0.06).next_to(c, UP, 0.15)
            return c, sp, t

        ca = np.array([1.35, 0.35, 0]); cb = np.array([5.05, 0.35, 0])
        h_c, h_s, h_t = dial(ca, ROLL_H, C_BASE, "Hopf grid", f"{len(ROLL_H)} anchors within 4° of d")
        t_c, t_s, t_t = dial(cb, ROLL_T, C_PRED, "tesseract leaves", f"{len(ROLL_T)} leaves within 4° of d")
        rl = lbl("in-plane roll of each anchor", 22, GREY_A).move_to([3.2, 3.4, 0])
        b.play(FadeIn(rl), ShowCreation(h_c), FadeIn(h_t), th.animate.set_value(0.9), rt=0.8)
        b.play(LaggedStart(*[ShowCreation(s) for s in h_s], lag_ratio=0.01), th.animate.set_value(1.3), rt=1.4)
        twelve = lbl("only 12 rolls, 30° apart", 22, C_FLIP).next_to(h_c, DOWN, 0.2)
        b.play(FadeIn(twelve), Indicate(h_s, color=C_FLIP, scale_factor=1.05), th.animate.set_value(1.6), rt=1.0)
        b.play(ShowCreation(t_c), FadeIn(t_t), LaggedStart(*[ShowCreation(s) for s in t_s], lag_ratio=0.01),
               th.animate.set_value(2.1), rt=1.5)
        spread = lbl("rolls spread evenly", 22, C_PRED).next_to(t_c, DOWN, 0.2)
        b.play(FadeIn(spread), th.animate.set_value(2.3), rt=0.6)
        fh = VGroup(lbl("floor", 22, C_BASE), Tex(r"6.06^\circ", font_size=40, color=C_BASE)).arrange(RIGHT, buff=0.15)
        ft = VGroup(lbl("floor", 22, C_PRED), Tex(r"2.51^\circ", font_size=40, color=C_PRED)).arrange(RIGHT, buff=0.15)
        fh.next_to(twelve, DOWN, 0.25); ft.next_to(spread, DOWN, 0.25)
        b.play(FadeIn(fh, shift=0.2 * UP), FadeIn(ft, shift=0.2 * UP), th.animate.set_value(2.6), rt=1.0)
        # pending control
        pend_c = np.array([1.75, -3.45, 0])
        spin = Arc(0, 1.5 * PI, radius=0.2, arc_center=pend_c).set_stroke(C_GT, 4)
        pend = lbl("same-grid control: still running", 24, C_GT).next_to(pend_c, RIGHT, 0.3)
        b.play(FadeIn(spin), FadeIn(pend), th.animate.set_value(2.8), rt=0.7)
        b.play(Rotate(spin, -2 * PI, about_point=pend_c), th.animate.set_value(3.3), rt=max(b.left - 0.3, 0.3))
        cloud.clear_updaters()
        b.done()
