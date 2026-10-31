"""S13_General: results beyond spacecraft, all animated. Numbers: paper/tables/tless_sota.tex (A0/A2 orbit, LiePose),
symsol_sota.tex (per shape), ycbv.tex + STORYBOARD (real-only test 69.4 -> with renders 12.9, 85% within 10 deg; val 8.97
-> 11.53 with orbit), trans_xobj.tex / sec/4_experiments.tex (T@2/5/10, YCB-V 1.6 cm vs box-only 8.1 cm)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401,E702
from _c_helpers import *  # noqa: F401,F403


def legend(items, fs=22):
    return VGroup(*[VGroup(Square(0.26).set_fill(c, 0.9).set_stroke(width=0), lbl(t, fs, c)).arrange(RIGHT, buff=0.12)
                    for t, c in items]).arrange(RIGHT, buff=0.5)


def shape_icon(kind, c=GREY_A):
    s = 0.42
    if kind == "tet":
        a, b_, d, e = [np.array(p) * s for p in [(-1, -0.8, 0), (1, -0.8, 0), (0.15, 1, 0), (0.3, -0.25, 0)]]
        g = VGroup(Polygon(a, b_, d), Line(a, e), Line(b_, e), Line(d, e))
    elif kind == "cube":
        sq = Square(1.3 * s).shift([-0.15 * s, -0.15 * s, 0]); sq2 = sq.copy().shift([0.4 * s, 0.4 * s, 0])
        g = VGroup(sq, sq2, *[Line(sq.get_vertices()[i], sq2.get_vertices()[i]) for i in range(4)])
    elif kind == "icosa":
        hexa = RegularPolygon(6).scale(s * 1.1); v = hexa.get_vertices()
        tri = Polygon(v[0], v[2], v[4]); tri2 = Polygon(v[1], v[3], v[5]).set_stroke(opacity=0.4)
        g = VGroup(hexa, tri, tri2)
    elif kind == "cone":
        e = Ellipse(width=1.8 * s, height=0.5 * s).shift(0.8 * s * DOWN)
        g = VGroup(e, Line(e.get_left(), [0, 1.0 * s, 0]), Line(e.get_right(), [0, 1.0 * s, 0]))
    else:
        e1 = Ellipse(width=1.5 * s, height=0.45 * s).shift(0.8 * s * DOWN); e2 = e1.copy().shift(1.6 * s * UP)
        g = VGroup(e1, e2, Line(e1.get_left(), e2.get_left()), Line(e1.get_right(), e2.get_right()))
    return g.set_stroke(c, 2.5).set_fill(opacity=0)


class S13_General(Scene):
    default_camera_config = dict(background_color=BG)

    def construct(self):
        setup_scene(self)
        # ------------------------------------------------------------------ b1: T-LESS recall at 2/5/10 deg
        b = Beat(self, "S13_General_b1")
        part = VGroup(Polygon([0, 0, 0], [1.6, 0, 0], [1.6, 0.4, 0], [0.45, 0.4, 0], [0.45, 1.3, 0], [0, 1.3, 0])
                      .set_fill(GREY_C, 0.9).set_stroke(GREY_A, 2),
                      Circle(radius=0.12).move_to([1.1, 0.2, 0]).set_fill(BG, 1).set_stroke(GREY_A, 1.5)).move_to([-5.6, 2.6, 0])
        ttl = VGroup(lbl("T-LESS", 34), lbl("textureless industrial parts, real test images", 20, GREY_B)).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        ttl.next_to(part, RIGHT, 0.35)
        b.play(DrawBorderThenFill(part), FadeIn(ttl), rt=1.0)
        y0, s = -2.75, 0.048; gx = [-2.6, 1.2, 5.0]; th = ["2°", "5°", "10°"]
        ax = Line([-4.6, y0, 0], [6.8, y0, 0]).set_stroke(GREY_B, 2)
        yl = lbl("% of rotations within", 20, GREY_B).move_to([-5.6, y0 + 0.4, 0])
        tl = VGroup(*[lbl(t, 26, GREY_A).move_to([x, y0 - 0.35, 0]) for x, t in zip(gx, th)])
        grid = VGroup(*[DashedLine([-4.6, y0 + v * s, 0], [6.8, y0 + v * s, 0], dash_length=0.06).set_stroke(GREY_D, 1, 0.6)
                        for v in (25, 50, 75)])
        gt = VGroup(*[lbl(f"{v}%", 16, GREY_C).next_to([-4.6, y0 + v * s, 0], LEFT, 0.08) for v in (25, 50, 75)])
        leg = legend([("single label + orbit (A0)", C_BASE), ("multi-path + orbit (A2)", C_PRED), ("LiePose (fine-tuned)", C_FT)], 20)
        leg.move_to([1.2, 3.55, 0]).shift(1.2 * RIGHT)
        b.play(ShowCreation(ax), FadeIn(tl), FadeIn(grid), FadeIn(gt), FadeIn(yl), rt=0.8)
        data = [("4.0", "36.0", "73.5", C_BASE), ("5.7", "48.1", "79.0", C_PRED), ("34.9", "78.2", "88.9", C_FT)]
        bars, vals = [], []
        for j, (*vs, c) in enumerate(data):
            bb, vv = VGroup(), VGroup()
            for i, v in enumerate(vs):
                r = vbar(gx[i] + (j - 1) * 0.95, y0, float(v) * s, 0.82, c)
                bb.add(r); vv.add(lbl(v, 21, c).next_to(r, UP, 0.08))
            bars.append(bb); vals.append(vv)
        b.play(FadeIn(leg[0]), *[grow_up(r) for r in bars[0]], rt=1.6)
        b.play(FadeIn(vals[0]), rt=0.6)
        b.play(FadeIn(leg[1]), *[TransformFromCopy(a, r) for a, r in zip(bars[0], bars[1])], rt=1.8)
        b.play(FadeIn(vals[1]), rt=0.6)
        # highlight 73.5 -> 79.0 at 10 deg
        hl = SurroundingRectangle(VGroup(bars[0][2], bars[1][2], vals[0][2], vals[1][2]), buff=0.12).set_stroke(C_GT, 3)
        up = Arrow(vals[0][2].get_top() + 0.05 * UP, vals[1][2].get_left() + 0.05 * LEFT, buff=0.05, stroke_width=4).set_color(C_GT)
        b.play(ShowCreation(hl), GrowArrow(up), rt=1.0)
        ups = VGroup(*[Arrow(vals[0][i].get_top() + 0.05 * UP, vals[1][i].get_left() + 0.05 * LEFT, buff=0.05, stroke_width=3)
                       .set_color(C_PRED) for i in (0, 1)])
        b.play(LaggedStart(*[GrowArrow(a) for a in ups], lag_ratio=0.3), rt=0.9)
        b.play(FadeIn(leg[2]), *[grow_up(r) for r in bars[2]], bars[2].animate.set_opacity(0.45), rt=1.6)
        b.play(FadeIn(vals[2]), rt=0.6)
        b.play(Flash(vals[1][2], color=C_PRED), Indicate(vals[1][2], color=C_GT), rt=0.7)
        b.wait(max(b.left - 0.3, 0))
        b.done()

        # ------------------------------------------------------------------ b2: SYMSOL-T per shape
        b = Beat(self, "S13_General_b2")
        b.play(*[FadeOut(m) for m in self.mobjects], rt=0.5)
        ttl = VGroup(lbl("SYMSOL-T", 34), lbl("symmetric solids, mean rotation error", 20, GREY_B)).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        ttl.to_corner(UL, buff=0.45)
        shapes = ["tet", "cube", "icosa", "cone", "cyl"]
        A0 = ["3.06", "2.90", "3.71", "1.79", "1.65", "2.62"]
        A2 = ["2.46", "2.30", "2.52", "1.57", "1.48", "2.07"]
        LP = ["0.59", "0.58", "0.64", "0.54", "0.41", "0.55"]
        y0, s = -2.3, 1.05; xs = [-5.4, -3.2, -1.0, 1.2, 3.4, 5.75]
        ax = Line([-6.6, y0, 0], [6.9, y0, 0]).set_stroke(GREY_B, 2)
        icons = VGroup(*[shape_icon(k).move_to([xs[i], y0 - 0.65, 0]) for i, k in enumerate(shapes)])
        names = VGroup(*[lbl(n, 18, GREY_B).move_to([xs[i], y0 - 1.3, 0]) for i, n in
                         enumerate(["tetrahedron", "cube", "icosahedron", "cone", "cylinder", "average"])])
        avg_icon = Tex(r"\bar{E}", font_size=44, color=GREY_A).move_to([xs[5], y0 - 0.65, 0])
        sep = DashedLine([4.55, y0, 0], [4.55, 2.3, 0]).set_stroke(GREY_D, 1.5)
        b.play(FadeIn(ttl), ShowCreation(ax), LaggedStart(*[DrawBorderThenFill(i) for i in icons], lag_ratio=0.15),
               FadeIn(names), FadeIn(avg_icon), ShowCreation(sep), rt=1.4)
        leg = legend([("A0 + orbit", C_BASE), ("A2 + orbit (DINOv3)", C_PRED), ("LiePose", C_FT)], 20).move_to([2.4, 3.55, 0])
        bA0 = VGroup(*[vbar(x - 0.35, y0, float(v) * s, 0.62, C_BASE) for x, v in zip(xs, A0)])
        vA0 = VGroup(*[Tex(v, font_size=26, color=C_BASE).next_to(r, UP, 0.07) for v, r in zip(A0, bA0)])
        b.play(FadeIn(leg[0]), LaggedStart(*[grow_up(r) for r in bA0], lag_ratio=0.1), rt=1.5)
        b.play(FadeIn(vA0), rt=0.6)
        # A2: start as copies of the A0 bars, then shrink to the A2 values ("every solid improves")
        bA2 = VGroup(*[vbar(x + 0.35, y0, float(v) * s, 0.62, C_PRED) for x, v in zip(xs, A0)])
        bA2t = VGroup(*[vbar(x + 0.35, y0, float(v) * s, 0.62, C_PRED) for x, v in zip(xs, A2)])
        vA2 = VGroup(*[Tex(v, font_size=26, color=C_PRED).next_to(r, UP, 0.07) for v, r in zip(A2, bA2t)])
        b.play(FadeIn(leg[1]), *[TransformFromCopy(a, c) for a, c in zip(bA0, bA2)], rt=1.0)
        b.play(*[Transform(a, c) for a, c in zip(bA2, bA2t)], rt=1.8)
        b.play(LaggedStart(*[FadeIn(v, shift=0.15 * DOWN) for v in vA2], lag_ratio=0.12), rt=1.2)
        avg = VGroup(Tex(r"2.62^\circ", font_size=40, color=C_BASE), Tex(r"\rightarrow", font_size=40),
                     Tex(r"2.07^\circ", font_size=40, color=C_PRED)).arrange(RIGHT, buff=0.2).move_to([5.2, 2.75, 0])
        b.play(FadeIn(avg, shift=0.2 * LEFT), Indicate(VGroup(bA0[5], bA2[5]), color=WHITE, scale_factor=1.05), rt=1.2)
        # LiePose, a thin bar in front of each pair
        bLP = VGroup(*[vbar(x, y0, float(v) * s, 0.42, C_FT) for x, v in zip(xs, LP)])
        vLP = VGroup(*[Tex(v, font_size=22, color=C_FT).next_to(r, UP, 0.05) for v, r in zip(LP, bLP)])
        b.play(FadeIn(leg[2]), *[grow_up(r) for r in bLP], rt=1.4)
        b.play(FadeIn(vLP), rt=0.6)
        four = VGroup(lbl("diffusion model, fine-tuned backbone:", 20, C_FT), lbl("about 4x more accurate", 24, C_FT)).arrange(DOWN, buff=0.08)
        four.move_to([-0.6, 2.8, 0])
        b.play(FadeIn(four), rt=0.9)
        b.wait(max(b.left - 0.3, 0))
        b.done()

        # ------------------------------------------------------------------ b3: YCB-V
        b = Beat(self, "S13_General_b3")
        b.play(*[FadeOut(m) for m in self.mobjects], rt=0.5)
        ttl = VGroup(lbl("YCB-V", 34), lbl("household objects, new test sequences", 20, GREY_B)).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        ttl.to_corner(UL, buff=0.45)
        y0, s = -2.6, 0.068
        ax = Line([-6.5, y0, 0], [-0.6, y0, 0]).set_stroke(GREY_B, 2)
        film = VGroup(*[Rectangle(0.42, 0.3).set_stroke(GREY_A, 1.5) for _ in range(4)]).arrange(RIGHT, buff=0.04)
        film.add(Line(film.get_corner(UL) + 0.06 * UP, film.get_corner(UR) + 0.06 * UP).set_stroke(GREY_A, 1.5),
                 Line(film.get_corner(DL) + 0.06 * DOWN, film.get_corner(DR) + 0.06 * DOWN).set_stroke(GREY_A, 1.5))
        film.move_to([-4.9, y0 - 0.45, 0])
        b.play(FadeIn(ttl), ShowCreation(ax), FadeIn(film), rt=0.8)
        br = vbar(-4.9, y0, 69.4 * s, 1.3, C_FLIP)
        vr = Tex(r"69.4^\circ", font_size=40, color=C_FLIP).next_to(br, UP, 0.1)
        nr = lbl("real videos only", 20, C_FLIP).next_to(film, DOWN, 0.15)
        b.play(grow_up(br), FadeIn(nr), rt=1.4)
        b.play(FadeIn(vr), rt=0.5)
        c9 = np.array([-4.9, -0.2, 0])
        d9 = Circle(radius=0.55).move_to(c9).set_fill(BG, 1).set_stroke(GREY_D, 9)
        a9 = Arc(PI / 2, -2 * PI * 0.09, radius=0.55, arc_center=c9).set_stroke(WHITE, 9)
        t9 = Tex(r"9\%", font_size=30).move_to(c9)
        l9 = lbl("within 10°", 16, WHITE).next_to(d9, DOWN, 0.08)
        b.play(FadeIn(d9), FadeIn(l9), rt=0.4)
        b.play(ShowCreation(a9), FadeIn(t9), rt=0.8)
        wire = VGroup(Square(0.36), Square(0.36).shift([0.12, 0.12, 0]))
        wire.add(*[Line(wire[0].get_vertices()[i], wire[1].get_vertices()[i]) for i in range(4)])
        icon2 = VGroup(film.copy(), lbl("+", 26), wire.set_stroke(C_PRED, 2)).arrange(RIGHT, buff=0.12).move_to([-2.4, y0 - 0.45, 0])
        bs = vbar(-2.4, y0, 12.9 * s, 1.3, C_PRED)
        vs = Tex(r"12.9^\circ", font_size=40, color=C_PRED).next_to(bs, UP, 0.1)
        ns = lbl("+ synthetic renders", 20, C_PRED).next_to(icon2, DOWN, 0.15)
        b.play(FadeIn(icon2), FadeIn(ns), TransformFromCopy(br, bs), rt=1.6)
        b.play(FadeIn(vs), rt=0.5)
        # 85% within 10 deg
        cen = np.array([1.35, 0.4, 0])
        ring_bg = Circle(radius=1.05).move_to(cen).set_stroke(GREY_D, 14)
        ring = Arc(PI / 2, -2 * PI * 0.85, radius=1.05, arc_center=cen).set_stroke(C_PRED, 14)
        rt85 = Tex(r"85\%", font_size=44, color=C_PRED).move_to(cen)
        rl = lbl("within 10°", 22, C_PRED).next_to(ring_bg, DOWN, 0.2)
        b.play(FadeIn(ring_bg), FadeIn(rl), rt=0.4)
        b.play(ShowCreation(ring), FadeIn(rt85), rt=1.4)
        # orbit voting hurts here: validation 8.97 -> 11.53, so validation switches it off
        y1, s1 = -2.6, 0.33; ox = [3.75, 5.6]
        ax2 = Line([2.8, y1, 0], [6.6, y1, 0]).set_stroke(GREY_B, 2)
        vt = lbl("validation error", 22, GREY_B).move_to([4.7, 2.9, 0])
        b1_ = vbar(ox[0], y1, 8.97 * s1, 1.1, C_PRED); b2_ = vbar(ox[1], y1, 11.53 * s1, 1.1, C_BASE)
        v1 = Tex(r"8.97^\circ", font_size=34, color=C_PRED).next_to(b1_, UP, 0.08)
        v2 = Tex(r"11.53^\circ", font_size=34, color=C_BASE).next_to(b2_, UP, 0.08)
        n1 = lbl("1 view", 20, C_PRED).next_to([ox[0], y1, 0], DOWN, 0.15)
        n2 = lbl("orbit (4)", 20, C_BASE).next_to([ox[1], y1, 0], DOWN, 0.15)
        b.play(ShowCreation(ax2), FadeIn(vt), grow_up(b1_), FadeIn(n1), FadeIn(v1), rt=1.0)
        b.play(grow_up(b2_), FadeIn(n2), FadeIn(v2), rt=1.0)
        cross = VGroup(Line(UL, DR), Line(UR, DL)).scale(0.45).move_to(n2).set_stroke(C_FLIP, 5)
        off = lbl("validation turns orbit off", 22, C_FLIP).move_to([4.7, -3.55, 0])
        b.play(ShowCreation(cross), FadeIn(off), b2_.animate.set_opacity(0.35), rt=0.9)
        b.wait(max(b.left - 0.3, 0))
        b.done()

        # ------------------------------------------------------------------ b4: translation from frozen tokens
        b = Beat(self, "S13_General_b4")
        b.play(*[FadeOut(m) for m in self.mobjects], rt=0.5)
        ttl = lbl("translation from the frozen tokens", 30).to_edge(UP, buff=0.35)
        b.play(FadeIn(ttl), rt=0.6)
        y0, s = -2.3, 0.04; gx = [-5.2, -3.4, -1.6]
        ax = Line([-6.4, y0, 0], [-0.6, y0, 0]).set_stroke(GREY_B, 2)
        sub = lbl("T-LESS: % of objects within", 22, GREY_A).move_to([-3.5, 2.75, 0])
        tl = VGroup(*[lbl(t, 24, GREY_A).move_to([x, y0 - 0.33, 0]) for x, t in zip(gx, ["2 cm", "5 cm", "10 cm"])])
        b.play(ShowCreation(ax), FadeIn(sub), FadeIn(tl), rt=0.7)
        ours = ["57.3", "88.0", "96.7"]; lp = ["68.0", "88.4", "95.2"]
        bo = VGroup(*[vbar(x - 0.38, y0, float(v) * s, 0.7, C_PRED) for x, v in zip(gx, ours)])
        bl_ = VGroup(*[vbar(x + 0.38, y0, float(v) * s, 0.7, C_FT) for x, v in zip(gx, lp)])
        vo = VGroup(*[lbl(v, 19, C_PRED).next_to(r, UP, 0.06) for v, r in zip(ours, bo)])
        vl = VGroup(*[lbl(v, 19, C_FT).next_to(r, UP, 0.06) for v, r in zip(lp, bl_)])
        leg = VGroup(lbl("ours (orbit mean)", 20, C_PRED), lbl("LiePose", 20, C_FT)).arrange(RIGHT, buff=0.5).move_to([-3.5, -3.4, 0])
        b.play(LaggedStart(*[grow_up(r) for r in bo], lag_ratio=0.2), FadeIn(leg[0]), rt=1.5)
        b.play(FadeIn(vo), rt=0.6)
        b.play(LaggedStart(*[grow_up(r) for r in bl_], lag_ratio=0.2), FadeIn(leg[1]), rt=1.5)
        b.play(FadeIn(vl), rt=0.6)
        hl = SurroundingRectangle(VGroup(bo[1], bl_[1], vo[1], vl[1]), buff=0.1).set_stroke(C_GT, 3)
        hlt = lbl("close to LiePose at 5 and 10 cm", 22, C_GT).move_to([-3.4, 2.25, 0])
        b.play(ShowCreation(hl), FadeIn(hlt), rt=1.0)
        # YCB-V: mean translation error drawn to scale as circles around the true centre
        cen = np.array([3.5, 0.0, 0]); cm = 0.3
        sub2 = lbl("YCB-V: mean translation error (to scale)", 22, GREY_A).move_to([3.5, 2.75, 0])
        tgt = Dot(cen, radius=0.07).set_fill(C_GT, 1)
        tl2 = lbl("true centre", 18, C_GT).move_to(cen + DOWN * (1.6 * 0.3 + 0.25))
        c8 = Circle(radius=8.1 * cm).move_to(cen).set_stroke(C_BASE, 3).set_fill(C_BASE, 0.08)
        c16 = Circle(radius=1.6 * cm).move_to(cen).set_stroke(C_PRED, 4).set_fill(C_PRED, 0.25)
        l8 = VGroup(lbl("box only", 20, C_BASE), Tex(r"8.1\,\mathrm{cm}", font_size=32, color=C_BASE)).arrange(DOWN, buff=0.05)
        l8.move_to(cen + np.array([1.75, 1.75, 0]) * 1.05)
        l16 = VGroup(lbl("ours", 20, C_PRED), Tex(r"1.6\,\mathrm{cm}", font_size=32, color=C_PRED)).arrange(RIGHT, buff=0.12)
        l16.next_to(c16, RIGHT, 0.1).shift(0.0 * UP)
        b.play(FadeIn(sub2), GrowFromCenter(tgt), FadeIn(tl2), rt=0.6)
        b.play(GrowFromCenter(c8), FadeIn(l8), rt=1.2)
        b.play(GrowFromCenter(c16), FadeIn(l16), rt=1.2)
        rad = Line(cen, cen + 8.1 * cm * np.array([np.cos(-0.6), np.sin(-0.6), 0])).set_stroke(C_BASE, 2)
        rad2 = Line(cen, cen + 1.6 * cm * np.array([np.cos(-0.6), np.sin(-0.6), 0])).set_stroke(C_PRED, 3)
        b.play(ShowCreation(rad), ShowCreation(rad2), rt=0.7)
        # caveat, drawn: the allocentric rotation needs the viewing ray, i.e. the translation t
        b.play(*[FadeOut(m) for m in self.mobjects if m is not ttl], rt=0.6)
        camp = np.array([-5.6, -1.2, 0]); obj = np.array([2.2, 1.0, 0])
        cam = Polygon(camp, camp + [-0.6, 0.4, 0], camp + [-0.6, -0.4, 0]).set_fill(GREY_C, 1).set_stroke(WHITE, 1)
        oax = DashedLine(camp, camp + 9.0 * RIGHT).set_stroke(GREY_B, 1.5)
        oal = lbl("optical axis", 18, GREY_B).next_to(camp + 8.2 * RIGHT, DOWN, 0.1)
        b.play(FadeIn(cam), ShowCreation(oax), FadeIn(oal), rt=0.7)
        part = VGroup(Polygon([0, 0, 0], [1.2, 0, 0], [1.2, 0.3, 0], [0.34, 0.3, 0], [0.34, 1.0, 0], [0, 1.0, 0])
                      .set_fill(GREY_C, 0.9).set_stroke(GREY_A, 2)).move_to(obj)
        ray = Arrow(camp, obj, buff=0.55, stroke_width=5).set_color(C_GT)
        tl_ = Tex(r"t", font_size=40, color=C_GT).next_to(ray.get_center(), UP, 0.15)
        b.play(FadeIn(part), GrowArrow(ray), FadeIn(tl_), rt=0.8)
        eq = Tex(r"\text{tree predicts } R_{\mathrm{ray}}(t)^\top R", font_size=38).move_to([-1.6, 1.7, 0])
        b.play(Write(eq), Rotate(part, 0.4), rt=1.1)
        gtt = lbl("on T-LESS and YCB-V, this t is the ground truth", 24, C_GT).move_to([1.0, -2.2, 0])
        b.play(FadeIn(gtt, shift=0.2 * UP), Indicate(tl_, color=C_GT, scale_factor=1.5), rt=0.9)
        # the conversion depends on where the object is: move it, the ray (and R_ray) follow
        ray.add_updater(lambda m: m.put_start_and_end_on(camp + 0.55 * normalize(part.get_center() - camp),
                                                         part.get_center() - 0.55 * normalize(part.get_center() - camp)))
        tl_.add_updater(lambda m: m.next_to(ray.get_center(), UP, 0.15))
        b.play(part.animate.move_to([3.6, 2.4, 0]), rt=0.8)
        b.play(part.animate.move_to(obj), rt=0.7)
        ray.clear_updaters(); tl_.clear_updaters()
        pend_c = np.array([-2.6, -3.2, 0])
        spin = Arc(0, 1.5 * PI, radius=0.17, arc_center=pend_c).set_stroke(C_GT, 4)
        pend = lbl("with the predicted t: evaluation pending", 22, C_GT).next_to(spin, RIGHT, 0.2)
        b.play(FadeIn(spin), FadeIn(pend), rt=0.6)
        b.play(Rotate(spin, -2 * PI), rt=max(b.left - 0.3, 0.3))
        b.done()
