"""S11_Translation: the SPACE-HOP translation target u, computed and inverted exactly on a real SPEED+ lightbox frame
(img004459, ground-truth pose and the 15%-margin square crop box of src/tfpose/data.py; numbers from
_prep_S11_S14_assets.py -> assets/s11_s14.json; formulas as in src/tfpose/trans.py)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401,E702
from _c_helpers import *  # noqa: F401,F403

D = assets_json(); S = D["s11"]
K = np.array(D["K"]); W, H, ALPHA = D["W"], D["H"], D["alpha"]
BX, BY, BW, BH = S["box"]; PCX, PCY = S["proj_centre"]; U = np.array(S["u"]); TT = np.array(S["t"])
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
ZS = 0.5 * (ALPHA * BW / W + BH / H)                      # 0.5 (alpha b_w/W + b_h/H)
assert np.allclose([(fx * TT[0] / TT[2] + cx - BX) / BW, (fy * TT[1] / TT[2] + cy - BY) / BH, ZS * TT[2]], U)


def f3(v):
    return f"{v:.3f}"


class S11_Translation(Scene):
    default_camera_config = dict(background_color=BG)

    def construct(self):
        setup_scene(self)
        # ------------------------------------------------------------------ b1: tokens -> query -> u
        b = Beat(self, "S11_Translation_b1")
        img = ImageMobject(str(ASSETS / "s11_full.png")).set_width(7.4).move_to([-3.1, 0.35, 0])
        cam_lbl = lbl("real SPEED+ lightbox frame", 20, GREY_B).next_to(img, DOWN, 0.15)
        box = Square(side_length=BW / W * img.get_width()).move_to(px(img, BX, BY)).set_stroke(C_GT, 3)
        b.play(FadeIn(img, scale=1.05), FadeIn(cam_lbl), rt=0.8)
        b.play(ShowCreation(box), rt=0.6)
        crop = ImageMobject(str(ASSETS / "s11_crop.png")).replace(box)
        self.add(crop)
        # crop flies out and becomes the 8x8 encoded token grid
        b.play(crop.animate.set_height(3.2).move_to([4.2, 0.3, 0]), rt=0.9)
        n = 8; s = 3.2 / n
        rng = np.random.default_rng(4)
        tokens = VGroup(*[Square(s * 0.86).move_to(crop.get_corner(UL) + (j + .5) * s * RIGHT + (i + .5) * s * DOWN)
                          for i in range(n) for j in range(n)])
        tokens.set_stroke(C_DINO, 1.2).set_fill(C_DINO, 0.15)
        b.play(LaggedStart(*[FadeIn(t, scale=0.5) for t in tokens], lag_ratio=0.01), crop.animate.set_opacity(0.35), rt=0.9)
        enc_lbl = lbl("encoded tokens  (from the frozen tree)", 20, C_DINO).next_to(crop, DOWN, 0.2)
        b.play(FadeIn(enc_lbl), FadeOut(img), FadeOut(box), FadeOut(cam_lbl), rt=0.5)
        # one learned query attends (attention weight illustrative: brighter on object tokens)
        query = Circle(radius=0.28).set_fill(C_PRED, 0.9).set_stroke(WHITE, 1.5).move_to([-0.6, 0.3, 0])
        ql = lbl("learned query", 20, C_PRED).next_to(query, UP, 0.15)
        cen = crop.get_center() + 0.3 * LEFT + 0.1 * UP
        lines = VGroup()
        for t in tokens:
            w = np.exp(-np.linalg.norm(t.get_center() - cen) ** 2 / 0.9) + 0.05 * rng.random()
            lines.add(Line(query.get_right(), t.get_center()).set_stroke(C_PRED, 0.5 + 3 * w, min(0.08 + w, 1)))
        b.play(GrowFromCenter(query), FadeIn(ql), rt=0.5)
        b.play(LaggedStart(*[ShowCreation(l) for l in lines], lag_ratio=0.01), rt=1.1)
        # small MLP -> 3 numbers u
        mlp = VGroup(*[Dot(radius=0.06, color=GREY_B).move_to([-2.3, y, 0]) for y in np.linspace(-0.8, 1.4, 6)],
                     *[Dot(radius=0.06, color=GREY_B).move_to([-3.4, y, 0]) for y in np.linspace(-0.5, 1.1, 3)])
        mlp_e = VGroup(*[Line(query.get_left(), d.get_center()) for d in mlp[:6]],
                       *[Line(a.get_center(), c.get_center()) for a in mlp[:6] for c in mlp[6:]]).set_stroke(GREY_B, 1, 0.5)
        b.play(ShowCreation(mlp_e), FadeIn(mlp), lines.animate.set_stroke(opacity=0.25), rt=0.8)
        names = [r"U_x", r"U_y", r"U_z"]
        scale = [5.0, 5.0, 0.55]; x0 = -5.3
        zero = Line([x0, 1.45, 0], [x0, -0.95, 0]).set_stroke(GREY_B, 2)
        bars, bl = VGroup(), VGroup()
        for k in range(3):
            y = 1.1 - 0.8 * k; L = max(abs(U[k]) * scale[k], 0.05)
            r = hbar(x0 if U[k] >= 0 else x0 - L, y, L, 0.42, [C_PRED, C_PRED, C_GT][k])
            bars.add(r)
            bl.add(VGroup(Tex(names[k], font_size=32).next_to([-6.2, y, 0], LEFT, 0.1),
                          Tex(f3(U[k]), font_size=28).next_to(r if U[k] >= 0 else [x0, y, 0], RIGHT, 0.2)))
        bars.add(zero)
        u_lbl = Tex(r"u", font_size=56, color=C_PRED).move_to([-5.9, 2.0, 0])
        b.play(ShowCreation(zero), *[GrowFromEdge(r, LEFT if U[k] >= 0 else RIGHT) for k, r in enumerate(bars[:3])], FadeIn(bl), Write(u_lbl), rt=1.0)
        b.done()

        # ------------------------------------------------------------------ b2: geometry of u, then invert
        b = Beat(self, "S11_Translation_b2")
        b.play(*[FadeOut(m) for m in [tokens, crop, enc_lbl, query, ql, lines, mlp, mlp_e]],
               Group(bars, bl, u_lbl).animate.scale(0.7).to_corner(DR, buff=0.4), rt=0.6)
        img = ImageMobject(str(ASSETS / "s11_full.png")).set_width(8.0).move_to([-2.9, 0.6, 0])
        box = Square(side_length=BW / W * img.get_width()).move_to(px(img, BX, BY)).set_stroke(C_GT, 3)
        b.play(FadeIn(img), ShowCreation(box), rt=0.6)
        bc = Dot(px(img, BX, BY), radius=0.07, color=C_GT)
        bcl = Tex(r"(b_x,b_y)", font_size=28, color=C_GT).next_to(bc, DOWN, 0.1)
        brw = Brace(box, UP, buff=0.05).set_color(C_GT)
        brwl = Tex(r"b_w = " + f"{BW:.1f}" + r"\,\mathrm{px}", font_size=26, color=C_GT).next_to(brw, UP, 0.12)
        b.play(GrowFromCenter(bc), FadeIn(bcl), GrowFromCenter(brw), FadeIn(brwl), rt=1.0)
        # project the true 3D centre with the pinhole
        pc = Dot(px(img, PCX, PCY), radius=0.08, color=C_PRED)
        pcl = Tex(r"\big(f_x\tfrac{X}{Z}+c_x,\ f_y\tfrac{Y}{Z}+c_y\big)", font_size=26, color=C_PRED)
        pcl.next_to(box, RIGHT, 0.15).set_y(pc.get_y() + 0.35)
        pcl_bg = BackgroundRectangle(pcl, fill_opacity=0.7, buff=0.05)
        ring = Circle(radius=0.6).move_to(pc).set_stroke(C_PRED, 3)
        b.play(GrowFromCenter(pc), ring.animate.scale(0.15).set_stroke(opacity=0), FadeIn(pcl_bg), Write(pcl), rt=1.1)
        off = Arrow(bc.get_center(), pc.get_center(), buff=0.05, stroke_width=4).set_color(WHITE)
        dx = DashedLine(bc.get_center(), [pc.get_x(), bc.get_y(), 0]).set_stroke(C_PRED, 2)
        dy = DashedLine([pc.get_x(), bc.get_y(), 0], pc.get_center()).set_stroke(C_PRED, 2)
        b.play(GrowArrow(off), ShowCreation(dx), ShowCreation(dy), rt=0.8)
        # equations on the right
        ex = Tex(r"U_x=\tfrac{1}{b_w}\big(f_x\tfrac{X}{Z}+c_x-b_x\big)", font_size=30)
        ey = Tex(r"U_y=\tfrac{1}{b_h}\big(f_y\tfrac{Y}{Z}+c_y-b_y\big)", font_size=30)
        ez = Tex(r"U_z=\tfrac{1}{2}\big(\tfrac{\alpha b_w}{W}+\tfrac{b_h}{H}\big)Z", font_size=30)
        eqs = VGroup(ex, ey, ez).arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to([4.6, 1.9, 0])
        vx = Tex(r"=" + f3(U[0]), font_size=30, color=C_PRED).next_to(ex, RIGHT, 0.15)
        vy = Tex(r"=" + f3(U[1]), font_size=30, color=C_PRED).next_to(ey, RIGHT, 0.15)
        vz = Tex(r"=" + f3(U[2]), font_size=30, color=C_GT).next_to(ez, RIGHT, 0.15)
        grp = VGroup(eqs, vx, vy, vz)
        if grp.get_right()[0] > 6.9:
            grp.shift((grp.get_right()[0] - 6.9) * LEFT)
        b.play(Write(ex), rt=1.0)
        b.play(FadeIn(vx, shift=0.2 * LEFT), rt=0.4)
        b.play(Write(ey), rt=0.9)
        b.play(FadeIn(vy, shift=0.2 * LEFT), rt=0.4)
        al = Tex(r"\alpha=1.6", font_size=28, color=C_GT).next_to(ez, DOWN, 0.2).align_to(ez, LEFT)
        b.play(Write(ez), FadeIn(al), rt=1.0)
        # "grows with the box size times the depth": the box pulses
        b.play(box.animate.scale(1.25), brw.animate.scale(1.25), rt=0.35)
        b.play(box.animate.scale(1 / 1.25), brw.animate.scale(1 / 1.25), FadeIn(vz, shift=0.2 * LEFT), rt=0.45)
        # ---- inversion: side view (x-z plane) of the camera, ray and the placed 3D point
        b.play(*[FadeOut(m) for m in [img, box, bc, bcl, brw, brwl, pc, pcl, pcl_bg, off, dx, dy]], rt=0.5)
        inv = Tex(r"Z=\frac{U_z}{\tfrac12(\alpha b_w/W+b_h/H)},\quad X=\frac{(U_x b_w+b_x-c_x)\,Z}{f_x},\quad "
                  r"Y=\frac{(U_y b_h+b_y-c_y)\,Z}{f_y}", font_size=28).move_to([0, -0.15, 0])
        b.play(Write(inv), Group(bars, bl, u_lbl).animate.to_corner(UL, buff=0.45), rt=1.4)
        # top-down schematic: camera at left, optical axis to the right (scale: 1 m = 1.1 units)
        sc = 1.1; o = np.array([-6.2, -2.55, 0])
        axis = Arrow(o, o + RIGHT * 6.6 * sc, buff=0, stroke_width=2).set_color(GREY_B)
        zl = Tex(r"Z\ (\mathrm{m})", font_size=24, color=GREY_B).next_to(axis, RIGHT, 0.1)
        camm = Polygon(o, o + [-0.5, 0.35, 0], o + [-0.5, -0.35, 0]).set_fill(GREY_C, 1).set_stroke(WHITE, 1)
        ticks = VGroup(*[VGroup(Line(o + RIGHT * k * sc + 0.08 * UP, o + RIGHT * k * sc + 0.08 * DOWN).set_stroke(GREY_B, 1.5),
                                Tex(str(k), font_size=20, color=GREY_B).next_to(o + RIGHT * k * sc, DOWN, 0.12)) for k in range(1, 7)])
        b.play(FadeIn(camm), GrowArrow(axis), FadeIn(ticks), FadeIn(zl), rt=0.6)
        # ray through the projected centre: x/z = X/Z (exact, from U_x)
        Xz = (U[0] * BW + BX - cx) / fx                      # X / Z
        dirv = np.array([1.0, -Xz, 0.0])                       # screen up = camera -x  (top view, x to the left)
        ray = Line(o, o + dirv / np.linalg.norm(dirv) * 7.0 * sc).set_stroke(C_PRED, 2, 0.8)
        b.play(ShowCreation(ray), rt=0.4)
        zt = ValueTracker(0.3)
        pt = always_redraw(lambda: Dot(o + dirv * zt.get_value() * sc, radius=0.1, color=C_GT))
        zval = always_redraw(lambda: Tex(f"Z={zt.get_value():.3f}" + r"\,\mathrm{m}", font_size=26, color=C_GT)
                             .next_to(o + dirv * zt.get_value() * sc, UP, 0.15))
        self.add(pt, zval)
        Zb = U[2] / ZS
        b.play(zt.animate.set_value(Zb), rt=1.0)
        pt.clear_updaters(); zval.clear_updaters()
        Xb = (U[0] * BW + BX - cx) * Zb / fx; Yb = (U[1] * BH + BY - cy) * Zb / fy
        res = Tex(r"t=(" + f3(Xb) + ",\ " + f3(Yb) + ",\ " + f3(Zb) + r")\,\mathrm{m}", font_size=30, color=C_GT)
        res.move_to([4.0, -1.55, 0])
        gt = lbl("= ground truth, exactly", 22, C_GT).next_to(res, DOWN, 0.15)
        b.play(Write(res), rt=0.8)
        b.play(FadeIn(gt), rt=0.4)
        # check: reprojecting the recovered t lands exactly on the projected centre
        rp = Tex(r"\big(f_x\tfrac{X}{Z}+c_x,\ f_y\tfrac{Y}{Z}+c_y\big)=(" + f"{fx * Xb / Zb + cx:.1f}" + r",\ " +
                 f"{fy * Yb / Zb + cy:.1f}" + r")\,\mathrm{px}", font_size=26, color=C_PRED).next_to(gt, DOWN, 0.2)
        rp.shift((6.9 - rp.get_right()[0]) * RIGHT)
        b.play(Write(rp), rt=0.9)
        par = lbl("translation block: 0.33 M parameters", 26, C_PRED).move_to([4.0, -3.45, 0])
        b.play(FadeIn(par, shift=0.2 * UP), Flash(pt, color=C_GT), rt=0.7)
        b.done()
