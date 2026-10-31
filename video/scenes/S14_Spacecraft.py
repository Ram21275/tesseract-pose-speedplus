"""S14_Spacecraft: SPEED+ application. Real lightbox / sunlamp / synthetic crops from the dataset (assets made by
_prep_S11_S14_assets.py), a labelled schematic satellite (box + panel + antennas, not the Tango model) drawn with real
rotation matrices, and the numbers of paper/tables/main.tex, flips.tex, spp_sota.tex and sec/4_experiments.tex."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401,E702
from _c_helpers import *  # noqa: F401,F403

D = assets_json()
R0 = rotm([0, 1, 0], 0.35) @ rotm([1, 0, 0], 0.25)


class S14_Spacecraft(Scene):
    default_camera_config = dict(background_color=BG)

    def construct(self):
        setup_scene(self)
        # ------------------------------------------------------------------ b1: renders -> schematic -> two real domains
        b = Beat(self, "S14_Spacecraft_b1")
        ang = ValueTracker(0.0)
        sat = sat_vg(R0, 1.05, [0, 0.4, 0])
        sat.add_updater(lambda m: m.become(sat_vg(rotm([0, 0, 1], ang.get_value()) @ R0, 1.05, [0, 0.4, 0])))
        sl = lbl("schematic satellite (stand-in, not the real model)", 20, GREY_B).move_to([0, -2.35, 0])
        b.play(FadeIn(sat, scale=0.8), FadeIn(sl), ang.animate.set_value(0.5), rt=1.0)
        syn = ImageMobject(str(ASSETS / "s14_synthetic.png")).set_height(3.3).move_to([-5.0, 0.4, 0])
        synl = lbl("train: renders only", 24, C_PRED).next_to(syn, DOWN, 0.2)
        ar1 = Arrow(syn.get_right(), [-2.0, 0.4, 0], buff=0.15).set_color(C_PRED)
        b.play(FadeIn(syn, shift=0.3 * RIGHT), FadeIn(synl), GrowArrow(ar1), ang.animate.set_value(1.1), rt=1.6)
        lb = ImageMobject(str(ASSETS / "s14_lightbox.png")).set_height(2.75).move_to([5.0, 1.85, 0])
        su = ImageMobject(str(ASSETS / "s14_sunlamp.png")).set_height(2.75).move_to([5.0, -1.85, 0])
        lbl1 = lbl("real: lightbox", 22, C_GT).next_to(lb, DOWN, 0.1)
        lbl2 = lbl("real: sunlamp", 22, C_GT).next_to(su, DOWN, 0.1)
        ar2 = Arrow([2.0, 0.6, 0], lb.get_left() + 0.2 * DOWN, buff=0.1).set_color(C_GT)
        ar3 = Arrow([2.0, 0.2, 0], su.get_left() + 0.2 * UP, buff=0.1).set_color(C_GT)
        b.play(FadeIn(lb, shift=0.3 * LEFT), GrowArrow(ar2), FadeIn(lbl1), ang.animate.set_value(1.7), rt=1.3)
        b.play(FadeIn(su, shift=0.3 * LEFT), GrowArrow(ar3), FadeIn(lbl2), ang.animate.set_value(2.3), rt=1.3)
        hil = lbl("two hardware-in-the-loop facilities", 22, C_GT).move_to([1.0, 3.55, 0])
        b.play(FadeIn(hil), ang.animate.set_value(3.0), rt=1.2)
        b.wait(max(b.left - 0.35, 0))
        b.done()

        # ------------------------------------------------------------------ b2: flips; oracle vs prediction
        b = Beat(self, "S14_Spacecraft_b2")
        sat.clear_updaters()
        Rs = rotm([0, 0, 1], 3.0) @ R0
        c0 = np.array([-3.6, 0.3, 0])
        b.play(*[FadeOut(m) for m in [syn, synl, ar1, lb, su, lbl1, lbl2, ar2, ar3, hil, sl]],
               sat.animate.become(sat_vg(Rs, 1.35, c0)), rt=0.8)
        ghost = sat_silhouette(Rs, 1.35, c0, C_GT, 7)
        gl = lbl("true pose", 22, C_GT).move_to(c0 + [-1.9, 2.6, 0])
        b.play(FadeIn(ghost), FadeIn(gl), rt=0.6)
        # half turn about the panel normal (body z axis), exact rotation matrices
        flip = ValueTracker(0.0)
        sat.add_updater(lambda m: m.become(sat_vg(Rs @ rotm([0, 0, 1], flip.get_value()), 1.35, c0)))
        fl = Tex(r"180^\circ", font_size=40, color=C_FLIP).move_to(c0 + [1.9, 2.6, 0])
        b.play(flip.animate.set_value(PI), FadeIn(fl), rt=1.8)
        sat.clear_updaters()
        after = sat_silhouette(Rs @ rotm([0, 0, 1], PI), 1.35, c0, C_FLIP, 2.5)
        self.add(ghost)
        b.play(ShowCreation(after), sat.animate.set_opacity(0.25), rt=0.8)
        # the only difference: the small antenna stub
        M = VIEW @ Rs
        stub_a = c0 + 1.35 * (M @ np.array([1.0, -0.4, -0.6]))[[0, 1, 2]] * [1, 1, 0]
        stub_b = c0 + 1.35 * (M @ rotm([0, 0, 1], PI) @ np.array([1.0, -0.4, -0.6])) * [1, 1, 0]
        ca = Circle(radius=0.38).move_to(stub_a).set_stroke(C_GT, 3)
        cb = Circle(radius=0.38).move_to(stub_b).set_stroke(C_FLIP, 3)
        dl = lbl("almost the same image", 24).move_to(c0 + [0, -3.0, 0])
        b.play(ShowCreation(ca), ShowCreation(cb), FadeIn(dl), rt=1.0)
        # bars: oracle 9.8 vs prediction 26.9 (lightbox)
        y0 = -2.4; s = 0.17; xs = [2.4, 4.9]
        ax = Line([1.3, y0, 0], [6.4, y0, 0]).set_stroke(GREY_B, 2)
        ttl = lbl("lightbox, mean rotation error", 22, GREY_B).move_to([3.85, 3.55, 0])
        b.play(ShowCreation(ax), FadeIn(ttl), rt=0.6)
        bp = vbar(xs[1], y0, 26.9 * s, 1.3, C_FLIP)
        bpl = Tex(r"26.9^\circ", font_size=40, color=C_FLIP).next_to(bp, UP, 0.12)
        bpn = lbl("prediction", 22, C_FLIP).next_to([xs[1], y0, 0], DOWN, 0.2)
        b.play(grow_up(bp), FadeIn(bpn), rt=1.4)
        b.play(FadeIn(bpl, shift=0.2 * UP), rt=0.4)
        bo = vbar(xs[0], y0, 9.8 * s, 1.3, C_GT)
        bol = Tex(r"9.8^\circ", font_size=40, color=C_GT).next_to(bo, UP, 0.12)
        bon = VGroup(lbl("oracle: best of the", 20, C_GT), lbl("tree's own candidates", 20, C_GT)).arrange(DOWN, buff=0.08)
        bon.next_to([xs[0], y0, 0], DOWN, 0.2)
        b.play(grow_up(bo), FadeIn(bon), rt=1.4)
        b.play(FadeIn(bol, shift=0.2 * UP), rt=0.4)
        cand = lbl("candidates: beam leaves and their 180° flips", 18, C_GT).move_to([2.5, -3.75, 0])
        b.play(FadeIn(cand), Indicate(ca, color=C_GT), Indicate(cb, color=C_FLIP), rt=1.0)
        _p1, _p2 = bp.get_top() + LEFT * 0.9, np.array([xs[1] - 0.9, bo.get_top()[1], 0]); _m = (_p1 + _p2) / 2
        gap = VGroup(Arrow(_m, _p1, buff=0), Arrow(_m, _p2, buff=0)).set_color(WHITE)
        gl2 = VGroup(lbl("the right mode is there,", 21), lbl("it is not selected", 21, C_FLIP)).arrange(DOWN, buff=0.1)
        gl2.move_to([2.3, 1.0, 0])
        b.play(GrowFromCenter(gap), rt=0.7)
        b.play(FadeIn(gl2[0]), rt=0.8)
        b.play(FadeIn(gl2[1]), Indicate(after, color=C_FLIP), rt=1.2)
        b.play(ghost.animate.set_stroke(C_GT, 9), Flash(bo.get_top(), color=C_GT), rt=0.8)
        b.done()

        # ------------------------------------------------------------------ b3: error staircase + flip repair
        b = Beat(self, "S14_Spacecraft_b3")
        b.play(*[FadeOut(m) for m in self.mobjects], rt=0.5)
        y0, s = -2.55, 0.13
        X = [-5.4, -2.6, 0.2, 3.0, 5.6]                         # stage boundaries
        stages = ["clean tree", "+ augmentation", "+ orbit voting", "+ distillation"]
        vals = {"lightbox": ([26.9, 17.3, 12.3, 11.21], ["26.9", "17.3", "12.3", "11.21"], C_PRED),
                "sunlamp": ([36.7, 20.2, 12.3, 11.09], ["36.7", "20.2", "12.3", "11.09"], C_POE)}
        yax = Arrow([X[0], y0, 0], [X[0], y0 + 41 * s, 0], buff=0, stroke_width=3).set_color(GREY_B)
        xax = Line([X[0], y0, 0], [X[-1] + 0.3, y0, 0]).set_stroke(GREY_B, 2)
        yl = Tex(r"E_R\ (^\circ)", font_size=28, color=GREY_B).next_to(yax, UP, 0.1)
        ticks = VGroup(*[VGroup(Line([X[0] - 0.08, y0 + v * s, 0], [X[0], y0 + v * s, 0]).set_stroke(GREY_B, 2),
                                Tex(str(v), font_size=22, color=GREY_B).next_to([X[0] - 0.08, y0 + v * s, 0], LEFT, 0.08))
                         for v in [10, 20, 30, 40]])
        grid = VGroup(*[DashedLine([X[0], y0 + v * s, 0], [X[-1], y0 + v * s, 0], dash_length=0.06)
                        .set_stroke(GREY_D, 1, 0.6) for v in [10, 20, 30, 40]])
        leg = VGroup(*[VGroup(Line(ORIGIN, 0.5 * RIGHT).set_stroke(c, 5), lbl(k, 22, c)).arrange(RIGHT, buff=0.15)
                       for k, (_, _, c) in vals.items()]).arrange(RIGHT, buff=0.6).move_to([3.0, 3.55, 0])
        b.play(GrowArrow(yax), ShowCreation(xax), FadeIn(ticks), FadeIn(yl), FadeIn(grid), FadeIn(leg), rt=0.9)
        slabels = VGroup(*[lbl(st, 20, GREY_A).move_to([(X[i] + X[i + 1]) / 2, y0 - 0.35, 0]) for i, st in enumerate(stages)])
        drawn = {}
        for i in range(4):
            anims = [FadeIn(slabels[i], shift=0.1 * UP)]
            for k, (v, sv, c) in vals.items():
                yv = y0 + v[i] * s
                seg = VGroup()
                if i > 0:
                    seg.add(Line([X[i], y0 + v[i - 1] * s, 0], [X[i], yv, 0]).set_stroke(c, 5))
                seg.add(Line([X[i], yv, 0], [X[i + 1], yv, 0]).set_stroke(c, 5))
                off = (0.3 if k == "sunlamp" else -0.3) if i < 2 else (0.3 if k == "lightbox" else -0.3)
                tag = Tex(sv[i] + r"^\circ", font_size=30, color=c).move_to([(X[i] + X[i + 1]) / 2, yv + off, 0])
                drawn[(k, i)] = tag
                anims += [ShowCreation(seg), FadeIn(tag)]
            b.play(*anims, rt=[2.1, 2.2, 2.2, 2.2][i])
        # flip repair: severe flips (>150 deg) fixed by orbit voting (tables/flips.tex)
        b.play(VGroup(*[m for m in self.mobjects if isinstance(m, VMobject)]).animate.scale(0.72, about_point=[-6.6, -3.6, 0]),
               rt=0.8)
        # severe flips (> 150 deg) as 50 marks each; orbit voting repairs 62% (31/50) and 80% (40/50) (tables/flips.tex)
        rt_ = lbl("severe flips repaired by orbit voting", 24).move_to([4.4, 3.35, 0])
        b.play(FadeIn(rt_), rt=0.4)
        waffles = []
        for j, (name, pct, c) in enumerate([("lightbox", 62, C_PRED), ("sunlamp", 80, C_POE)]):
            org = np.array([2.9 + 2.3 * j, 2.75, 0])
            marks = VGroup(*[Square(0.27).set_fill(C_FLIP, 0.9).set_stroke(BG, 2).move_to(org + [0.32 * (k % 5), -0.32 * (k // 5), 0])
                             for k in range(50)])
            nl = lbl(name, 22, c).next_to(marks, DOWN, 0.6)
            waffles.append((marks, nl, pct, c))
        b.play(*[LaggedStart(*[FadeIn(m, scale=0.5) for m in w[0]], lag_ratio=0.02) for w in waffles],
               *[FadeIn(w[1]) for w in waffles], rt=1.0)
        for marks, nl, pct, c in waffles:
            n_fix = pct * 50 // 100
            t = Tex(f"{pct}\\%", font_size=34, color=c).next_to(marks, DOWN, 0.1)
            b.play(LaggedStart(*[m.animate.set_fill(c, 0.95) for m in marks[:n_fix]], lag_ratio=0.03), FadeIn(t), rt=1.2)
        # good predictions broken: 0.4% = 1 in 250
        strip = VGroup(*[Square(0.1).set_fill(C_GT, 0.85).set_stroke(width=0) for _ in range(250)])
        strip.arrange_in_grid(10, 25, buff=0.025).move_to([4.15, -2.45, 0])
        sl = lbl("good predictions (250 shown)", 18, C_GT).next_to(strip, UP, 0.12)
        b.play(LaggedStart(*[FadeIn(m) for m in strip], lag_ratio=0.002), FadeIn(sl), rt=0.9)
        br = VGroup(lbl("broken:", 22, GREY_A), Tex(r"0.4\%", font_size=40, color=C_FLIP)).arrange(RIGHT, buff=0.2)
        br.next_to(strip, DOWN, 0.2)
        b.play(strip[137].animate.set_fill(C_FLIP, 1).scale(2.2), FadeIn(br, shift=0.2 * UP), rt=0.9)
        b.done()

        # ------------------------------------------------------------------ b4: detector boxes vs ground-truth boxes
        b = Beat(self, "S14_Spacecraft_b4")
        b.play(*[FadeOut(m) for m in self.mobjects], rt=0.4)
        imgs, boxes = Group(), []
        for j, e in enumerate(D["s14_det"]):
            name = e["img"][:-4]
            im = ImageMobject(str(ASSETS / f"s14_det_{name}.png")).set_width(5.9).move_to([-3.25 + 6.5 * j, 1.55, 0])
            g = Square(side_length=e["gt"][2] / 1920 * im.get_width()).move_to(px(im, e["gt"][0], e["gt"][1])).set_stroke(C_GT, 3)
            d = Square(side_length=e["det"][2] / 1920 * im.get_width()).move_to(px(im, e["det"][0], e["det"][1])).set_stroke(C_PRED, 3)
            iou = Tex(r"\mathrm{IoU}=" + f"{e['iou']:.2f}", font_size=30, color=C_PRED).next_to(im, DOWN, 0.1)
            imgs.add(im); boxes.append((g, d, iou))
        b.play(FadeIn(imgs), *[ShowCreation(g) for g, _, _ in boxes], rt=0.9)
        legend = VGroup(VGroup(Line(ORIGIN, 0.45 * RIGHT).set_stroke(C_GT, 4), lbl("ground-truth box", 20, C_GT)).arrange(RIGHT, buff=0.12),
                        VGroup(Line(ORIGIN, 0.45 * RIGHT).set_stroke(C_PRED, 4), lbl("detector (trained on renders)", 20, C_PRED)).arrange(RIGHT, buff=0.12)
                        ).arrange(RIGHT, buff=0.6).move_to([0, 3.65, 0])
        b.play(*[TransformFromCopy(g, d) for g, d, _ in boxes], FadeIn(legend), rt=1.1)
        mean = lbl("lightbox: mean IoU 0.76", 24, C_PRED).move_to([0, -0.85, 0])
        b.play(*[FadeIn(i) for _, _, i in boxes], rt=0.5)
        b.play(FadeIn(mean), rt=0.5)
        # GT crop vs detector crop error, tables/spp_sota.tex
        x0, s = -2.2, 0.2
        rows = [("lightbox", 11.21, 23.88, "11.21", "23.88", -1.85), ("sunlamp", 11.09, 16.42, "11.09", "16.42", -2.95)]
        anims1, anims2 = [], []
        for name, a, d, sa, sd, y in rows:
            n = lbl(name, 22, GREY_A).move_to([x0 - 1.3, y, 0])
            ba = hbar(x0, y + 0.17, a * s, 0.3, C_GT, 0.85)
            bd = hbar(x0, y - 0.17, d * s, 0.3, C_PRED, 0.85)
            ta = Tex(sa + r"^\circ", font_size=26, color=C_GT).next_to(ba, RIGHT, 0.1)
            td = Tex(sd + r"^\circ", font_size=26, color=C_PRED).next_to(bd, RIGHT, 0.1)
            anims1 += [FadeIn(n), grow_right(ba), FadeIn(ta)]; anims2 += [grow_right(bd), FadeIn(td)]
            if name == "lightbox":
                x2 = Tex(r"\times 2", font_size=36, color=C_FLIP).next_to(td, RIGHT, 0.3)
        b.play(*anims1, rt=1.0)
        b.play(*anims2, rt=1.4)
        bn = lbl("detection, not the heads, is the bottleneck", 24, C_FLIP).move_to([0, -3.65, 0])
        b.play(FadeIn(x2), FadeIn(bn), rt=0.8)
        link = CurvedArrow(boxes[1][2].get_bottom() + 0.05 * DOWN, x2.get_top() + 0.1 * UP, angle=-0.6).set_color(C_PRED)
        b.play(ShowCreation(link), Indicate(boxes[1][1], color=C_FLIP, scale_factor=1.04), rt=1.0)
        b.done()
