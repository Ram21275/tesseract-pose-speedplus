"""S15_FineTuning: frozen heads vs fine-tuning. Numbers from paper/tables/ft_cost.tex and spp_sota.tex
(frozen DINOv3-L + head 15.47/24.49; Church & Nikolian DINOv3-H+ 840 M LoRA 1.17/1.56; ours 11.21/11.09; backprop passes
1.4 M / 1.0 M / 4.3 M / 4.3 M / 7.2 M / 0; PAID-ViT 12.29/11.76, FastPose-ViT 12.26/18.28) and sec/4_experiments.tex
(gap closed 30% / 58%: (15.47-11.21)/(15.47-1.17), (24.49-11.09)/(24.49-1.56))."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401,E702
from _c_helpers import *  # noqa: F401,F403

GAP = [(15.47 - 11.21) / (15.47 - 1.17), (24.49 - 11.09) / (24.49 - 1.56)]
assert [round(100 * g) for g in GAP] == [30, 58]


def lock(c=GREY_B):
    body = RoundedRectangle(width=0.34, height=0.26, corner_radius=0.04).set_fill(c, 1).set_stroke(width=0)
    shackle = Arc(0, PI, radius=0.11).set_stroke(c, 4).next_to(body, UP, 0)
    return VGroup(shackle, body)


def net(rows, cols, w, h, color=GREY_B):
    return lattice(rows, cols, w, h, color=color, dot_r=0.03, stroke=0.8, opacity=0.45)


def pulses(start, end, color, n=6, spread=0.9, rng=None, size=0.07):
    rng = rng or np.random.default_rng(0)
    ds, anims = [], []
    for k in range(n):
        dy = (rng.random() - 0.5) * spread * UP
        d = Dot(start + dy, radius=size).set_fill(color, 1)
        ds.append(d); anims.append(Succession(d.animate(rate_func=linear).move_to(end + dy * 0.6), FadeOut(d, run_time=0.05)))
    return ds, anims


class S15_FineTuning(Scene):
    default_camera_config = dict(background_color=BG)

    def construct(self):
        setup_scene(self)
        rng = np.random.default_rng(3)
        # ------------------------------------------------------------------ b1: same features, frozen head vs LoRA vs ours
        b = Beat(self, "S15_FineTuning_b1")
        bb1 = net(9, 16, 4.2, 1.7).move_to([-4.0, 1.75, 0])
        lk1 = lock().next_to(bb1, UP, 0.1).align_to(bb1, LEFT)
        bl1 = lbl("DINOv3-L, 303 M, frozen", 21, GREY_A).next_to(lk1, RIGHT, 0.15)
        b.play(FadeIn(bb1, lag_ratio=0.02), FadeIn(lk1), FadeIn(bl1), rt=1.4)
        hd1 = net(4, 3, 0.6, 0.8, C_BASE).next_to(bb1, RIGHT, 0.45)
        hl1 = lbl("head", 20, C_BASE).next_to(hd1, UP, 0.12)
        ds, an = pulses(bb1.get_left(), hd1.get_center(), C_DINO, 7, 1.4, rng)
        self.add(*ds)
        b.play(FadeIn(hd1), FadeIn(hl1), *an, rt=1.3)
        # bars (error axis on the right)
        y0, s = -3.0, 0.215; gx = [2.0, 5.0]
        ax = Line([0.6, y0, 0], [6.6, y0, 0]).set_stroke(GREY_B, 2)
        dl = VGroup(*[lbl(t, 24, GREY_A).move_to([x, y0 - 0.35, 0]) for x, t in zip(gx, ["lightbox", "sunlamp"])])
        yt = lbl("mean rotation error, real images", 20, GREY_B).move_to([3.6, 3.55, 0])
        b.play(ShowCreation(ax), FadeIn(dl), FadeIn(yt), rt=0.7)
        cols = {"frozen": C_BASE, "lora": C_FT, "ours": C_PRED}
        vals = {"frozen": ["15.47", "24.49"], "lora": ["1.17", "1.56"], "ours": ["11.21", "11.09"]}
        off = {"frozen": -0.75, "lora": 0.0, "ours": 0.75}
        bars, tags = {}, {}
        for k in vals:
            bars[k] = VGroup(*[vbar(x + off[k], y0, float(v) * s, 0.66, cols[k]) for x, v in zip(gx, vals[k])])
            tags[k] = VGroup(*[Tex(v + r"^\circ", font_size=28, color=cols[k]).next_to(r, UP, 0.08) for v, r in zip(vals[k], bars[k])])
        b.play(*[grow_up(r) for r in bars["frozen"]], hd1.animate.set_color(C_BASE), rt=1.3)
        b.play(FadeIn(tags["frozen"]), rt=0.5)
        # LoRA: bigger backbone, adapters, backward pulses through the whole network
        bb2 = net(9, 16, 4.2, 1.7).move_to([-4.0, -1.55, 0])
        bl2 = lbl("DINOv3-H+, 840 M, + LoRA adapters", 21, C_FT).next_to(bb2, UP, 0.12).align_to(bb2, LEFT)
        adapters = VGroup(*[Rectangle(0.07, 1.8).set_fill(C_FT, 0.85).set_stroke(width=0).move_to(bb2.get_center() + x * RIGHT)
                            for x in np.linspace(-1.75, 1.75, 6)])
        hd2 = net(4, 3, 0.6, 0.8, C_FT).next_to(bb2, RIGHT, 0.45)
        b.play(FadeIn(bb2, lag_ratio=0.02), FadeIn(bl2), FadeIn(hd2), rt=1.1)
        b.play(LaggedStart(*[GrowFromCenter(a) for a in adapters], lag_ratio=0.1), rt=0.8)
        ds, an = pulses(hd2.get_center(), bb2.get_left(), C_FT, 9, 1.4, rng)
        self.add(*ds)
        grad = lbl("gradients through the backbone", 20, C_FT).next_to(bb2, DOWN, 0.15)
        b.play(*an, FadeIn(grad), rt=1.4)
        b.play(*[grow_up(r) for r in bars["lora"]], rt=1.0)
        b.play(FadeIn(tags["lora"]), rt=0.5)
        # ours on the same frozen DINOv3 features
        ol = lbl("ours: frozen, selection-aware heads", 21, C_PRED).next_to(bb1, DOWN, 0.15).align_to(bb1, LEFT)
        b.play(*[grow_up(r) for r in bars["ours"]], hd1.animate.set_color(C_PRED), FadeIn(ol), rt=1.3)
        b.play(FadeIn(tags["ours"]), rt=0.5)
        ds, an = pulses(bb1.get_left(), hd1.get_center(), C_PRED, 8, 1.4, rng)
        self.add(*ds)
        b.play(*an, rt=1.1)
        same = VGroup(*[ArcBetweenPoints(np.array([x - 0.75, y0 - 0.62, 0]), np.array([x + 0.75, y0 - 0.62, 0]), angle=PI / 2).set_stroke(C_PRED, 3)
                        for x in gx])
        sl_ = lbl("frozen head and ours: same frozen DINOv3-L features", 18, C_PRED).move_to([3.6, 3.1, 0])
        b.play(ShowCreation(same), FadeIn(sl_), rt=0.8)
        b.play(Indicate(bars["lora"], color=WHITE), Indicate(tags["lora"], color=WHITE), rt=0.8)
        b.wait(max(b.left - 0.3, 0))
        b.done()

        # ------------------------------------------------------------------ b2: backprop passes per target (log scale)
        b = Beat(self, "S15_FineTuning_b2")
        b.play(*[FadeOut(m) for m in self.mobjects], rt=0.5)
        ttl = lbl("back-propagation passes through the backbone, per target", 26).to_edge(UP, buff=0.35)
        xl, xr = -1.6, 6.3                        # log10 5 .. 7
        def X(v):  # noqa: E306
            return xl + (np.log10(v) - 5.0) / 2.0 * (xr - xl)
        ya = -2.75
        ax = Line([xl, ya, 0], [xr + 0.2, ya, 0]).set_stroke(GREY_B, 2)
        ticks = VGroup()
        for v, t in [(1e5, "100 k"), (1e6, "1 M"), (1e7, "10 M")]:
            ticks.add(Line([X(v), ya - 0.1, 0], [X(v), ya + 0.1, 0]).set_stroke(GREY_B, 2), lbl(t, 20, GREY_B).next_to([X(v), ya, 0], DOWN, 0.15))
            ticks.add(DashedLine([X(v), ya, 0], [X(v), 2.75, 0], dash_length=0.06).set_stroke(GREY_D, 1, 0.6))
        for v in [2e5, 3e5, 5e5, 2e6, 3e6, 5e6]:
            ticks.add(Line([X(v), ya - 0.06, 0], [X(v), ya + 0.06, 0]).set_stroke(GREY_C, 1.5))
        logl = lbl("log scale", 18, GREY_C).move_to([xr - 0.3, ya - 0.75, 0])
        circ_h = lbl("backbone", 18, C_FT).move_to([-2.45, 3.0, 0])
        b.play(FadeIn(ttl), ShowCreation(ax), FadeIn(ticks), FadeIn(logl), FadeIn(circ_h), rt=0.8)
        # rows: (name, passes, label, backbone label, backbone M for the circle area, None = size not reported)
        meth = [("Church & Nikolian", 1.4e6, "1.4 M", "840 M", 840), ("SPNv2", 1.0e6, "1.0 M", "12 M", 12),
                ("SPNv3", 4.3e6, "4.3 M", "86 M", 86), ("PAID-ViT", 4.3e6, "4.3 M", "ViT-B/16", None),
                ("FastPose-ViT", 7.2e6, "7.2 M", "86 M", 86)]
        ys = [2.25 - 0.82 * i for i in range(6)]
        for (name, v, vl, bk, M), y in zip(meth, ys):
            nm = lbl(name, 22).move_to([-4.6, y, 0])
            r = 0.05 + 0.37 * np.sqrt((M or 86) / 840)
            circ = Circle(radius=r).move_to([-2.45, y, 0]).set_fill(C_FT, 0.35).set_stroke(C_FT, 2)
            if M is None:
                circ = DashedVMobject(circ, num_dashes=12).set_stroke(C_FT, 2)
            ct = lbl(bk, 15, C_FT).next_to(circ, DOWN, 0.03) if M != 840 else lbl(bk, 15, C_FT).move_to(circ)
            bar = hbar(xl, y, X(v) - xl, 0.36, C_FT, 0.8)
            vt = lbl(vl, 20, C_FT).next_to(bar, RIGHT, 0.12)
            ds, an = pulses(np.array([X(v) - 0.1, y, 0]), np.array([xl, y, 0]), WHITE, 3, 0.2, rng, 0.045)
            self.add(*ds)
            b.play(FadeIn(nm), GrowFromCenter(circ), FadeIn(ct), grow_right(bar), FadeIn(vt), rt=1.0)
            b.play(*an, rt=0.35)
        rng_l = lbl("backbones of 12 to 840 M parameters", 20, C_FT).move_to([-4.0, -3.55, 0])
        b.play(FadeIn(rng_l), rt=0.5)
        # ours: zero passes -- cannot sit on a log axis, so it is written out
        y = ys[5]
        nm = lbl("ours", 24, C_PRED).move_to([-4.6, y, 0])
        fr = net(3, 4, 0.55, 0.4).move_to([-2.45, y, 0]); flk = lock().scale(0.7).next_to(fr, LEFT, 0.1)
        z0 = VGroup(Tex("0", font_size=46, color=C_PRED), lbl("no gradient reaches any backbone", 22, C_PRED)).arrange(RIGHT, buff=0.3)
        z0.next_to([xl, y, 0], RIGHT, 0.2)
        b.play(FadeIn(nm), FadeIn(fr), FadeIn(flk), FadeIn(z0, shift=0.2 * RIGHT), rt=1.0)
        allp = []
        for (name, v, vl, bk, M), y in zip(meth, ys[:5]):
            ds, an = pulses(np.array([X(v) - 0.1, y, 0]), np.array([xl, y, 0]), WHITE, 4, 0.2, rng, 0.045)
            self.add(*ds); allp += an
        b.play(*allp, Indicate(z0[0], color=C_PRED, scale_factor=1.4), rt=1.3)
        b.wait(max(b.left - 0.3, 0))
        b.done()

        # ------------------------------------------------------------------ b3: gap closed; direct regressors; one pass, many heads
        b = Beat(self, "S15_FineTuning_b3")
        b.play(*[FadeOut(m) for m in self.mobjects], rt=0.5)
        x0, s = -6.0, 0.2
        def XE(e):  # noqa: E306
            return x0 + e * s
        ttl = lbl("frozen head to LoRA: the gap our heads close", 22).move_to([-3.3, 3.55, 0])
        info = [("lightbox", 1.5, 15.47, 1.17, 11.21, "11.21", [("PAID-ViT 12.29, FastPose-ViT 12.26", 12.275)]),
                ("sunlamp", -1.6, 24.49, 1.56, 11.09, "11.09", [("PAID-ViT 11.76", 11.76), ("FastPose-ViT 18.28", 18.28)])]
        gauges, fills, olabs, regs, pcts = VGroup(), [], [], [], []
        for k, (dom, y, fz, lo, ours, so, rg) in enumerate(info):
            g = Rectangle(XE(fz) - XE(lo), 0.55).move_to([(XE(fz) + XE(lo)) / 2, y, 0]).set_stroke(GREY_B, 2)
            name = lbl(dom, 24, GREY_A).move_to([XE(lo) + 0.6, y + 0.85, 0])
            ft = VGroup(lbl("frozen + head", 17, C_BASE), Tex(f"{fz:.2f}" + r"^\circ", font_size=26, color=C_BASE)).arrange(DOWN, buff=0.05)
            ft.next_to(g, RIGHT, 0.12)
            lt = VGroup(lbl("LoRA", 17, C_FT), Tex(f"{lo:.2f}" + r"^\circ", font_size=26, color=C_FT)).arrange(RIGHT, buff=0.08)
            lt.next_to([XE(lo), y - 0.28, 0], DOWN, 0.1)
            gauges.add(VGroup(g, name, ft, lt))
            fills.append(Rectangle(XE(fz) - XE(ours), 0.55).move_to([XE(fz), y, 0], aligned_edge=RIGHT)
                         .set_fill(C_PRED, 0.85).set_stroke(width=0))
            olabs.append(VGroup(lbl("ours", 17, C_PRED), Tex(so + r"^\circ", font_size=26, color=C_PRED)).arrange(RIGHT, buff=0.08)
                         .next_to([XE(ours), y + 0.28, 0], UP, 0.08))
            pcts.append(Tex(f"{round(100 * GAP[k])}\\%", font_size=30, color=BG).move_to(fills[-1]))
            rr = VGroup()
            for j, (t, v) in enumerate(rg):
                tk = Line([XE(v), y - 0.28, 0], [XE(v), y - 0.6 - 0.32 * j, 0]).set_stroke(C_FT, 3)
                rr.add(VGroup(tk, lbl(t, 16, C_FT).next_to(tk, DOWN, 0.05)))
            regs.append(rr)
        b.play(FadeIn(ttl), *[FadeIn(g) for g in gauges], rt=0.9)
        for k in range(2):
            b.play(GrowFromEdge(fills[k], RIGHT), FadeIn(olabs[k]), rt=1.1)
            b.play(FadeIn(pcts[k], scale=1.3), rt=0.4)
        b.play(*[FadeIn(r, shift=0.15 * UP) for r in regs], rt=0.9)
        reg = lbl("fine-tuned direct regressors: matched or beaten", 20, C_FT).move_to([-3.3, -3.6, 0])
        b.play(FadeIn(reg), rt=0.6)
        # one backbone pass serves every object's heads
        bb = net(9, 10, 2.0, 2.2).move_to([3.0, 0.3, 0])
        lk = lock().next_to(bb, UP, 0.1)
        bl_ = lbl("one frozen backbone pass", 21, GREY_A).next_to(bb, DOWN, 0.2)
        b.play(FadeIn(bb, lag_ratio=0.02), FadeIn(lk), FadeIn(bl_), rt=0.8)
        n = 9; heads = Group(); lines = VGroup()
        for i in range(n):
            ang = (i - (n - 1) / 2) * 0.3
            p = bb.get_right() + 2.0 * np.array([np.cos(ang), np.sin(ang), 0]) + 0.3 * RIGHT
            c = [C_PRED, C_DINO, C_GT, C_POE, C_MOGE][i % 5]
            h = net(3, 2, 0.28, 0.34, c).move_to(p)
            heads.add(h); lines.add(Line(bb.get_right(), h.get_left(), buff=0.05).set_stroke(c, 1.8, 0.8))
        hl_ = lbl("a small head per object", 20, C_PRED).move_to([4.6, 3.4, 0])
        b.play(LaggedStart(*[ShowCreation(l) for l in lines], lag_ratio=0.08),
               LaggedStart(*[FadeIn(h, scale=0.5) for h in heads], lag_ratio=0.08), FadeIn(hl_), rt=1.6)
        ds = [Dot(bb.get_right(), radius=0.06).set_fill(l.get_color(), 1) for l in lines]
        self.add(*ds)
        b.play(*[MoveAlongPath(d, l) for d, l in zip(ds, lines)], rt=1.0)
        self.remove(*ds)
        b.play(*[Indicate(f, color=WHITE, scale_factor=1.03) for f in fills], *[Flash(h.get_center(), color=l.get_color(), line_length=0.12)
                                                                               for h, l in zip(heads, lines)], rt=1.0)
        ds = [Dot(bb.get_right(), radius=0.06).set_fill(l.get_color(), 1) for l in lines]
        self.add(*ds)
        b.play(*[MoveAlongPath(d, l) for d, l in zip(ds, lines)], rt=max(min(1.0, b.left - 0.3), 0.3))
        self.remove(*ds)
        b.done()
