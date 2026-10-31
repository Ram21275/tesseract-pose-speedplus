"""S16_Synthesis: recap. Hypotheses in the axis-angle ball (true mode and its exact 180-degree look-alike; the spread of
candidates around them is illustrative), then the tesseract cube with a decoded path (encode / cell_box / free_coords from
common.py) plus the path of a symmetric equivalent q*s, the orbit ring (real lightbox crop rotated by 0/90/180/270), and
the frozen backbone with many small heads."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401,E702
from _c_helpers import *  # noqa: F401,F403

Q = quat([0.37, -0.52, 0.77], 1.83)
CH, PATH = encode(Q)
_, _, U = free_coords(Q)
for _l in range(6):
    _lo, _hi = cell_box(PATH[:_l]); assert ((U > _lo) & (U < _hi)).all()
QS = qmul(Q, quat([0, 0, 1], PI)); CHS, PATHS = encode(QS); _, _, US = free_coords(QS)
CUBE_VIEW = rotm([1, 0, 0], -1.1) @ rotm([0, 0, 1], 0.55)


def axis_angle(q):
    q = np.asarray(q, float); q = q * np.sign(q[0]) if q[0] != 0 else q
    ang = 2 * np.arccos(np.clip(q[0], -1, 1)); s = np.linalg.norm(q[1:])
    return np.zeros(3) if s < 1e-9 else q[1:] / s * ang


def box_edges(lo, hi, M, scale, center, color, width):
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    P = corners @ M.T * scale
    g = VGroup()
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(corners[i] != corners[j]) == 1:
                g.add(Line(center + [P[i, 0], P[i, 1], 0], center + [P[j, 0], P[j, 1], 0]))
    return g.set_stroke(color, width)


def cube_path(path, u, scale, center, label):
    cube = box_edges(-np.ones(3), np.ones(3), CUBE_VIEW, scale, center, GREY_B, 1.5)
    boxes = VGroup(*[box_edges(*cell_box(path[:l + 1]), CUBE_VIEW, scale, center, LEVEL_COLORS[l + 1], 2.5) for l in range(5)])
    p = CUBE_VIEW @ u * scale
    dot = Dot(center + [p[0], p[1], 0], radius=0.06).set_fill(C_GT, 1)
    t = Tex(label, font_size=30).next_to(cube, DOWN, 0.15)
    return cube, boxes, dot, t


class S16_Synthesis(Scene):
    default_camera_config = dict(background_color=BG)

    def construct(self):
        setup_scene(self)
        rng = np.random.default_rng(5)
        # ------------------------------------------------------------------ b1: the right mode was among the hypotheses
        b = Beat(self, "S16_Synthesis_b1")
        part = VGroup(Polygon([0, 0, 0], [1.6, 0, 0], [1.6, 0.4, 0], [0.45, 0.4, 0], [0.45, 1.3, 0], [0, 1.3, 0])
                      .set_fill(GREY_C, 0.9).set_stroke(GREY_A, 2),
                      Circle(radius=0.12).move_to([1.1, 0.2, 0]).set_fill(BG, 1).set_stroke(GREY_A, 1.5)).move_to([-5.3, 0.6, 0])
        sat = sat_vg(rotm([0, 1, 0], 0.35), 0.8, [5.2, 0.6, 0])
        sl = lbl("schematic", 16, GREY_C).next_to(sat, DOWN, 0.15)
        b.play(DrawBorderThenFill(part), FadeIn(sat), FadeIn(sl), rt=1.0)
        cen = np.array([0, 0.2, 0]); Rb = 2.4
        ball = Circle(radius=Rb).move_to(cen).set_stroke(GREY_B, 2).set_fill(GREY_E, 0.25)
        bl = Tex(r"\|\omega\|\le\pi", font_size=28, color=GREY_B).next_to(ball, DOWN, 0.1)
        b.play(ShowCreation(ball), FadeIn(bl), rt=0.7)
        V = rotm([1, 0, 0], -0.4) @ rotm([0, 1, 0], 0.3)
        def to2d(w):  # noqa: E306
            p = V @ w / PI * Rb; return cen + [p[0], p[1], 0]
        qt = quat([0.2, 0.9, 0.3], 1.1); qf = qmul(qt, quat([0, 0, 1], PI))       # look-alike: exact half turn
        modes = [(qt, C_GT, 14), (qf, C_FLIP, 10)]
        cloud = VGroup()
        for q, c, n in modes:
            for _ in range(n):
                dq = quat(rng.normal(size=3), np.radians(rng.uniform(8, 32)))
                cloud.add(Dot(to2d(axis_angle(qmul(q, dq))), radius=0.06).set_fill(C_PRED, 0.85))
        for _ in range(8):     # a few scattered, unlikely hypotheses
            cloud.add(Dot(to2d(axis_angle(quat(rng.normal(size=3), rng.uniform(0.3, 3.0)))), radius=0.05).set_fill(C_PRED, 0.5))
        hyp = lbl("hypotheses (illustrative spread)", 20, C_PRED).move_to(cen + [0, Rb + 0.45, 0])
        b.play(LaggedStart(*[FadeIn(d, scale=2) for d in cloud], lag_ratio=0.04), FadeIn(hyp), rt=1.7)
        dt = Dot(to2d(axis_angle(qt)), radius=0.13).set_fill(C_GT, 1)
        df = Dot(to2d(axis_angle(qf)), radius=0.12).set_fill(C_FLIP, 1)
        tl = lbl("correct", 20, C_GT).next_to(dt, UR, 0.05); fl = lbl("look-alike", 20, C_FLIP).next_to(df, DR, 0.05)
        b.play(GrowFromCenter(dt), GrowFromCenter(df), FadeIn(tl), FadeIn(fl), rt=0.9)
        sel = Circle(radius=0.38).move_to(dt).set_stroke(C_GT, 4)
        ch = lbl("the hard part: choosing it", 26, C_GT).move_to([0, -3.35, 0])
        b.play(ShowCreation(sel), FadeIn(ch), Indicate(dt, color=C_GT), rt=1.3)
        b.play(cloud.animate.set_opacity(0.25), df.animate.set_opacity(0.5), fl.animate.set_opacity(0.5),
               sel.animate.scale(1.3).set_stroke(width=6), rt=1.0)
        b.wait(max(b.left - 0.3, 0))
        b.done()

        # ------------------------------------------------------------------ b2: tree + multi-path, orbit ring, frozen backbone
        b = Beat(self, "S16_Synthesis_b2")
        b.play(*[FadeOut(m) for m in self.mobjects], rt=0.5)
        slots = [np.array([-4.6, 0.5, 0]), np.array([0.0, 0.5, 0]), np.array([4.6, 0.5, 0])]
        # (1) hierarchical partition: the decoded path, level by level
        c1 = np.array([-1.8, 0.3, 0])
        cube, boxes, dot, t = cube_path(PATH, U, 1.55, c1, r"q:\ \text{chart } " + str(CH) + r",\ " + ",".join(map(str, PATH)))
        b.play(ShowCreation(cube), FadeIn(t), rt=0.6)
        for l in range(5):
            b.play(ShowCreation(boxes[l]), boxes[:l].animate.set_stroke(opacity=0.35), rt=0.45)
        b.play(GrowFromCenter(dot), rt=0.3)
        # multi-path: the symmetric equivalent q s (s = half turn) has its own path
        c2 = np.array([3.2, 0.3, 0])
        cube2, boxes2, dot2, t2 = cube_path(PATHS, US, 1.2, c2, r"q\,s:\ \text{chart } " + str(CHS) + r",\ " + ",".join(map(str, PATHS)))
        mp = lbl("multi-path targets: every equivalent path", 22, C_GT).move_to([0.6, 3.4, 0])
        b.play(ShowCreation(cube2), FadeIn(t2), FadeIn(mp), LaggedStart(*[ShowCreation(x) for x in boxes2], lag_ratio=0.2),
               GrowFromCenter(dot2), rt=1.4)
        g1 = VGroup(cube, boxes, dot, t, cube2, boxes2, dot2, t2)
        cap1 = lbl("hierarchical choice", 22, LEVEL_COLORS[0])
        b.play(g1.animate.set_width(4.1).move_to(slots[0]), FadeOut(mp), rt=0.8)
        cap1.move_to([slots[0][0], -1.55, 0])
        b.play(FadeIn(cap1), rt=0.3)
        # (2) orbit ring of four rotated crops around the object
        ring_c = np.array([0.0, 0.4, 0]); rr = 1.65
        crops = Group()
        for k in range(4):
            im = ImageMobject(str(ASSETS / "s14_lightbox.png")).set_height(1.2)
            im.rotate(k * PI / 2).move_to(ring_c + rr * np.array([np.cos(PI / 2 - k * PI / 2), np.sin(PI / 2 - k * PI / 2), 0]))
            crops.add(im)
        circ = Circle(radius=rr).move_to(ring_c).set_stroke(C_PRED, 2, 0.6)
        angs = VGroup(*[Tex(f"{90 * k}^\\circ", font_size=24, color=C_PRED).next_to(crops[k], UR, 0.02)
                        for k in range(4)])
        b.play(ShowCreation(circ), LaggedStart(*[FadeIn(c, scale=0.6) for c in crops], lag_ratio=0.25), FadeIn(angs), rt=1.3)
        vote = VGroup(*[vbar(ring_c[0] - 0.45 + 0.3 * i, ring_c[1] - 0.45, h, 0.22, c) for i, (h, c) in
                        enumerate([(1.0, C_GT), (0.4, C_FLIP), (0.28, C_PRED), (0.22, C_PRED)])])
        vl = lbl("vote", 18, GREY_A).next_to(vote, DOWN, 0.08)
        b.play(*[grow_up(v) for v in vote], FadeIn(vl), *[Indicate(c, scale_factor=1.06, color=C_PRED) for c in crops], rt=1.2)
        cost = lbl("look-alikes: orbit voting, 4x cost", 22, C_PRED).move_to([0, 3.4, 0])
        b.play(FadeIn(cost), rt=0.5)
        g2 = Group(crops, circ, angs, vote, vl)
        cap2 = lbl("orbit voting", 22, C_PRED)
        b.play(g2.animate.set_height(3.0).move_to(slots[1]), FadeOut(cost), rt=0.8)
        cap2.move_to([slots[1][0], -1.55, 0])
        b.play(FadeIn(cap2), rt=0.3)
        # (3) one frozen backbone, many small heads
        bb = lattice(8, 8, 1.5, 1.6, color=GREY_B, dot_r=0.025, stroke=0.8, opacity=0.45).move_to([3.35, 0.5, 0])
        lk_body = RoundedRectangle(width=0.34, height=0.26, corner_radius=0.04).set_fill(GREY_B, 1).set_stroke(width=0)
        lk = VGroup(Arc(0, PI, radius=0.11).set_stroke(GREY_B, 4).next_to(lk_body, UP, 0), lk_body).next_to(bb, UP, 0.12)
        fz = lbl("DINOv3 + MoGe-2", 18, GREY_A).next_to(bb, DOWN, 0.15)
        b.play(FadeIn(bb, lag_ratio=0.02), FadeIn(lk), FadeIn(fz), rt=0.8)
        heads, lines = Group(), VGroup()
        n = 12
        for i in range(n):
            a = (i - (n - 1) / 2) * 0.2
            p = bb.get_right() + 1.75 * np.array([np.cos(a), np.sin(a), 0]) + 0.2 * RIGHT
            c = [C_PRED, C_DINO, C_GT, C_POE, C_MOGE, C_FT][i % 6]
            h = lattice(3, 2, 0.2, 0.24, color=c, dot_r=0.025, stroke=0.8, opacity=0.7).move_to(p)
            heads.add(h); lines.add(Line(bb.get_right(), h.get_left(), buff=0.04).set_stroke(c, 1.6, 0.8))
        mins = lbl("a small head per object, trained in minutes", 22, C_PRED).move_to([0.4, 3.4, 0])
        b.play(LaggedStart(*[ShowCreation(l) for l in lines], lag_ratio=0.06),
               LaggedStart(*[FadeIn(h, scale=0.4) for h in heads], lag_ratio=0.06), FadeIn(mins), rt=1.4)
        g3 = Group(bb, lk, fz, heads, lines)
        cap3 = lbl("frozen backbones, small heads", 22, GREY_A)
        b.play(FadeOut(mins), rt=0.5)
        cap3.move_to([slots[2][0], -1.55, 0])
        b.play(FadeIn(cap3), rt=0.3)
        for g, c in [(g1, cap1), (g2, cap2), (g3, cap3)]:
            b.play(Indicate(c, color=WHITE), g.animate.scale(1.06), rt=0.4)
            b.play(g.animate.scale(1 / 1.06), rt=0.3)
        close = lbl("The right pose is usually there. Selecting it is the problem.", 30).move_to([0, -2.9, 0])
        b.play(FadeIn(close, shift=0.2 * UP), rt=1.0)
        b.wait(max(b.left - 0.3, 0))
        b.done()
