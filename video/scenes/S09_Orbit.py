"""S09: orbit voting (paper eq. orbit).
Computed: the four rolled views q_k = Exp(-theta_k d) (x) q of the true pose and of its 180-degree look-alike q F; the
schematic object's projection in each rolled crop; every candidate is a real leaf centre (decode(encode(.))) of a
perturbed pose in its view and is mapped back exactly with c = Exp(+theta_k d) (x) c_k, along the true roll path; the
axis-angle coordinates of every point. Which poses are in each beam and all per-view log-probabilities are illustrative
(labelled so)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa: F401,F403,E402


def Tx(*a, color=None, **k):
    """Tex with a colour (the color kwarg is ignored by this manimgl build)."""
    t = Tex(*a, **k)
    return t.set_color(color) if color is not None else t
import itertools  # noqa: E402

SC = "S09_Orbit"
D = np.array([0.0, 0.0, 1.0])                              # ray through the crop centre (camera frame)
QT = np.array([-0.697, 0.6974, -0.1167, 0.1195]); QT = QT / np.linalg.norm(QT)
FLIP = quat([1, 0, 0], np.pi)                               # 180 deg about the body x axis (look-alike)
QF = qmul(QT, FLIP)
THETAS = [0, 90, 180, 270]
VIEW_COLS = ["#4CC9F0", "#5E9BFF", "#8C7BFF", "#B388FF"]
BOXES = [(np.array([0, 0, 0]), np.array([0.5, 0.36, 0.34])),
         (np.array([1.05, 0, 0]), np.array([0.52, 0.03, 0.3])),
         (np.array([-1.05, 0, 0]), np.array([0.52, 0.03, 0.3])),
         (np.array([0.0, -0.36 - 0.18, 0.12]), np.array([0.05, 0.18, 0.05]))]
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]


def roll(q, deg):
    """label of the view whose crop is rotated by deg (cv2 sense): Exp(-theta d) (x) q, sign-normalised."""
    r = qmul(quat(D, -np.radians(deg)), q); return r * (1 if r[0] >= 0 else -1)


def unroll(q, deg):
    r = qmul(quat(D, np.radians(deg)), q); return r * (1 if r[0] >= 0 else -1)


def leaf(q):
    c, p = encode(q); r = centre(c, p); return r * (1 if r[0] >= 0 else -1)


def rotvec(q):
    q = q * (1 if q[0] >= 0 else -1); a = 2 * np.arccos(np.clip(q[0], -1, 1)); n = np.linalg.norm(q[1:])
    return np.zeros(3) if n < 1e-9 else q[1:] / n * a


def perturb(q, sig_deg, rng):
    ax = rng.normal(size=3); return qmul(quat(ax, np.radians(abs(rng.normal(0, sig_deg)))), q)


# -------- illustrative beams: 8 real leaves per view, around the true pose and around the look-alike
rng = np.random.default_rng(4)
N_TRUE = [5, 5, 3, 5]                                       # in the 180-degree view the look-alike leads the beam
CANDS = []                                                  # (view k, kind, c_k in view frame, c back in common frame)
for k, th in enumerate(THETAS):
    for i in range(8):
        kind = "T" if i < N_TRUE[k] else "F"
        base = roll(QT if kind == "T" else QF, th)
        ck = leaf(perturb(base, 4.0 + 3.0 * (i % 3), rng))
        CANDS.append((k, kind, ck, unroll(ck, th)))


def proj_segments(q):
    R = qmat(q); segs = []
    for c, h in BOXES:
        P = np.array([c + h * np.array(s) for s in itertools.product([-1, 1], repeat=3)]) @ R.T
        segs += [(P[i], P[j]) for i, j in EDGES]
    return segs


BOX_COLS = [GREY_B, "#3A6EA5", "#3A6EA5", GREY_A]
LIGHT = np.array([-0.3, -0.6, -0.75]); LIGHT = LIGHT / np.linalg.norm(LIGHT)
FACE_IDX = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]


def crop_drawing(q, centre_pt, size, col=None, fill=GREY_E, border=GREY_B, tint=1.0):
    """opaque orthographic drawing of the schematic object in a square crop (camera frame: x right, y down, z forward)."""
    sq = Square(size).move_to(centre_pt).set_stroke(border, 2.5).set_fill(fill, 0.85)
    s = size / 3.1; R = qmat(q); items = []
    for (c, h), bc in zip(BOXES, BOX_COLS):
        P = np.array([c + h * np.array(sg) for sg in itertools.product([-1, 1], repeat=3)]) @ R.T
        for f in FACE_IDX:
            pts = P[list(f)]
            n = np.cross(pts[1] - pts[0], pts[2] - pts[0]); n = n / (np.linalg.norm(n) + 1e-9)
            if np.dot(n, pts.mean(0) - R @ c) < 0:
                n = -n
            if n[2] >= -1e-6:                       # facing away from the camera
                continue
            shade = (0.3 + 0.7 * max(np.dot(n, LIGHT), 0)) * tint
            poly = Polygon(*[centre_pt + s * np.array([p[0], -p[1], 0]) for p in pts])
            poly.set_fill(interpolate_color(BLACK, bc, shade), 1.0).set_stroke(WHITE, 0.6, 0.4)
            items.append((-pts[:, 2].mean(), poly))
    items.sort(key=lambda t: t[0])
    return VGroup(sq, VGroup(*[m for _, m in items]))


class Ball:
    """orthographic drawing of the axis-angle ball (radius pi) with a slowly turning view."""

    def __init__(self, centre_pt, radius, spin, move=None):
        self.c0, self.r, self.spin, self.move = np.asarray(centre_pt, float), radius, spin, move

    @property
    def c(self):
        if self.move is None:
            return self.c0
        tgt, tr = self.move
        return interpolate(self.c0, tgt, smooth(tr.get_value()))

    def V(self):
        a = self.spin.get_value(); tilt = np.radians(-18)
        Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
        return Rx @ Rz

    def pt(self, rv):
        p = self.V() @ (np.asarray(rv) / np.pi) * self.r
        return self.c + np.array([p[0], p[2], 0])

    def shell(self):
        g = VGroup(Circle(radius=self.r).move_to(self.c).set_stroke(GREY_B, 1.5).set_fill(BLUE_E, 0.12))
        th = np.linspace(0, TAU, 60)
        eq = [self.pt(np.pi * np.array([np.cos(t), np.sin(t), 0])) for t in th]
        g.add(VMobject().set_points_as_corners(eq).set_stroke(GREY_C, 1, 0.6))
        return g


class S09_Orbit(Scene):
    default_camera_config = dict(background_color=BG)

    def construct(self):
        spin = ValueTracker(0.4)
        spin.add_updater(lambda m, dt: m.increment_value(0.12 * dt))
        self.add(spin)
        # ring layout
        CROP = {0: np.array([-0.9, 2.55, 0]), 1: np.array([4.2, -0.15, 0]), 2: np.array([-0.9, -2.75, 0]), 3: np.array([-4.2, -0.15, 0])}
        BALLC = {0: np.array([0.9, 2.55, 0]), 1: np.array([5.95, -0.15, 0]), 2: np.array([0.9, -2.75, 0]),
                 3: np.array([-5.95, -0.15, 0])}
        CS = 1.45
        balls = {k: Ball(BALLC[k], 0.72, spin) for k in range(4)}

        # ------------------------------------------------ b1: four rotated crops, beam-8 in each
        b = Beat(self, f"{SC}_b1")
        main = crop_drawing(QT, np.array([0, -0.1, 0]), 2.1)
        sch = Text("schematic object", font_size=18).set_color(GREY_B).next_to(main, DOWN, buff=0.08)
        b.play(FadeIn(main), FadeIn(sch), frac=0.08)
        tl = VGroup(*[Tx(f"{th}^\\circ", font_size=28, color=VIEW_COLS[k]).next_to(Square(CS).move_to(CROP[k]),
                                                                                  UP if k != 2 else DOWN, buff=0.08)
                      for k, th in enumerate(THETAS)])
        # each view is the main crop turned by theta_k (counter-clockwise): copies fly out, then turn
        flyers = [main.copy() for _ in range(4)]
        b.play(*[flyers[k].animate.scale(CS / 2.1).move_to(CROP[k]) for k in range(4)], frac=0.16)
        b.play(*[Rotate(flyers[k], np.radians(THETAS[k])) for k in range(4)],
               *[flyers[k][0].animate.set_stroke(VIEW_COLS[k], 3) for k in range(4)], FadeIn(tl), frac=0.2)
        crops = VGroup(*flyers)
        shells = VGroup(*[balls[k].shell() for k in range(4)])
        shells_live = [always_redraw(lambda k=k: balls[k].shell()) for k in range(4)]
        bl = Text("beam search: 8 leaves per view", font_size=22).set_color(C_PRED).move_to([3.9, 2.95, 0])
        b.play(*[FadeIn(s) for s in shells_live], FadeIn(bl), frac=0.1)
        pts = []
        for j, (k, kind, ck, cb) in enumerate(CANDS):
            col = C_GT if kind == "T" else C_FLIP
            d = always_redraw(lambda k=k, ck=ck, col=col: Dot(balls[k].pt(rotvec(ck)), radius=0.045, fill_color=col))
            pts.append(d)
        for k in range(4):
            seg = [pts[j] for j in range(len(CANDS)) if CANDS[j][0] == k]
            b.play(LaggedStart(*[FadeIn(p, scale=3) for p in seg], lag_ratio=0.25), rt=0.085 * b.T)
        il = Text("which leaves enter each beam: illustrative", font_size=18).set_color(GREY_B).to_corner(DR, buff=0.2)
        b.play(FadeIn(il), frac=0.05)
        b.done()

        # ------------------------------------------------ b2: map back through the inverse roll -> 32 candidates
        b = Beat(self, f"{SC}_b2")
        shift_t = ValueTracker(0.0)
        big = Ball(np.array([0, -0.1, 0]), 1.7, spin, move=(np.array([-4.6, -0.6, 0]), shift_t))
        big_live = always_redraw(big.shell)
        eqb = Tx(r"c=\mathrm{Exp}(+\theta_k d)\otimes c_k", font_size=36).to_corner(UL, buff=0.3)
        b.play(FadeOut(main), FadeOut(sch), FadeIn(big_live), Write(eqb), FadeOut(bl), frac=0.2)
        s_t = ValueTracker(0.0)
        movers = []
        for j, (k, kind, ck, cb) in enumerate(CANDS):
            col = C_GT if kind == "T" else C_FLIP

            def mk(k=k, ck=ck, col=col):
                s = s_t.get_value()
                q = qmul(quat(D, np.radians(THETAS[k]) * s), ck)          # along the exact inverse roll
                p_small = balls[k].pt(rotvec(q)); p_big = big.pt(rotvec(q))
                w = smooth(min(s * 1.6, 1.0))
                return Dot(interpolate(p_small, p_big, w), radius=0.045 + 0.02 * w, fill_color=col)
            movers.append(always_redraw(mk))
        for p in pts:
            p.clear_updaters()
        self.remove(*pts); self.add(*movers)
        b.play(s_t.animate.set_value(1.0), frac=0.5, rate_func=linear)
        n32 = Text("32 candidates, one frame", font_size=24).set_color(WHITE).next_to(eqb, DOWN, aligned_edge=LEFT, buff=0.2)
        mt = Dot(big.pt(rotvec(QT)), radius=0.1, fill_color=C_GT)
        b.play(FadeIn(n32), FadeOut(VGroup(*shells_live)), FadeOut(crops),
               FadeOut(tl), frac=0.2)
        b.done()

        # ------------------------------------------------ b3: score every candidate in every view
        b = Beat(self, f"{SC}_b3")
        eq = Tx(r"s(c)=\sum_{k=1}^{K}\log p_k\big(\mathrm{path}(\mathrm{Exp}(-\theta_k d)\otimes c)\big)", font_size=40)
        eq.to_edge(UP, buff=0.25).shift(RIGHT * 1.6)
        kk = Tx(r"K=4", font_size=32, color=GREY_A).next_to(eq, DOWN, buff=0.12)
        # move the ball to the left
        b.play(FadeOut(eqb), FadeOut(n32), Write(eq), FadeIn(kk), shift_t.animate.set_value(1.0), frac=0.16)
        # six representative candidates: the three true-cluster leaves closest to q and three look-alikes
        idx_t = sorted([j for j in range(32) if CANDS[j][1] == "T"], key=lambda j: geodesic_deg(CANDS[j][3], QT))[:3]
        idx_f = sorted([j for j in range(32) if CANDS[j][1] == "F"], key=lambda j: geodesic_deg(CANDS[j][3], QF))[:3]
        sel = [idx_t[1], idx_f[0], idx_t[0], idx_f[1], idx_t[2], idx_f[2]]
        # illustrative per-view log-probabilities: true-pose candidates are consistent; look-alikes peak in the 180 view
        LP = np.zeros((6, 4))
        r2 = np.random.default_rng(9)
        for i, j in enumerate(sel):
            kind = CANDS[j][1]
            err = geodesic_deg(CANDS[j][3], QT if kind == "T" else QF)
            for v in range(4):
                if kind == "T":
                    LP[i, v] = -(2.2 + 0.25 * err + r2.uniform(0, 1.4))
                else:
                    LP[i, v] = -(1.4 + 0.2 * err + r2.uniform(0, 0.4)) if v == 2 else -(7.5 + r2.uniform(0, 2.5))
        S = LP.sum(1)
        XB, ZY, SC_ = -0.95, 1.75, 0.105
        cols_x = [XB + 0.98 * i for i in range(6)]
        zero = Line([XB - 0.5, ZY, 0], [cols_x[-1] + 0.5, ZY, 0], color=GREY_B, stroke_width=2)
        clabs = VGroup(*[Tx(f"c_{i + 1}", font_size=30, color=C_GT if CANDS[j][1] == "T" else C_FLIP)
                         .move_to([cols_x[i], ZY + 0.3, 0]) for i, j in enumerate(sel)])
        # mark the six in the ball
        mtex = [Tx(f"c_{i + 1}", font_size=22, color=WHITE) for i in range(6)]

        def marks_f():
            g = VGroup()
            for i, j in enumerate(sel):
                p = big.pt(rotvec(CANDS[j][3]))
                g.add(mtex[i].copy().move_to(p + (UP if i % 2 == 0 else DOWN) * 0.22 + RIGHT * 0.2))
            return g
        marks = always_redraw(marks_f)
        b.play(ShowCreation(zero), FadeIn(clabs), frac=0.08)
        b.play(FadeOut(il), frac=0.06)
        segs = [[None] * 4 for _ in range(6)]
        vl = VGroup(*[Tx(r"\text{view } " + f"{THETAS[v]}^\\circ", font_size=24, color=VIEW_COLS[v]) for v in range(4)])
        vl.arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([cols_x[-1] + 1.55, ZY - 0.8, 0])
        ill = Text("log-probabilities\nillustrative", font_size=18).set_color(GREY_B).move_to([cols_x[-1] + 1.5, ZY - 2.0, 0])
        tops = [ZY] * 6
        for v in range(4):
            rects = []
            for i in range(6):
                h = -LP[i, v] * SC_
                r = Rectangle(width=0.55, height=h).set_stroke(BG, 1).set_fill(VIEW_COLS[v], 0.95)
                r.move_to([cols_x[i], tops[i], 0], aligned_edge=UP)
                tops[i] -= h; segs[i][v] = r; rects.append(r)
            b.play(*[GrowFromEdge(r, UP) for r in rects], FadeIn(vl[v]), *([FadeIn(ill)] if v == 0 else []), frac=0.155)
        b.done()

        # ------------------------------------------------ b4: the look-alike wins one view, the true pose wins the sum
        b = Beat(self, f"{SC}_b4")
        fi = 1   # c_2 is a look-alike
        win1 = SurroundingRectangle(VGroup(*[segs[i][2] for i in range(6)]), buff=0.06, color=WHITE)
        t1 = Text("look-alike wins\nthe 180° view", font_size=22).set_color(C_FLIP).move_to([cols_x[-1] + 1.6, -2.3, 0])
        b.play(ShowCreation(win1), *[segs[i][v].animate.set_opacity(0.25) for i in range(6) for v in (0, 1, 3)], FadeIn(t1), frac=0.15)
        b.play(Indicate(segs[fi][2], color=C_FLIP), frac=0.08)
        t2 = Text("but unlikely in the\nother three views", font_size=22).set_color(C_FLIP).move_to([cols_x[-1] + 1.6, -2.3, 0])
        b.play(FadeOut(win1), *[segs[i][v].animate.set_opacity(0.95) for i in range(6) for v in (0, 1, 3)],
               *[segs[i][2].animate.set_opacity(0.95) for i in range(6)],
               *[segs[i][v].animate.set_stroke(WHITE, 2) for i in (1, 3, 5) for v in (0, 1, 3)],
               Transform(t1, t2), frac=0.15)
        totals = VGroup(*[DecimalNumber(S[i], num_decimal_places=1, font_size=24).next_to(segs[i][3], DOWN, buff=0.08)
                          for i in range(6)])
        b.play(FadeIn(totals), frac=0.1)
        w = int(np.argmax(S))
        box = SurroundingRectangle(VGroup(clabs[w], segs[w][3], totals[w]), buff=0.08, color=C_GT, stroke_width=4)
        am = Tx(r"\hat q=\arg\max_c s(c)", font_size=36, color=C_GT).move_to([cols_x[-1] + 1.45, -1.2, 0])
        jw = sel[w]
        star = always_redraw(lambda: Dot(big.pt(rotvec(CANDS[jw][3])), radius=0.14, fill_color=C_GT).set_stroke(WHITE, 2))
        b.play(ShowCreation(box), Write(am), FadeIn(star, scale=3), FadeOut(t1), frac=0.17)
        # cost: four backbone + tree passes instead of one
        icons = VGroup(*[crop_drawing(roll(QT, th), ORIGIN, 0.62).set_stroke(width=1) for th in THETAS]).arrange(RIGHT, buff=0.12)
        icons.move_to([-0.6, -3.4, 0])
        cost = Text("cost: 4 passes instead of 1", font_size=26).set_color(WHITE).next_to(icons, RIGHT, buff=0.3)
        b.play(LaggedStart(*[FadeIn(i, scale=0.5) for i in icons], lag_ratio=0.2), FadeIn(cost), frac=0.2)
        b.done()
