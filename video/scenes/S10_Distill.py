"""S10: orbit distillation (paper eq. distill, beta = 1).
Computed: three training views of one image (clean and two rolled copies, theta from the U(-180,180) augmentation range);
8 real leaf centres per view mapped back exactly to the clean frame with Exp(+theta d) (x) c_k; the exact GT soft targets
of the clean view (tau from phase3.TAU_DEG); the teacher's level marginals along the GT prefix from the 24 weighted
candidates; and the per-level minimiser of the loss at the GT prefix, (t + (1/(L+1)) sum_j w_j e_child_j) / (1 + (1/(L+1))
sum_j w_j), since L_GT sums the cross-entropies of the L+1 levels (scripts/train_consistency.py). The orbit scores s(c_j)
(hence the weights) and the student's initial distributions are illustrative."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa: F401,F403,E402


def Tx(*a, color=None, **k):
    """Tex with a colour (the color kwarg is ignored by this manimgl build)."""
    t = Tex(*a, **k)
    return t.set_color(color) if color is not None else t
import itertools  # noqa: E402

SC = "S10_Distill"
D = np.array([0.0, 0.0, 1.0])
TAUS = [39.27, 39.27, 19.95, 10.01, 5.01, 2.51]
QT = np.array([0.6605, 0.6182, -0.1147, 0.4104]); QT = QT / np.linalg.norm(QT)
FLIP = quat([1, 0, 0], np.pi); QF = qmul(QT, FLIP)
VIEWS = [0.0, 37.0, -112.0]                         # clean + two augmented rolls (example angles)
VIEW_NAMES = ["clean", "augmented", "augmented"]
VIEW_COLS = ["#4CC9F0", "#5E9BFF", "#8C7BFF"]
BOXES = [(np.array([0, 0, 0]), np.array([0.5, 0.36, 0.34])),
         (np.array([1.05, 0, 0]), np.array([0.52, 0.03, 0.3])),
         (np.array([-1.05, 0, 0]), np.array([0.52, 0.03, 0.3])),
         (np.array([0.0, -0.54, 0.12]), np.array([0.05, 0.18, 0.05]))]
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]


def norm_w(q):
    return q * (1 if q[0] >= 0 else -1)


def roll(q, deg):
    return norm_w(qmul(quat(D, -np.radians(deg)), q))


def unroll(q, deg):
    return norm_w(qmul(quat(D, np.radians(deg)), q))


def leaf(q):
    c, p = encode(q); return norm_w(centre(c, p))


def rotvec(q):
    q = norm_w(q); a = 2 * np.arccos(np.clip(q[0], -1, 1)); n = np.linalg.norm(q[1:])
    return np.zeros(3) if n < 1e-9 else q[1:] / n * a


def perturb(q, sig_deg, rng):
    return qmul(quat(rng.normal(size=3), np.radians(abs(rng.normal(0, sig_deg)))), q)


rng = np.random.default_rng(11)
CANDS = []          # (view, kind, c_k, c in clean frame, illustrative score)
for v, th in enumerate(VIEWS):
    for i in range(8):
        kind = "T" if i < (6 if v != 2 else 4) else "F"
        ck = leaf(perturb(roll(QT if kind == "T" else QF, th), 2.0 + 1.5 * (i % 3), rng))
        c = unroll(ck, th)
        err = geodesic_deg(c, QT if kind == "T" else QF)
        s = -(9.0 + 0.6 * err + rng.uniform(0, 1.0)) if kind == "T" else -(13.5 + rng.uniform(0, 2.5))
        CANDS.append((v, kind, ck, c, s))
S = np.array([c[4] for c in CANDS]); WTS = np.exp(S - S.max()); WTS = WTS / WTS.sum()

# exact GT soft targets of the clean view and the teacher's level marginals along the GT prefix
CH, PATH = encode(QT)
TRUE = [CH] + PATH


def gt_targets():
    out = []
    cands = [centre(k, []) for k in range(4)]
    for l in range(6):
        if l > 0:
            cands = [centre(CH, PATH[:l - 1] + [k]) for k in range(8)]
        d = np.array([geodesic_deg(QT, x) for x in cands]); t = np.exp(-(d / TAUS[l]) ** 2); out.append(t / t.sum())
    return out


GT_T = gt_targets()
CPATHS = [encode(c[3]) for c in CANDS]


def teacher_mass():
    """per level: sum of w_j over candidates whose prefix matches the GT prefix, binned by the candidate's next choice."""
    out = []
    for l in range(6):
        m = np.zeros(4 if l == 0 else 8)
        for (c, p), w in zip(CPATHS, WTS):
            seq = [c] + p
            if seq[:l] == TRUE[:l]:
                m[seq[l]] += w
        out.append(m)
    return out


TM = teacher_mass()
L1 = 6.0                                           # L + 1 levels
OPT = [(GT_T[l] + TM[l] / L1) / (1 + TM[l].sum() / L1) for l in range(6)]


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


def ball_pt(c, r, rv, spin):
    a = spin; tilt = np.radians(-18)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    p = Rx @ Rz @ (np.asarray(rv) / np.pi) * r
    return c + np.array([p[0], p[2], 0])


class S10_Distill(Scene):
    default_camera_config = dict(background_color=BG)

    def construct(self):
        spin = ValueTracker(0.6)
        spin.add_updater(lambda m, dt: m.increment_value(0.12 * dt))
        self.add(spin)

        # ------------------------------------------------ b1: the orbit teacher over three training views
        b = Beat(self, f"{SC}_b1")
        CY = [2.35, 0.0, -2.35]
        crops = VGroup()
        looks = [(1.0, GREY_E), (1.45, "#3A372C"), (0.55, "#0B0F14")]       # photometric changes (schematic)
        for v, th in enumerate(VIEWS):
            crops.add(crop_drawing(roll(QT, th), np.array([-5.7, CY[v], 0]), 1.75, tint=looks[v][0], fill=looks[v][1],
                                   border=VIEW_COLS[v]))
        labs = VGroup(*[Tx(r"\text{" + VIEW_NAMES[v] + "}" + (r",\ \theta=" + f"{VIEWS[v]:.0f}" + r"^\circ" if v else ""),
                            font_size=24, color=VIEW_COLS[v]).next_to(crops[v], RIGHT, buff=0.15).shift(UP * 0.62)
                        for v in range(3)])
        b.play(LaggedStart(*[FadeIn(c, shift=RIGHT * 0.3) for c in crops], lag_ratio=0.3), FadeIn(labs), frac=0.14)
        BC, BR = np.array([-1.1, -0.35, 0]), 1.55
        shell = always_redraw(lambda: VGroup(Circle(radius=BR).move_to(BC).set_stroke(GREY_B, 1.5).set_fill(BLUE_E, 0.12),
                                             VMobject().set_points_as_corners(
                                                 [ball_pt(BC, BR, np.pi * np.array([np.cos(t), np.sin(t), 0]), spin.get_value())
                                                  for t in np.linspace(0, TAU, 60)]).set_stroke(GREY_C, 1, 0.6)))
        bl = Tx(r"24 \text{ candidates, clean frame}", font_size=26).next_to(BC + UP * BR, UP, buff=0.15)
        s_t = ValueTracker(0.0)

        def mover(j):
            v, kind, ck, c, s = CANDS[j]
            col = C_GT if kind == "T" else C_FLIP

            def f():
                a = s_t.get_value()
                q = qmul(quat(D, np.radians(VIEWS[v]) * a), ck)
                start = crops[v].get_center(); end = ball_pt(BC, BR, rotvec(q), spin.get_value())
                return Dot(interpolate(start, end, smooth(min(1.0, a * 1.5))), radius=0.05, fill_color=col)
            return always_redraw(f)

        movers = [mover(j) for j in range(24)]
        b.play(FadeIn(shell), FadeIn(bl), *[FadeIn(m) for m in movers], frac=0.08)
        b.play(s_t.animate.set_value(1.0), frac=0.3, rate_func=linear)
        # weights from the orbit scores
        XB, YB, step = 1.35, -1.6, 0.225
        order = list(np.argsort(-WTS))
        wb = VGroup()
        for r_, j in enumerate(order):
            col = C_GT if CANDS[j][1] == "T" else C_FLIP
            h = max(WTS[j] / WTS.max() * 3.0, 0.02)
            wb.add(Rectangle(width=0.16, height=h).set_stroke(VIEW_COLS[CANDS[j][0]], 1.5).set_fill(col, 0.9)
                   .move_to([XB + r_ * step, YB, 0], aligned_edge=DOWN))
        base = Line([XB - 0.2, YB, 0], [XB + 23 * step + 0.2, YB, 0], color=GREY_B, stroke_width=1.5)
        wf = Tx(r"w_j=\mathrm{softmax}_j\, s(c_j)", font_size=40).move_to([XB + 11.5 * step, 2.55, 0])
        ill = Text("orbit scores illustrative", font_size=18).set_color(GREY_B).next_to(base, DOWN, buff=0.15)
        b.play(Write(wf), ShowCreation(base), frac=0.14)
        b.play(LaggedStart(*[GrowFromEdge(r, DOWN) for r in wb], lag_ratio=0.04), FadeIn(ill), frac=0.2)
        b.done()

        # ------------------------------------------------ b2: a single-view student learns the level marginals + GT
        b = Beat(self, f"{SC}_b2")
        for m in movers:
            m.clear_updaters()
        eq = Tx(r"\begin{aligned}\mathcal{L}=\;&\mathcal{L}_{\text{GT}}(v)\\&-\frac{\beta}{L{+}1}\sum_j w_j\sum_{\ell=0}^{L}"
                 r"\log p^{(\ell)}\big(\mathrm{pre}_\ell(\mathrm{Exp}(-\theta_v d)\otimes c_j)\big)\end{aligned}", font_size=36)
        eq.to_corner(UL, buff=0.3)
        beta = Tx(r"\beta=1", font_size=32, color=GREY_A).next_to(eq, RIGHT, buff=0.4).align_to(eq, DOWN)
        b.play(*[FadeOut(m) for m in [crops, labs, shell, bl, wb, base, wf, ill, *movers]], Write(eq), FadeIn(beta), frac=0.2)
        # student level distributions (initial values illustrative), GT target (yellow), teacher marginal (purple)
        widths = [0.22] * 6
        starts = [-6.6]
        for l in range(1, 6):
            starts.append(starts[-1] + (4 if l == 1 else 8) * 0.26 + 0.38)
        YB2, HS = -2.3, 2.6
        r2 = np.random.default_rng(3)
        init = [np.ones(4 if l == 0 else 8) + r2.uniform(0, 1.0, 4 if l == 0 else 8) for l in range(6)]
        init = [x / x.sum() for x in init]

        def bars_for(vals, l, kind):
            g = VGroup()
            for k, v in enumerate(vals):
                x = starts[l] + k * 0.26
                h = max(v * HS, 0.015)
                r = Rectangle(width=0.2, height=h).move_to([x, YB2, 0], aligned_edge=DOWN)
                if kind == "student":
                    r.set_stroke(width=0).set_fill(C_PRED, 0.9)
                elif kind == "gt":
                    r.set_stroke(C_GT, 2).set_fill(opacity=0)
                else:
                    r.set_stroke(C_POE, 2).set_fill(opacity=0)
                g.add(r)
            return g

        stud = VGroup(*[bars_for(init[l], l, "student") for l in range(6)])
        heads = VGroup(*[Tx(r"\text{root}" if l == 0 else r"\ell=" + str(l), font_size=24, color=LEVEL_COLORS[l])
                         .move_to([starts[l] + (1.5 if l == 0 else 3.5) * 0.26, YB2 - 0.3, 0]) for l in range(6)])
        floor = Line([-6.8, YB2, 0], [6.8, YB2, 0], color=GREY_B, stroke_width=1.5)
        b.play(FadeIn(stud), FadeIn(heads), ShowCreation(floor), frac=0.12)
        gt = VGroup(*[bars_for(GT_T[l], l, "gt") for l in range(6)])
        tm = VGroup(*[bars_for(TM[l] / max(TM[l].sum(), 1e-9), l, "t") for l in range(6)])
        key = VGroup(Text("GT soft target (exact)", font_size=18).set_color(C_GT),
                     Text("teacher level marginal (illustrative weights)", font_size=18).set_color(C_POE),
                     Text("student (start illustrative)", font_size=18).set_color(C_PRED)).arrange(RIGHT, buff=0.35)
        key.move_to([0, YB2 - 0.85, 0])
        b.play(FadeIn(gt), FadeIn(tm), FadeIn(key), frac=0.1)
        opt = VGroup(*[bars_for(OPT[l], l, "student") for l in range(6)])
        b.play(Transform(stud, opt), frac=0.3)
        # test time: one view, one pass
        test = crop_drawing(QT, np.array([5.4, 2.2, 0]), 1.6, border=C_PRED)
        tl = Text("test: 1 view", font_size=24).set_color(C_PRED).next_to(test, DOWN, buff=0.12)
        b.play(FadeIn(test, scale=0.8), FadeIn(tl), frac=0.14)
        b.done()
