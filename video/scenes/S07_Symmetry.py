"""S07: symmetric objects. Contradictory single labels, the discontinuous canonical label, and multi-path targets (eq. symloss).
Computed: a square tower with exact 4-fold symmetry S = {Rz(k 90 deg)} about its body axis; equivalents q s via qmul
(right multiplication; the rotated mesh coincides to 3e-16); the canonical member = argmax_s |w(q s)| (closest to the
identity) at every object angle, so the sawtooth and its 90 deg jumps are exact; the tree paths of all members from encode;
the root target averaged over the members with the code's tau_0 = 39.27 deg."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa: F401,F403,E402

SC = "S07_Symmetry"


def Tx(*a, color=None, **k):
    """Tex with a colour (the color kwarg is ignored by this manimgl build)."""
    t = Tex(*a, **k)
    return t.set_color(color) if color is not None else t
Q0 = quat([1, 0.3, 0], np.radians(28))                        # tilt about a horizontal axis
SYM = [quat([0, 0, 1], np.radians(90 * k)) for k in range(4)]
TAU0 = 39.27
LIGHT = np.array([-0.4, -0.8, 0.9]); LIGHT = LIGHT / np.linalg.norm(LIGHT)
MEMBER_COLS = [C_GT, "#FFE9A8", "#F6B44B", "#E8C547"]
NAMES = ["w", "x", "y", "z"]


def q_at(phi_deg):
    return qmul(Q0, quat([0, 0, 1], np.radians(phi_deg)))


def canonical(q):
    ms = [qmul(q, s) for s in SYM]
    k = int(np.argmax([abs(m[0]) for m in ms]))
    return k, ms[k]


def tower_faces(marker=False):
    """faces (list of 3D point arrays) of a square tower with a pyramid roof; exact C4 about +z."""
    h, a, top = 0.75, 0.5, 1.3
    c = [np.array([x, y, 0.0]) for x, y in [(-a, -a), (a, -a), (a, a), (-a, a)]]
    F = []
    for i in range(4):
        p, q = c[i], c[(i + 1) % 4]
        F.append(np.array([p - [0, 0, h], q - [0, 0, h], q + [0, 0, h], p + [0, 0, h]]))
        F.append(np.array([p + [0, 0, h], q + [0, 0, h], [0, 0, top]]))
    F.append(np.array([x - [0, 0, h] for x in c[::-1]]))
    if marker:   # an asymmetric stub on the +x face (breaks the symmetry)
        s = 0.2
        F.append(np.array([[-s, -a - 0.002, 0.1 - s], [s, -a - 0.002, 0.1 - s], [s, -a - 0.002, 0.1 + s], [-s, -a - 0.002, 0.1 + s]]))
    return F


def _rx(a):
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def _rz(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


VIEW = _rx(np.radians(-72)) @ _rz(np.radians(-25))          # orthographic view: world z up, looking slightly down


def to_screen(P, pos):
    v = np.atleast_2d(P) @ VIEW.T
    out = np.c_[v[:, 0], v[:, 1], np.zeros(len(v))] + pos
    return out, v[:, 2]


def tower(q, pos, scale=1.0, marker=False, base=GREY_B):
    """opaque orthographic drawing (back faces culled, painter's order); identical for equivalent rotations."""
    R = qmat(q); faces = tower_faces(marker); items = []
    for i, f in enumerate(faces):
        P = (f * scale) @ R.T
        n = np.cross(P[1] - P[0], P[2] - P[0]); n = n / (np.linalg.norm(n) + 1e-9)
        if (VIEW @ n)[2] <= 1e-6:
            continue
        sp, dep = to_screen(P, pos)
        shade = 0.35 + 0.65 * max(np.dot(n, LIGHT), 0)
        col = C_FLIP if (marker and i == len(faces) - 1) else base
        items.append((dep.mean() + (0.01 if marker and i == len(faces) - 1 else 0),
                      Polygon(*sp).set_fill(interpolate_color(BLACK, col, shade), 1.0).set_stroke(WHITE, 0.8, 0.5)))
    items.sort(key=lambda t: t[0])
    return VGroup(*[m for _, m in items])


def axes_of(q, pos, scale=1.0, col=C_GT, width=5, axis=True):
    R = qmat(q) @ np.eye(3); R = VIEW @ R; g = VGroup()
    ex, ey, ez = R[:, 0] * [1, 1, 0], R[:, 1] * [1, 1, 0], R[:, 2] * [1, 1, 0]
    g.add(Line(pos, pos + ex * 1.25 * scale, stroke_width=width, color=col))
    g.add(Dot(pos + ex * 1.25 * scale, radius=0.07 * scale, fill_color=col))
    g.add(Line(pos, pos + ey * 0.95 * scale, stroke_width=width * 0.55, color=col))
    if axis:
        g.add(DashedLine(pos - ez * 1.0 * scale, pos + ez * 1.9 * scale, stroke_width=1.5, color=WHITE).set_stroke(opacity=0.5))
    return g


class S07_Symmetry(Scene):
    default_camera_config = dict(background_color=BG)

    def construct(self):

        # ------------------------------------------------ b1: identical images, two different labels
        b = Beat(self, f"{SC}_b1")
        qa = q_at(20.0); qb = qmul(qa, SYM[1])
        PA, PB = np.array([-3.1, 0, 0]), np.array([3.1, 0, 0])
        ang = ValueTracker(-60.0)
        op = ValueTracker(0.0)
        obj_a = always_redraw(lambda: tower(q_at(20.0 + ang.get_value()), PA, 1.25).set_opacity(op.get_value()))
        self.add(obj_a)
        b.play(op.animate.set_value(1.0), ang.animate.set_value(0.0), frac=0.16)
        obj_a.clear_updaters()
        obj_b = tower(qb, PA, 1.25)          # drawn from q s: pixel-identical to the drawing from q
        self.add(obj_b)
        b.play(obj_b.animate.shift(PB - PA), frac=0.12)
        ax_a = axes_of(qa, PA, 1.25); ax_b = axes_of(qb, PB, 1.25)
        la = Tx("q", font_size=56, color=C_GT).move_to([-3.1, 2.7, 0]).fix_in_frame()
        lb = Tx(r"q\,s", font_size=56, color=C_GT).move_to([3.1, 2.7, 0]).fix_in_frame()
        sl = Tx(r"s = 90^\circ \text{ about the axis}", font_size=30, color=GREY_A).next_to(lb, DOWN, buff=0.15).fix_in_frame()
        b.play(ShowCreation(ax_a), FadeIn(la), frac=0.11)
        b.play(ShowCreation(ax_b), FadeIn(lb), FadeIn(sl), frac=0.11)
        img = Text("same image", font_size=30).set_color(WHITE).move_to([0, -2.55, 0]).fix_in_frame()
        # slide the right copy onto the left one: the shapes coincide exactly, the frames do not
        b.play(obj_b.animate.shift(PA - PB), ax_b.animate.shift(PA - PB), frac=0.16)
        b.play(FadeIn(img), Indicate(ax_b, color=C_FLIP), frac=0.1)
        b.play(obj_b.animate.shift(PB - PA), ax_b.animate.shift(PB - PA), frac=0.12)
        ct = Text("contradictory targets", font_size=40).set_color(C_FLIP).move_to([0, -3.3, 0]).fix_in_frame()
        b.play(Write(ct), frac=0.1)
        b.done()

        # ------------------------------------------------ b2: the canonical label jumps
        b = Beat(self, f"{SC}_b2")
        P0 = np.array([-3.6, -0.3, 0.0])
        phi = ValueTracker(0.0)
        b.play(*[FadeOut(m) for m in [obj_b, ax_b, ax_a, la, lb, sl, img, ct]], FadeOut(obj_a), frac=0.05)
        obj = always_redraw(lambda: tower(q_at(phi.get_value()), P0, 1.35))
        raw = always_redraw(lambda: axes_of(q_at(phi.get_value()), P0, 1.35, col=C_PRED, width=2.5, axis=False))
        can = always_redraw(lambda: axes_of(canonical(q_at(phi.get_value()))[1], P0, 1.35, col=C_GT, width=6))
        title = Text("canonical label: the member closest to the identity", font_size=28).set_color(C_GT)
        title.to_edge(UP, buff=0.3).fix_in_frame()
        key = VGroup(Text("annotated q (turns with the object)", font_size=20).set_color(C_PRED),
                     Text("canonical label", font_size=20).set_color(C_GT)).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        key.to_corner(DL, buff=0.35).fix_in_frame()
        b.play(FadeIn(obj), FadeIn(raw), FadeIn(can), Write(title), FadeIn(key), frac=0.1)
        # plot: canonical label's angle about the axis vs object angle (exact argmax over the 4 members)
        X0, X1, Y0, YS = 0.6, 6.5, -0.9, 1.9 / 60

        def px(p):
            return X0 + (X1 - X0) * p / 180.0

        def psi(p):
            k, _ = canonical(q_at(p))
            return ((p + 90 * k + 180) % 360) - 180

        axx = Line([X0, Y0 - 65 * YS, 0], [X1 + 0.2, Y0 - 65 * YS, 0], color=GREY_B, stroke_width=2)
        zl0 = Line([X0, Y0, 0], [X1, Y0, 0], color=GREY_D, stroke_width=1)
        axy = Line([X0, Y0 - 65 * YS, 0], [X0, Y0 + 65 * YS, 0], color=GREY_B, stroke_width=2)
        tks = VGroup()
        for v in [0, 45, 90, 135, 180]:
            tks.add(Tx(f"{v}^\\circ", font_size=22).move_to([px(v), Y0 - 65 * YS - 0.25, 0]))
            tks.add(DashedLine([px(v), Y0 - 60 * YS, 0], [px(v), Y0 + 60 * YS, 0], stroke_width=1, color=GREY_D))
        for v in [-45, 45]:
            tks.add(Tx(f"{v}^\\circ", font_size=22).next_to([X0, Y0 + v * YS, 0], LEFT, buff=0.1))
        xl = Text("object angle about its axis", font_size=20).set_color(GREY_A).next_to(axx, DOWN, buff=0.45)
        yl = Text("canonical label angle", font_size=20).set_color(C_GT).next_to(axy, UP, buff=0.12).align_to(axy, LEFT)
        plot = VGroup(axx, zl0, axy, tks, xl, yl).fix_in_frame()
        b.play(FadeIn(plot), frac=0.07)

        # exact breakpoints of the canonical choice (bisection on the argmax switch); psi is linear between them
        grid = np.linspace(0, 180, 721)
        ks = [canonical(q_at(p))[0] for p in grid]
        brk = []
        for i in range(720):
            if ks[i] != ks[i + 1]:
                lo, hi = grid[i], grid[i + 1]
                for _ in range(40):
                    mid = (lo + hi) / 2
                    if canonical(q_at(mid))[0] == ks[i]:
                        lo = mid
                    else:
                        hi = mid
                brk.append((lo + hi) / 2)
        pieces = list(zip([0.0] + brk, brk + [180.0]))

        def psi_k(x, k):
            return ((x + 90 * k + 180) % 360) - 180

        far = [0.0]

        def trace():
            far[0] = max(far[0], phi.get_value()); p_end = far[0]; g = VGroup()
            for a0, a1 in pieces:
                if a0 >= p_end:
                    break
                e = min(a1, p_end)
                k = canonical(q_at((a0 + e) / 2))[0]
                g.add(Line([px(a0), Y0 + psi_k(a0 + 1e-6, k) * YS, 0], [px(e), Y0 + psi_k(e - 1e-6, k) * YS, 0],
                           stroke_width=4, color=C_GT))
                if a1 < p_end:
                    k2 = canonical(q_at(a1 + 1e-6))[0]
                    g.add(DashedLine([px(a1), Y0 + psi_k(a1 - 1e-6, k) * YS, 0], [px(a1), Y0 + psi_k(a1 + 1e-6, k2) * YS, 0],
                                     stroke_width=3, color=C_FLIP))
            pc = phi.get_value()
            g.add(Dot([px(pc), Y0 + psi(pc) * YS, 0], radius=0.08, fill_color=C_GT))
            return g.fix_in_frame()

        tr = always_redraw(trace)
        self.add(tr)
        b.play(phi.animate.set_value(180.0), rate_func=linear, frac=0.42)
        # two nearly identical views either side of a switch
        b.play(phi.animate.set_value(44.0), frac=0.1)
        m44 = Dot([px(44), Y0 + psi(44) * YS, 0], radius=0.09, fill_color=WHITE).fix_in_frame()
        g44 = axes_of(canonical(q_at(44))[1], P0, 1.35, col=C_GT, width=3).set_opacity(0.45)
        self.add(m44, g44)
        b.play(phi.animate.set_value(46.0), frac=0.06)
        m46 = Dot([px(46), Y0 + psi(46) * YS, 0], radius=0.09, fill_color=WHITE).fix_in_frame()
        dv = 2.0
        dl = geodesic_deg(canonical(q_at(44))[1], canonical(q_at(46))[1])
        msg = VGroup(Tx(r"\text{views } 44^\circ \text{ and } 46^\circ\text{: } 2^\circ \text{ apart}", font_size=30),
                     Tx(r"\text{their labels: } " + f"{dl:.0f}" + r"^\circ \text{ apart}", font_size=30, color=C_FLIP))
        msg.arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([3.5, 2.0, 0]).fix_in_frame()
        b.play(FadeIn(m46), FadeIn(msg[0]), frac=0.08)
        b.play(FadeIn(msg[1]), m46.animate.scale(1.6).set_fill(C_FLIP), frac=0.08)
        b.done()

        # ------------------------------------------------ b3: multi-path targets
        b = Beat(self, f"{SC}_b3")
        for m in (obj, raw, can, tr):
            m.clear_updaters()
        b.play(*[FadeOut(m) for m in self.mobjects], frac=0.05)
        q = q_at(20.0)
        members = [qmul(q, s) for s in SYM]
        paths = [encode(m) for m in members]
        P1 = np.array([-4.7, -0.5, 0.0])
        body = tower(q, P1, 1.2)
        frames = VGroup(*[axes_of(m, P1, 1.2, col=MEMBER_COLS[k], width=5, axis=(k == 0)) for k, m in enumerate(members)])
        b.play(FadeIn(body), ShowCreation(frames[0]), frac=0.06)
        b.play(*[TransformFromCopy(frames[0], frames[k]) for k in range(1, 4)], frac=0.08)
        # tree lattice (fixed in frame): chart column + five columns of 8 children
        CX = [-1.6 + 1.5 * l for l in range(6)]
        RY = [1.25 - 0.47 * r for r in range(8)]
        CY4 = [RY[1] - 0.47 * 0.0, RY[3], RY[5], RY[7] - 0.0]
        CY4 = [1.25 - 0.47 * (2 * i + 0.5) for i in range(4)]

        def node(l, k):
            return np.array([CX[l], CY4[k] if l == 0 else RY[k], 0])

        nodes = VGroup()
        for l in range(6):
            for k in range(4 if l == 0 else 8):
                nodes.add(Dot(node(l, k), radius=0.06, fill_color=GREY_C))
        heads = VGroup(*[Tx(r"\text{chart}" if l == 0 else r"\ell=" + str(l), font_size=26, color=LEVEL_COLORS[l])
                         .move_to([CX[l], RY[0] + 0.45, 0]) for l in range(6)])
        clab = VGroup(*[Tx(NAMES[k], font_size=24, color=GREY_B).next_to(node(0, k), LEFT, buff=0.12) for k in range(4)])
        lat = VGroup(nodes, heads, clab).fix_in_frame()
        b.play(FadeIn(lat, lag_ratio=0.02), frac=0.07)

        def path_pts(cp, dx=0.0, dy=0.0):
            c, p = cp
            return [node(0, c) + [dx, dy, 0]] + [node(l + 1, p[l]) + [dx, dy, 0] for l in range(5)]

        plines = VGroup()
        for k, cp in enumerate(paths):
            off = (k - 1.5) * 0.035
            plines.add(VMobject().set_points_as_corners(path_pts(cp, 0, off)).set_stroke(MEMBER_COLS[k], 4).fix_in_frame())
        tags = VGroup(*[Tx(r"q\,s_{" + str(k) + "}", font_size=26, color=MEMBER_COLS[k]).next_to(path_pts(cp)[-1], RIGHT, buff=0.15)
                        .fix_in_frame() for k, cp in enumerate(paths)])
        # tags may collide when two members end in the same leaf row; spread them vertically if needed
        for i in range(4):
            for j in range(i):
                if abs(tags[i].get_y() - tags[j].get_y()) < 0.25:
                    tags[i].shift(DOWN * 0.25)
        for k in range(4):
            b.play(ShowCreation(plines[k]), FadeIn(tags[k]), Indicate(frames[k], scale_factor=1.0, color=MEMBER_COLS[k]),
                   rt=0.055 * b.T)
        eq = Tx(r"\mathcal{L}_{\text{sym}}=\frac{1}{|\mathcal{S}'|}\sum_{s\in\mathcal{S}'}\mathcal{L}_{\text{GT}}\big(q\,s\big)",
                 font_size=46).move_to([2.2, 2.85, 0]).fix_in_frame()
        sub = Tx(r"\mathcal{S}'=\mathcal{S} \text{ if } |\mathcal{S}|\le 8,\ \text{else 8 sampled}", font_size=28, color=GREY_A)
        sub.next_to(eq, DOWN, buff=0.12).fix_in_frame()
        b.play(Write(eq), frac=0.12)
        b.play(FadeIn(sub), frac=0.05)
        # probability split: exact root target averaged over the members, and edge weights = share of members
        roots = [centre(k, []) for k in range(4)]

        def root_t(m):
            d = np.array([geodesic_deg(m, r) for r in roots]); t = np.exp(-(d / TAU0) ** 2); return t / t.sum()

        avg = np.mean([root_t(m) for m in members], 0)
        rbars = VGroup(*[Rectangle(width=max(avg[k] * 1.6, 0.02), height=0.22).set_stroke(width=0)
                         .set_fill(C_GT, 0.9).next_to(clab[k], LEFT, buff=0.1) for k in range(4)]).fix_in_frame()
        rv = VGroup(*[DecimalNumber(avg[k], num_decimal_places=2, font_size=20).next_to(rbars[k], UP, buff=0.03)
                      for k in range(4)]).fix_in_frame()
        rlab = Text("root target, averaged", font_size=18).set_color(GREY_B).next_to(rbars, DOWN, buff=0.25).fix_in_frame()
        edges = {}
        for cp in paths:
            pts = path_pts(cp)
            for l in range(5):
                key_ = (tuple(np.round(pts[l], 3)), tuple(np.round(pts[l + 1], 3)))
                edges[key_] = edges.get(key_, 0) + 1
        wlines = VGroup(*[Line(np.array(a), np.array(c), stroke_width=3 + 9 * n / 4, color=C_GT).set_stroke(opacity=0.35 + 0.65 * n / 4)
                          for (a, c), n in edges.items()]).fix_in_frame()
        share = Text("each equivalent path: 1/4 of the probability", font_size=22).set_color(C_GT).move_to([2.2, -2.75, 0]).fix_in_frame()
        b.play(GrowFromEdge(rbars, RIGHT), FadeIn(rv), FadeIn(rlab), frac=0.08)
        b.play(FadeIn(wlines), plines.animate.set_stroke(opacity=0.0), FadeIn(share), frac=0.08)
        # no symmetry: the same loss with a single member
        body2 = tower(q, P1, 1.2, marker=True)
        single = VMobject().set_points_as_corners(path_pts(paths[0])).set_stroke(C_GT, 12).fix_in_frame()
        rt1 = root_t(members[0])
        rbars1 = VGroup(*[Rectangle(width=max(rt1[k] * 1.6, 0.02), height=0.22).set_stroke(width=0)
                          .set_fill(C_GT, 0.9).next_to(clab[k], LEFT, buff=0.1) for k in range(4)]).fix_in_frame()
        nos = Tx(r"\text{no symmetry: } |\mathcal{S}|=1 \;\Rightarrow\; \mathcal{L}_{\text{sym}}=\mathcal{L}_{\text{GT}}(q)",
                  font_size=32, color=WHITE).move_to([2.2, -2.75, 0]).fix_in_frame()
        b.play(FadeOut(body), FadeIn(body2), FadeOut(frames[1:]), FadeOut(tags[1:]), FadeOut(wlines), FadeIn(single),
               Transform(rbars, rbars1), FadeOut(rv), FadeOut(share),
               Transform(rlab, Text("root target", font_size=18).set_color(GREY_B).move_to(rlab).fix_in_frame()), frac=0.1)
        b.play(Write(nos), frac=0.08)
        b.done()
