"""S01_Problem: two applications (bin picking, spacecraft docking) need the 6D pose (R, t) from one image;
fine-tuning a large network per object vs frozen shared backbones with small per-object heads.

Layout: the camera frame stays 2-D; 3-D scenes are pre-tilted by VIEW. Pinhole projections are computed with real
perspective division onto each camera's image plane, and they update live while the objects move.
Lattices: one dot = 303 M / 960 = 0.316 M parameters, so the backbone and head lattices are drawn to scale
(DINOv3 303 M -> 960 dots, MoGe-2 331 M -> 1049 dots, head 2.4 M -> 8 dots)."""
from common import *  # noqa: F401,F403


def rot_x(a):
    c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


VIEW = rot_x(-62 * DEGREES) @ rot_z(-35 * DEGREES)
AX_COL = ["#E06C75", "#98C379", "#61AFEF"]
PER_DOT = 303.0 / 960.0          # million parameters per lattice dot


def solid(m):
    m._depth = True
    return m


def box_verts(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)


def convex_hull_2d(P):
    P = sorted(map(tuple, P))
    if len(P) < 3:
        return np.array(P)
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


class PoseRig:
    """An object (list of boxes in its body frame) seen by a pinhole camera; everything in model coordinates,
    mapped to the screen by S(p) = centre + VIEW @ p."""

    def __init__(self, centre, boxes, colors, cam_screen, f=1.05, plane=(1.5, 1.1), K=1.45):
        self.centre = np.asarray(centre, float); self.K = K
        self.boxes = boxes                      # list of (lo, hi) in body frame
        self.colors = colors
        self.cam = VIEW.T @ np.asarray(cam_screen, float)          # camera centre (model coordinates)
        self.f, self.plane = f, plane
        self.R = np.eye(3)
        self.pos = np.zeros(3)                  # object position (model coordinates)
        a = self.pos - self.cam; self.a = a / np.linalg.norm(a)     # optical axis (fixed)
        u = np.cross(self.a, [0, 0, 1.0]); self.u = u / np.linalg.norm(u)
        self.v = np.cross(self.u, self.a)

    def S(self, p):
        return self.centre + self.K * (VIEW @ np.asarray(p, float))

    def world_verts(self, k):
        lo, hi = self.boxes[k]
        return (self.R @ box_verts(lo, hi).T).T + self.pos

    def project(self, X):
        """Pinhole: intersection of the ray X -> camera centre with the image plane (perspective division)."""
        d = X - self.cam
        depth = d @ self.a
        return self.cam + self.f * d / depth

    def object_mob(self, opacity=1.0):
        g = Group()
        for (lo, hi), col in zip(self.boxes, self.colors):
            lo, hi = np.array(lo, float), np.array(hi, float)
            p = Prism(*(hi - lo), color=col, opacity=opacity, shading=(0.35, 0.55, 0.2))
            p.move_to((lo + hi) / 2)
            g.add(p)
        g.apply_matrix(self.K * VIEW @ self.R, about_point=ORIGIN)
        g.shift(self.S(self.pos))
        return solid(g)

    def frustum(self, color=GREY_B):
        c = self.cam + self.f * self.a
        w, h = self.plane
        corners = [c + sx * w / 2 * self.u + sy * h / 2 * self.v for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        quad = Polygon(*[self.S(p) for p in corners], stroke_color=color, stroke_width=2, fill_color="#1C2633", fill_opacity=0.85)
        edges = VGroup(*[Line(self.S(self.cam), self.S(p), stroke_width=1.5, color=color) for p in corners])
        apex = Dot(self.S(self.cam), radius=0.06, color=WHITE)
        return VGroup(edges, quad, apex)

    def image(self):
        """Projected silhouette in the image plane: union of the convex hulls of each box's projected vertices."""
        g = VGroup()
        for k, col in enumerate(self.colors):
            P = np.array([self.project(X) for X in self.world_verts(k)])
            uv = np.stack([(P - self.cam) @ self.u, (P - self.cam) @ self.v], 1)
            hull = convex_hull_2d(uv)
            pts = [self.S(self.cam + self.f * self.a + x * self.u + y * self.v) for x, y in hull]
            g.add(Polygon(*pts, stroke_width=1, stroke_color=WHITE, fill_color=col, fill_opacity=0.9))
        return g

    def inset(self, centre, width=2.7):
        """The same projected image drawn flat and enlarged (u to the right, v up)."""
        w, h = self.plane; k = width / w
        frame = Rectangle(width=width, height=h * k, stroke_color=GREY_B, stroke_width=2, fill_color="#1C2633", fill_opacity=1)
        frame.move_to(centre)
        g = VGroup()
        for kk, col in enumerate(self.colors):
            P = np.array([self.project(X) for X in self.world_verts(kk)])
            uv = np.stack([(P - self.cam) @ self.u, (P - self.cam) @ self.v], 1)
            hull = convex_hull_2d(uv)
            g.add(Polygon(*[np.array(centre) + k * np.array([x, y, 0]) for x, y in hull],
                          stroke_width=1, stroke_color=WHITE, fill_color=col, fill_opacity=0.95))
        return VGroup(frame, g)

    def rays(self, opacity=0.5):
        g = VGroup()
        for k in range(len(self.boxes)):
            for X in self.world_verts(k):
                g.add(Line(self.S(X), self.S(self.cam), stroke_width=1.2, color=C_PRED).set_opacity(opacity))
        return g

    def triad(self, L=0.9):
        o = self.S(self.pos)
        return VGroup(*[Arrow(o, o + self.K * VIEW @ (self.R @ (L * e)), buff=0, stroke_width=4, color=c)
                        for e, c in zip([RIGHT, UP, OUT], AX_COL)])


def make_lattice(n, cols, origin, sp):
    """Dot positions filled row by row (top-left first): n dots, `cols` per row."""
    idx = np.arange(n)
    r, c = idx // cols, idx % cols
    return np.stack([origin[0] + c * sp, origin[1] - r * sp, np.zeros(n)], 1)


def lattice_lines(pts_grid_shape, origin, sp, rows_full, cols):
    g = VGroup()
    for r in range(rows_full):
        g.add(Line(origin + DOWN * r * sp, origin + DOWN * r * sp + RIGHT * (cols - 1) * sp))
    for c in range(cols):
        g.add(Line(origin + RIGHT * c * sp, origin + RIGHT * c * sp + DOWN * (rows_full - 1) * sp))
    g.set_stroke(GREY_D, width=0.6, opacity=0.5)
    return g


def lock_icon(color=GREY_B, s=1.0):
    body = RoundedRectangle(width=0.5 * s, height=0.4 * s, corner_radius=0.06 * s, fill_color=color, fill_opacity=1, stroke_width=0)
    shackle = Arc(0, PI, radius=0.16 * s, stroke_color=color, stroke_width=5 * s).next_to(body, UP, buff=-0.02 * s)
    hole = Dot(body.get_center() + UP * 0.03 * s, radius=0.045 * s, color=BG)
    return VGroup(shackle, body, hole)


def icon_bracket(c=C_BASE):
    return Polygon([0, 0, 0], [0.9, 0, 0], [0.9, 0.22, 0], [0.22, 0.22, 0], [0.22, 0.75, 0], [0, 0.75, 0],
                   fill_color=c, fill_opacity=1, stroke_width=0)


def icon_satellite(c=C_BASE):
    body = Rectangle(width=0.42, height=0.42, fill_color=c, fill_opacity=1, stroke_width=0)
    pl = Rectangle(width=0.36, height=0.2, fill_color="#3A5A80", fill_opacity=1, stroke_width=0).next_to(body, LEFT, buff=0.06)
    pr = pl.copy().next_to(body, RIGHT, buff=0.06)
    ant = Line(body.get_top(), body.get_top() + UP * 0.22 + RIGHT * 0.1, stroke_width=3, color=c)
    return VGroup(pl, pr, body, ant)


def icon_mug(c=C_BASE):
    cup = Rectangle(width=0.5, height=0.6, fill_color=c, fill_opacity=1, stroke_width=0)
    handle = Arc(-PI / 2, PI, radius=0.15, stroke_color=c, stroke_width=6).next_to(cup, RIGHT, buff=-0.02)
    top = Ellipse(width=0.5, height=0.12, fill_color=GREY_D, fill_opacity=1, stroke_width=0).move_to(cup.get_top())
    return VGroup(handle, cup, top)


class S01_Problem(ThreeDScene):
    default_camera_config = dict(background_color=BG)

    def add(self, *mobjects, **kw):
        for m in mobjects:
            if getattr(m, "_depth", False):
                ThreeDScene.add(self, m)
            else:
                Scene.add(self, m)
        return self

    def construct(self):
        self.frame.reorient(0, 0, 0)
        # ================================================================== b1
        b = Beat(self, "S01_Problem_b1")
        # bracket: L-shaped part (two boxes); satellite: bus, two panels, antenna stub (all schematic)
        L = PoseRig(centre=[-4.35, 0.8, 0],
                    boxes=[([-0.8, -0.35, -0.12], [0.8, 0.35, 0.12]), ([-0.8, -0.35, 0.12], [-0.56, 0.35, 1.0])],
                    colors=["#8D99A6", "#8D99A6"], cam_screen=[2.35, -0.45, 0.6])
        Sat = PoseRig(centre=[4.15, 0.9, 0],
                      boxes=[([-0.45, -0.45, -0.4], [0.45, 0.45, 0.4]), ([-1.35, -0.03, -0.32], [-0.5, 0.03, 0.32]),
                             ([0.5, -0.03, -0.32], [1.35, 0.03, 0.32]), ([-0.04, -0.04, 0.4], [0.04, 0.04, 0.95])],
                      colors=["#C9B37E", "#3A5A80", "#3A5A80", "#DDDDDD"], cam_screen=[-2.35, -0.45, 0.6])
        q_L = quat([0.3, 0.2, 1.0], 25 * DEGREES); L.R = qmat(q_L)
        q_S = quat([1.0, 0.4, 0.2], 20 * DEGREES); Sat.R = qmat(q_S)

        objL = L.object_mob()
        labL = TexText(r"industrial part (schematic)", font_size=26, color=GREY_B).move_to([-5.45, -3.1, 0])
        b.play(FadeIn(objL, shift=0.3 * DOWN), FadeIn(labL), rt=1.0)
        # a bin of identical parts: a few copies at other poses behind it
        bin_parts = Group()
        for k, (dx, ax, an) in enumerate([(-1.6, [1, 0.2, 0.3], 70), (1.2, [0.1, 1, 0.4], 120), (-0.4, [0.5, 0.5, 1], 200)]):
            g = Group(*[Prism(*(np.array(hi) - np.array(lo)), color="#5E6873", shading=(0.3, 0.5, 0.2)).move_to((np.array(lo) + np.array(hi)) / 2)
                        for lo, hi in L.boxes]).scale(0.55, about_point=ORIGIN)
            g.apply_matrix(VIEW @ qmat(quat(ax, an * DEGREES)), about_point=ORIGIN).shift([-4.35 + dx, -1.0 + 0.15 * k, -1.5])
            bin_parts.add(solid(g))
        b.play(LaggedStartMap(FadeIn, bin_parts, lag_ratio=0.3), rt=1.4)
        frL = L.frustum()
        camL = TexText("robot camera", font_size=24, color=GREY_B).next_to(frL, DOWN, buff=0.15)
        b.play(ShowCreation(frL), FadeIn(camL), rt=1.2)

        objS = Sat.object_mob()
        labS = TexText(r"spacecraft (schematic)", font_size=26, color=GREY_B).move_to([5.45, -3.1, 0])
        b.play(FadeIn(objS, shift=0.3 * LEFT), FadeIn(labS), FadeOut(bin_parts), rt=1.2)
        frS = Sat.frustum()
        camS = TexText("chaser camera", font_size=24, color=GREY_B).next_to(frS, DOWN, buff=0.15)
        # the target closes in: slides along the optical axis toward the chaser camera
        approach = ValueTracker(0.0)
        Sat.pos = np.zeros(3)
        start_pos = 0.6 * Sat.a      # starts further away along the optical axis

        def upd_sat(m):
            Sat.pos = interpolate(start_pos, np.zeros(3), approach.get_value())
            m.become(Sat.object_mob())
        Sat.pos = start_pos
        objS.become(Sat.object_mob())
        objS.add_updater(upd_sat)
        b.play(ShowCreation(frS), FadeIn(camS), approach.animate.set_value(0.5), rt=1.4)
        b.play(approach.animate.set_value(1.0), rt=1.2)
        objS.clear_updaters()

        # pinhole projection: rays from every vertex to the camera centre, hits on the image plane
        raysL, raysS = L.rays(), Sat.rays()
        imgL, imgS = L.image(), Sat.image()
        b.play(LaggedStartMap(ShowCreation, raysL, lag_ratio=0.03), LaggedStartMap(ShowCreation, raysS, lag_ratio=0.03), rt=1.6)
        insL, insS = L.inset([-2.3, -2.4, 0]), Sat.inset([2.3, -2.4, 0])
        one = TexText("one image", font_size=26, color=C_PRED)
        oneL = one.copy().next_to(insL, DOWN, buff=0.1)
        oneS = one.copy().next_to(insS, DOWN, buff=0.1)
        linkL = DashedLine(frL[1].get_center(), insL.get_top(), dash_length=0.08, stroke_width=1.5, color=GREY_B)
        linkS = DashedLine(frS[1].get_center(), insS.get_top(), dash_length=0.08, stroke_width=1.5, color=GREY_B)
        b.play(FadeIn(imgL), FadeIn(imgS), raysL.animate.set_opacity(0.15), raysS.animate.set_opacity(0.15), rt=0.6)
        b.play(ShowCreation(linkL), ShowCreation(linkS), TransformFromCopy(imgL, insL[1]), TransformFromCopy(imgS, insS[1]),
               FadeIn(insL[0]), FadeIn(insS[0]), FadeIn(oneL), FadeIn(oneS), rt=0.9)

        # the pose: R (body axes) and t (camera centre -> object centre)
        triL, triS = L.triad(), Sat.triad(0.8)
        tL = Arrow(L.S(L.cam), L.S(L.pos), buff=0.05, stroke_width=5, color=C_GT)
        tS = Arrow(Sat.S(Sat.cam), Sat.S(Sat.pos), buff=0.05, stroke_width=5, color=C_GT)
        tLl = Tex("t", font_size=40, color=C_GT).next_to(tL.get_center(), DOWN, buff=0.12)
        tSl = Tex("t", font_size=40, color=C_GT).next_to(tS.get_center(), DOWN, buff=0.12)
        eq = Tex(r"(R,\,t)\in SO(3)\times\mathbb{R}^3", font_size=54).to_edge(UP, buff=0.35)
        sixd = TexText("6D pose of a known object", font_size=30, color=GREY_A).next_to(eq, DOWN, buff=0.15)
        b.play(FadeOut(raysL), FadeOut(raysS), GrowArrow(tL), GrowArrow(tS), FadeIn(tLl), FadeIn(tSl), Write(eq), rt=1.4)
        RL = Tex("R", font_size=40, color=WHITE).next_to(objL, UP, buff=0.1)
        RS = Tex("R", font_size=40, color=WHITE).next_to(objS, UP, buff=0.1).shift(LEFT * 0.6)
        b.play(*[GrowArrow(a) for a in (*triL, *triS)], FadeIn(RL), FadeIn(RS), FadeIn(sixd), rt=1.0)

        # rotate both objects; the projected images and body axes follow (recomputed every frame)
        rt_ = ValueTracker(0.0)
        axL, axS = np.array([0.2, 1.0, 0.5]), np.array([0.3, 0.2, 1.0])

        def upd_L(m):
            L.R = qmat(qmul(quat(axL, rt_.get_value() * 70 * DEGREES), q_L)); m.become(L.object_mob())

        def upd_S(m):
            Sat.R = qmat(qmul(quat(axS, -rt_.get_value() * 80 * DEGREES), q_S)); m.become(Sat.object_mob())
        objL.add_updater(upd_L); objS.add_updater(upd_S)
        imgL.add_updater(lambda m: m.become(L.image())); imgS.add_updater(lambda m: m.become(Sat.image()))
        insL.add_updater(lambda m: m.become(L.inset([-2.3, -2.4, 0]))); insS.add_updater(lambda m: m.become(Sat.inset([2.3, -2.4, 0])))
        triL.add_updater(lambda m: m.become(L.triad())); triS.add_updater(lambda m: m.become(Sat.triad(0.8)))
        b.play(rt_.animate.set_value(1.0), rt=2.6, rate_func=there_and_back_with_pause)
        for m in (objL, objS, imgL, imgS, triL, triS, insL, insS):
            m.clear_updaters()
        b.done()

        # ================================================================== b2
        b = Beat(self, "S01_Problem_b2")
        everything = Group(objL, objS, labL, labS, frL, frS, camL, camS, imgL, imgS, oneL, oneS, triL, triS, insL, insS,
                           linkL, linkS, tL, tS, tLl, tSl, RL, RS, sixd)
        COLS = 40
        n_dino, n_moge, n_head = 960, int(round(331 / PER_DOT)), int(round(2.4 / PER_DOT))   # 960, 1049, 8
        icons = VGroup(icon_bracket(), icon_satellite(), icon_mug())
        icon_pos = [np.array([6.1, 2.3, 0]), np.array([6.1, 0.0, 0]), np.array([6.1, -2.3, 0])]
        for ic, p in zip(icons, icon_pos):
            ic.move_to(p)
        b.play(FadeOut(everything), FadeOut(eq), rt=0.6)

        # --- fine-tuning: one full copy of the 303 M backbone per object, gradients through all of it
        SPc = 0.088
        tints = ["#A9C7E8", "#E3CF9A", "#C9B8F0"]
        copies, copy_pts = [], []
        for k, p in enumerate(icon_pos):
            o = np.array([1.7 - (COLS - 1) * SPc, p[1] + 11.5 * SPc, 0])
            pts = make_lattice(n_dino, COLS, o, SPc)
            dc = DotCloud(pts, radius=0.03, color=tints[k]).set_opacity(0.85)
            copies.append(dc); copy_pts.append(pts)
        ft_lab = TexText(r"fine-tuning: a full copy of the network per object", font_size=30, color=C_FT).to_edge(UP, buff=0.2)
        cnt = VGroup(*[TexText("303 M trained", font_size=28, color=C_FT).next_to(c, LEFT, buff=0.25) for c in copies])
        b.play(*[FadeIn(c, shift=0.3 * LEFT) for c in copies], LaggedStartMap(FadeIn, icons, lag_ratio=0.2), FadeIn(ft_lab), rt=0.8)

        def pulse_updater(pts, tracker, base_rgb):
            band = 0.45
            x_right, x_left = pts[:, 0].max(), pts[:, 0].min()

            def f(m):
                xw = x_right + band - tracker.get_value() * (x_right - x_left + 2 * band)
                w = np.exp(-((pts[:, 0] - xw) / band) ** 2)
                rgba = np.zeros((len(pts), 4))
                rgba[:, :3] = (1 - w[:, None]) * np.array(base_rgb) + w[:, None] * np.array(color_to_rgb(C_FT))
                rgba[:, 3] = 0.7 + 0.3 * w
                m.set_rgba_array(rgba)
                m.set_radii(0.028 + 0.035 * w)
            return f
        trk = [ValueTracker(0.0) for _ in range(3)]
        for k in range(3):
            copies[k].add_updater(pulse_updater(copy_pts[k], trk[k], color_to_rgb(tints[k])))
        # backward pass: the orange gradient wave runs from the output (right) back through every copy, twice
        b.play(LaggedStart(*[t.animate.set_value(2.0) for t in trk], lag_ratio=0.25), FadeIn(cnt), rt=2.7, rate_func=linear)
        for c in copies:
            c.clear_updaters()

        # --- ours: the large networks never change -> one shared grey lattice, frozen
        SP = 0.105
        dino_o = np.array([-4.3, 3.15, 0])
        moge_o = np.array([-4.3, -0.05, 0])
        dino_target = DotCloud(make_lattice(n_dino, COLS, dino_o, SP), radius=0.03, color="#6B7682").set_opacity(0.95)
        b.play(*[Transform(c, dino_target) for c in copies], FadeOut(ft_lab), FadeOut(cnt), rt=1.4)
        dino = copies[0]; self.remove(copies[1], copies[2]); self.add(dino)
        dino_lines = lattice_lines(None, dino_o, SP, 24, COLS)
        dino_lab = VGroup(TexText("DINOv3", font_size=34, color=C_DINO), TexText("303 M", font_size=26, color=GREY_A),
                          TexText("appearance", font_size=24, color=GREY_B))
        dino_lab.arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(dino_lines, LEFT, buff=0.3).align_to(dino_lines, UP)
        lock1 = lock_icon().next_to(dino_lab, DOWN, buff=0.25).align_to(dino_lab, LEFT)
        fz1 = TexText("frozen", font_size=24, color=GREY_B).next_to(lock1, RIGHT, buff=0.12)
        b.play(FadeIn(dino_lines), FadeIn(lock1, scale=1.5), FadeIn(fz1), rt=0.9)
        b.play(FadeIn(dino_lab), rt=0.8)

        moge = DotCloud(make_lattice(n_moge, COLS, moge_o, SP), radius=0.03, color="#6B7682").set_opacity(0.95)
        moge_lines = lattice_lines(None, moge_o, SP, 26, COLS)
        moge_lab = VGroup(TexText("MoGe-2", font_size=34, color=C_MOGE), TexText("331 M", font_size=26, color=GREY_A),
                          TexText("surface geometry", font_size=24, color=GREY_B))
        moge_lab.arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(moge_lines, LEFT, buff=0.3).align_to(moge_lines, UP)
        lock2 = lock_icon().next_to(moge_lab, DOWN, buff=0.25).align_to(moge_lab, LEFT)
        fz2 = TexText("frozen", font_size=24, color=GREY_B).next_to(lock2, RIGHT, buff=0.12)
        b.play(FadeIn(moge, shift=0.3 * UP), FadeIn(moge_lines), FadeIn(moge_lab), FadeIn(lock2, scale=1.5), FadeIn(fz2), rt=1.2)

        # --- shared by every application: feature streams from both backbones reach every object
        x_out = dino_o[0] + (COLS - 1) * SP + 0.12
        srcs = [np.array([x_out, dino_o[1] - 11.5 * SP, 0]), np.array([x_out, moge_o[1] - 13 * SP, 0])]
        head_pos = [p + LEFT * 1.7 for p in icon_pos]
        paths = VGroup()
        for s_, col in zip(srcs, [C_DINO, C_MOGE]):
            for hp in head_pos:
                path = CubicBezier(s_, s_ + RIGHT * 1.5, hp + LEFT * 1.5, hp + LEFT * 0.5)
                path.set_stroke(col, width=2, opacity=0.35)
                paths.add(path)
        b.play(LaggedStartMap(ShowCreation, paths, lag_ratio=0.12), rt=1.2)
        flow = ValueTracker(0.0)
        n_part = 7

        def particles():
            g = VGroup()
            for i, pth in enumerate(paths):
                col = C_DINO if i < 3 else C_MOGE
                for j in range(n_part):
                    a = (flow.get_value() + j / n_part + 0.05 * i) % 1.0
                    g.add(Dot(pth.point_from_proportion(a), radius=0.045).set_fill(col, 1).set_stroke(width=0))
            return g
        parts = particles(); parts.add_updater(lambda m: m.become(particles()))
        self.add(parts)
        shared = TexText("shared by every application", font_size=30, color=GREY_A).move_to([3.2, 3.6, 0])
        b.play(flow.animate.set_value(1.4), FadeIn(shared), rt=5.0, rate_func=linear)

        # --- small per-object heads (to scale: 8 dots ~ 2.4 M); gradients only inside the heads
        heads, head_pts, head_labs = [], [], VGroup()
        for hp in head_pos:
            pts = make_lattice(n_head, 4, hp + np.array([-1.5 * 0.2, 0.1, 0]), 0.2)
            dc = DotCloud(pts, radius=0.055, color=WHITE)
            heads.append(dc); head_pts.append(pts)
            head_labs.add(TexText("head 2.4 M", font_size=24, color=GREY_A).next_to(dc, DOWN, buff=0.14))
        scale_note = TexText(r"1 dot $\approx$ 0.32 M parameters", font_size=22, color=GREY_B).to_corner(DL, buff=0.25)
        b.play(*[GrowFromCenter(h) for h in heads], LaggedStartMap(FadeIn, head_labs, lag_ratio=0.2), FadeIn(scale_note),
               flow.animate.increment_value(0.5), rt=1.6, rate_func=linear)
        trained = TexText("trained per object", font_size=28, color=C_FT).move_to([4.4, -3.6, 0])
        htr = [ValueTracker(0.0) for _ in range(3)]

        def head_pulse(pts, tracker):
            def f(m):
                ph = tracker.get_value()
                w = 0.5 + 0.5 * np.cos(2 * PI * (ph * 3 + (pts[:, 0] - pts[:, 0].min()) / 0.8))
                rgba = np.zeros((len(pts), 4))
                rgba[:, :3] = (1 - w[:, None]) * np.array(color_to_rgb(WHITE)) + w[:, None] * np.array(color_to_rgb(C_FT))
                rgba[:, 3] = 1.0
                m.set_rgba_array(rgba)
            return f
        for h, p, t in zip(heads, head_pts, htr):
            h.add_updater(head_pulse(p, t))
        b.play(*[t.animate.set_value(1.0) for t in htr], FadeIn(trained), flow.animate.increment_value(1.0),
               rt=b.left - 0.05, rate_func=linear)
        for h in heads:
            h.clear_updaters()
        parts.clear_updaters()
        b.done()

        # ================================================================== b3
        b = Beat(self, "S01_Problem_b3")
        target = heads[1]
        q1 = TexText("How far can small heads go?", font_size=48).to_edge(UP, buff=0.45)
        q2 = TexText("What do they need to get there?", font_size=40, color=C_PRED).next_to(q1, DOWN, buff=0.25)
        q1.fix_in_frame(); q2.fix_in_frame()
        b.play(self.frame.animate.move_to(target.get_center() + DOWN * 0.45 + RIGHT * 0.6).set_height(3.4),
               FadeOut(VGroup(shared, trained, scale_note)), paths.animate.set_stroke(opacity=0.15),
               parts.animate.set_opacity(0.25), rt=1.8)
        b.play(Write(q1), rt=1.4)
        b.play(FadeIn(q2, shift=0.2 * UP), rt=1.0)
        heads[1].add_updater(head_pulse(head_pts[1], htr[1]))
        htr[1].set_value(0.0)
        b.play(htr[1].animate.set_value(0.6), rt=1.5, rate_func=linear)
        heads[1].clear_updaters()
        b.done()
