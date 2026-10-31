"""S02_Ambiguity: exact symmetry (cube, 24 equivalent rotations) vs near symmetry (a box with a small stub);
beliefs as multi-modal densities in the axis-angle ball of radius pi.

Everything is computed: the cube's rotation group (24 signed permutation matrices with det +1), the "image" insets
(a small orthographic renderer with back-face culling and Lambert shading, so the before/after images are exactly
equal for the cube), the axis-angle vectors (log map of the actual quaternions) and the density blobs (perturbed
quaternions mapped through the same log map). Rotations of exactly 180 deg sit on the ball's boundary, where
+pi*axis and -pi*axis are the same rotation; we draw one representative (sign chosen so the first non-zero axis
component is positive)."""
from common import *  # noqa: F401,F403


def rot_x(a):
    c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


VIEW = rot_x(-62 * DEGREES) @ rot_z(-35 * DEGREES)
AX_COL = ["#E06C75", "#98C379", "#61AFEF"]
OBJ_RGB = np.array(color_to_rgb("#8FA3B8"))
LIGHT = np.array([-0.35, 0.55, 1.0]); LIGHT = LIGHT / np.linalg.norm(LIGHT)


def solid(m):
    m._depth = True
    return m


def cube_group():
    """The 24 proper rotations of the cube: signed permutation matrices with det +1."""
    import itertools
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for i, (j, s) in enumerate(zip(perm, signs)):
                M[i, j] = s
            if np.linalg.det(M) > 0:
                mats.append(M)
    return mats


def mat_to_quat(M):
    """Shepperd's method (robust at 180 deg); verified against qmat."""
    tr = np.trace(M)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        q = [0.25 * S, (M[2, 1] - M[1, 2]) / S, (M[0, 2] - M[2, 0]) / S, (M[1, 0] - M[0, 1]) / S]
    elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
        S = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
        q = [(M[2, 1] - M[1, 2]) / S, 0.25 * S, (M[0, 1] + M[1, 0]) / S, (M[0, 2] + M[2, 0]) / S]
    elif M[1, 1] > M[2, 2]:
        S = np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
        q = [(M[0, 2] - M[2, 0]) / S, (M[0, 1] + M[1, 0]) / S, 0.25 * S, (M[1, 2] + M[2, 1]) / S]
    else:
        S = np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
        q = [(M[1, 0] - M[0, 1]) / S, (M[0, 2] + M[2, 0]) / S, (M[1, 2] + M[2, 1]) / S, 0.25 * S]
    q = np.array(q); q = q / np.linalg.norm(q)
    assert np.allclose(qmat(q), M, atol=1e-6)
    return q


def log_map(q):
    """Axis-angle vector (axis * angle, angle in [0, pi]) of a unit quaternion."""
    q = np.asarray(q, float); q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    v = q[1:]; s = np.linalg.norm(v)
    if s < 1e-9:
        return np.zeros(3)
    ang = 2 * np.arctan2(s, q[0]); ax = v / s
    if abs(ang - PI) < 1e-6:                          # boundary: pick one representative of +-pi*axis
        nz = ax[np.argmax(np.abs(ax) > 1e-6)]
        ax = ax * np.sign(nz)
    return ax * ang


def box_faces(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    faces = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        for side, val in ((-1, lo[i]), (1, hi[i])):
            quad = []
            for a, b_ in ((0, 0), (1, 0), (1, 1), (0, 1)):
                p = np.zeros(3); p[i] = val
                p[j] = lo[j] if a == 0 else hi[j]; p[k] = lo[k] if b_ == 0 else hi[k]
                quad.append(p)
            n = np.zeros(3); n[i] = side
            faces.append((np.array(quad), n))
    return faces


def render_image(boxes, R, centre, k, light=1.0, contrast=1.0, rgb=OBJ_RGB):
    """Orthographic render as seen along the screen axis (same VIEW as the 3-D objects): visible faces only,
    Lambert shading, painter's order by depth."""
    M = VIEW @ R
    polys = []
    for lo, hi in boxes:
        for quad, n in box_faces(lo, hi):
            ns = M @ n
            if ns[2] <= 1e-6:
                continue
            P = (M @ quad.T).T
            shade = 0.25 + 0.75 * max(0.0, ns @ LIGHT)
            col = rgb * shade * light
            grey = np.full(3, 0.3 * light)
            col = contrast * col + (1 - contrast) * grey
            polys.append((P[:, 2].mean(), P, np.clip(col, 0, 1)))
    polys.sort(key=lambda t: t[0])
    g = VGroup()
    for _, P, col in polys:
        g.add(Polygon(*[np.array(centre) + k * np.array([p[0], p[1], 0]) for p in P],
                      stroke_width=0.5, stroke_color=rgb_to_hex(col), fill_color=rgb_to_hex(col), fill_opacity=1))
    return g


def object_mob(boxes, R, centre, K, color="#8FA3B8", opacity=1.0):
    g = Group()
    for lo, hi in boxes:
        lo, hi = np.array(lo, float), np.array(hi, float)
        p = Prism(*(hi - lo), color=color, opacity=opacity, shading=(0.3, 0.55, 0.25)); p.move_to((lo + hi) / 2)
        g.add(p)
    g.apply_matrix(K * VIEW @ R, about_point=ORIGIN).shift(centre)
    return solid(g)


def triad(R, centre, L, K=1.0, width=4, opacity=1.0):
    return VGroup(*[Arrow(centre, centre + K * VIEW @ (R @ (L * e)), buff=0, stroke_width=width, color=c).set_opacity(opacity)
                    for e, c in zip([RIGHT, UP, OUT], AX_COL)])


def hull_outline(boxes, R, centre, K, **kw):
    pts = []
    for lo, hi in boxes:
        for quad, _ in box_faces(lo, hi):
            pts += list((K * VIEW @ R @ quad.T).T[:, :2])
    P = np.array(pts)
    # 2-D convex hull (monotone chain)
    P = sorted(map(tuple, np.round(P, 6)))
    def cr(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo_, up_ = [], []
    for p in P:
        while len(lo_) >= 2 and cr(lo_[-2], lo_[-1], p) <= 0:
            lo_.pop()
        lo_.append(p)
    for p in reversed(P):
        while len(up_) >= 2 and cr(up_[-2], up_[-1], p) <= 0:
            up_.pop()
        up_.append(p)
    H = lo_[:-1] + up_[:-1]
    return DashedVMobject(Polygon(*[np.array(centre) + np.array([x, y, 0]) for x, y in H], **kw), num_dashes=60)


def inset_frame(centre, w=3.2, h=2.6, label=None, color=GREY_B):
    fr = Rectangle(width=w, height=h, stroke_color=color, stroke_width=2, fill_color="#151C24", fill_opacity=1).move_to(centre)
    g = VGroup(fr)
    if label is not None:
        g.add(TexText(label, font_size=26, color=color).next_to(fr, DOWN, buff=0.12))
    return g


def check_mark(color=C_GT, s=1.0):
    return VMobject(stroke_color=color, stroke_width=7 * s).set_points_as_corners([LEFT * 0.18 * s, DOWN * 0.16 * s, UR * 0.25 * s + RIGHT * 0.1 * s])


def cross_mark(color=C_FLIP, s=1.0):
    return VGroup(Line(UL * 0.17 * s, DR * 0.17 * s), Line(UR * 0.17 * s, DL * 0.17 * s)).set_stroke(color, 7 * s)


class S02_Ambiguity(ThreeDScene):
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
        G = cube_group()
        assert len(G) == 24
        CUBE = [([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])]
        OC = np.array([-3.7, -0.3, 0]); K = 2.3
        IN1, IN2 = np.array([1.45, 0.35, 0]), np.array([5.2, 0.35, 0])
        KI = 1.4

        # ================================================================== b1
        b = Beat(self, "S02_Ambiguity_b1")
        rot = ValueTracker(0.0)
        R0 = np.eye(3)

        def Rcur():
            return rot_z(rot.get_value() * PI / 2) @ R0
        cube = object_mob(CUBE, Rcur(), OC, K)
        tri = triad(Rcur(), OC, 0.95, K, width=5)
        b.play(FadeIn(cube, scale=0.85), rt=0.9)
        # hypotheses flicker around the cube: representing a pose is easy, choosing one is the problem
        rng = np.random.default_rng(5)
        hyp = VGroup(*[triad(qmat(quat(rng.normal(size=3), rng.uniform(0.3, 3.0))), OC, 0.95, K, width=3, opacity=0.35)
                       for _ in range(6)])
        b.play(LaggedStartMap(FadeIn, hyp, lag_ratio=0.15), rt=1.1)
        b.play(FadeOut(hyp, lag_ratio=0.1), GrowFromCenter(tri), rt=0.9)

        # face axis and the 90-degree turn
        ax_line = DashedLine(OC - K * 1.25 * (VIEW @ OUT), OC + K * 1.25 * (VIEW @ OUT), dash_length=0.12, color=WHITE, stroke_width=2.5)
        arc_pts = [OC + K * (VIEW @ np.array([0.95 * np.cos(a), 0.95 * np.sin(a), 0.62])) for a in np.linspace(-0.3, PI / 2 - 0.3, 30)]
        turn_arc = VGroup(VMobject(stroke_color=C_GT, stroke_width=4).set_points_smoothly(arc_pts[:-2]),
                          Arrow(arc_pts[-4], arc_pts[-1], buff=0, stroke_width=4, color=C_GT, max_tip_length_to_length_ratio=0.9))
        turn_lab = Tex(r"90^\circ", font_size=40, color=C_GT).next_to(arc_pts[-1], UR, buff=0.05)
        outline = hull_outline(CUBE, Rcur(), OC, K, stroke_color=WHITE, stroke_width=3)
        im1 = render_image(CUBE, Rcur(), IN1, KI)
        fr1 = inset_frame(IN1, label="image before")
        b.play(ShowCreation(ax_line), FadeIn(fr1), FadeIn(im1), ShowCreation(outline), rt=1.2)
        cube.add_updater(lambda m: m.become(object_mob(CUBE, Rcur(), OC, K)))
        tri.add_updater(lambda m: m.become(triad(Rcur(), OC, 0.95, K, width=5)))
        b.play(ShowCreation(turn_arc), FadeIn(turn_lab), rot.animate.set_value(1.0), rt=2.2)
        cube.clear_updaters(); tri.clear_updaters()
        im2 = render_image(CUBE, Rcur(), IN2, KI)
        fr2 = inset_frame(IN2, label="image after")
        eqs = Tex("=", font_size=70).move_to((IN1 + IN2) / 2)
        b.play(FadeIn(fr2), FadeIn(im2), Write(eqs), Indicate(outline, color=C_GT), rt=1.1)
        # the two images are the same pixels: slide one onto the other
        im1c = im1.copy().set_fill(opacity=0.45).set_stroke(WHITE, 1.5)
        b.play(im1c.animate.move_to(im2.get_center()), rt=0.8)
        self.remove(im1c)

        # all 24 rotations of the cube give this image
        b.play(FadeOut(VGroup(ax_line, turn_arc, turn_lab, outline, fr1, fr2, im1, im2, eqs)), FadeOut(cube), FadeOut(tri), rt=0.6)
        cols, rows = 8, 3
        xs = np.linspace(-5.9, 5.9, cols); ys = [1.55, -0.55, -2.65]
        minis, ticks = Group(), VGroup()
        for idx, g in enumerate(G):
            c = np.array([xs[idx % cols], ys[idx // cols], 0])
            m = Group(object_mob(CUBE, g, c, 0.95), triad(g, c, 0.85, 0.95, width=3))
            minis.add(m)
            ticks.add(check_mark(s=0.7).move_to(c + np.array([0.55, -0.55, 0])))
        title = TexText("24 rotations, one image, all correct", font_size=40, color=C_GT).to_edge(UP, buff=0.35)
        b.play(LaggedStartMap(FadeIn, minis, lag_ratio=0.06), FadeIn(title), rt=2.4)
        b.play(LaggedStartMap(ShowCreation, ticks, lag_ratio=0.04), rt=b.left - 0.05)
        b.done()

        # ================================================================== b2
        b = Beat(self, "S02_Ambiguity_b2")
        NEAR = [([-0.85, -0.5, -0.45], [0.85, 0.5, 0.45]), ([0.5, -0.07, 0.45], [0.62, 0.07, 0.72])]   # body + small stub
        q0 = quat([0.25, 0.15, 1.0], 20 * DEGREES); R_true = qmat(q0)
        q_flip = qmul(q0, quat([0, 0, 1], PI)); R_flip = qmat(q_flip)
        OC2 = np.array([-3.9, -0.2, 0]); K2 = 2.1
        b.play(FadeOut(minis), FadeOut(ticks), FadeOut(title), rt=0.5)
        near = object_mob(NEAR, R_true, OC2, K2)
        nlab = TexText("nearly symmetric (schematic)", font_size=28, color=GREY_B).move_to(OC2 + DOWN * 2.75)
        stub_pt = OC2 + K2 * (VIEW @ (R_true @ np.array([0.56, 0, 0.6])))
        ring = Circle(radius=0.35, stroke_color=C_GT, stroke_width=3).move_to(stub_pt)
        ring_lab = TexText("small feature", font_size=28, color=C_GT).next_to(ring, UL, buff=0.15)
        b.play(FadeIn(near, scale=0.9), FadeIn(nlab), rt=0.8)
        b.play(ShowCreation(ring), FadeIn(ring_lab), rt=0.7)

        lt = ValueTracker(1.0)        # lighting level
        ct = ValueTracker(1.0)        # contrast

        def imA():
            return render_image(NEAR, R_true, IN1, 1.25, light=lt.get_value(), contrast=ct.get_value())

        def imB():
            return render_image(NEAR, R_flip, IN2, 1.25, light=lt.get_value(), contrast=ct.get_value())
        frA = inset_frame(IN1, label=r"$q$")
        frB = inset_frame(IN2, label=r"$q\cdot$ half turn")
        iA = imA()
        b.play(FadeIn(frA), FadeIn(iA), rt=0.6)
        # half turn about the body's long-face axis
        ht = ValueTracker(0.0)
        zb = R_true @ np.array([0, 0, 1.0])
        near.add_updater(lambda m: m.become(object_mob(NEAR, qmat(qmul(q0, quat([0, 0, 1], ht.get_value() * PI))), OC2, K2)))
        ring.add_updater(lambda m: m.move_to(OC2 + K2 * (VIEW @ (qmat(qmul(q0, quat([0, 0, 1], ht.get_value() * PI))) @ np.array([0.56, 0, 0.6])))))
        iB = imB()
        b.play(ht.animate.set_value(1.0), FadeOut(ring_lab), rt=1.6)
        b.play(FadeIn(frB), FadeIn(iB), rt=0.6)
        b.play(ht.animate.set_value(0.0), rt=1.0)
        near.clear_updaters(); ring.clear_updaters()
        okA = check_mark(s=1.3).next_to(frA[0], UP, buff=0.15)
        noB = cross_mark(s=1.3).next_to(frB[0], UP, buff=0.15)
        triA = triad(R_true, OC2, 1.0, K2, width=5)
        b.play(ShowCreation(okA), ShowCreation(noB), GrowFromCenter(triA), frA[0].animate.set_stroke(C_GT), frB[0].animate.set_stroke(C_FLIP), rt=1.0)

        # lighting change: darker, lower contrast, sensor noise -> the stub is what disappears
        iA.add_updater(lambda m: m.become(imA())); iB.add_updater(lambda m: m.become(imB()))
        noise_rng = np.random.default_rng(1)
        npts = []
        for c in (IN1, IN2):
            P = noise_rng.uniform([-1.55, -1.25], [1.55, 1.25], size=(800, 2))
            npts.append(np.c_[P[:, 0] + c[0], P[:, 1] + c[1], np.zeros(len(P))])
        noise = DotCloud(np.vstack(npts), radius=0.018, color=GREY_B).set_opacity(0.0)
        self.add(noise)
        nt = ValueTracker(0.0)
        noise.add_updater(lambda m: m.set_opacity(0.55 * nt.get_value()))
        light_lab = TexText("lighting change", font_size=30, color=GREY_A).move_to([3.35, -2.75, 0])
        bar_bg = Rectangle(width=3.0, height=0.14, stroke_color=GREY_C, stroke_width=1).next_to(light_lab, DOWN, buff=0.15)
        bar = Rectangle(width=3.0, height=0.14, stroke_width=0, fill_color=C_GT, fill_opacity=0.9).move_to(bar_bg)
        bar.add_updater(lambda m: m.stretch_to_fit_width(max(3.0 * lt.get_value(), 0.01)).align_to(bar_bg, LEFT))
        b.play(FadeIn(light_lab), FadeIn(bar_bg), FadeIn(bar), rt=0.6)
        b.play(lt.animate.set_value(0.3), ct.animate.set_value(0.45), nt.animate.set_value(1.0), rt=2.4)
        iA.clear_updaters(); iB.clear_updaters(); bar.clear_updaters(); noise.clear_updaters()
        unreliable = TexText("the separating detail is what becomes unreliable", font_size=28, color=C_FLIP).to_edge(UP, buff=0.4)
        b.play(FadeIn(unreliable), Indicate(VGroup(iA, iB), color=GREY_C), rt=b.left - 0.05)
        b.done()

        # ================================================================== b3
        b = Beat(self, "S02_Ambiguity_b3")
        BC = np.array([-2.2, -0.25, 0]); BR = 3.15          # ball centre and radius on screen (radius pi)
        sc = BR / PI
        theta = ValueTracker(0.0)                           # slow turn of the ball for depth

        def V():
            return VIEW @ rot_z(theta.get_value())

        def to_screen(P):
            P = np.atleast_2d(P)
            return BC + sc * (V() @ P.T).T

        b.play(FadeOut(Group(frA, frB, iA, iB, okA, noB, light_lab, bar_bg, bar, unreliable, nlab, ring, noise)),
               FadeOut(near), FadeOut(triA), rt=0.4)
        self.frame.set_field_of_view(6 * DEGREES)           # near-orthographic so the drawn ball has radius exactly pi
        ball = VGroup(*[Circle(radius=BR * f, stroke_width=0, fill_color="#2A3A4C", fill_opacity=0.10) for f in np.linspace(1.0, 0.3, 8)],
                      Circle(radius=BR, stroke_color=GREY_B, stroke_width=2)).move_to(BC)
        mesh = SurfaceMesh(Sphere(radius=BR, resolution=(40, 20)), resolution=(13, 7), stroke_width=1, stroke_color=GREY_C)
        mesh.apply_matrix(VIEW, about_point=ORIGIN).shift(BC).set_stroke(opacity=0.3)
        rad_line = Line(BC, BC + BR * normalize(np.array([0.6, 0.8, 0])), stroke_width=2, color=GREY_B)
        rad_lab = Tex(r"\pi", font_size=40, color=GREY_A).next_to(rad_line.get_center(), UL, buff=0.06)
        b.play(FadeIn(ball), ShowCreation(mesh), ShowCreation(rad_line), FadeIn(rad_lab), rt=0.8)

        # one rotation: direction = axis, length = angle (the object on the right turns with it)
        ax_a, ax_b = normalize(VIEW.T @ np.array([0.8, 0.55, 0.25])), normalize(VIEW.T @ np.array([-0.75, 0.6, 0.2]))
        sw = ValueTracker(0.0)
        ang = ValueTracker(0.0)

        def demo_axis():
            return normalize(interpolate(ax_a, ax_b, sw.get_value()))
        OBJ_R = np.array([4.6, 0.6, 0])

        def demo_vec():
            p = to_screen(demo_axis() * ang.get_value())[0]
            return VGroup(Arrow(BC, p, buff=0, stroke_width=5, color=C_PRED, max_tip_length_to_length_ratio=0.15),
                          Dot(p, radius=0.08, color=C_PRED))
        dv = demo_vec(); dv.add_updater(lambda m: m.become(demo_vec()))
        dobj = object_mob(NEAR, np.eye(3), OBJ_R, 1.5)
        dobj.add_updater(lambda m: m.become(object_mob(NEAR, qmat(quat(demo_axis(), ang.get_value())), OBJ_R, 1.5)))
        axis_lab = TexText("direction = axis", font_size=28, color=C_PRED).move_to([4.6, -1.55, 0])
        len_lab = TexText("length = angle", font_size=28, color=C_PRED).next_to(axis_lab, DOWN, buff=0.12)
        ang_num = DecimalNumber(0, num_decimal_places=0, font_size=34, color=C_PRED).move_to([4.35, -2.55, 0])
        ang_deg = Tex(r"^\circ", font_size=34, color=C_PRED)
        ang_num.add_updater(lambda m: m.set_value(np.degrees(ang.get_value())))
        ang_deg.add_updater(lambda m: m.next_to(ang_num, RIGHT, buff=0.04).align_to(ang_num, UP))
        self.add(dv)
        b.play(FadeIn(dobj), FadeIn(axis_lab), FadeIn(len_lab), FadeIn(ang_num), FadeIn(ang_deg), rt=0.6)
        b.play(ang.animate.set_value(0.8 * PI), theta.animate.set_value(0.25), rt=3.0)
        b.play(sw.animate.set_value(1.0), ang.animate.set_value(0.45 * PI), rt=2.0)
        for m in (dv, dobj, ang_num, ang_deg):
            m.clear_updaters()

        # helpers for modes and density blobs (perturbed quaternions through the same log map)
        def splat_pts(q_mode, n, sigma_deg, seed):
            r = np.random.default_rng(seed)
            out = []
            for _ in range(n):
                axr = r.normal(size=3)
                d = quat(axr, np.radians(abs(r.normal(0, sigma_deg))))
                out.append(log_map(qmul(q_mode, d)))
            return np.array(out)

        def make_cloud(P, color, op=0.22, rad=0.05):
            dc = DotCloud(to_screen(P), radius=rad, color=color).set_opacity(op)
            dc._model = P
            dc.add_updater(lambda m: m.set_points(to_screen(m._model)))
            return dc

        def make_modes(P, color, rad=0.09):
            dc = DotCloud(to_screen(P), radius=rad, color=color).set_opacity(1.0)
            dc._model = P
            dc.add_updater(lambda m: m.set_points(to_screen(m._model)))
            return dc

        # symmetry: the cube's 24 equivalent rotations -> 24 correct modes
        cube_q = [mat_to_quat(g) for g in G]
        cube_P = np.array([log_map(q) for q in cube_q])
        assert np.all(np.linalg.norm(cube_P, axis=1) <= PI + 1e-6)
        cube_modes = make_modes(cube_P, C_GT)
        cube_cloud = make_cloud(np.vstack([splat_pts(q, 120, 9, 10 + i) for i, q in enumerate(cube_q)]), C_GT)
        bump = TexText("belief: several bumps", font_size=32, color=GREY_A).to_corner(UR, buff=0.4)
        ill_lab = TexText("(illustrative belief)", font_size=24, color=GREY_B).next_to(bump, DOWN, buff=0.1).align_to(bump, RIGHT)
        sym_lab = TexText(r"symmetry: 24 modes, all correct", font_size=32, color=C_GT).next_to(ill_lab, DOWN, buff=0.2).align_to(bump, RIGHT)
        b.play(FadeOut(Group(dv, dobj, axis_lab, len_lab, ang_num, ang_deg)), FadeIn(bump), rt=0.6)
        self.add(cube_cloud, cube_modes)
        cube_cloud.set_opacity(0); cube_modes.set_opacity(0)
        b.play(cube_cloud.animate.set_opacity(0.22), cube_modes.animate.set_opacity(1.0), FadeIn(ill_lab),
               theta.animate.set_value(0.5), rt=2.0)
        b.play(theta.animate.set_value(0.65), rt=0.8)
        b.play(FadeIn(sym_lab), theta.animate.set_value(0.85), rt=1.6)
        b.play(theta.animate.set_value(1.0), rt=0.6)

        # near symmetry: the true mode and a look-alike at a half turn
        q_true_b = quat([1.0, 0.3, 0.2], 60 * DEGREES)
        q_look = qmul(q_true_b, quat([0, 0, 1], PI))
        tP, fP = log_map(q_true_b)[None], log_map(q_look)[None]
        t_modes, f_modes = make_modes(tP, C_GT, 0.12), make_modes(fP, C_FLIP, 0.12)
        t_cloud = make_cloud(splat_pts(q_true_b, 500, 11, 3), C_GT, 0.25, 0.06)
        f_cloud = make_cloud(splat_pts(q_look, 380, 11, 4), C_FLIP, 0.25, 0.06)
        near_lab = TexText(r"near symmetry: a look-alike mode, wrong", font_size=32, color=C_FLIP).move_to(sym_lab).align_to(sym_lab, RIGHT)
        sep = DashedLine(to_screen(tP)[0], to_screen(fP)[0], dash_length=0.1, stroke_width=2, color=GREY_B)
        sep_lab = Tex(r"180^\circ", font_size=32, color=GREY_A).next_to(sep.get_center(), UP, buff=0.08)
        b.play(cube_cloud.animate.set_opacity(0), cube_modes.animate.set_opacity(0), FadeOut(sym_lab),
               theta.animate.set_value(1.15), rt=0.6)
        self.remove(cube_cloud, cube_modes)
        self.add(t_cloud, f_cloud, t_modes, f_modes)
        t_cloud.set_opacity(0); f_cloud.set_opacity(0); t_modes.set_opacity(0); f_modes.set_opacity(0)
        b.play(t_cloud.animate.set_opacity(0.25), f_cloud.animate.set_opacity(0.25), t_modes.animate.set_opacity(1),
               f_modes.animate.set_opacity(1), FadeIn(near_lab), theta.animate.set_value(1.35), rt=1.6)
        for m in (t_cloud, f_cloud, t_modes, f_modes):
            m.clear_updaters()
        sep = DashedLine(t_modes.get_points()[0], f_modes.get_points()[0], dash_length=0.1, stroke_width=2, color=GREY_B)
        sep_lab.next_to(sep.get_center(), UP, buff=0.08)
        b.play(ShowCreation(sep), FadeIn(sep_lab), rt=0.9)

        # a single prediction has to select one of them
        sel_lab = TexText("select one", font_size=40, color=C_PRED).move_to([4.4, -1.6, 0])
        tp = t_modes.get_points()[0]
        sel = Arrow(sel_lab.get_left() + LEFT * 0.1, tp + RIGHT * 0.12, buff=0.05, stroke_width=6, color=C_PRED)
        ring_t = Circle(radius=0.32, stroke_color=C_PRED, stroke_width=4).move_to(tp)
        b.play(FadeIn(sel_lab), GrowArrow(sel), ShowCreation(ring_t), f_cloud.animate.set_opacity(0.08),
               f_modes.animate.set_opacity(0.35), rt=1.4)
        b.play(Indicate(t_modes, color=C_GT, scale_factor=1.6), ring_t.animate.scale(1.25), rt=0.9)
        b.done()
