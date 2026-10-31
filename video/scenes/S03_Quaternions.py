"""S03_Quaternions: unit quaternions as points on S^3 (drawn as S^2), q ~ -q, sandwich product, geodesic distance.

Layout: the camera frame stays in its 2-D orientation; 3-D objects are pre-tilted by a fixed view rotation VIEW so
screen positions are easy to control while shading stays 3-D. Only solid objects (marked with ._depth) are depth
tested; flat overlays (dots, labels, arcs on the sphere) are drawn on top.
Every rotation shown is computed with quat/qmat/qmul/geodesic_deg from common.py."""
from common import *  # noqa: F401,F403


def rot_x(a):
    c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


VIEW = rot_x(-62 * DEGREES) @ rot_z(-35 * DEGREES)   # model -> screen (tilted, looking slightly from above)
SPH_C = np.array([-3.6, -0.55, 0.0])
SPH_R = 2.35
OBJ_C = np.array([3.55, 1.35, 0.0])                  # final object position (after the sphere appears)
OBJ_C0 = np.array([-3.0, 0.2, 0.0])                  # first position (large, left)
BXF = np.array([2.05, -2.05, 0.0])                   # final bar-chart origin
BX0 = np.array([1.6, -0.4, 0.0])                     # first bar-chart origin
AX_COL = ["#E06C75", "#98C379", "#61AFEF"]          # body x, y, z axes
BW, BH, GAP = 0.55, 1.15, 0.42


def solid(m):
    m._depth = True
    return m


def make_object(R, center=OBJ_C, scale=0.62, opacity=1.0):
    """Schematic object: box body with a small stub and a body triad, rotated by R, then by VIEW, at center."""
    body = Prism(2.0 * scale, 1.3 * scale, 0.9 * scale, color="#5C6B7A", opacity=opacity, shading=(0.35, 0.5, 0.2))
    stub = Prism(0.22 * scale, 0.22 * scale, 0.6 * scale, color=C_GT, opacity=opacity, shading=(0.3, 0.5, 0.2))
    stub.move_to(np.array([0.55, 0.3, 0.75]) * scale)
    axes = VGroup(*[Line(ORIGIN, 1.55 * scale * v, stroke_width=5, color=c).set_opacity(opacity)
                    for v, c in zip([RIGHT, UP, OUT], AX_COL)])
    g = Group(body, stub, axes)
    g.apply_matrix(VIEW @ R, about_point=ORIGIN)
    g.shift(center)
    return solid(g)


def sph_pt(v, lift=0.0):
    v = np.asarray(v, float); v = v / np.linalg.norm(v)
    p = SPH_C + (SPH_R + lift) * (VIEW @ v)
    p[2] = 0.0                     # flattened onto the screen plane (orthographic look, no perspective offset)
    return p


def great_arc(a, b, n=60, **kw):
    a = np.asarray(a, float) / np.linalg.norm(a); b = np.asarray(b, float) / np.linalg.norm(b)
    om = np.arccos(np.clip(a @ b, -1, 1))
    pts = [(np.sin((1 - s) * om) * a + np.sin(s * om) * b) / np.sin(om) for s in np.linspace(0, 1, n)]
    return VMobject(**kw).set_points_smoothly([sph_pt(p) for p in pts])


def bars_for(q, BX):
    g = VGroup()
    for i in range(4):
        x0 = BX + RIGHT * (i * (BW + GAP)); h = q[i] * BH
        r = Rectangle(width=BW, height=max(abs(h), 0.004), stroke_width=0, fill_color=CHART_COLORS[i], fill_opacity=0.85)
        r.move_to(x0 + RIGHT * BW / 2 + UP * h / 2); g.add(r)
    return g


def vals_for(q, BX):
    g = VGroup()
    for i in range(4):
        d = DecimalNumber(q[i], num_decimal_places=2, font_size=24, include_sign=True, color=CHART_COLORS[i])
        d.move_to(BX + RIGHT * (BW / 2 + i * (BW + GAP)) + UP * (BH + 0.3)); g.add(d)
    return g


def axis_for(BX):
    base = Line(BX + LEFT * 0.35, BX + RIGHT * (4 * BW + 3 * GAP + 0.35), stroke_width=1.5, color=GREY_C)
    letters = VGroup(*[Tex(n, font_size=34, color=CHART_COLORS[i]).move_to(BX + RIGHT * (BW / 2 + i * (BW + GAP)) + DOWN * (BH + 0.35))
                       for i, n in enumerate("wxyz")])
    return VGroup(base, letters)


class S03_Quaternions(ThreeDScene):
    default_camera_config = dict(background_color=BG)

    def add(self, *mobjects, **kw):
        # depth-test only solid 3-D objects; everything else is a flat overlay drawn in order
        for m in mobjects:
            if getattr(m, "_depth", False):
                ThreeDScene.add(self, m)
            else:
                Scene.add(self, m)
        return self

    def construct(self):
        self.frame.reorient(0, 0, 0)
        AXIS = np.array([1.0, 2.0, 0.6]); ANG = 115 * DEGREES
        q1 = quat(AXIS, ANG)
        t = ValueTracker(0.0)      # rotation progress along the axis
        mv = ValueTracker(0.0)     # layout: 0 = large first layout, 1 = final layout

        def cur_q():
            return quat(AXIS, t.get_value() * ANG)

        def cur_obj_c():
            return interpolate(OBJ_C0, OBJ_C, mv.get_value())

        def cur_scale():
            return interpolate(1.15, 0.62, mv.get_value())

        def cur_bx():
            return interpolate(BX0, BXF, mv.get_value())

        # ------------------------------------------------------------------ b1
        b = Beat(self, "S03_Quaternions_b1")
        obj = make_object(np.eye(3), cur_obj_c(), cur_scale())
        obj_lab = Tex(r"\text{schematic object}", font_size=26, color=GREY_B)
        obj_lab.add_updater(lambda m: m.move_to(cur_obj_c() + DOWN * (2.15 * cur_scale())))
        obj_lab.update()
        obj.add_updater(lambda m: m.become(make_object(qmat(cur_q()), cur_obj_c(), cur_scale())))
        b.play(FadeIn(obj, shift=0.3 * UP), FadeIn(obj_lab), rt=0.9)

        bars = bars_for(cur_q(), cur_bx()); bars.add_updater(lambda m: m.become(bars_for(cur_q(), cur_bx())))
        vals = vals_for(cur_q(), cur_bx()); vals.add_updater(lambda m: m.become(vals_for(cur_q(), cur_bx())))
        axis = axis_for(cur_bx()); axis.add_updater(lambda m: m.become(axis_for(cur_bx())))
        b.play(FadeIn(axis), FadeIn(bars), FadeIn(vals), rt=0.8)
        # the object turns from the identity to q1 about a fixed axis; the bars track the four components of q
        b.play(t.animate.set_value(1.0), rt=3.0, rate_func=smooth)

        eq = Tex(r"q = w + x\,i + y\,j + z\,k", font_size=50,
                 t2c={"w": CHART_COLORS[0], "x": CHART_COLORS[1], "y": CHART_COLORS[2], "z": CHART_COLORS[3]})
        eq.to_corner(UR, buff=0.35).shift(LEFT * 0.2)
        b.play(Write(eq), t.animate.set_value(0.8), rt=1.3)

        # unit norm: the squared components stack up to exactly 1
        SX = np.array([1.75, -3.55, 0]); SL = 3.9

        def make_sq():
            q = cur_q(); g = VGroup(); x = 0.0
            for i in range(4):
                w = q[i] ** 2 * SL
                r = Rectangle(width=max(w, 0.003), height=0.22, stroke_width=0, fill_color=CHART_COLORS[i], fill_opacity=0.9)
                r.move_to(SX + RIGHT * (x + w / 2)); g.add(r); x += w
            return g
        sq = make_sq(); sq.add_updater(lambda m: m.become(make_sq()))
        sq_frame = Rectangle(width=SL, height=0.22, stroke_color=GREY_B, stroke_width=1.5).move_to(SX + RIGHT * SL / 2)
        norm = Tex(r"\|q\|^2 = w^2+x^2+y^2+z^2 = 1", font_size=30).next_to(sq_frame, UP, buff=0.12).align_to(sq_frame, RIGHT)
        b.play(TransformFromCopy(bars, sq), FadeIn(sq_frame), t.animate.set_value(1.0), rt=1.1)
        b.play(Write(norm), rt=0.9)

        # S^3 drawn as S^2, the point q
        # translucent ball drawn as concentric discs (a translucent Surface blends in arbitrary order), highlight to the upper left
        sphere = VGroup(*[Circle(radius=SPH_R * f, stroke_width=0, fill_color=interpolate_color("#22303E", "#4A5E73", 1 - f),
                                 fill_opacity=0.35).move_to(SPH_C + (1 - f) * SPH_R * 0.45 * np.array([-0.6, 0.8, 0]))
                          for f in np.linspace(1.0, 0.2, 9)],
                        Circle(radius=SPH_R, stroke_color=GREY_B, stroke_width=1.5).move_to(SPH_C))
        mesh = SurfaceMesh(Sphere(radius=SPH_R, resolution=(48, 24)), resolution=(17, 9), stroke_width=1, stroke_color=GREY_C)
        mesh.apply_matrix(VIEW, about_point=ORIGIN).shift(SPH_C)
        mesh.apply_function(lambda p: np.array([p[0], p[1], 0.0]))
        mesh.set_stroke(opacity=0.35)
        s_lab = Tex(r"S^3\ \text{(drawn as } S^2\text{)}", font_size=36, color=GREY_A).move_to(SPH_C + UP * (SPH_R + 0.5))
        s_note = Tex(r"\text{unit quaternions in } \mathbb{R}^4", font_size=26, color=GREY_B).move_to(SPH_C + DOWN * (SPH_R + 0.45))
        Q_DIR = np.array([0.45, -0.35, 0.82])        # schematic placement of q on the stand-in sphere
        qdot = Dot(sph_pt(Q_DIR), radius=0.11, color=C_PRED)
        qlab = Tex("q", font_size=42, color=C_PRED).next_to(qdot, UR, buff=0.06)
        sq.clear_updaters()
        b.play(mv.animate.set_value(1.0), FadeIn(sphere), ShowCreation(mesh), FadeIn(s_lab),
               FadeOut(VGroup(sq, sq_frame, norm)), rt=1.3)
        b.play(FadeIn(s_note), GrowFromCenter(qdot), Write(qlab), rt=0.8)
        b.done()

        # ------------------------------------------------------------------ b2
        b = Beat(self, "S03_Quaternions_b2")
        for m in (obj, bars, vals, axis, obj_lab):
            m.clear_updaters()
        # -q: antipodal point, joined through the centre
        mq_dot = Dot(sph_pt(-Q_DIR), radius=0.11, color=C_PRED)
        mq_lab = Tex("-q", font_size=42, color=C_PRED).next_to(mq_dot, DL, buff=0.06)
        chord = DashedLine(qdot.get_center(), mq_dot.get_center(), dash_length=0.12, stroke_width=2.5, color=C_PRED)
        cdot = Dot(SPH_C, radius=0.05, color=GREY_B)
        b.play(ShowCreation(chord), FadeIn(cdot), rt=0.8)
        b.play(GrowFromCenter(mq_dot), Write(mq_lab), rt=0.5)

        # every component negated, yet the rotation matrix is the same: a copy rotated by qmat(-q1) lands on the object
        ghost = make_object(qmat(-q1), OBJ_C + LEFT * 2.0 + DOWN * 0.4, opacity=0.5)
        arr_q = CurvedArrow(qdot.get_center() + RIGHT * 0.2, OBJ_C + LEFT * 1.3 + UP * 0.3, angle=-TAU / 8, color=C_PRED, stroke_width=3)
        arr_mq = CurvedArrow(mq_dot.get_center() + RIGHT * 0.25, OBJ_C + LEFT * 1.2 + DOWN * 0.4, angle=TAU / 10, color=C_PRED, stroke_width=3)
        b.play(Transform(bars, bars_for(-q1, BXF)), Transform(vals, vals_for(-q1, BXF)), rt=0.9)
        b.play(ShowCreation(arr_q), ShowCreation(arr_mq), FadeIn(ghost), rt=0.8)
        same = Tex(r"R(q) = R(-q)", font_size=38, color=C_PRED).move_to(np.array([3.3, 2.75, 0]))
        b.play(ghost.animate.shift(RIGHT * 2.0 + UP * 0.4).set_opacity(0.0), Write(same), rt=1.0)
        self.remove(ghost)

        # double cover: several antipodal pairs, one colour per pair
        rng = np.random.default_rng(3)
        pairs = VGroup()
        for k in range(5):
            v = rng.normal(size=3); v /= np.linalg.norm(v)
            if (VIEW @ v)[2] < 0:
                v = -v
            col = [GREY_A, "#F4A261", "#E76F51", "#A8DADC", "#CDB4DB"][k]   # neutral: not the GT/branch colours
            a, c = Dot(sph_pt(v), radius=0.075, color=col), Dot(sph_pt(-v), radius=0.075, color=col)
            c.set_opacity(0.55)
            pairs.add(VGroup(DashedLine(a.get_center(), c.get_center(), dash_length=0.08, stroke_width=1.5, color=col).set_opacity(0.5), a, c))
        two = Tex(r"\text{each rotation twice}", font_size=28, color=GREY_A).move_to(SPH_C + DOWN * (SPH_R + 0.9))
        b.play(LaggedStartMap(FadeIn, pairs, lag_ratio=0.25), FadeIn(two), rt=1.3)

        # sandwich product: rotate a vector v by q
        b.play(FadeOut(pairs), FadeOut(arr_q), FadeOut(arr_mq), FadeOut(two),
               FadeOut(same), rt=0.5)
        sand = Tex(r"v' = q\,v\,\bar q", font_size=48).next_to(eq, DOWN, buff=0.25).align_to(eq, LEFT)
        v0 = VIEW.T @ np.array([-1.0, -0.45, 0.25]); v0 = v0 / np.linalg.norm(v0) * 1.9   # starts nearly in the screen plane
        s = ValueTracker(0.0)
        vec0 = Arrow(OBJ_C, OBJ_C + VIEW @ v0, buff=0, stroke_width=6, color=WHITE)
        vlab = Tex("v", font_size=36).next_to(vec0.get_end(), LEFT, buff=0.1)

        def vec_now():
            R = qmat(quat(AXIS, s.get_value() * ANG))
            return Arrow(OBJ_C, OBJ_C + VIEW @ (R @ v0), buff=0, stroke_width=6, color=C_PRED)
        vec = vec_now(); vec.add_updater(lambda m: m.become(vec_now()))
        trace = TracedPath(lambda: vec.get_end(), stroke_color=C_PRED, stroke_width=3)
        # v is turned by the same rotation q (axis-angle path from identity to q), ending at v' = R(q) v
        b.play(Write(sand), GrowArrow(vec0), FadeIn(vlab), obj.animate.set_opacity(0.25), rt=0.9)
        self.add(trace, vec)
        b.play(s.animate.set_value(1.0), rt=1.6)
        vec.clear_updaters(); trace.clear_updaters()
        vlab2 = Tex("v'", font_size=36, color=C_PRED).next_to(vec.get_end(), RIGHT, buff=0.1)
        b.play(FadeIn(vlab2), rt=0.35)

        # geodesic distance between two rotations
        qa = q1
        qb = qmul(quat([0.3, -1.0, 0.5], 50 * DEGREES), q1)
        dot_ab = float(np.dot(qa, qb))
        delta = geodesic_deg(qa, qb)
        phi = np.arccos(abs(dot_ab))               # angle on S^3 between q_1 and the nearer of +-q_2
        perp = np.cross(Q_DIR, [0.0, 0.0, 1.0]); perp /= np.linalg.norm(perp)
        qa_dir = Q_DIR / np.linalg.norm(Q_DIR)
        qb_dir = np.cos(phi) * qa_dir + np.sin(phi) * perp
        if (VIEW @ qb_dir)[2] < 0:
            qb_dir = np.cos(phi) * qa_dir - np.sin(phi) * perp
        q2dot = Dot(sph_pt(qb_dir), radius=0.11, color=C_GT)
        q2lab = Tex("q_2", font_size=40, color=C_GT).next_to(q2dot, UL, buff=0.06)
        q1lab = Tex("q_1", font_size=40, color=C_PRED).move_to(qlab)
        arc = great_arc(qa_dir, qb_dir, color=WHITE, stroke_width=5)
        dist = Tex(r"\delta(q_1,q_2) = 2\arccos|\langle q_1, q_2\rangle|", font_size=44,
                   t2c={"q_1": C_PRED, "q_2": C_GT}).to_edge(DOWN, buff=0.4).shift(RIGHT * 2.9)
        objA = make_object(qmat(qa), OBJ_C + LEFT * 1.0 + DOWN * 0.1)
        objB = make_object(qmat(qb), OBJ_C + RIGHT * 1.9 + DOWN * 0.1)
        la = Tex("q_1", font_size=34, color=C_PRED).move_to(OBJ_C + LEFT * 1.0 + DOWN * 1.45)
        lb = Tex("q_2", font_size=34, color=C_GT).move_to(OBJ_C + RIGHT * 1.9 + DOWN * 1.45)
        b.play(FadeOut(VGroup(vec0, vlab, vlab2, trace, vec, sand, bars, vals, axis, mq_dot, mq_lab, chord, obj_lab)),
               FadeOut(obj), FadeIn(objA), FadeIn(objB), FadeIn(la), FadeIn(lb),
               ReplacementTransform(qlab, q1lab), GrowFromCenter(q2dot), Write(q2lab), rt=0.9)
        b.play(ShowCreation(arc), Write(dist), rt=1.4)

        ip = Tex(r"\langle q_1,q_2\rangle =", font_size=36).move_to(np.array([2.3, -1.2, 0]))
        ipv = DecimalNumber(dot_ab, num_decimal_places=3, include_sign=True, font_size=36).next_to(ip, RIGHT, buff=0.15)
        dl = Tex(r"\delta =", font_size=36).next_to(ip, DOWN, buff=0.35).align_to(ip, LEFT)
        dv = DecimalNumber(delta, num_decimal_places=1, font_size=36, color=WHITE).next_to(dl, RIGHT, buff=0.15)
        deg = Tex(r"^\circ", font_size=36).next_to(dv, RIGHT, buff=0.04).align_to(dv, UP)
        b.play(FadeIn(ip), FadeIn(ipv), FadeIn(dl), FadeIn(dv), FadeIn(deg), rt=0.7)
        # swap q_2 for -q_2: the dot product flips sign, its absolute value and delta do not change
        mq2 = Dot(sph_pt(-qb_dir), radius=0.11, color=C_GT)
        mq2lab = Tex("-q_2", font_size=40, color=C_GT).next_to(mq2, DOWN, buff=0.08)
        chord2 = DashedLine(q2dot.get_center(), mq2.get_center(), dash_length=0.1, stroke_width=2, color=C_GT)
        ip2 = Tex(r"\langle q_1,-q_2\rangle =", font_size=36).move_to(ip).align_to(ip, LEFT)
        b.play(ShowCreation(chord2), GrowFromCenter(mq2), Write(mq2lab), rt=0.7)
        b.play(Transform(ip, ip2), ipv.animate.set_value(-dot_ab).next_to(ip2, RIGHT, buff=0.15),
               Indicate(VGroup(dv, deg), color=C_GT, scale_factor=1.25), rt=1.0)
        b.done()
