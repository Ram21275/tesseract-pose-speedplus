"""HERO scene: the Tesseract partition of SO(3), built from the paper's own encoder/decoder (src/tfpose/tesseract.py).
Everything drawn is computed: 4D hypercube vertices/edges under a real 4D rotation and perspective projection, the chart
choice argmax|q_i|, the sign canonicalisation, the projection x = q/q_c, the recursive cell boxes (cell_box), the decoded
path (encode), cell centres (decode) and the per-level geodesic error of the example rotation."""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa: F401,F403,E402

SC = "S04_Tesseract"
Q = quat([0.62, -0.35, 0.70], 2.25)            # example rotation (fixed); its chart/path are computed below
CH, QS, U = free_coords(Q)
CHART, PATH = encode(Q, 5)
assert CHART == CH
IDX_NAMES = ["w", "x", "y", "z"]
FREE = [i for i in range(4) if i != CH]
S = 2.7                                        # world units per unit of projected coordinate
CUBE_C = LEFT * 3.2                            # where the projected cube lives in the world
MEAN_ERR = [39.27, 19.95, 10.01, 5.01, 2.51]  # paper Tab. (depth 1..5, mean own-cell error, 2e6 Haar rotations)


def to_world(u):
    return CUBE_C + S * np.array([u[0], u[1], u[2]])


def box_mob(lo, hi, color, fill=0.0, width=2.0):
    side = (hi[0] - lo[0]) * S
    b = VCube(side_length=side)
    b.set_stroke(color, width, opacity=1.0).set_fill(color, opacity=fill)
    b.move_to(to_world((lo + hi) / 2))
    return b


def tesseract_points(t, dist=3.2, scale=1.25):
    verts = np.array(list(itertools.product([-1, 1], repeat=4)), float)
    a, b = t, 0.6 * t
    R = np.eye(4)
    R[[0, 0, 3, 3], [0, 3, 0, 3]] = [np.cos(a), -np.sin(a), np.sin(a), np.cos(a)]      # rotation in the (0,3) plane
    R2 = np.eye(4)
    R2[[1, 1, 2, 2], [1, 2, 1, 2]] = [np.cos(b), -np.sin(b), np.sin(b), np.cos(b)]     # rotation in the (1,2) plane
    v = verts @ (R2 @ R).T
    w = v[:, 0]
    p = v[:, 1:] / (dist - w)[:, None] * dist * scale                                    # 4D -> 3D perspective
    return verts, p


EDGES = [(i, j) for i, j in itertools.combinations(range(16), 2)
         if np.sum(np.abs(np.array(list(itertools.product([-1, 1], repeat=4)))[i]
                          - np.array(list(itertools.product([-1, 1], repeat=4)))[j])) == 2]


def rel_axis_angle(R):
    """Axis-angle of a rotation matrix (for a true rigid rotation animation)."""
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if ang < 1e-8:
        return 0.0, OUT
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    return float(ang), ax / np.linalg.norm(ax)


class S04_Tesseract(ThreeDScene):
    default_camera_config = dict(background_color=BG)

    def construct(self):
        frame = self.frame
        frame.set_euler_angles(theta=-35 * DEGREES, phi=68 * DEGREES)
        frame.set_height(11).move_to(RIGHT * 0.8)

        # ------------------------------------------------ b1: tesseract and the chart rule
        b = Beat(self, f"{SC}_b1")
        tt = ValueTracker(0.0)

        def build_tess():
            verts, p = tesseract_points(tt.get_value(), scale=1.9)
            g = VGroup()
            for i, j in EDGES:
                on_pos = verts[i][CH] == 1 and verts[j][CH] == 1
                on_neg = verts[i][CH] == -1 and verts[j][CH] == -1
                col = CHART_COLORS[CH] if on_pos else (interpolate_color(CHART_COLORS[CH], BG, 0.55) if on_neg else GREY_B)
                g.add(Line(p[i] + LEFT * 1.5, p[j] + LEFT * 1.5, stroke_width=4.5 if on_pos else (3 if on_neg else 1.6), color=col))
            return g

        tess = always_redraw(build_tess)
        sph = Sphere(radius=1.45, resolution=(48, 24)).set_color(BLUE_E, opacity=0.22).move_to(LEFT * 1.5)
        lbl_t = Tex(r"S^3 \subset \text{tesseract } [-1,1]^4", font_size=44).to_corner(UL).fix_in_frame()
        b.play(ShowCreation(tess), FadeIn(sph), Write(lbl_t), frac=0.22)
        BX = np.array([3.6, -3.0, 0])
        def make_bars(vec, fixed=True):
            g = VGroup()
            for i in range(4):
                r = Rectangle(width=0.62, height=max(abs(vec[i]) * 2.2, 0.03)).set_stroke(width=0)
                r.set_fill(CHART_COLORS[i] if i == CH else GREY_C, opacity=0.95)
                r.move_to(BX + RIGHT * 0.95 * i, aligned_edge=DOWN)
                g.add(r)
            return g.fix_in_frame() if fixed else g
        bars = make_bars(np.abs(Q))
        names = VGroup(*[Tex(IDX_NAMES[i], font_size=36).next_to(bars[i], DOWN, buff=0.15).set_y(BX[1] - 0.3) for i in range(4)]).fix_in_frame()
        vals = VGroup(*[DecimalNumber(Q[i], num_decimal_places=2, font_size=28).move_to(BX + RIGHT * 0.95 * i + UP * 2.55)
                        for i in range(4)]).fix_in_frame()
        qlab = Tex(r"q=(w,x,y,z)", font_size=36).next_to(vals, UP, buff=0.3).fix_in_frame()
        b.play(tt.animate.set_value(0.9), FadeIn(bars, lag_ratio=0.2), FadeIn(names), FadeIn(vals), Write(qlab), frac=0.33)
        eq_c = Tex(r"c=\arg\max_i |q_i| = " + IDX_NAMES[CH], font_size=46).set_y(2.9).to_edge(RIGHT, buff=0.45).fix_in_frame()
        eq_c[-1].set_color(CHART_COLORS[CH])
        b.play(tt.animate.set_value(1.7), Write(eq_c), bars[CH].animate.set_stroke(WHITE, 3), frac=0.33)
        b.done()

        # ------------------------------------------------ b2: sign flip and projection onto the facet cube
        b = Beat(self, f"{SC}_b2")
        X = QS / QS[CH]
        eq_sign = Tex(r"q \leftarrow \operatorname{sign}(q_c)\, q\qquad (q \equiv -q)", font_size=40).set_y(2.0).to_edge(RIGHT, buff=0.45).fix_in_frame()
        b.play(FadeOut(qlab), Write(eq_sign), tt.animate.set_value(2.2), frac=0.16)
        b.play(Transform(bars, make_bars(QS)), *[ChangeDecimalToValue(vals[i], QS[i]) for i in range(4)], frac=0.12)
        eq_proj = Tex(r"x = q / q_c,\qquad u=(x_j)_{j\ne c}\in[-1,1]^3", font_size=40).next_to(eq_sign, DOWN, buff=0.35).to_edge(RIGHT, buff=0.45).fix_in_frame()
        b.play(Write(eq_proj), Transform(bars, make_bars(X)), *[ChangeDecimalToValue(vals[i], X[i]) for i in range(4)], frac=0.2)
        tess.clear_updaters()
        cube = box_mob(-np.ones(3), np.ones(3), CHART_COLORS[CH], fill=0.0, width=3.5)
        axes = Group(*[Line3D(to_world(-np.eye(3)[k] * 1.2), to_world(np.eye(3)[k] * 1.2), width=0.012, color=GREY_B) for k in range(3)])
        ax_l = VGroup(*[Tex(f"u_{k}", font_size=40).move_to(to_world(np.eye(3)[k] * 1.38)) for k in range(3)])
        for t_ in ax_l:
            t_.rotate(PI / 2, RIGHT)
        pt = Sphere(radius=0.09).set_color(C_GT).move_to(to_world(U))
        b.play(FadeOut(tess), FadeOut(sph), FadeOut(lbl_t), ShowCreation(cube), ShowCreation(axes), FadeIn(ax_l),
               frame.animate.set_euler_angles(theta=-50 * DEGREES, phi=66 * DEGREES).move_to(CUBE_C + RIGHT * 2.9).set_height(10.8),
               frac=0.25)
        drops = VGroup(*[DashedLine(to_world(U), to_world(np.where(np.arange(3) == k, U, [U[0], U[1], -1.0])), color=C_GT)
                         for k in range(2)])
        b.play(FadeIn(pt, scale=3), ShowCreation(drops), frac=0.12)
        b.play(frame.animate.increment_theta(25 * DEGREES), FadeOut(drops), frac=0.12)
        b.done()

        # ------------------------------------------------ b3: first split into 8 children (bits)
        b = Beat(self, f"{SC}_b3")
        planes = Group()
        for k in range(3):
            sq = Square3D(side_length=2 * S).set_color(WHITE, opacity=0.08)
            if k == 0:
                sq.rotate(PI / 2, UP)
            elif k == 1:
                sq.rotate(PI / 2, RIGHT)
            planes.add(sq.move_to(to_world(np.zeros(3))))
        children = VGroup(*[box_mob(*cell_box([kid]), GREY_B, fill=0.0, width=1.4) for kid in range(8)])
        b.play(FadeIn(planes, lag_ratio=0.3), ShowCreation(children, lag_ratio=0.06), frac=0.3)
        bits = [(PATH[0] >> j) & 1 for j in range(3)]
        sel = box_mob(*cell_box(PATH[:1]), LEVEL_COLORS[1], fill=0.25, width=3.5)
        bit_rule = Tex(r"b_i = [\,u_i \ge \text{midpoint}\,]", font_size=40).set_y(-1.2).to_edge(RIGHT, buff=0.45).fix_in_frame()
        eq_k = Tex(r"k = b_0 + 2b_1 + 4b_2 = " + f"{bits[0]} + 2\\cdot{bits[1]} + 4\\cdot{bits[2]} = {PATH[0]}",
                   font_size=40).next_to(bit_rule, DOWN, buff=0.35).to_edge(RIGHT, buff=0.45).fix_in_frame()
        b.play(FadeIn(sel), Write(bit_rule), FadeOut(eq_sign), FadeOut(eq_proj), FadeOut(bars), FadeOut(names), FadeOut(vals), frac=0.3)
        b.play(Write(eq_k), frac=0.25)
        b.done()

        # ------------------------------------------------ b4: recurse to depth 5 by re-scaling the chosen child
        b = Beat(self, f"{SC}_b4")
        digits = [IDX_NAMES[CH]] + [str(k) for k in PATH]
        path_tex = Tex(r"\text{path}(q) = (" + digits[0] + r";\ " + r",\ ".join(digits[1:]) + ")", font_size=42)
        path_tex.set_y(2.9).to_edge(RIGHT, buff=0.45).fix_in_frame()
        count = Tex(r"4 \times 8^{5} = 131{,}072 \text{ cells}", font_size=44).set_y(-2.9).to_edge(RIGHT, buff=0.45).fix_in_frame()
        lvl_lab = Tex(r"\ell = 1", font_size=40).set_y(2.1).to_edge(RIGHT, buff=0.45).fix_in_frame()
        b.play(FadeOut(bit_rule), FadeOut(eq_k), FadeOut(eq_c), FadeIn(path_tex), FadeIn(lvl_lab), frac=0.1)
        cur = sel
        per = (0.82 * b.T) / 4
        for lvl in range(2, 6):
            # grow the chosen child to the full cube outline (re-scaling = zoom), keeping u inside it
            lo, hi = cell_box(PATH[:lvl - 1])
            u_rel = (U - lo) / (hi - lo) * 2 - 1
            big = box_mob(-np.ones(3), np.ones(3), LEVEL_COLORS[lvl - 1], fill=0.0, width=3)
            b.play(Transform(cur, big), FadeOut(children), FadeOut(planes), pt.animate.move_to(to_world(u_rel)),
                   Transform(lvl_lab, Tex(rf"\ell = {lvl}", font_size=40).move_to(lvl_lab).fix_in_frame()), rt=per * 0.45)
            children = VGroup(*[box_mob(*cell_box([k]), GREY_B, fill=0.0, width=1.2) for k in range(8)])
            planes = Group()
            ksel = PATH[lvl - 1]
            nsel = box_mob(*cell_box([ksel]), LEVEL_COLORS[min(lvl, 5)], fill=0.25, width=3.5)
            b.play(ShowCreation(children, lag_ratio=0.04), rt=per * 0.25)
            b.play(FadeIn(nsel), path_tex.animate.set_opacity(1), rt=per * 0.3)
            self.remove(cur); cur = nsel
        b.play(Write(count), frac=0.08)
        b.done()

        # ------------------------------------------------ b5: centres, not corners; nested cells at true scale
        b = Beat(self, f"{SC}_b5")
        nested = VGroup(*[box_mob(*cell_box(PATH[:l]), LEVEL_COLORS[min(l, 5)], fill=0.12 if l == 5 else 0.0, width=2.5)
                          for l in range(1, 6)])
        b.play(FadeOut(children), FadeOut(cur), FadeOut(lvl_lab), pt.animate.move_to(to_world(U)), FadeIn(cube),
               ShowCreation(nested, lag_ratio=0.25), frac=0.3)
        lo5, hi5 = cell_box(PATH)
        c5 = Sphere(radius=0.06).set_color(C_PRED).move_to(to_world((lo5 + hi5) / 2))
        lift = Tex(r"\hat q = x/\|x\|,\quad x_c = 1,\ \ x_{j\ne c} = \text{cell centre}", font_size=38).set_y(1.9).to_edge(RIGHT, buff=0.45).fix_in_frame()
        b.play(FadeIn(c5, scale=4), Write(lift), frac=0.22)
        corner = Sphere(radius=0.11).set_color(C_FLIP).move_to(to_world(np.zeros(3)))
        kids1 = VGroup(*[box_mob(*cell_box([k]), C_FLIP, fill=0.04, width=1.5) for k in range(8)])
        note = VGroup(Tex(r"\text{corner: shared by 8 cells}", font_size=36, color=C_FLIP),
                      Tex(r"\text{centre: in exactly one cell}", font_size=36, color=C_PRED)).arrange(DOWN, aligned_edge=LEFT)
        note.set_y(-1.6).to_edge(RIGHT, buff=0.45).fix_in_frame()
        b.play(FadeIn(corner, scale=3), ShowCreation(kids1, lag_ratio=0.05), Write(note), frac=0.3)
        b.play(FadeOut(kids1), FadeOut(corner), frac=0.1)
        b.done()

        # ------------------------------------------------ b6: decoding: the object turns to each level's centre
        b = Beat(self, f"{SC}_b6")
        def bracket():
            base = Prism(2.6, 0.8, 0.5).set_color(GREY_B)
            arm = Prism(0.5, 0.8, 1.8).set_color(GREY_B).shift(RIGHT * 1.05 + OUT * 1.15)
            stub = Prism(0.45, 0.45, 0.45).set_color(C_PRED).shift(LEFT * 0.95 + OUT * 0.47 + UP * 0.18)
            g = Group(base, arm, stub); g.scale(1.6); g.apply_depth_test(); return g
        OBJ_C = CUBE_C + RIGHT * 8.2
        gt_ghost = bracket()
        for m in gt_ghost:
            m.set_color(C_GT, opacity=0.5)
        gt_ghost.apply_matrix(qmat(Q)).move_to(OBJ_C)
        R0 = qmat(centre(CHART, []))
        pred = bracket(); pred.apply_matrix(R0).move_to(OBJ_C)
        b.play(FadeOut(lift), FadeOut(note), FadeOut(path_tex), FadeOut(count),
               frame.animate.move_to((CUBE_C + OBJ_C) / 2).set_height(11.5).set_euler_angles(theta=-15 * DEGREES, phi=62 * DEGREES),
               FadeIn(gt_ghost), FadeIn(pred), frac=0.14)
        errs = [geodesic_deg(Q, centre(CHART, PATH[:l])) for l in range(0, 6)]
        lab_obj = VGroup(Tex(r"\text{true pose}", font_size=34, color=C_GT), Tex(r"\text{decoded cell centre}", font_size=34, color=C_PRED))
        lab_obj[0].set_color(C_GT); lab_obj[1].set_color(C_PRED)
        lab_obj.arrange(RIGHT, buff=0.6).to_edge(UP).shift(RIGHT * 2.5).fix_in_frame()
        err_t = Tex(r"\delta(q,\hat q_\ell)=", font_size=44).to_corner(UL).shift(DOWN * 0.2).fix_in_frame()
        err_v = DecimalNumber(errs[0], num_decimal_places=1, font_size=44).next_to(err_t, RIGHT).fix_in_frame()
        deg = Tex(r"^\circ", font_size=44).next_to(err_v, RIGHT, buff=0.05).align_to(err_v, UP).fix_in_frame()
        deg.add_updater(lambda m: m.next_to(err_v, RIGHT, buff=0.05).align_to(err_v, UP))
        lvl_t = Tex(r"\ell = 0 \ (\text{chart})", font_size=38).next_to(err_t, DOWN, aligned_edge=LEFT).fix_in_frame()
        mean_t = Tex(r"\text{mean over random rotations: } 39.3^\circ \to 20.0^\circ \to 10.0^\circ \to 5.0^\circ \to 2.51^\circ",
                     font_size=34).to_edge(DOWN).fix_in_frame()
        b.play(Write(err_t), FadeIn(err_v), FadeIn(deg), FadeIn(lvl_t), FadeIn(lab_obj), frac=0.08)
        Rprev = R0
        for l in range(1, 6):
            Rl = qmat(centre(CHART, PATH[:l]))
            ang, axv = rel_axis_angle(Rl @ Rprev.T)
            b.play(Rotate(pred, angle=ang, axis=axv, about_point=OBJ_C), ChangeDecimalToValue(err_v, errs[l]),
                   Transform(lvl_t, Tex(rf"\ell = {l}", font_size=38).move_to(lvl_t, aligned_edge=LEFT).fix_in_frame()),
                   nested[l - 1].animate.set_stroke(WHITE, 4).set_fill(opacity=0.3), rt=0.11 * b.T)
            Rprev = Rl
        b.play(Write(mean_t), frame.animate.increment_theta(12 * DEGREES), frac=0.12)
        b.done()
