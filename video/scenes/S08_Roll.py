"""S08: an in-plane rotation of the crop is a camera roll about the crop-centre ray d (paper eq. roll).
Computed: pinhole projections with a real intrinsic matrix K; the rolled labels q' = Exp(-theta d) (x) q (qmul/quat) and
t' = R(d,-theta) t; the rolled camera's projections; the crop rotated by theta in the cv2 convention (counter-clockwise
on screen); and the homography K R(d,-theta) K^-1. Convention verified numerically (scratch check): on-axis the affine
crop rotation matches the rolled camera to 0.014 px, off-axis only the homography is exact (0 px)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa: F401,F403,E402


def Tx(*a, color=None, **k):
    """Tex with a colour (the color kwarg is ignored by this manimgl build)."""
    t = Tex(*a, **k)
    return t.set_color(color) if color is not None else t

SC = "S08_Roll"
FPX, CX, CY = 400.0, 320.0, 240.0                  # intrinsics of a 640x480 schematic camera
K = np.array([[FPX, 0, CX], [0, FPX, CY], [0, 0, 1.0]])
FW = 1.5                                            # image-plane distance in world units (FPX px <-> FW units)
PXU = FPX / FW
Q0 = quat([0.35, 0.85, -0.25], 1.0)                # object rotation (camera frame)
T0 = np.array([0.9, -0.55, 4.6])                    # object translation (camera frame: x right, y down, z forward)

# schematic object: body box + side panel + antenna stub (object frame); (centre, half sizes)
BOXES = [(np.array([0, 0, 0]), np.array([0.45, 0.32, 0.3])),
         (np.array([0.95, 0, 0]), np.array([0.5, 0.03, 0.26])),
         (np.array([-0.2, -0.47, 0.1]), np.array([0.06, 0.15, 0.06]))]
BOX_COL = [GREY_B, "#3A6EA5", C_GT]


def corners(c, h):
    return np.array([c + h * np.array(s) for s in itertools.product([-1, 1], repeat=3)])


import itertools  # noqa: E402

EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]


def Rcam(q):
    return qmat(q)


def W(p):
    """camera coordinates (x right, y down, z forward) -> world (x, z, -y)."""
    p = np.asarray(p, float)
    return np.array([p[..., 0], p[..., 2], -p[..., 1]]).T if p.ndim > 1 else np.array([p[0], p[2], -p[1]])


M = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)     # W(p) = M p


def proj_px(q, t, Km=None):
    """pixel coordinates of every object edge endpoint: list of (2,2) arrays."""
    Km = K if Km is None else Km
    R = Rcam(q); segs = []
    for c, h in BOXES:
        P = corners(c, h) @ R.T + t
        px = (P @ Km.T); px = px[:, :2] / px[:, 2:]
        segs += [np.stack([px[i], px[j]]) for i, j in EDGES]
    return segs


def unit(v):
    return v / np.linalg.norm(v)


C_PX = (K @ T0)[:2] / T0[2]                       # crop centre = projected object centre
D = unit(np.linalg.solve(K, np.r_[C_PX, 1.0]))    # unit ray through the crop centre
CROP = 270.0                                       # crop side in px


def rot2(th):   # cv2 rotation acting on y-down pixel offsets (counter-clockwise on screen)
    return np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])


class S08_Roll(ThreeDScene):
    default_camera_config = dict(background_color=BG)

    def construct(self):
        frame = self.frame
        frame.set_euler_angles(theta=-62 * DEGREES, phi=72 * DEGREES)
        frame.move_to(np.array([0.7, 1.6, 0.2])).set_height(5.6)
        th = ValueTracker(0.0)              # camera roll (b1) / label roll (b2), radians

        # ---------------- camera rig in camera coordinates, drawn in world coordinates
        def rig(theta):
            Rr = qmat(quat(D, theta))       # the camera rotates by +theta about d (points transform by R(d,-theta))
            hw, hh = 320 / PXU, 240 / PXU
            cs = [np.array([sx * hw, sy * hh, FW]) for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
            cs = [Rr @ c for c in cs]
            g = VGroup()
            plane = Polygon(*[W(c) for c in cs]).set_fill(GREY_E, 0.55).set_stroke(GREY_A, 2)
            g.add(plane)
            for c in cs:
                g.add(Line(ORIGIN, W(c) * 1.0, stroke_width=1.5, color=GREY_B))
            # crop square on the image plane, centred on the projected object centre
            cc = np.r_[(C_PX - [CX, CY]) / PXU, FW]
            a = CROP / 2 / PXU
            sq = [cc + np.array([sx * a, sy * a, 0]) for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
            sq = [Rr @ s for s in sq]
            g.add(Polygon(*[W(s) for s in sq]).set_stroke(C_PRED, 3).set_fill(C_PRED, 0.08))
            # small camera body
            g.add(Dot(ORIGIN, radius=0.06, fill_color=WHITE))
            # camera "up" tick (shows the roll): from image-plane top edge centre
            up0 = Rr @ np.array([0, -hh, FW]); up1 = Rr @ np.array([0, -hh - 0.25, FW])
            g.add(Line(W(up0), W(up1), stroke_width=4, color=C_PRED))
            return g

        def obj(q, t, col_override=None):
            R = M @ Rcam(q) @ M.T
            g = Group()
            for (c, h), col in zip(BOXES, BOX_COL):
                pr = Prism(*(2 * np.array([h[0], h[2], h[1]])))           # world box dims (x, z_cam, y_cam)
                pr.set_color(col_override or col, opacity=0.95)
                pr.apply_matrix(R)
                pr.move_to(W(t) + R @ W(c))
                g.add(pr)
            return g

        def img_outline(q, t, theta, color):
            """projection of the object onto the (rolled) image plane, drawn in 3D."""
            Rr = qmat(quat(D, theta))
            g = VGroup()
            for s in proj_px(q, t):
                p3 = [Rr @ np.r_[(x - [CX, CY]) / PXU, FW] for x in s]
                g.add(Line(W(p3[0]), W(p3[1]), stroke_width=1.6, color=color))
            return g

        # ---------------- 2D inset: what the crop looks like
        IN_C = np.array([4.6, 1.2, 0]); IN_S = 2.6                  # inset centre and size on screen

        def inset_lines(segs_px, centre_px, color, width=2.2, rot=0.0):
            g = VGroup()
            Rm = rot2(rot)
            for s in segs_px:
                pts = []
                for x in s:
                    o = Rm @ (x - centre_px)                       # rotate the crop content (cv2 convention)
                    pts.append(IN_C + np.array([o[0], -o[1], 0]) / CROP * IN_S)
                g.add(Line(pts[0], pts[1], stroke_width=width, color=color))
            return g

        in_box = Square(IN_S).move_to(IN_C).set_stroke(C_PRED, 3).set_fill(BG, 0.92)
        in_lab = Text("crop", font_size=24).set_color(C_PRED).next_to(in_box, UP, buff=0.12).align_to(in_box, LEFT)
        segs0 = proj_px(Q0, T0)

        # ------------------------------------------------ b1: crop rotation = camera roll about d
        b = Beat(self, f"{SC}_b1")
        cam = rig(0.0)
        body = obj(Q0, T0)
        lab_s = Text("schematic object", font_size=20).set_color(GREY_B).fix_in_frame().to_corner(DL, buff=0.3)
        b.play(ShowCreation(cam), FadeIn(body), FadeIn(lab_s), frame.animate.increment_theta(8 * DEGREES), frac=0.16)
        # projection rays from the object's corners to the camera centre, then the image
        rays = VGroup()
        Rq = Rcam(Q0)
        for P in corners(*BOXES[0]) @ Rq.T + T0:
            rays.add(Line(W(P), ORIGIN, stroke_width=1.2, color=GREY_A).set_stroke(opacity=0.5))
        im = img_outline(Q0, T0, 0.0, WHITE)
        b.play(ShowCreation(rays, lag_ratio=0.05), frac=0.1)
        b.play(ShowCreation(im), FadeOut(rays), frac=0.08)
        dline = Line(ORIGIN, W(D * np.linalg.norm(T0) * 1.18), stroke_width=4, color=C_GT)
        dlab = Tx("d", font_size=44, color=C_GT).move_to(W(D * np.linalg.norm(T0) * 1.18) + np.array([0, 0, 0.3]))
        dlab.rotate(PI / 2, RIGHT).rotate(-62 * DEGREES + 90 * DEGREES, OUT)
        b.play(ShowCreation(dline), FadeIn(dlab), frac=0.1)
        # 2D inset of the crop: rotated crop (white) vs the rolled camera's own view (blue, dashed)
        self.add(in_box.fix_in_frame(), in_lab.fix_in_frame())
        inset_b = None
        inset_a = always_redraw(lambda: inset_lines(segs0, C_PX, WHITE, 1.6, rot=th.get_value()).fix_in_frame())

        def rolled_view():
            qq = qmul(quat(D, -th.get_value()), Q0); tt = qmat(quat(D, -th.get_value())) @ T0
            return inset_lines(proj_px(qq, tt), C_PX, C_PRED, 6.0).set_stroke(opacity=0.6).fix_in_frame()

        inset_b = always_redraw(rolled_view)
        self.add(inset_b)
        key = VGroup(Text("crop rotated by θ", font_size=20).set_color(WHITE),
                     Text("view of the rolled camera", font_size=20).set_color(C_PRED)).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        key.next_to(in_box, DOWN, buff=0.15).fix_in_frame()
        thv = DecimalNumber(0, num_decimal_places=0, font_size=34).fix_in_frame()
        thl = Tx(r"\theta =", font_size=34).fix_in_frame()
        thd = Tx(r"^\circ", font_size=34).fix_in_frame()

        def place_th():
            thl.next_to(in_box, UP, buff=0.12).align_to(in_box, RIGHT).shift(LEFT * 0.9)
            thv.set_value(np.degrees(th.get_value())).next_to(thl, RIGHT, buff=0.12)
            thd.next_to(thv, RIGHT, buff=0.04).align_to(thv, UP)
        place_th()
        thv.add_updater(lambda m: place_th())
        b.play(FadeIn(inset_a), FadeIn(key), FadeIn(thl), FadeIn(thv), FadeIn(thd), frac=0.1)
        cam_live = always_redraw(lambda: rig(th.get_value()))
        im_live = always_redraw(lambda: img_outline(qmul(quat(D, -th.get_value()), Q0), qmat(quat(D, -th.get_value())) @ T0,
                                                    th.get_value(), WHITE))
        self.remove(cam, im); self.add(cam_live, im_live)
        b.play(th.animate.set_value(50 * DEGREES), frame.animate.increment_theta(-10 * DEGREES), frac=0.3, rate_func=smooth)
        same = Text("identical", font_size=26).set_color(C_GT).next_to(key, DOWN, buff=0.12).fix_in_frame()
        b.play(FadeIn(same), in_box.animate.set_stroke(C_GT, 5), frac=0.1)
        b.done()

        # ------------------------------------------------ b2: the labels transform exactly
        b = Beat(self, f"{SC}_b2")
        eq = Tx(r"q' = \mathrm{Exp}(-\theta\, d)\otimes q,\qquad t'=R(d,-\theta)\,t", font_size=42).fix_in_frame()
        eq.to_edge(UP, buff=0.3).shift(LEFT * 1.2)
        b.play(th.animate.set_value(0.0), FadeOut(same), in_box.animate.set_stroke(C_PRED, 3), Write(eq), frac=0.18)
        # camera-frame view: the camera is fixed, the object and t turn by R(d, -theta) about the camera centre
        self.remove(cam_live, im_live); cam = rig(0.0); self.add(cam)
        im_fix = always_redraw(lambda: img_outline(qmul(quat(D, -th.get_value()), Q0), qmat(quat(D, -th.get_value())) @ T0,
                                                   0.0, WHITE))
        self.add(im_fix)

        def frame_axes(q, t, color):
            R = M @ Rcam(q) @ M.T
            return VGroup(*[Line(W(t), W(t) + R @ np.eye(3)[k] * 0.9, stroke_width=5, color=color) for k in range(3)])

        def body_live():
            a = th.get_value(); qq = qmul(quat(D, -a), Q0); tt = qmat(quat(D, -a)) @ T0
            return Group(obj(qq, tt), frame_axes(qq, tt, C_PRED), Line(ORIGIN, W(tt), stroke_width=3, color=C_PRED))

        ghost = VGroup(frame_axes(Q0, T0, C_GT), Line(ORIGIN, W(T0), stroke_width=3, color=C_GT))
        self.remove(body); bl = always_redraw(body_live); self.add(bl)
        b.play(FadeIn(ghost), frac=0.08)
        # live readouts of q' (sign-normalised, w >= 0) and theta
        names = ["w", "x", "y", "z"]
        qn = VGroup(*[DecimalNumber(0, num_decimal_places=3, font_size=28, include_sign=True) for _ in range(4)])
        qlab = Tx(r"q' =", font_size=32, color=C_PRED)
        rows = VGroup(*[VGroup(Tx(names[i], font_size=28, color=GREY_B), qn[i]).arrange(RIGHT, buff=0.15) for i in range(4)])
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        panel = VGroup(qlab, rows).arrange(RIGHT, buff=0.2).to_corner(DL, buff=0.45).shift(UP * 0.4).fix_in_frame()

        def upd_q(m):
            a = th.get_value(); qq = qmul(quat(D, -a), Q0); qq = qq * (1 if qq[0] >= 0 else -1)
            for i in range(4):
                qn[i].set_value(qq[i])
        upd_q(None); panel.add_updater(upd_q)
        b.play(FadeIn(panel), FadeOut(lab_s), frac=0.08)
        b.play(th.animate.set_value(70 * DEGREES), frac=0.36, rate_func=smooth)
        b.play(th.animate.set_value(35 * DEGREES), frame.animate.increment_theta(8 * DEGREES), frac=0.22, rate_func=smooth)
        b.done()

        # ------------------------------------------------ b3: off-axis object -> homography K R K^-1
        b = Beat(self, f"{SC}_b3")
        panel.clear_updaters(); thv.clear_updaters()
        fade3d = [m for m in self.mobjects if not m.is_fixed_in_frame()]
        for m in fade3d:
            m.clear_updaters()
        b.play(*[FadeOut(m) for m in self.mobjects], frac=0.07)
        frame.set_euler_angles(0, 0, 0); frame.move_to(ORIGIN).set_height(8)
        # wide camera, object far off-axis
        K2 = np.array([[330.0, 0, 320], [0, 330.0, 240], [0, 0, 1]])
        T2 = np.array([2.1, 1.35, 3.0]) * 1.5
        c2 = (K2 @ T2)[:2] / T2[2]
        d2 = unit(np.linalg.solve(K2, np.r_[c2, 1]))
        side = 200.0

        # full image thumbnail (top-left) with principal point and the crop
        TH_C = np.array([-4.9, 1.9, 0]); TH_S = 3.6 / 640

        def thumb(px):
            return TH_C + np.array([(px[0] - 320) * TH_S, -(px[1] - 240) * TH_S, 0])

        tb = Rectangle(width=640 * TH_S, height=480 * TH_S).move_to(TH_C).set_stroke(GREY_A, 2)
        pp = Dot(thumb([320, 240]), radius=0.05, fill_color=WHITE)
        ppl = Text("optical axis", font_size=16).set_color(GREY_A).next_to(pp, DOWN, buff=0.08)
        cr = Square(side * TH_S).move_to(thumb(c2)).set_stroke(C_PRED, 2.5)
        objt = VGroup(*[Line(thumb(s[0]), thumb(s[1]), stroke_width=1, color=WHITE) for s in proj_px(Q0, T2, K2)
                        if True])
        b.play(ShowCreation(tb), FadeIn(pp), FadeIn(ppl), FadeIn(objt), ShowCreation(cr), frac=0.08)

        # main panel: zoom on the crop region, pixel -> screen
        MC = np.array([2.3, -0.45, 0]); MS = 3.7 / side

        def scr(px):
            return MC + np.array([(px[0] - c2[0]) * MS, -(px[1] - c2[1]) * MS, 0])

        zoom = Square(side * MS).move_to(MC).set_stroke(C_PRED, 2.5)
        zl = DashedLine(cr.get_corner(DR), zoom.get_corner(UL), stroke_width=1.5, color=C_PRED)
        n = 7
        grid_px = []
        for i in range(n):
            u = -side / 2 + side * i / (n - 1)
            grid_px.append(np.array([[c2[0] + u, c2[1] - side / 2 + side * k / 40] for k in range(41)]))
            grid_px.append(np.array([[c2[0] - side / 2 + side * k / 40, c2[1] + u] for k in range(41)]))

        def poly(pts, color, w):
            return VMobject().set_points_as_corners([scr(p) for p in pts]).set_stroke(color, w)

        grid0 = VGroup(*[poly(g, GREY_C, 1.5) for g in grid_px])
        b.play(ShowCreation(zoom), ShowCreation(zl), ShowCreation(grid0, lag_ratio=0.02), frac=0.1)
        th3 = ValueTracker(0.0)

        def H(a):
            return K2 @ qmat(quat(d2, -a)) @ np.linalg.inv(K2)

        def warp(pts, Hm):
            p = np.c_[pts, np.ones(len(pts))] @ Hm.T
            return p[:, :2] / p[:, 2:]

        def affine(pts, a):
            return (pts - c2) @ rot2(a).T + c2

        g_aff = always_redraw(lambda: VGroup(*[poly(affine(g, th3.get_value()), GREY_B, 2.0) for g in grid_px]))
        g_hom = always_redraw(lambda: VGroup(*[poly(warp(g, H(th3.get_value())), C_PRED, 2.4) for g in grid_px]))

        def obj_lines(a):
            qq = qmul(quat(d2, -a), Q0); tt = qmat(quat(d2, -a)) @ T2
            return VGroup(*[Line(scr(s[0]), scr(s[1]), stroke_width=5, color=C_GT) for s in proj_px(qq, tt, K2)])

        def obj_aff(a):
            return VGroup(*[Line(scr(p[0]), scr(p[1]), stroke_width=1.6, color=GREY_A).set_stroke(opacity=0.8)
                            for p in [affine(s, a) for s in proj_px(Q0, T2, K2)]])

        o_true = always_redraw(lambda: obj_lines(th3.get_value()))

        def obj_hom(a):            # the original outline warped by H(theta): lies exactly on the rolled camera's view
            Hm = H(a)
            return VGroup(*[Line(scr(p[0]), scr(p[1]), stroke_width=1.4, color=C_PRED)
                            for p in [warp(np.asarray(sg), Hm) for sg in proj_px(Q0, T2, K2)]])

        o_hom = always_redraw(lambda: obj_hom(th3.get_value()))
        o_aff = always_redraw(lambda: obj_aff(th3.get_value()))
        b.play(FadeOut(grid0), FadeIn(g_aff), FadeIn(o_aff), frac=0.06)
        lab_a = Text("affine crop rotation", font_size=22).set_color(GREY_B)
        lab_h = Text("homography warp (exact)", font_size=22).set_color(C_PRED)
        heq = Tx(r"H = K\,R(d,-\theta)\,K^{-1}", font_size=48, color=C_PRED).move_to([2.3, 3.35, 0])
        lab_t = Text("rolled camera's true view", font_size=22).set_color(C_GT)
        legend = VGroup(lab_a, lab_h, lab_t).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([-4.6, -1.4, 0])
        b.play(th3.animate.set_value(38 * DEGREES), FadeIn(lab_a), frac=0.2)
        b.play(FadeIn(g_hom), FadeIn(o_true), FadeIn(o_hom), FadeIn(lab_t), frac=0.1)
        b.play(Write(heq), FadeIn(lab_h), frac=0.12)
        b.play(th3.animate.set_value(-30 * DEGREES), frac=0.22, rate_func=smooth)
        b.done()
