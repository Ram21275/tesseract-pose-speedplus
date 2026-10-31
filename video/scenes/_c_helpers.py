"""Small shared helpers for scenes S11-S16 (labels, bars, image-pixel mapping). Not a scene."""
import json
import os
from pathlib import Path

from common import *  # noqa: F401,F403

ASSETS = Path("/media/kavinder/hdd2/RAM/video/assets")
FONT = "DejaVu Sans"


def lbl(s, fs=28, color=WHITE, **kw):
    return Text(s, font=FONT, font_size=fs, color=color, **kw)


def assets_json():
    return json.loads((ASSETS / "s11_s14.json").read_text())


def px(img, x, y, W=1920, H=1200):
    """Scene point of image pixel (x, y) for an ImageMobject `img` showing a W x H image."""
    ul = img.get_corner(UL); w, h = img.get_width(), img.get_height()
    return ul + RIGHT * (x / W * w) + DOWN * (y / H * h)


def vbar(x, y0, h, w=0.6, color=C_PRED, opacity=0.9):
    r = Rectangle(width=w, height=max(h, 1e-3), fill_color=color, fill_opacity=opacity, stroke_width=0)
    r.move_to([x, y0, 0], aligned_edge=DOWN); return r


def hbar(x0, y, L, h=0.4, color=C_PRED, opacity=0.9):
    r = Rectangle(width=max(L, 1e-3), height=h, fill_color=color, fill_opacity=opacity, stroke_width=0)
    r.move_to([x0, y, 0], aligned_edge=LEFT); return r


def grow_up(m):
    return GrowFromEdge(m, DOWN)


def grow_right(m):
    return GrowFromEdge(m, LEFT)


def lattice(rows, cols, w, h, color=GREY_B, dot_r=0.025, stroke=0.6, opacity=0.6):
    """A rectangular lattice of dots joined by thin edges (a 'network' drawing)."""
    xs = np.linspace(-w / 2, w / 2, cols); ys = np.linspace(-h / 2, h / 2, rows)
    pts = np.array([[x, y, 0] for y in ys for x in xs])
    dots = DotCloud(pts, radius=dot_r, color=color)
    edges = VGroup(*[Line([xs[0], y, 0], [xs[-1], y, 0]) for y in ys],
                   *[Line([x, ys[0], 0], [x, ys[-1], 0]) for x in xs])
    edges.set_stroke(color, stroke, opacity)
    return Group(edges, dots)


# ---------------------------------------------------------------- schematic satellite (box + panel + antennas)
def rotm(axis, ang):
    return qmat(quat(axis, ang))


VIEW = rotm([1, 0, 0], -1.05) @ rotm([0, 0, 1], 0.6)      # fixed oblique view: body -> screen (x right, y up, z to viewer)
_BOX = np.array([0.8, 0.8, 0.45])
_ANT = [((0.8, 0.8, -0.45), (1.6, 1.6, -0.9), 1.0), ((-0.8, -0.8, -0.45), (-1.6, -1.6, -0.9), 1.0),
        ((0.8, -0.3, -0.45), (1.15, -0.45, -0.75), 0.8)]                 # last one: the small asymmetric stub


def _faces():
    f = []
    for ax in range(3):
        for sgn in (-1, 1):
            n = np.zeros(3); n[ax] = sgn
            u, v = [i for i in range(3) if i != ax]
            q = []
            for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                p = np.zeros(3); p[ax] = sgn * _BOX[ax]; p[u] = a * _BOX[u]; p[v] = b * _BOX[v]; q.append(p)
            f.append((np.array(q), n, "body"))
    z = _BOX[2] + 0.06; s = 1.0
    f.append((np.array([[-s, -s, z], [s, -s, z], [s, s, z], [-s, s, z]]), np.array([0, 0, 1.0]), "panel"))
    return f


FACES = _faces()


def sat_vg(R, scale=1.0, center=ORIGIN, body_color="#C9CED6", panel_color="#2B3A67", stroke=WHITE, ant_color="#E0E0E0",
           opacity=1.0):
    """Orthographic drawing of the schematic satellite under body rotation R (then the fixed VIEW)."""
    M = VIEW @ R; light = np.array([0.3, 0.5, 0.8]); light /= np.linalg.norm(light)
    items = []
    for pts, n, kind in FACES:
        P = pts @ M.T; nn = M @ n
        if kind == "body" and nn[2] <= 0:
            continue
        shade = 0.45 + 0.55 * abs(float(nn @ light))
        col = interpolate_color(BLACK, body_color if kind == "body" else panel_color, shade)
        poly = Polygon(*[np.array([p[0], p[1], 0]) for p in P]).set_fill(col, opacity).set_stroke(stroke, 1.2, 0.7 * opacity)
        if kind == "panel":   # panel cell lines
            grid = VGroup()
            for t in np.linspace(-0.6, 0.6, 4):
                for a, b in [((t, -1, 0), (t, 1, 0)), ((-1, t, 0), (1, t, 0))]:
                    A = M @ (np.array(a) + [0, 0, _BOX[2] + 0.06]); B = M @ (np.array(b) + [0, 0, _BOX[2] + 0.06])
                    grid.add(Line([A[0], A[1], 0], [B[0], B[1], 0]).set_stroke("#8FA3D9", 0.8, 0.6 * opacity))
            poly = VGroup(poly, grid)
        items.append((float(P[:, 2].mean()), poly))
    for a, b, w in _ANT:
        A = M @ np.array(a); B = M @ np.array(b)
        items.append((float((A[2] + B[2]) / 2) + 0.05, Line([A[0], A[1], 0], [B[0], B[1], 0]).set_stroke(ant_color, 3 * w, opacity)))
    items.sort(key=lambda x: x[0])
    g = VGroup(*[m for _, m in items])
    g.scale(scale, about_point=ORIGIN).shift(center)
    return g


def sat_silhouette(R, scale=1.0, center=ORIGIN, color=C_GT, width=3):
    """Outline-only drawing (for overlays)."""
    g = sat_vg(R, scale, center, opacity=0.0)
    for m in g.family_members_with_points():
        m.set_fill(opacity=0)
    g.set_stroke(color, width, 1.0)
    return g


if os.environ.get("BEAT_SLACK"):                      # diagnostics only: print the unanimated tail of every beat
    import common as _common
    _orig_done = _common.Beat.done

    def _done(self):
        # the trailing b.wait(...) padding counts as idle too: it is recorded in self._pad by the scenes via b.wait
        print(f"[slack] {self.key}: T={self.T:.2f} animated={self.used - getattr(self, '_pad', 0):.2f}", flush=True)
        _orig_done(self)
    _orig_wait = _common.Beat.wait

    def _wait(self, t):
        self._pad = getattr(self, "_pad", 0) + t
        _orig_wait(self, t)
    _common.Beat.done = _done; _common.Beat.wait = _wait
