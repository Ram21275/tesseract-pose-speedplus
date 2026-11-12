"""Shared style for the manim paper figures (static, white background). Colours match the paper's other figures:
ground truth green, prediction magenta (scripts/fig_qualitative.py); frozen modules grey and dashed, trained modules coloured
(paper/figures/pipeline.pdf). Geometry comes from src/tfpose via ../common.py."""
import sys
from pathlib import Path

PF_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PF_DIR.parent))
from common import *  # noqa: F401,F403,E402  (manimlib, tesseract math, encode/decode helpers)

ASSET = PF_DIR / "assets"
INK = "#1A1A1A"; MUTED = "#6B7280"; PAPER = "#FFFFFF"
GT = "#1FAE3C"; PRED = "#E0189F"; FLIP = "#D62828"
OV_GT = "#22DD44"; OV_PRED = "#FF2BD6"                                    # overlay colours of scripts/fig_qualitative.py
FROZEN = "#8A94A6"; FROZEN_FILL = "#EEF1F5"
DINO = "#2F6DB5"; DINO_FILL = "#E3EEFA"; MOGE = "#C0662B"; MOGE_FILL = "#FBEBDD"
ORBIT = "#2E8B57"; ORBIT_FILL = "#E4F4EA"; TRANS = "#7B4FBF"; TRANS_FILL = "#EFE7FA"
LV = ["#264653", "#2A9D8F", "#8AB17D", "#E9C46A", "#F4A261", "#E76F51"]       # chart, levels 1..5
CH4 = ["#3A86FF", "#FF006E", "#2A9D8F", "#FB8500"]                              # charts w, x, y, z


FS = 1.3                                       # global text scale: full-width figures print at ~7 in, keep labels >= ~6 pt


def tx(s, size=30, color=INK, **k):
    """Math (LaTeX math mode)."""
    return Tex(s, font_size=size * FS, **k).set_color(color)


def tt(s, size=26, color=INK, bold=False, **k):
    """Text, typeset by LaTeX so it matches the paper's font."""
    return TexText(r"\textbf{" + s + "}" if bold else s, font_size=size * FS, **k).set_color(color)


ICE = "#3B9AD9"; FIRE = "#F26B1D"; FIRE_IN = "#FFC93C"
TRAIN = "#D97706"                                                              # training-only flow (amber); test-time flow uses ORBIT


def snowflake(size=0.34, color=ICE, sw=2.6):
    """Frozen-module icon (the usual snowflake): 6 arms with two side branches each."""
    g = VGroup(); L = size / 2
    for k in range(6):
        a = k * PI / 3; u = np.array([np.cos(a), np.sin(a), 0])
        g.add(Line(ORIGIN, L * u, stroke_width=sw))
        for s in (1, -1):
            b = a + s * PI / 4; v = np.array([np.cos(b), np.sin(b), 0])
            g.add(Line(0.55 * L * u, 0.55 * L * u + 0.38 * L * v, stroke_width=sw * 0.8))
    return g.set_color(color)


def _drop(h, w):
    s = np.linspace(0, 1, 60)                                   # height fraction; tip leans slightly right
    hw = w / 2 * np.sin(np.pi * s) ** 0.9 * (1 - s) ** 0.25
    x0 = 0.06 * w * s ** 2
    pts = [np.array([x0[i] + hw[i], h * s[i] - h / 2, 0]) for i in range(60)] + \
          [np.array([x0[i] - hw[i], h * s[i] - h / 2, 0]) for i in range(59, -1, -1)]
    return Polygon(*pts)


def flame(size=0.36):
    """Trained-module icon (the usual flame)."""
    o = _drop(size, size * 0.72).set_fill(FIRE, 1).set_stroke(width=0)
    i = _drop(size * 0.55, size * 0.4).set_fill(FIRE_IN, 1).set_stroke(width=0).move_to(o.get_bottom(), aligned_edge=DOWN).shift(UP * size * 0.04)
    return VGroup(o, i)


def badge(mob, icon, corner=UR, buff=0.08):
    """Put an icon inside the top-right corner of a box/block."""
    return icon.move_to(mob.get_corner(corner) + (-corner[0] * (icon.get_width() / 2 + buff)) * RIGHT + (-corner[1] * (icon.get_height() / 2 + buff)) * UP)


def box(w, h, color, fill, frozen=False, r=0.12, sw=3):
    bg = RoundedRectangle(width=w, height=h, corner_radius=r).set_fill(fill, 1).set_stroke(width=0)
    ol = RoundedRectangle(width=w, height=h, corner_radius=r).set_fill(opacity=0).set_stroke(color, sw)
    if frozen:
        ol = DashedVMobject(ol, num_dashes=int(2 * (w + h) / 0.16)).set_stroke(color, sw)
    return VGroup(bg, ol)


def block(title, lines=(), color=DINO, fill=DINO_FILL, w=None, h=None, frozen=False, tsize=26, lsize=21, buff=0.1, icon=None):
    """icon: None (default: snowflake on frozen blocks), "ice", "fire" or False."""
    content = VGroup(tt(title, tsize, bold=True), *[tt(l, lsize) for l in lines]).arrange(DOWN, buff=buff)
    icon = ("ice" if frozen else False) if icon is None else icon
    extra = 0.5 if icon else 0.0                                   # room for the corner icon
    W = w if w is not None else content.get_width() + 0.45 + extra; H = h if h is not None else content.get_height() + 0.4
    b = box(W, H, color, fill, frozen)
    g = VGroup(b, content.move_to(b).shift(LEFT * extra / 2 if w is None else ORIGIN))
    if icon:
        g.add(badge(b, snowflake() if icon == "ice" else flame()))
    return g


def arrow(a, b, color=INK, buff=0.08, sw=3, arc=0.0):
    return Arrow(a, b, buff=buff, thickness=sw, path_arc=arc).set_color(color)


def img(name, h=1.6, border=None):
    m = ImageMobject(str(ASSET / f"{name}.png")); m.set_height(h)
    if border:
        fr = SurroundingRectangle(m, buff=0.0).set_stroke(border, 4)
        return Group(m, fr)
    return m


def letter(s, size=30):
    return tt(f"({s})", size, bold=True)


def fit(scene, mobs, pad=0.25):
    """Point the camera at the content: the frame's aspect comes from the render resolution."""
    g = Group(*mobs); fr = scene.frame; aspect = fr.get_width() / fr.get_height()
    w, h = g.get_width() + 2 * pad, g.get_height() + 2 * pad
    fr.move_to(g.get_center()); fr.set_height(max(h, w / aspect))


def iso(p, az=-35, el=24, s=1.0, o=ORIGIN):
    """Orthographic view of a 3-D point (x right, y depth, z up) for flat 2-D drawings of cubes."""
    a, e = np.radians(az), np.radians(el); x, y, z = p
    xr = x * np.cos(a) - y * np.sin(a); yr = x * np.sin(a) + y * np.cos(a)
    return o + s * np.array([xr, z * np.cos(e) + yr * np.sin(e), 0.0])


CUBE_EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def cube_lines(lo, hi, color, sw=2.5, **kw):
    C = [np.array([x, y, z]) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    # vertex index = 4*ix + 2*iy + iz
    return VGroup(*[Line(iso(C[a], **kw), iso(C[b], **kw), stroke_width=sw).set_color(color) for a, b in CUBE_EDGES])


def cube_faces(lo, hi, color, op=0.25, **kw):
    """The three faces seen from the default view (front y=lo, right x=hi, top z=hi)."""
    x0, y0, z0 = lo; x1, y1, z1 = hi
    F = [[(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
         [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]]
    return VGroup(*[Polygon(*[iso(np.array(v), **kw) for v in f]).set_fill(color, op).set_stroke(width=0) for f in F])


class PaperScene(Scene):
    default_camera_config = dict(background_color=PAPER)
