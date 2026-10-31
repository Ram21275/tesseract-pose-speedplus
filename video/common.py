"""Shared style, narration timing and verified math for the explainer video.

Math: the Tesseract encoder/decoder and rotation helpers are imported from the paper's own code (src/tfpose), so every
cell, path and centre drawn in the video is computed exactly as in the experiments.
Narration: `Beat` adds the beat's audio clip at the current scene time and lets the scene spend the clip's duration on
animations; the clip lengths come from audio/durations.json written by tts.py."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
import manim_compat  # noqa: F401,E402  (headless EGL + LaTeX shims; must precede manimlib)
from manimlib import *  # noqa: F401,F403,E402
from tfpose import tesseract as T  # noqa: E402
from tfpose import rotations as ROT  # noqa: E402

AUDIO = Path("/media/kavinder/hdd2/RAM/video/audio")
DUR = json.loads((AUDIO / "durations.json").read_text()) if (AUDIO / "durations.json").exists() else {}

# ---------------------------------------------------------------- colour conventions (fixed for the whole video)
BG = "#0E1116"
C_GT = "#FFD166"         # ground truth / true pose
C_PRED = "#4CC9F0"       # our prediction / hypotheses
C_FLIP = "#EF476F"       # wrong look-alike mode / failure
C_DINO = "#06D6A0"       # DINOv3 appearance branch
C_MOGE = "#F78C6B"       # MoGe-2 geometry branch
C_POE = "#B388FF"        # fused (product of experts)
C_BASE = "#9AA5B1"       # baselines
C_FT = "#FF9F1C"         # fine-tuned methods
CHART_COLORS = ["#4CC9F0", "#F72585", "#7BE495", "#FFD166"]   # charts 0..3 (w, x, y, z largest)
LEVEL_COLORS = ["#FFD166", "#F4A261", "#E76F51", "#B388FF", "#4CC9F0", "#06D6A0"]


class Beat:
    """One narration clip. Usage:  b = Beat(self, "S04_Tesseract_b1"); b.play(anim, frac=0.4); ...; b.done()"""

    def __init__(self, scene, key, gap=0.15):
        self.scene, self.key, self.T, self.used = scene, key, DUR.get(key, 4.0), 0.0
        wav = AUDIO / f"{key}.wav"
        if wav.exists():
            scene.add_sound(str(wav))
        self.gap = gap

    @property
    def left(self):
        return max(self.T - self.used, 0.0)

    def play(self, *anims, frac=None, rt=None, **kw):
        rt = rt if rt is not None else max((frac if frac is not None else 0.25) * self.T, 0.2)
        self.scene.play(*anims, run_time=rt, **kw); self.used += rt

    def wait(self, t):
        self.scene.wait(t); self.used += t

    def done(self):
        if self.used > self.T + 0.05:
            print(f"[beat] {self.key}: animations ({self.used:.2f}s) exceed narration ({self.T:.2f}s)", flush=True)
        rem = self.T - self.used + self.gap
        if rem > 0.02:
            self.scene.wait(rem)


def beat(scene, scene_name, bid, **kw):
    return Beat(scene, f"{scene_name}_{bid}", **kw)


# ---------------------------------------------------------------- math helpers (verified against src/tfpose)
def quat(axis, angle):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    return np.r_[np.cos(angle / 2), np.sin(angle / 2) * axis]


def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qmat(q):
    return ROT.quat_to_matrix(np.asarray(q, float)[None])[0]


def geodesic_deg(a, b):
    return float(np.degrees(2 * np.arccos(np.clip(abs(np.dot(a, b)), 0, 1))))


def encode(q, depth=5):
    c, p = T.encode(np.asarray(q, float)[None], depth); return int(c[0]), [int(x) for x in p[0]]


def centre(chart, path):
    return T.decode(np.array([chart]), np.array([path], dtype=np.int64).reshape(1, -1))[0]


def free_coords(q):
    """Chart, sign-normalised quaternion and the 3 free projected coordinates u in [-1,1]^3."""
    q = np.asarray(q, float); q = q / np.linalg.norm(q); c = int(np.argmax(np.abs(q)))
    q = q * np.sign(q[c]); x = q / q[c]; return c, q, np.delete(x, c)


def cell_box(path):
    """Projected-cube box (lo, hi) of the cell reached by `path` (list of child ids), each in [-1,1]^3."""
    lo, hi = -np.ones(3), np.ones(3)
    for k in path:
        bits = np.array([(k >> 0) & 1, (k >> 1) & 1, (k >> 2) & 1]); mid = (lo + hi) / 2
        lo = np.where(bits == 1, mid, lo); hi = np.where(bits == 1, hi, mid)
    return lo, hi


def apply_rot(mob, R, about=ORIGIN):
    """Apply a 3x3 rotation matrix to a mobject's points about a centre."""
    mob.apply_matrix(R, about_point=about); return mob


def setup_scene(scene):
    scene.camera.background_rgba = list(color_to_rgba(BG)) if hasattr(scene, "camera") else None
