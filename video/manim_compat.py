"""Compatibility shims for rendering 3b1b/manim (manimgl 1.7) on this headless machine.
- Headless OpenGL through EGL (pyglet headless mode).
- No dvisvgm available: LaTeX is compiled with tectonic and converted with poppler's pdftocairo -svg.
Import this module before manimlib."""
import os
import re
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("PYGLET_HEADLESS", "1")

import manimlib.utils.tex_file_writing as tfw  # noqa: E402


def _full_tex_to_svg(full_tex: str, compiler: str = "latex", message: str = ""):
    with tempfile.TemporaryDirectory() as d:
        # pdfTeX-only \DisableLigatures, and tipa (redefines \| under XeTeX) are dropped
        full_tex = "\n".join(l for l in full_tex.splitlines() if "DisableLigatures" not in l and "{tipa}" not in l)
        tex = Path(d, "working.tex"); tex.write_text(full_tex)
        # tectonic (self-contained TeX engine that fetches missing packages) instead of the minimal local TeX Live
        p = subprocess.run([str(Path.home() / ".local/bin/tectonic"), "--chatter", "minimal", "--outdir", d, str(tex)],
                           capture_output=True, text=True)
        if p.returncode != 0:
            raise tfw.LatexError((p.stderr or p.stdout)[-800:] or "LaTeX compilation failed")
        svg = Path(d, "working.svg")
        subprocess.run(["pdftocairo", "-svg", str(tex.with_suffix(".pdf")), str(svg)], check=True, capture_output=True)
        return svg.read_text()


tfw.full_tex_to_svg = _full_tex_to_svg

# moderngl standalone contexts default to X11; force EGL.
import moderngl  # noqa: E402

_orig_cc = moderngl.create_context
_orig_csc = moderngl.create_standalone_context


def _create_context(*a, **k):
    if k.get("standalone"):
        k.setdefault("backend", "egl")
    return _orig_cc(*a, **k)


def _create_standalone_context(*a, **k):
    k.setdefault("backend", "egl")
    return _orig_csc(*a, **k)


moderngl.create_context = _create_context
moderngl.create_standalone_context = _create_standalone_context


def _patch_text_colour():
    """manimgl 1.7.2 here ignores the `color=` keyword of Tex/TexText/Text (labels render white). Apply it explicitly."""
    import manimlib.mobject.svg.tex_mobject as tm
    import manimlib.mobject.svg.text_mobject as xm
    for cls in (tm.Tex, tm.TexText, xm.Text):
        orig = cls.__init__
        if getattr(orig, "_colour_patched", False):
            continue

        def init(self, *a, __orig=orig, **k):
            col = k.get("color")
            __orig(self, *a, **k)
            if col is not None:
                self.set_color(col)
        init._colour_patched = True
        cls.__init__ = init


_patch_text_colour()
