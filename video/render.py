"""Entry point: python render.py <manimgl args>. Applies the headless/LaTeX shims, fixes manimgl 1.7.2's string-typed
--fps argument, then runs manimgl."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import manim_compat  # noqa: F401,E402
import manimlib.config as _cfg  # noqa: E402
import manimlib.__main__ as _main  # noqa: E402

_orig_parse = _cfg.parse_cli


def _parse_cli():
    a = _orig_parse()
    if getattr(a, "fps", None) is not None:
        a.fps = int(a.fps)
    return a


_cfg.parse_cli = _parse_cli
_main.parse_cli = _parse_cli

if __name__ == "__main__":
    _main.main()
