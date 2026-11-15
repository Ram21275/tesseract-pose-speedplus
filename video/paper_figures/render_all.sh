#!/usr/bin/env bash
# Render the manim paper figures into paper/figures/method/: a vector PDF (text, equations, lines and arrows as paths; photos
# embedded at source resolution; used by the paper) and a trimmed PNG preview.
# Usage: bash video/paper_figures/render_all.sh [FigName ...]     (needs: conda env manim311; project .venv for trimming)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT/video"
OUT="$ROOT/paper/figures/method"; RAW="${TMPDIR:-/tmp}/paper_figs_raw"; mkdir -p "$OUT" "$RAW"
PY="$HOME/miniconda3/envs/manim311/bin/python"
declare -A RES=([FigArch]=4200x1700 [FigTeaser]=4000x1500 [FigTesseract]=3600x1350 [FigTree]=3600x1400 [FigOrbit]=3600x1400)
declare -A SRC=([FigArch]=fig_arch.py [FigTeaser]=fig_teaser.py [FigTesseract]=fig_tesseract.py [FigTree]=fig_tree.py [FigOrbit]=fig_orbit.py)
declare -A NAME=([FigArch]=architecture [FigTeaser]=teaser [FigTesseract]=tesseract_partition [FigTree]=tree_decoder_multipath [FigOrbit]=orbit_voting)
FIGS=("$@"); [ ${#FIGS[@]} -eq 0 ] && FIGS=(FigArch FigTeaser FigTesseract FigTree FigOrbit)
for f in "${FIGS[@]}"; do
  PF_VEC="$RAW/$f.pkl" "$PY" render.py "paper_figures/${SRC[$f]}" "$f" -s -w -r "${RES[$f]}" --video_dir "$RAW" > "$RAW/$f.log" 2>&1 || { tail -20 "$RAW/$f.log"; exit 1; }
  "$ROOT/.venv/bin/python" - "$RAW/$f.png" "$OUT/${NAME[$f]}" <<'PYT'
import sys
from PIL import Image, ImageChops
im = Image.open(sys.argv[1]).convert("RGB"); bg = Image.new("RGB", im.size, (255, 255, 255))
x0, y0, x1, y1 = ImageChops.difference(im, bg).getbbox(); p = 30
im = im.crop((max(x0 - p, 0), max(y0 - p, 0), min(x1 + p, im.width), min(y1 + p, im.height)))
im.save(sys.argv[2] + ".png", dpi=(300, 300))
print(sys.argv[2], im.size)
PYT
  "$ROOT/.venv/bin/python" "$ROOT/video/paper_figures/vec2pdf.py" "$RAW/$f.pkl" "$OUT/${NAME[$f]}.pdf"   # vector PDF (used by the paper)
done
