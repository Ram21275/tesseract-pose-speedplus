# Explainer video: "Selecting the Right Mode"

This folder is the source of the 3Blue1Brown-style explainer video, made with 3b1b/manim (manimgl 1.7.2).
The rendered MP4 is not in git. It is at `/media/kavinder/hdd2/RAM/video/selecting_the_right_mode_explainer.mp4`: 10.5 min, 1920×1080, 60 fps, H.264/AAC.

## Files
- `script.py`: narration, one beat per audio clip. Every number comes from `paper/CLAIMS.md` and the paper's tables.
- `tts.py`: offline neural voice (Piper `en_US-ryan-high`), loudness-normalised. It writes `audio/durations.json`.
- `common.py`:
  - colours;
  - the `Beat` helper, which adds each clip and times that beat's animations to it;
  - math imported from `src/tfpose` (Tesseract encode/decode), so the geometry on screen is exactly the code's.
- `scenes/S01…S16_*.py`: one manimgl Scene per narration scene. `STORYBOARD.md` gives the visual plan and the rules.
- `assemble.py`: concatenates the scenes and runs loudness and format checks.
- `manim_compat.py`, `render.py`, `custom_config.yml`: workarounds for this headless server.
  - **Display:** EGL instead of X11.
  - **LaTeX:** `tectonic` instead of `latex` + `dvisvgm`, dropping the `tipa` package and the `\DisableLigatures` command, which fail under XeTeX.
  - **Text colour:** Tex/Text ignore `color=` in this build, so it is applied explicitly.
  - **Frame rate:** set to 60 fps through the config file, because `--fps` crashes manimgl 1.7.2.

## Rebuild
```
conda activate manim311          # Python 3.11, manimgl 1.7.2, piper-tts, conda-forge pango/glib/manimpango
cd video
python tts.py                    # narration audio
python render.py scenes/S04_Tesseract.py S04_Tesseract -w --hd --video_dir /media/kavinder/hdd2/RAM/video/scenes   # per scene
python assemble.py
```
