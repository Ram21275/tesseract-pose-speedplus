"""Concatenate the rendered scenes (in script order) into the final H.264/AAC MP4 and run basic checks.
Output: /media/kavinder/hdd2/RAM/video/selecting_the_right_mode_explainer.mp4"""
import json
import subprocess
import sys
from pathlib import Path

from script import SCENES

ROOT = Path("/media/kavinder/hdd2/RAM/video")
SC_DIR = ROOT / "scenes"
OUT = ROOT / "selecting_the_right_mode_explainer.mp4"


def probe(f):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,r_frame_rate:format=duration",
                        "-of", "json", str(f)], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


if __name__ == "__main__":
    files, total = [], 0.0
    for name in SCENES:
        f = SC_DIR / f"{name}.mp4"
        if not f.exists():
            sys.exit(f"missing scene {f}")
        p = probe(f); v = [s for s in p["streams"] if s["codec_type"] == "video"][0]
        has_a = any(s["codec_type"] == "audio" for s in p["streams"])
        d = float(p["format"]["duration"]); total += d
        print(f"{name:20s} {d:6.1f}s {v['width']}x{v['height']} {v['r_frame_rate']} audio={has_a}")
        if (v["width"], v["height"], v["r_frame_rate"]) != (1920, 1080, "60/1") or not has_a:
            sys.exit(f"bad format {name}")
        files.append(f)
    print(f"total {total / 60:.2f} min")
    ins = sum([["-i", str(f)] for f in files], [])
    n = len(files)
    filt = "".join(f"[{i}:v][{i}:a]" for i in range(n)) + f"concat=n={n}:v=1:a=1[v][a0];[a0]loudnorm=I=-16:TP=-1.5:LRA=11[a]"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", filt, "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "60",
                    "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", str(OUT)], check=True)
    p = probe(OUT)
    print("OUT", OUT, json.dumps(p["format"]), [(s["codec_type"], s.get("width"), s.get("r_frame_rate")) for s in p["streams"]])
