"""Synthesize one WAV per narration beat (script.py) with the offline Piper voice en_US-ryan-high, normalise loudness,
and write audio/durations.json. Usage: python tts.py [scene ...]  (default: all scenes; existing files are reused)."""
import json
import subprocess
import sys
from pathlib import Path

from script import SCENES

ROOT = Path("/media/kavinder/hdd2/RAM/video")
VOICE = ROOT / "voices/en_US-ryan-high.onnx"
OUT = ROOT / "audio"
PY = sys.executable
LENGTH_SCALE = "1.0"       # natural speaking rate
SENT_SIL = "0.25"


def synth(text, wav):
    raw = wav.with_suffix(".raw.wav")
    subprocess.run([PY, "-m", "piper", "-m", str(VOICE), "-f", str(raw), "--length-scale", LENGTH_SCALE,
                    "--sentence-silence", SENT_SIL], input=text.encode(), check=True, capture_output=True)
    # EBU R128 loudness normalisation, 48 kHz mono, short tail of silence
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-af", "loudnorm=I=-18:TP=-1.5:LRA=11,apad=pad_dur=0.25",
                    "-ar", "48000", "-ac", "1", str(wav)], check=True)
    raw.unlink()


def duration(wav):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(wav)],
                                capture_output=True, text=True, check=True).stdout)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    dfile = OUT / "durations.json"
    dur = json.loads(dfile.read_text()) if dfile.exists() else {}
    for scene in (sys.argv[1:] or list(SCENES)):
        for bid, text in SCENES[scene]:
            wav = OUT / f"{scene}_{bid}.wav"; key = f"{scene}_{bid}"
            stamp = OUT / f"{scene}_{bid}.txt"
            if not wav.exists() or not stamp.exists() or stamp.read_text() != text:
                synth(text, wav); stamp.write_text(text)
            dur[key] = duration(wav)
            print(key, round(dur[key], 2), flush=True)
    dfile.write_text(json.dumps(dur, indent=1))
    print("total min", round(sum(dur.values()) / 60, 2))
