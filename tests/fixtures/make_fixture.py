#!/usr/bin/env python3
"""Generate the synthetic test media (no binary blobs are checked in).

Creates in OUT_DIR:
  song.mp4 — 5 s testsrc video + audio: sine-chord bed with a "voice" on top
             (espeak if available, otherwise a vibrato formant-ish tone)
  song.mp3 — the same audio, no video stream

Usage: make_fixture.py OUT_DIR
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DURATION = 5


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def _make_voice(tmp: Path) -> tuple[Path, str]:
    """A vocal-ish track: espeak when present, else a vibrato tone.

    Returns (wav path, source name). Demucs classifies synthetic tones as
    "other", not "vocals" — tests that assert actual vocal reduction must
    check the source name (written to voice_source.txt) and only assert
    strictly when real speech was available.
    """
    voice = tmp / "voice.wav"
    espeak = shutil.which("espeak") or shutil.which("espeak-ng")
    if espeak:
        _run([espeak, "-s", "120", "-w", str(voice),
              "Mercury Venus Earth and Mars, Jupiter Saturn Uranus Neptune"])
        return voice, "espeak"
    _run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
          "-i", f"sine=frequency=440:duration={DURATION},"
                "vibrato=f=6:d=0.5,volume=0.8",
          str(voice)])
    return voice, "synthetic"


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        voice, voice_source = _make_voice(tmp)
        (out_dir / "voice_source.txt").write_text(voice_source + "\n")
        bed_and_voice = (
            "[1][2][3][4]amix=inputs=4:duration=first:normalize=0,"
            "volume=2,aformat=sample_rates=44100:channel_layouts=stereo[a]"
        )
        common = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"testsrc=duration={DURATION}:size=320x240:rate=10",
            "-f", "lavfi", "-i", f"sine=frequency=220:duration={DURATION},volume=0.25",
            "-f", "lavfi", "-i", f"sine=frequency=277:duration={DURATION},volume=0.25",
            "-f", "lavfi", "-i", f"sine=frequency=330:duration={DURATION},volume=0.25",
            "-i", str(voice),
            "-filter_complex", bed_and_voice,
        ]
        # mpeg4 video encoder: available in every ffmpeg build, unlike libx264
        _run(common + ["-map", "0:v", "-map", "[a]", "-c:v", "mpeg4",
                       "-c:a", "aac", "-b:a", "128k", str(out_dir / "song.mp4")])
        _run(common + ["-map", "[a]", "-c:a", "libmp3lame", "-b:a", "128k",
                       str(out_dir / "song.mp3")])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
