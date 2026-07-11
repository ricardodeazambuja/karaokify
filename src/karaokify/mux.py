"""ffmpeg/ffprobe subprocess helpers: probe, audio extraction, encoding, remuxing."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

from . import EnvError, KaraokifyError

log = logging.getLogger("karaokify")

# codec key -> (ffmpeg encoder, container extension for audio-only outputs)
CODECS = {
    "aac": ("aac", ".m4a"),
    "mp3": ("libmp3lame", ".mp3"),
    "opus": ("libopus", ".opus"),
    "flac": ("flac", ".flac"),
}

# ffprobe codec_name of the input audio -> our codec key (for --audio-codec copy-format)
_PROBE_TO_CODEC = {
    "aac": "aac",
    "mp3": "mp3",
    "opus": "opus",
    "vorbis": "opus",
    "flac": "flac",
}


def check_env() -> None:
    """Fail fast (exit 3) when ffmpeg/ffprobe are not on PATH."""
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise EnvError(
                f"{tool} not found on PATH — install ffmpeg >= 4.4 "
                "(e.g. `mamba install ffmpeg` or `apt install ffmpeg`)"
            )


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    log.debug("$ %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = " | ".join(proc.stderr.strip().splitlines()[-4:])
        raise KaraokifyError(f"{cmd[0]} failed: {tail}")
    return proc


def probe(path: Path) -> dict:
    """Return {duration_s, has_video, audio_codec, sample_rate} or raise KaraokifyError."""
    proc = _run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)]
    )
    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    # attached_pic covers cover-art "video" streams in mp3/m4a files
    video = [
        s for s in streams
        if s.get("codec_type") == "video"
        and not s.get("disposition", {}).get("attached_pic", 0)
    ]
    if not audio:
        raise KaraokifyError("no audio stream found")
    duration = data.get("format", {}).get("duration")
    return {
        "duration_s": round(float(duration), 3) if duration else None,
        "has_video": bool(video),
        "audio_codec": audio[0].get("codec_name"),
        "sample_rate": int(audio[0].get("sample_rate") or 0) or None,
    }


def resolve_codec(codec: str, probe_info: dict) -> str:
    """Map --audio-codec copy-format to a concrete codec key using the probed input."""
    if codec != "copy-format":
        return codec
    resolved = _PROBE_TO_CODEC.get(probe_info["audio_codec"])
    if resolved is None:
        log.warning(
            "copy-format: no encoder mapping for input codec %r, falling back to aac",
            probe_info["audio_codec"],
        )
        resolved = "aac"
    return resolved


def extract_audio(path: Path, wav_out: Path, sample_rate: int = 44100) -> None:
    """Decode the first audio stream to a stereo float32 wav for Demucs."""
    _run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(path),
         "-map", "0:a:0", "-vn", "-ac", "2", "-ar", str(sample_rate),
         "-c:a", "pcm_f32le", str(wav_out)]
    )


def _codec_args(codec: str, bitrate: str) -> list[str]:
    encoder, _ = CODECS[codec]
    args = ["-c:a", encoder]
    if codec != "flac":  # flac is lossless; -b:a is meaningless there
        args += ["-b:a", bitrate]
    return args


def encode_audio(mixed_wav: Path, output: Path, codec: str, bitrate: str) -> None:
    """Audio-only output: encode the mixed wav directly."""
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(mixed_wav),
          *_codec_args(codec, bitrate), str(output)])


def remux(input_path: Path, mixed_wav: Path, output: Path, codec: str, bitrate: str) -> None:
    """Copy every original stream except audio; add the new audio track encoded."""
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-i", str(input_path), "-i", str(mixed_wav),
           "-map", "0", "-map", "-0:a", "-map", "1:a:0",
           "-c", "copy", *_codec_args(codec, bitrate)]
    if output.suffix.lower() in {".mp4", ".m4v", ".mov", ".m4a"}:
        cmd += ["-movflags", "+faststart"]
    cmd.append(str(output))
    _run(cmd)
