"""karaokify — remove or attenuate vocals from audio/video files, locally.

Wraps Demucs (source separation) + ffmpeg (demux/remux) into a single CLI.
"""

__version__ = "0.3.0"


class KaraokifyError(Exception):
    """Per-input processing failure. Reported as action="failed" (exit code 1)."""


class EnvError(Exception):
    """Missing environment dependency (no ffmpeg, no demucs, CUDA absent). Exit code 3."""
