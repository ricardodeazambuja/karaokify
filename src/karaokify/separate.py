"""Demucs integration: model loading, separation, stem mixing, stem export.

torch/demucs are imported lazily inside functions so that --help, argument
validation, and the friendly exit-3 environment check all work without them.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from . import EnvError, KaraokifyError

log = logging.getLogger("karaokify")

_local = threading.local()

# stderr stream for the inline progress meter; None disables it (set by pipeline)
PROGRESS_STREAM = None


def check_env() -> str:
    try:
        import demucs
        import torch  # noqa: F401
    except ImportError as exc:
        raise EnvError(
            f"demucs/torch not importable ({exc}) — install with "
            "`pip install demucs` inside an env that already has torch (see README)"
        ) from exc
    return demucs.__version__


def resolve_device(choice: str) -> str:
    import torch

    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise EnvError("CUDA requested but torch reports no CUDA device available")
    return choice


class _Progress:
    """Demucs callback -> coarse percentage meter on stderr (best effort)."""

    def __init__(self) -> None:
        self.label = ""

    def __call__(self, info: dict) -> None:
        stream = PROGRESS_STREAM
        if stream is None:
            return
        try:
            if info.get("state") != "end":
                return
            length = info.get("audio_length") or 0
            if not length:
                return
            models = info.get("models") or 1
            model_idx = info.get("model_idx_in_bag") or 0
            offset = min(info.get("segment_offset") or 0, length)
            frac = (model_idx + offset / length) / models
            print(f"\r  separating {self.label}: {frac * 100:3.0f}%",
                  end="", file=stream, flush=True)
        except Exception:  # progress must never break separation
            pass

    def finish(self) -> None:
        if PROGRESS_STREAM is not None:
            print(file=PROGRESS_STREAM)


def get_separator(model: str, device: str, shifts: int, overlap: float,
                  segment: int | None):
    """Per-thread cached (Separator, _Progress) — model loads are expensive."""
    from demucs.api import Separator

    cache = getattr(_local, "separators", None)
    if cache is None:
        cache = _local.separators = {}
    key = (model, device, shifts, overlap, segment)
    if key not in cache:
        log.info("loading model %s on %s ...", model, device)
        progress = _Progress()
        try:
            separator = Separator(model=model, device=device, shifts=shifts,
                                  overlap=overlap, segment=segment,
                                  callback=progress)
        except Exception as exc:
            raise KaraokifyError(f"could not load model {model!r}: {exc}") from exc
        cache[key] = (separator, progress)
    return cache[key]


def separate(separator, progress: _Progress, wav_path: Path, label: str):
    """Run separation; returns (samplerate, {stem_name: cpu tensor})."""
    progress.label = label
    try:
        _origin, stems = separator.separate_audio_file(wav_path)
    finally:
        progress.finish()
    return separator.samplerate, {name: t.cpu() for name, t in stems.items()}


def mix_stems(stems: dict, gains: dict[str, float | None], two_stem: bool):
    """Gain-and-sum the stems selected by `gains` (None gain = dropped).

    In two-stem mode the mixable material is {vocals, no_vocals} where
    no_vocals is the sum of every non-vocal model source.
    """
    if two_stem:
        if "vocals" not in stems:
            raise KaraokifyError(
                f"model has no 'vocals' source (sources: {sorted(stems)}); "
                "use --stems 4/6 with --keep/--drop instead"
            )
        rest = [t for name, t in stems.items() if name != "vocals"]
        material = {"vocals": stems["vocals"], "no_vocals": sum(rest)}
    else:
        material = stems

    out = None
    for name, gain_db in gains.items():
        if gain_db is None:
            continue
        if name not in material:
            raise KaraokifyError(
                f"stem {name!r} not produced by this model (sources: {sorted(material)})"
            )
        weighted = material[name] * (10.0 ** (gain_db / 20.0))
        out = weighted if out is None else out + weighted
    if out is None:
        raise KaraokifyError("empty mix: every stem is dropped")
    return material, out


def save_wav(tensor, samplerate: int, path: Path) -> None:
    from demucs.api import save_audio

    save_audio(tensor, path, samplerate=samplerate, as_float=True)


def save_stems(material: dict, samplerate: int, directory: Path) -> list[str]:
    """Write each mixable stem as <dir>/<stem>.wav; returns written paths."""
    from demucs.api import save_audio

    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, tensor in material.items():
        path = directory / f"{name}.wav"
        save_audio(tensor, path, samplerate=samplerate)
        written.append(str(path))
    return written


def progress_to(stream) -> None:
    """Enable/disable the inline progress meter (call once from the pipeline)."""
    global PROGRESS_STREAM
    PROGRESS_STREAM = stream if (stream is not None and stream.isatty()) else None
