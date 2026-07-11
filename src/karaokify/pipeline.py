"""Pipeline orchestration: probe -> separate -> mix -> remux, per input."""

from __future__ import annotations

import logging
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import KaraokifyError
from . import manifest, mux
from . import separate as sep

log = logging.getLogger("karaokify")


@dataclass
class Options:
    inputs: list[Path]
    stems_mode: str = "vocals"  # vocals | 4 | 6
    gains: dict[str, float | None] = field(default_factory=dict)  # stem -> dB, None = dropped
    model: str = "htdemucs"
    shifts: int = 1
    overlap: float = 0.25
    segment: int | None = None
    device: str = "auto"
    jobs: int = 1
    output: Path | None = None
    output_dir: Path | None = None
    suffix: str = " (no vocals)"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    save_stems: Path | bool | None = None  # None=off, True=default dir, Path=explicit dir
    audio_only: bool = False
    dry_run: bool = False
    overwrite: bool = False
    manifest_path: Path | None = None
    resolved_device: str | None = None  # filled in by run()


def resolve_output(input_path: Path, opts: Options, probe_info: dict, codec: str) -> Path:
    """Deterministic output path — an agent can predict it from a dry-run."""
    if opts.output is not None:
        return opts.output
    audio_only = opts.audio_only or not probe_info["has_video"]
    ext = mux.CODECS[codec][1] if audio_only else input_path.suffix
    directory = opts.output_dir or input_path.parent
    return directory / f"{input_path.stem}{opts.suffix}{ext}"


def resolve_stems_dir(input_path: Path, opts: Options) -> Path:
    if isinstance(opts.save_stems, Path):
        return opts.save_stems / input_path.stem
    base = opts.output_dir or input_path.parent
    return base / f"{input_path.stem} (stems)"


def process_one(input_path: Path, opts: Options) -> dict:
    started = time.monotonic()
    timings: dict[str, float] = {}
    common = dict(
        model=opts.model,
        device=opts.resolved_device,
        stems=dict(opts.gains),
    )
    try:
        if not input_path.exists():
            raise KaraokifyError("input file not found")

        probe_started = time.monotonic()
        info = mux.probe(input_path)
        timings["probe_s"] = round(time.monotonic() - probe_started, 3)
        common["input_duration_s"] = info["duration_s"]

        codec = mux.resolve_codec(opts.audio_codec, info)
        output = resolve_output(input_path, opts, info, codec)
        if output.resolve() == input_path.resolve():
            raise KaraokifyError(
                "output path equals input path — change --suffix or use -o/--output-dir"
            )

        if output.exists() and not opts.overwrite:
            log.info("skipped %s — output already exists: %s (use --overwrite to redo)",
                     input_path.name, output)
            return manifest.make_result(
                input_path, output, "skipped",
                reason="output exists (use --overwrite)", **common,
            )
        if opts.dry_run:
            log.info("planned %s -> %s", input_path.name, output)
            return manifest.make_result(input_path, output, "planned", **common)

        log.info("processing %s", input_path.name)
        separate_started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="karaokify-") as tmp:
            source_wav = Path(tmp) / "input.wav"
            mux.extract_audio(input_path, source_wav)
            separator, progress = sep.get_separator(
                opts.model, opts.resolved_device, opts.shifts, opts.overlap, opts.segment
            )
            samplerate, stems = sep.separate(separator, progress, source_wav, input_path.name)
            timings["separate_s"] = round(time.monotonic() - separate_started, 3)

            material, mixed = sep.mix_stems(
                stems, opts.gains, two_stem=(opts.stems_mode == "vocals")
            )
            if opts.save_stems is not None:
                stems_dir = resolve_stems_dir(input_path, opts)
                written = sep.save_stems(material, samplerate, stems_dir)
                log.info("saved %d stems to %s", len(written), stems_dir)

            mix_wav = Path(tmp) / "mix.wav"
            sep.save_wav(mixed, samplerate, mix_wav)

            remux_started = time.monotonic()
            output.parent.mkdir(parents=True, exist_ok=True)
            if info["has_video"] and not opts.audio_only:
                mux.remux(input_path, mix_wav, output, codec, opts.audio_bitrate)
            else:
                mux.encode_audio(mix_wav, output, codec, opts.audio_bitrate)
            timings["remux_s"] = round(time.monotonic() - remux_started, 3)

        timings["total_s"] = round(time.monotonic() - started, 3)
        log.info("done: %s (%.1fs)", output, timings["total_s"])
        return manifest.make_result(
            input_path, output, "processed",
            timings=timings, output_size_bytes=output.stat().st_size, **common,
        )
    except KaraokifyError as exc:
        reason = str(exc)
    except Exception as exc:  # keep batch runs alive; report the file as failed
        reason = f"{type(exc).__name__}: {exc}"
    timings["total_s"] = round(time.monotonic() - started, 3)
    log.error("failed: %s — %s", input_path, reason)
    return manifest.make_result(
        input_path, None, "failed", reason=reason, timings=timings, **common,
    )


def run(opts: Options) -> tuple[int, dict]:
    """Process every input; returns (exit_code, JSON result document)."""
    mux.check_env()
    sep.check_env()
    opts.resolved_device = sep.resolve_device(opts.device)
    log.info("device: %s, model: %s", opts.resolved_device, opts.model)

    jobs = opts.jobs
    if jobs > 1 and opts.resolved_device != "cpu":
        log.warning("--jobs applies to CPU batch runs only; running sequentially on %s",
                    opts.resolved_device)
        jobs = 1
    if jobs > 1 and not opts.dry_run and len(opts.inputs) > 1:
        sep.progress_to(None)  # interleaved progress meters would be garbage
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(lambda path: process_one(path, opts), opts.inputs))
    else:
        sep.progress_to(sys.stderr if log.getEffectiveLevel() <= logging.INFO else None)
        results = [process_one(path, opts) for path in opts.inputs]

    if opts.manifest_path is not None and not opts.dry_run:
        for result in results:
            manifest.append_manifest(opts.manifest_path, result)

    document = manifest.make_document(results)
    exit_code = 1 if document["summary"]["failed"] else 0
    return exit_code, document
