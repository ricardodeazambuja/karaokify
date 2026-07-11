"""Command-line interface for karaokify.

Contract (normative, see PLAN.md): stdout is data (--json), stderr is logs.
Exit codes: 0 success/skips, 1 at least one input failed, 2 bad invocation,
3 environment missing. Never prompts.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

from . import EnvError, __version__
from . import manifest
from .pipeline import Options, run

log = logging.getLogger("karaokify")

STEM_SETS = {
    "vocals": ("no_vocals", "vocals"),
    "4": ("drums", "bass", "other", "vocals"),
    "6": ("drums", "bass", "other", "vocals", "guitar", "piano"),
}
DEFAULT_MODELS = {"vocals": "htdemucs", "4": "htdemucs", "6": "htdemucs_6s"}


def parse_stem_list(value: str, valid: tuple[str, ...]) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise ValueError("empty stem list")
    for name in names:
        if name not in valid:
            raise ValueError(f"unknown stem {name!r} (valid: {', '.join(valid)})")
    return names


def parse_gain_arg(value: str, valid: tuple[str, ...]) -> tuple[str, float]:
    stem, sep_char, db_text = value.partition("=")
    stem = stem.strip()
    if not sep_char or stem not in valid:
        raise ValueError(
            f"--gain expects STEM=DB with STEM in {{{', '.join(valid)}}}, got {value!r}"
        )
    try:
        db = float(db_text)
    except ValueError:
        raise ValueError(f"--gain {value!r}: {db_text!r} is not a number") from None
    if math.isnan(db):
        raise ValueError(f"--gain {value!r}: NaN is not a gain")
    return stem, db


def resolve_gains(
    stems_mode: str,
    keep: str | None,
    drop: str | None,
    vocals_gain: float | None,
    gain_args: list[str],
) -> dict[str, float | None]:
    """Resolve the per-stem output gains (dB); None means the stem is dropped.

    Precedence: --keep/--drop set the baseline, then --vocals-gain, then --gain.
    """
    names = STEM_SETS[stems_mode]
    if keep is not None:
        kept = parse_stem_list(keep, names)
        gains: dict[str, float | None] = {n: (0.0 if n in kept else None) for n in names}
    else:
        dropped = parse_stem_list(drop, names) if drop is not None else ["vocals"]
        gains = {n: (None if n in dropped else 0.0) for n in names}

    if vocals_gain is not None and not math.isnan(vocals_gain):
        gains["vocals"] = None if vocals_gain == float("-inf") else vocals_gain

    for arg in gain_args:
        stem, db = parse_gain_arg(arg, names)
        gains[stem] = None if db == float("-inf") else db
    return gains


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="karaokify",
        description="Remove (or attenuate) vocals from any audio/video file, locally. "
                    "Wraps Demucs + ffmpeg; the video stream is copied untouched.",
        epilog="Exit codes: 0 ok, 1 input failed, 2 bad invocation, 3 environment missing.",
    )
    p.add_argument("inputs", nargs="+", metavar="INPUT", type=Path,
                   help="audio/video file(s) readable by ffmpeg")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    g = p.add_argument_group("separation knobs")
    g.add_argument("--stems", choices=("vocals", "4", "6"), default="vocals",
                   help="two-stem karaoke mode (default), or full 4/6-stem separation")
    keepdrop = g.add_mutually_exclusive_group()
    keepdrop.add_argument("--keep", metavar="STEM[,STEM]",
                          help="stems to keep in the output mix")
    keepdrop.add_argument("--drop", metavar="STEM[,STEM]",
                          help="stems to drop from the output mix (default: vocals)")
    g.add_argument("--vocals-gain", type=float, metavar="DB", default=None,
                   help="attenuate vocals instead of removing (e.g. -18 for a guide vocal)")
    g.add_argument("--gain", action="append", default=[], metavar="STEM=DB",
                   help="per-stem gain for the output mix (repeatable)")
    g.add_argument("--model", default=None,
                   help="Demucs model (default: htdemucs, or htdemucs_6s for --stems 6)")
    g.add_argument("--shifts", type=int, default=1, metavar="N",
                   help="test-time augmentation passes; better quality, linear slowdown")
    g.add_argument("--overlap", type=float, default=0.25, metavar="FLOAT",
                   help="segment overlap; quality vs speed (default: 0.25)")
    g.add_argument("--segment", type=int, default=None, metavar="SECONDS",
                   help="segment length; lower to fit small GPUs")
    g.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                   help="compute device (default: auto)")
    g.add_argument("--jobs", type=int, default=1, metavar="N",
                   help="parallel workers for batch inputs (CPU only)")

    o = p.add_argument_group("output knobs")
    o.add_argument("-o", "--output", type=Path, metavar="PATH",
                   help="explicit output path (single input only)")
    o.add_argument("--output-dir", type=Path, metavar="DIR",
                   help="where outputs land (default: alongside each input)")
    o.add_argument("--suffix", default=" (no vocals)", metavar="STR",
                   help='output name suffix (default: " (no vocals)")')
    o.add_argument("--audio-codec", default="aac",
                   choices=("aac", "mp3", "opus", "flac", "copy-format"),
                   help="output audio codec (copy-format matches the input's codec)")
    o.add_argument("--audio-bitrate", default="192k", metavar="RATE",
                   help="bitrate for lossy codecs (default: 192k)")
    o.add_argument("--save-stems", nargs="?", const=True, default=None, metavar="DIR",
                   help="also keep the separated stems as wav "
                        "(default dir: '<input> (stems)'; use --save-stems=DIR with a value)")
    o.add_argument("--audio-only", action="store_true",
                   help="emit just the mixed audio file, even for video inputs")

    a = p.add_argument_group("agent/automation knobs")
    a.add_argument("--json", action="store_true",
                   help="emit a single JSON result document on stdout (schema: karaokify/schema.json)")
    a.add_argument("--dry-run", action="store_true",
                   help="resolve inputs/model/device/outputs, report the plan, touch nothing")
    ow = a.add_mutually_exclusive_group()
    ow.add_argument("--overwrite", dest="overwrite", action="store_true",
                    help="replace existing outputs")
    ow.add_argument("--skip-existing", dest="overwrite", action="store_false",
                    help="skip inputs whose output exists (default)")
    p.set_defaults(overwrite=False)
    a.add_argument("--manifest", type=Path, metavar="PATH",
                   help="append a JSON-lines record per input (audit trail across runs)")
    vq = a.add_mutually_exclusive_group()
    vq.add_argument("--verbose", action="store_true", help="debug logging on stderr")
    vq.add_argument("--quiet", action="store_true", help="warnings and errors only")
    return p


class _StderrFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        prefix = f"{record.levelname.lower()}: " if record.levelno >= logging.WARNING else ""
        return prefix + record.getMessage()


def setup_logging(verbose: bool, quiet: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_StderrFormatter())
    log.handlers[:] = [handler]
    log.setLevel(level)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose, args.quiet)

    if args.output is not None and len(args.inputs) > 1:
        parser.error("-o/--output works with a single input; use --output-dir for batches")
    if args.output is not None and args.output_dir is not None:
        parser.error("-o/--output and --output-dir are mutually exclusive")
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    try:
        gains = resolve_gains(args.stems, args.keep, args.drop, args.vocals_gain, args.gain)
    except ValueError as exc:
        parser.error(str(exc))

    save_stems: Path | bool | None
    if args.save_stems is None or args.save_stems is True:
        save_stems = args.save_stems
    else:
        save_stems = Path(args.save_stems)

    opts = Options(
        inputs=args.inputs,
        stems_mode=args.stems,
        gains=gains,
        model=args.model or DEFAULT_MODELS[args.stems],
        shifts=args.shifts,
        overlap=args.overlap,
        segment=args.segment,
        device=args.device,
        jobs=args.jobs,
        output=args.output,
        output_dir=args.output_dir,
        suffix=args.suffix,
        audio_codec=args.audio_codec,
        audio_bitrate=args.audio_bitrate,
        save_stems=save_stems,
        audio_only=args.audio_only,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        manifest_path=args.manifest,
    )

    try:
        exit_code, document = run(opts)
    except EnvError as exc:
        log.error("%s", exc)
        if args.json:
            document = manifest.make_document([])
            document["error"] = str(exc)
            manifest.emit(document)
        return 3

    if args.json:
        manifest.emit(document)
    else:
        s = document["summary"]
        log.info("summary: %d processed, %d skipped, %d failed%s",
                 s["processed"], s["skipped"], s["failed"],
                 f", {s['planned']} planned (dry run)" if s["planned"] else "")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
