# karaokify — PLAN

Remove (or attenuate) vocals from any audio/video file, locally, with one command.
`karaokify` wraps **Demucs** (source separation) + **ffmpeg** (demux/remux) into a
single CLI that is pleasant for humans and trivially drivable by AI agents.

The origin story: "remove the vocals from this kids' song video" turned out to be
a two-command pipeline (`demucs --two-stems=vocals` → `ffmpeg -c:v copy` remux).
This project productizes that pipeline with sensible defaults and explicit knobs.

## Goals

1. **One command, batteries included**: `karaokify video.mp4` produces
   `video (no vocals).mp4` next to the input, video stream untouched (`-c:v copy`).
2. **Knobs, not forks**: every quality/speed/output decision the pipeline makes is
   overridable via a flag, but no flag is ever required.
3. **AI-agent friendly**: non-interactive, machine-readable output, stable exit
   codes, dry-run, idempotent. An agent should be able to plan, run, and verify
   without scraping human prose.
4. **Local & private**: nothing leaves the machine. GPU if available, CPU fallback.

## Non-goals

- Real-time / streaming separation.
- A GUI or web frontend (explicitly dropped; may live in a separate repo later).
- Training or fine-tuning separation models.

## Pipeline (what happens under the hood)

```
input (any ffmpeg-readable file)
  └─ probe (ffprobe: streams, duration, sample rate)          [stage: probe]
  └─ demucs separation → stems (vocals / no_vocals, or 4/6)   [stage: separate]
  └─ optional stem mixing (e.g. vocals at −18 dB, bass +3 dB) [stage: mix]
  └─ remux: original video stream copied + new audio encoded  [stage: remux]
output + JSON manifest
```

Audio-only inputs skip the remux stage and encode the mixed audio directly.

## CLI design

```
karaokify INPUT... [options]
```

### Core knobs

| Flag | Default | Purpose |
|---|---|---|
| `--stems {vocals,4,6}` | `vocals` | Two-stem karaoke mode, or full 4/6-stem separation |
| `--keep / --drop STEM[,STEM]` | `--drop vocals` | Which stems end up in the output mix |
| `--vocals-gain DB` | `-inf` | Attenuate instead of remove (e.g. `-18` leaves a guide vocal for sing-along) |
| `--gain STEM=DB` (repeatable) | — | Per-stem gain for the output mix (4/6-stem mode) |
| `--model NAME` | `htdemucs` | Any Demucs model: `htdemucs_ft` (better/4× slower), `htdemucs_6s`, `mdx_extra`, … |
| `--shifts N` | `1` | Demucs test-time augmentation; higher = better quality, linear slowdown |
| `--overlap FLOAT` | `0.25` | Segment overlap; quality vs speed |
| `--segment SECONDS` | model default | Lower to fit small GPUs / bound RAM |
| `--device {auto,cuda,cpu}` | `auto` | Compute device |
| `--jobs N` | `1` | Parallel workers for batch inputs (CPU) |

### Output knobs

| Flag | Default | Purpose |
|---|---|---|
| `-o, --output PATH` | `<input> (no vocals).<ext>` | Explicit output path (single input only) |
| `--output-dir DIR` | alongside input | Where outputs land for batch runs |
| `--suffix STR` | ` (no vocals)` | Output name suffix |
| `--audio-codec {aac,mp3,opus,flac,copy-format}` | `aac` | Output audio codec |
| `--audio-bitrate RATE` | `192k` | Bitrate for lossy codecs |
| `--save-stems [DIR]` | off | Also keep the raw separated stems (wav) |
| `--audio-only` | off | Emit just the mixed audio file, even for video inputs |

### Agent/automation knobs

| Flag | Default | Purpose |
|---|---|---|
| `--json` | off | Suppress human logs on stdout; emit a single JSON result document (schema below) |
| `--dry-run` | off | Resolve everything (inputs, model, device, output paths), print the plan, touch nothing |
| `--overwrite / --skip-existing` | `--skip-existing` | Idempotency policy; skipped files are reported, not errors |
| `--manifest PATH` | off | Append a JSON-lines record per processed file (audit trail across runs) |
| `--verbose / --quiet` | normal | Log level (logs always go to **stderr**; stdout is reserved for `--json`) |

## AI-agent contract

This section is normative — future changes must not break it without a major
version bump.

- **stdout is data, stderr is logs.** With `--json`, stdout contains exactly one
  JSON document and nothing else.
- **Exit codes**: `0` all inputs succeeded (or were policy-skipped) · `1` at least
  one input failed · `2` bad invocation (unknown flag, no inputs) · `3` environment
  missing (no ffmpeg, no demucs, CUDA requested but absent). Never asks a question,
  never opens a TTY prompt.
- **JSON result schema** (per input):
  `{input, output, action: "processed"|"skipped"|"failed", reason?, model, device,
  stems: {name: gain_db}, timings: {probe_s, separate_s, remux_s, total_s},
  input_duration_s, output_size_bytes}` — top level:
  `{version, results: [...], summary: {processed, skipped, failed}}`.
- **`--dry-run --json`** returns the same schema with `action: "planned"`, so an
  agent can validate a batch before spending GPU minutes.
- **Deterministic naming**: given the same input and flags, output paths are
  reproducible; an agent can predict them from the dry-run.
- Docs kept agent-parseable: this PLAN, a terse `README.md`, and `CLAUDE.md` with
  environment bootstrap (`conda activate local`, GPU expectations, test command).

## Implementation notes

- **Language**: Python ≥3.10, packaged with `pyproject.toml` (pip-installable,
  `karaokify` console script). Depends on `demucs>=4.1` and `torch` (declared but the
  README will document reusing an existing torch install — e.g. the `local` conda
  env already has torch 2.10 + CUDA and demucs 4.1.0).
- **Call demucs via its Python API** (`demucs.api.Separator`), not subprocess —
  gives progress callbacks, in-memory stems (no temp wav round-trip for the mix
  stage), and clean error propagation.
- **ffmpeg/ffprobe via subprocess** (system binary, min version 4.4): probe first,
  fail fast with exit 3 if missing. `-c:v copy` always; subtitle/attachment
  streams passed through with `-map 0` minus original audio.
- **Mixing** happens on the stem tensors before encoding (simple gain-and-sum),
  so `--vocals-gain -18` costs nothing extra.
- **Structure**:
  ```
  karaokify/
    pyproject.toml
    PLAN.md  README.md  CLAUDE.md
    src/karaokify/{__init__,cli,pipeline,separate,mux,manifest}.py
    tests/{test_cli.py, test_pipeline.py, fixtures/}
  ```

## Testing strategy

- **Fixture**: generate a 5 s synthetic "song" with ffmpeg (sine-wave chord bed +
  espeak/tts voice mixed on top, plus a `testsrc` video track) — checked-in
  generation script, not a binary blob. Total test runtime target: <60 s on CPU.
- **Unit**: output-path resolution, gain parsing, JSON schema (validate against a
  checked-in schema file), exit codes, skip/overwrite policy.
- **Integration** (marked `slow`, needs demucs): run the fixture end-to-end, assert
  output duration ≈ input duration, video stream bit-identical (`ffmpeg -c:v copy`
  hash), vocals band energy reduced vs input.
- Per global workflow: dry-run everything testable on fixtures before declaring done.

## Milestones

1. **M1 — MVP** (parity with the original two-command pipeline): single input,
   `--stems vocals`, remux, `--device`, `--dry-run`. Tag `v0.1`.
2. **M2 — Knobs**: gains/attenuation, model selection, shifts/overlap/segment,
   output codec/suffix, `--save-stems`, `--audio-only`.
3. **M3 — Agent surface**: `--json`, manifest, batch inputs + `--jobs`,
   skip/overwrite policy, exit-code contract, JSON schema file.
4. **M4 — Polish**: tests green in CI (CPU-only), README with recipes
   (karaoke, guide-vocal, drums-only practice track), CLAUDE.md, `v1.0`.

## Risks / open questions

- Demucs pulls a large torch; docs must steer users to an existing env rather than
  a fresh 3 GB install. Consider an optional `karaokify doctor` env-check subcommand.
- `htdemucs` weights download (~80 MB) on first run: surface progress on stderr,
  and document the cache location (`~/.cache/torch/hub/`... actually
  `torch.hub` dir) for offline/air-gapped use.
- Exotic containers (VOB, weird mkv audio layouts): rely on ffprobe stage to
  reject clearly with exit 1 + reason, rather than trying to be clever.
- Demucs is GPL-adjacent? No — MIT, and model weights are freely distributed;
  license the project MIT.
