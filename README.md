<img width="947" height="330" alt="image" src="https://github.com/user-attachments/assets/8a1bfadf-cc14-4d75-9843-fe3d65a2221f" />

# karaokify

Remove (or attenuate) vocals from any audio/video file, locally, with one command.
Wraps [Demucs](https://github.com/adefossez/demucs) (source separation) + ffmpeg
(demux/remux). The video stream is copied untouched (`-c:v copy`).

```
karaokify video.mp4          # -> "video (no vocals).mp4" next to the input
```

## Install

Demucs pulls a large torch. **Reuse an existing torch install** instead of letting
pip download ~3 GB:

```bash
# inside an env that already has torch (and ideally demucs>=4.1):
pip install --no-deps karaokify        # or: pip install -e . --no-deps
pip install demucs                     # only if not present yet
```

Also needs `ffmpeg`/`ffprobe` (>= 4.4) on PATH. First run downloads the model
weights (~80 MB) to the `torch.hub` cache (`~/.cache/torch/hub/checkpoints/`);
pre-populate that directory for air-gapped machines.

## Recipes

```bash
karaokify song.mp4                                   # karaoke: vocals removed
karaokify song.mp4 --vocals-gain -18                 # guide vocal for sing-along
karaokify song.mp3 --keep vocals --suffix " (acapella)"   # acapella
karaokify song.mp4 --stems 4 --keep drums --audio-only    # drums-only practice track
karaokify song.mp4 --stems 4 --gain bass=6 --drop vocals  # no vocals, bass boosted
karaokify *.mp4 --output-dir out/ --jobs 4 --device cpu   # batch on CPU
karaokify song.mp4 --model htdemucs_ft --shifts 2         # best quality, ~8x slower
karaokify song.mp4 --save-stems                           # keep the raw stems (wav)
```

`karaokify --help` lists every knob; no flag is ever required.

## Agent contract (normative)

- **stdout is data, stderr is logs.** With `--json`, stdout contains exactly one
  JSON document (schema: [`src/karaokify/schema.json`](src/karaokify/schema.json)).
- **Exit codes**: `0` all inputs succeeded or were policy-skipped · `1` at least one
  input failed · `2` bad invocation · `3` environment missing (no ffmpeg/demucs,
  CUDA requested but absent). Never prompts.
- **`--dry-run --json`** resolves inputs, model, device, and output paths and
  returns the same schema with `action: "planned"` — validate a batch before
  spending GPU minutes. Output naming is deterministic; predict it from the dry run.
- **`--skip-existing`** (default) makes reruns idempotent; skipped files are
  reported, not errors. `--manifest runs.jsonl` appends an audit record per input.
- With `--json`, an environment error (exit 3) still emits one JSON document,
  with an `error` string and empty `results`.

Example:

```bash
karaokify song.mp4 --dry-run --json | jq -r '.results[0].output'
```

## Development

```bash
pip install -e . --no-deps && pip install pytest jsonschema
pytest -m "not slow"     # unit tests, seconds
pytest                   # includes end-to-end Demucs runs on a 5 s synthetic fixture
```

MIT license.
