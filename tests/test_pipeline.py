"""Pipeline tests: dry-run contract on fixtures, plus slow end-to-end runs."""

import json
import subprocess
from pathlib import Path

import pytest

from karaokify import cli, manifest

jsonschema = pytest.importorskip("jsonschema")

SCHEMA = json.loads(manifest.schema_path().read_text())


def run_json(argv, capsys) -> tuple[int, dict]:
    code = cli.main(argv + ["--json"])
    document = json.loads(capsys.readouterr().out)
    jsonschema.validate(document, SCHEMA)
    return code, document


def ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return float(out)


def video_stream_md5(path: Path) -> str:
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
         "-c", "copy", "-f", "md5", "-"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def audio_rms(path: Path) -> float:
    proc = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(path),
         "-map", "0:a:0", "-af", "volumedetect", "-f", "null", "-"],
        check=True, capture_output=True, text=True,
    )
    for line in proc.stderr.splitlines():
        if "mean_volume" in line:
            return float(line.split("mean_volume:")[1].split("dB")[0])
    raise AssertionError("volumedetect produced no mean_volume")


# --- dry-run contract (fast, no separation) ---------------------------------

def test_dry_run_plans_and_predicts_output(fixture_media, capsys):
    song = fixture_media / "song.mp4"
    code, document = run_json([str(song), "--dry-run"], capsys)
    assert code == 0
    assert document["summary"] == {"processed": 0, "skipped": 0, "failed": 0, "planned": 1}
    (result,) = document["results"]
    assert result["action"] == "planned"
    assert result["output"] == str(fixture_media / "song (no vocals).mp4")
    assert result["model"] == "htdemucs"
    assert result["device"] in ("cpu", "cuda")
    assert result["stems"] == {"no_vocals": 0.0, "vocals": None}
    assert result["input_duration_s"] == pytest.approx(5.0, abs=0.5)


def test_dry_run_audio_input_gets_codec_extension(fixture_media, capsys):
    song = fixture_media / "song.mp3"
    code, document = run_json([str(song), "--dry-run"], capsys)
    assert code == 0
    assert document["results"][0]["output"] == str(fixture_media / "song (no vocals).m4a")


def test_dry_run_reports_planned_skip(fixture_media, capsys, tmp_path):
    song = fixture_media / "song.mp4"
    existing = tmp_path / "song (no vocals).mp4"
    existing.touch()
    code, document = run_json(
        [str(song), "--dry-run", "--output-dir", str(tmp_path)], capsys
    )
    assert code == 0
    (result,) = document["results"]
    assert result["action"] == "skipped"
    assert "exists" in result["reason"]


def test_dry_run_overwrite_plans_despite_existing(fixture_media, capsys, tmp_path):
    song = fixture_media / "song.mp4"
    (tmp_path / "song (no vocals).mp4").touch()
    code, document = run_json(
        [str(song), "--dry-run", "--overwrite", "--output-dir", str(tmp_path)], capsys
    )
    assert code == 0
    assert document["results"][0]["action"] == "planned"


def test_missing_input_fails_with_exit_1(capsys):
    code, document = run_json(["/nonexistent/nope.mp4", "--dry-run"], capsys)
    assert code == 1
    (result,) = document["results"]
    assert result["action"] == "failed"
    assert "not found" in result["reason"]


def test_output_equals_input_fails(fixture_media, capsys):
    song = fixture_media / "song.mp3"
    code, document = run_json([str(song), "--dry-run", "-o", str(song)], capsys)
    assert code == 1
    assert document["results"][0]["action"] == "failed"


def test_batch_mixes_planned_and_failed(fixture_media, capsys):
    code, document = run_json(
        [str(fixture_media / "song.mp4"), "/nonexistent/nope.mp4", "--dry-run"], capsys
    )
    assert code == 1
    assert document["summary"]["planned"] == 1
    assert document["summary"]["failed"] == 1


# --- end-to-end (slow: runs Demucs) -----------------------------------------

@pytest.mark.slow
def test_end_to_end_video(fixture_media, capsys, tmp_path):
    song = fixture_media / "song.mp4"
    manifest_path = tmp_path / "runs.jsonl"
    code, document = run_json(
        [str(song), "--output-dir", str(tmp_path), "--manifest", str(manifest_path)],
        capsys,
    )
    assert code == 0
    (result,) = document["results"]
    assert result["action"] == "processed"
    output = Path(result["output"])
    assert output.exists()
    assert result["output_size_bytes"] == output.stat().st_size

    # duration preserved, video stream bit-identical, audio energy not increased
    assert ffprobe_duration(output) == pytest.approx(ffprobe_duration(song), abs=0.5)
    assert video_stream_md5(output) == video_stream_md5(song)
    assert audio_rms(output) <= audio_rms(song) + 0.5
    if (fixture_media / "voice_source.txt").read_text().strip() == "espeak":
        # with real speech in the fixture, dropping vocals must reduce energy
        assert audio_rms(output) < audio_rms(song) - 1.0

    # manifest audit trail
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["action"] == "processed"
    assert "ts" in records[0]

    # idempotency: second run skips, --overwrite reprocesses
    code, document = run_json([str(song), "--output-dir", str(tmp_path)], capsys)
    assert code == 0
    assert document["results"][0]["action"] == "skipped"
    code, document = run_json(
        [str(song), "--output-dir", str(tmp_path), "--overwrite"], capsys
    )
    assert code == 0
    assert document["results"][0]["action"] == "processed"


@pytest.mark.slow
def test_keep_single_quiet_stem_discriminates(fixture_media, capsys, tmp_path):
    # the synthetic chord bed has essentially no bass-stem content, so keeping
    # only the bass stem must yield a much quieter output — proves the mix
    # honors --keep even when the fixture has no separable vocals
    song = fixture_media / "song.mp3"
    code, document = run_json(
        [str(song), "--stems", "4", "--keep", "bass",
         "--output-dir", str(tmp_path), "--suffix", " (bass)"],
        capsys,
    )
    assert code == 0
    (result,) = document["results"]
    assert result["action"] == "processed"
    assert result["stems"] == {"drums": None, "bass": 0.0, "other": None, "vocals": None}
    assert audio_rms(Path(result["output"])) < audio_rms(song) - 6.0


@pytest.mark.slow
def test_end_to_end_audio_only_with_stems(fixture_media, capsys, tmp_path):
    song = fixture_media / "song.mp3"
    code, document = run_json(
        [str(song), "--output-dir", str(tmp_path), "--save-stems",
         "--vocals-gain", "-18"],
        capsys,
    )
    assert code == 0
    (result,) = document["results"]
    assert result["action"] == "processed"
    assert result["stems"] == {"no_vocals": 0.0, "vocals": -18.0}
    assert Path(result["output"]).suffix == ".m4a"
    stems_dir = tmp_path / "song (stems)"
    assert sorted(p.name for p in stems_dir.iterdir()) == ["no_vocals.wav", "vocals.wav"]
