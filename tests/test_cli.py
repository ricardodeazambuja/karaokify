"""Unit tests: gain resolution, path resolution, exit codes. No demucs needed."""

import json
from pathlib import Path

import pytest

from karaokify import cli, mux
from karaokify.cli import resolve_gains
from karaokify.pipeline import Options, resolve_output

NEG_INF = float("-inf")


# --- gain resolution -------------------------------------------------------

def test_default_drops_vocals():
    assert resolve_gains("vocals", None, None, None, []) == {
        "no_vocals": 0.0, "vocals": None,
    }


def test_vocals_gain_attenuates_instead_of_dropping():
    gains = resolve_gains("vocals", None, None, -18.0, [])
    assert gains == {"no_vocals": 0.0, "vocals": -18.0}


def test_vocals_gain_minus_inf_is_dropped():
    gains = resolve_gains("vocals", None, None, NEG_INF, [])
    assert gains == {"no_vocals": 0.0, "vocals": None}


def test_keep_only_vocals_acapella():
    gains = resolve_gains("vocals", "vocals", None, None, [])
    assert gains == {"no_vocals": None, "vocals": 0.0}


def test_drop_list_four_stems():
    gains = resolve_gains("4", None, "vocals,drums", None, [])
    assert gains == {"drums": None, "bass": 0.0, "other": 0.0, "vocals": None}


def test_per_stem_gain_overrides():
    gains = resolve_gains("4", None, None, None, ["bass=3", "other=-6"])
    assert gains == {"drums": 0.0, "bass": 3.0, "other": -6.0, "vocals": None}


def test_gain_minus_inf_drops_stem():
    gains = resolve_gains("4", None, None, None, ["drums=-inf"])
    assert gains["drums"] is None


def test_gain_can_resurrect_dropped_stem():
    gains = resolve_gains("vocals", None, None, None, ["vocals=-12"])
    assert gains == {"no_vocals": 0.0, "vocals": -12.0}


def test_six_stem_names():
    gains = resolve_gains("6", None, None, None, [])
    assert set(gains) == {"drums", "bass", "other", "vocals", "guitar", "piano"}


@pytest.mark.parametrize("bad", ["synth", "vocals,flute", ""])
def test_unknown_stem_rejected(bad):
    with pytest.raises(ValueError):
        resolve_gains("4", bad, None, None, [])


@pytest.mark.parametrize("bad", ["bass", "bass=", "bass=loud", "flute=3", "bass=nan"])
def test_bad_gain_arg_rejected(bad):
    with pytest.raises(ValueError):
        resolve_gains("4", None, None, None, [bad])


# --- output path resolution ------------------------------------------------

VIDEO_PROBE = {"duration_s": 5.0, "has_video": True, "audio_codec": "aac", "sample_rate": 44100}
AUDIO_PROBE = {"duration_s": 5.0, "has_video": False, "audio_codec": "mp3", "sample_rate": 44100}


def test_output_video_keeps_container():
    opts = Options(inputs=[])
    out = resolve_output(Path("/x/song.mp4"), opts, VIDEO_PROBE, "aac")
    assert out == Path("/x/song (no vocals).mp4")


def test_output_audio_ext_follows_codec():
    opts = Options(inputs=[])
    out = resolve_output(Path("/x/song.mp3"), opts, AUDIO_PROBE, "aac")
    assert out == Path("/x/song (no vocals).m4a")


def test_output_audio_only_flag_wins_over_video():
    opts = Options(inputs=[], audio_only=True)
    out = resolve_output(Path("/x/song.mp4"), opts, VIDEO_PROBE, "mp3")
    assert out == Path("/x/song (no vocals).mp3")


def test_output_dir_and_suffix():
    opts = Options(inputs=[], output_dir=Path("/out"), suffix="_karaoke")
    out = resolve_output(Path("/x/song.mp4"), opts, VIDEO_PROBE, "aac")
    assert out == Path("/out/song_karaoke.mp4")


def test_explicit_output_wins():
    opts = Options(inputs=[], output=Path("/y/final.mp4"))
    out = resolve_output(Path("/x/song.mp4"), opts, VIDEO_PROBE, "aac")
    assert out == Path("/y/final.mp4")


def test_copy_format_codec_resolution():
    assert mux.resolve_codec("copy-format", AUDIO_PROBE) == "mp3"
    assert mux.resolve_codec("copy-format", {"audio_codec": "pcm_s16le"}) == "aac"
    assert mux.resolve_codec("opus", AUDIO_PROBE) == "opus"


# --- invocation / exit codes ----------------------------------------------

def test_no_inputs_exits_2(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2


def test_unknown_flag_exits_2():
    with pytest.raises(SystemExit) as exc:
        cli.main(["song.mp4", "--frobnicate"])
    assert exc.value.code == 2


def test_output_with_multiple_inputs_exits_2():
    with pytest.raises(SystemExit) as exc:
        cli.main(["a.mp4", "b.mp4", "-o", "out.mp4"])
    assert exc.value.code == 2


def test_keep_and_drop_conflict_exits_2():
    with pytest.raises(SystemExit) as exc:
        cli.main(["a.mp4", "--keep", "vocals", "--drop", "vocals"])
    assert exc.value.code == 2


def test_missing_ffmpeg_exits_3(monkeypatch, capsys):
    monkeypatch.setattr("karaokify.mux.shutil.which", lambda tool: None)
    assert cli.main(["song.mp4"]) == 3


def test_missing_ffmpeg_json_error_document(monkeypatch, capsys):
    monkeypatch.setattr("karaokify.mux.shutil.which", lambda tool: None)
    assert cli.main(["song.mp4", "--json"]) == 3
    document = json.loads(capsys.readouterr().out)
    assert "error" in document
    assert document["results"] == []


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
