from __future__ import annotations

import json
import subprocess
from pathlib import Path

from unrender.audio.separation.input_audio import normalize_full_audio_input
from unrender.project import RunPaths


def _fake_media_runner(monkeypatch, streams: list[dict[str, object]]) -> list[list[str]]:
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"streams": streams}),
                stderr="",
            )
        target = Path(command[-1])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"normalized pcm")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("unrender.audio.separation.input_audio.subprocess.run", fake_run)
    return commands


def _stream(index: int, channels: int, layout: str = "") -> dict[str, object]:
    return {
        "index": index,
        "channels": channels,
        "channel_layout": layout,
        "codec_name": "pcm_s24le",
        "sample_rate": "48000",
    }


def test_eight_mono_streams_select_first_two_as_existing_stereo_program(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "package.mov"
    source.write_bytes(b"media")
    commands = _fake_media_runner(monkeypatch, [_stream(index, 1) for index in range(1, 9)])
    run = RunPaths.from_path(tmp_path / "run")

    result = normalize_full_audio_input(run=run, source=source, prefix="show")

    assert result.selection == "first_mono_pair_2_0_program"
    assert result.selected_streams == (1, 2)
    assert result.normalized is True
    assert Path(result.prepared_source).parent == run.normalized_audio_dir
    ffmpeg_command = commands[1]
    assert (
        "[0:1][0:2]amerge=inputs=2" in ffmpeg_command[ffmpeg_command.index("-filter_complex") + 1]
    )
    assert "0:3" not in " ".join(ffmpeg_command)


def test_single_stereo_wav_remains_directly_usable(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "mix.wav"
    source.write_bytes(b"wav")
    commands = _fake_media_runner(monkeypatch, [_stream(0, 2, "stereo")])

    result = normalize_full_audio_input(
        run=RunPaths.from_path(tmp_path / "run"),
        source=source,
    )

    assert result.prepared_source == source
    assert result.normalized is False
    assert result.selection == "direct_stereo_wav"
    assert len(commands) == 1


def test_external_mono_pair_is_merged_as_stereo(monkeypatch, tmp_path: Path) -> None:
    left = tmp_path / "mix_L.wav"
    right = tmp_path / "mix_R.wav"
    left.write_bytes(b"left")
    right.write_bytes(b"right")
    commands = _fake_media_runner(monkeypatch, [_stream(0, 1, "mono")])
    run = RunPaths.from_path(tmp_path / "run")

    result = normalize_full_audio_input(
        run=run,
        source=(left, right),
        prefix="show",
    )

    assert result.source == (str(left), str(right))
    assert result.selection == "external_mono_pair_2_0_program"
    assert result.selected_streams == (0, 0)
    assert result.normalized is True
    ffmpeg_command = commands[2]
    assert ffmpeg_command.count("-i") == 2
    assert str(left) in ffmpeg_command
    assert str(right) in ffmpeg_command
    filter_graph = ffmpeg_command[ffmpeg_command.index("-filter_complex") + 1]
    assert filter_graph == "[0:a:0][1:a:0]amerge=inputs=2,pan=stereo|c0=c0|c1=c1[out]"
    assert ffmpeg_command.count("-map") == 1
    assert ffmpeg_command[ffmpeg_command.index("-map") + 1] == "[out]"


def test_external_pair_requires_one_mono_stream_per_side(monkeypatch, tmp_path: Path) -> None:
    left = tmp_path / "mix_L.wav"
    right = tmp_path / "mix_R.wav"
    left.write_bytes(b"left")
    right.write_bytes(b"right")
    _fake_media_runner(monkeypatch, [_stream(0, 2, "stereo")])

    try:
        normalize_full_audio_input(
            run=RunPaths.from_path(tmp_path / "run"),
            source=(left, right),
        )
    except ValueError as exc:
        assert "one mono stream per side" in str(exc)
    else:
        raise AssertionError("a non-mono external pair should be rejected")


def test_embedded_stereo_media_is_staged_as_pcm(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "mix.mxf"
    source.write_bytes(b"media")
    commands = _fake_media_runner(monkeypatch, [_stream(4, 2, "stereo")])

    result = normalize_full_audio_input(
        run=RunPaths.from_path(tmp_path / "run"),
        source=source,
    )

    assert result.selection == "stereo_stream"
    assert result.normalized is True
    assert commands[1][commands[1].index("-map") + 1] == "0:4"
    assert "pcm_s24le" in commands[1]


def test_single_5_1_stream_falls_back_to_stereo_downmix(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "surround.mov"
    source.write_bytes(b"media")
    commands = _fake_media_runner(monkeypatch, [_stream(2, 6, "5.1")])

    result = normalize_full_audio_input(
        run=RunPaths.from_path(tmp_path / "run"),
        source=source,
    )

    assert result.selection == "downmix_5_1_stream"
    ffmpeg_command = commands[1]
    assert ffmpeg_command[ffmpeg_command.index("-map") + 1] == "0:2"
    assert ffmpeg_command[ffmpeg_command.index("-ac") + 1] == "2"


def test_six_mono_streams_fall_back_to_5_1_downmix(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "surround.mov"
    source.write_bytes(b"media")
    commands = _fake_media_runner(monkeypatch, [_stream(index, 1) for index in range(1, 7)])

    result = normalize_full_audio_input(
        run=RunPaths.from_path(tmp_path / "run"),
        source=source,
    )

    assert result.selection == "downmix_six_mono_5_1"
    filter_graph = commands[1][commands[1].index("-filter_complex") + 1]
    assert "amerge=inputs=6" in filter_graph
    assert "pan=stereo" in filter_graph


def test_normalized_input_is_idempotent(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "mix.mov"
    source.write_bytes(b"media")
    commands = _fake_media_runner(monkeypatch, [_stream(1, 2, "stereo")])
    run = RunPaths.from_path(tmp_path / "run")

    first = normalize_full_audio_input(run=run, source=source)
    second = normalize_full_audio_input(run=run, source=source)

    assert first.prepared_source == second.prepared_source
    assert len([command for command in commands if command[0] == "ffmpeg"]) == 1


def test_changed_source_requires_force_before_replacing_staged_audio(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "mix.mov"
    source.write_bytes(b"first")
    _fake_media_runner(monkeypatch, [_stream(1, 2, "stereo")])
    run = RunPaths.from_path(tmp_path / "run")
    normalize_full_audio_input(run=run, source=source)
    source.write_bytes(b"changed source")

    try:
        normalize_full_audio_input(run=run, source=source)
    except ValueError as exc:
        assert "rerun with --force" in str(exc)
    else:
        raise AssertionError("changed source should require --force")
