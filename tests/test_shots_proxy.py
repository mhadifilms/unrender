from __future__ import annotations

from pathlib import Path

import pytest

from conftest import make_png_sequence, make_video, requires_ffmpeg
from unrender.editorial.shots.proxy import (
    create_master_proxy,
    default_proxy_path,
    master_usable_as_proxy,
    proxy_inputs,
)
from unrender.lib.video import probe_video
from unrender.project import RunPaths


@pytest.fixture
def run(tmp_path: Path) -> RunPaths:
    return RunPaths.from_path(tmp_path / "run")


@requires_ffmpeg
def test_video_proxy_from_prores(tmp_path: Path, run: RunPaths) -> None:
    master = make_video(
        tmp_path / "master.mov", codec="prores", frames=24, audio=True, timecode="00:59:58:00"
    )
    output = default_proxy_path(run, master)
    result = create_master_proxy(master=master, output=output, run=run)
    assert result.created and result.source_kind == "video"
    probe = probe_video(output)
    assert probe.codec == "h264"
    assert probe.has_audio
    assert probe.fps == pytest.approx(24.0)
    assert probe.start_timecode == "00:59:58:00"  # master TC carried over

    # Second run reuses the proxy.
    again = create_master_proxy(master=master, output=output, run=run)
    assert not again.created


@requires_ffmpeg
def test_video_proxy_warns_when_master_changed(
    tmp_path: Path, run: RunPaths, capsys: pytest.CaptureFixture[str]
) -> None:
    master = make_video(tmp_path / "master.mov", codec="prores", frames=24)
    output = default_proxy_path(run, master)
    create_master_proxy(master=master, output=output, run=run)
    make_video(tmp_path / "master.mov", codec="prores", frames=36)  # replace master
    result = create_master_proxy(master=master, output=output, run=run)
    assert not result.created
    assert "changed since" in capsys.readouterr().out


@requires_ffmpeg
def test_sequence_proxy(tmp_path: Path, run: RunPaths) -> None:
    seq = make_png_sequence(tmp_path / "seq", frames=24, start=1001, prefix="s.", padding=7)
    output = default_proxy_path(run, seq)
    result = create_master_proxy(
        master=seq, output=output, run=run, fps=24.0, timecode_start="01:00:00:00"
    )
    assert result.created and result.source_kind == "sequence"
    probe = probe_video(output)
    assert probe.codec == "h264"
    assert not probe.has_audio
    assert probe.start_timecode == "01:00:00:00"


@requires_ffmpeg
def test_sequence_proxy_starts_at_the_program_frame(tmp_path: Path, run: RunPaths) -> None:
    seq = make_png_sequence(tmp_path / "seq", frames=24, start=1001, prefix="s.", padding=7)
    output = default_proxy_path(run, seq)
    result = create_master_proxy(
        master=seq, output=output, run=run, fps=24.0, frame_range=(1005, 1012)
    )
    assert result.created
    probe = probe_video(output)
    assert probe.duration_sec == pytest.approx(8 / 24.0, abs=0.01)  # 1005..1012, not all 24 files


@requires_ffmpeg
def test_sequence_proxy_fingerprints_only_the_program_span(tmp_path: Path) -> None:
    seq = make_png_sequence(tmp_path / "seq", frames=24, start=1001, prefix="s.", padding=7)
    inside = proxy_inputs(seq, "sequence", (1005, 1012))["master"]
    assert [Path(p).name for p in inside] == ["s.0001005.png", "s.0001012.png"]


@requires_ffmpeg
def test_sequence_proxy_rejects_a_range_outside_the_sequence(tmp_path: Path, run: RunPaths) -> None:
    seq = make_png_sequence(tmp_path / "seq", frames=8, start=1001, prefix="s.", padding=7)
    with pytest.raises(ValueError, match="outside the sequence"):
        create_master_proxy(
            master=seq, output=run.media_dir / "p.mov", run=run, fps=24.0, frame_range=(1001, 1099)
        )


@requires_ffmpeg
def test_frame_range_is_meaningless_for_a_video_master(tmp_path: Path, run: RunPaths) -> None:
    master = make_video(tmp_path / "master.mov", codec="prores", frames=8)
    with pytest.raises(ValueError, match="only applies to an image sequence"):
        create_master_proxy(
            master=master, output=run.media_dir / "p.mov", run=run, frame_range=(2, 5)
        )


@requires_ffmpeg
def test_sequence_proxy_ignores_gaps_outside_the_span(tmp_path: Path, run: RunPaths) -> None:
    seq = make_png_sequence(tmp_path / "seq", frames=12, start=1001, prefix="s.", padding=7)
    (seq / "s.0001002.png").unlink()
    result = create_master_proxy(
        master=seq, output=run.media_dir / "p.mov", run=run, fps=24.0, frame_range=(1004, 1010)
    )
    assert result.created


@requires_ffmpeg
def test_sequence_proxy_requires_fps(tmp_path: Path, run: RunPaths) -> None:
    seq = make_png_sequence(tmp_path / "seq", frames=4)
    with pytest.raises(ValueError, match="explicit frame rate"):
        create_master_proxy(master=seq, output=run.media_dir / "p.mov", run=run)


@requires_ffmpeg
def test_sequence_proxy_refuses_gaps(tmp_path: Path, run: RunPaths) -> None:
    seq = make_png_sequence(tmp_path / "seq", frames=8, prefix="s.", padding=4)
    (seq / "s.0004.png").unlink()
    with pytest.raises(ValueError, match="missing frame"):
        create_master_proxy(master=seq, output=run.media_dir / "p.mov", run=run, fps=24.0)


def test_sequence_proxy_preflight_failure(tmp_path: Path, run: RunPaths) -> None:
    seq = tmp_path / "seq"
    seq.mkdir()
    for frame in (1, 2):
        (seq / f"bad.{frame:04d}.exr").write_bytes(b"not an exr")
    with pytest.raises(RuntimeError, match="cannot decode this sequence"):
        create_master_proxy(master=seq, output=run.media_dir / "p.mov", run=run, fps=24.0)


@requires_ffmpeg
def test_master_usable_as_proxy(tmp_path: Path) -> None:
    h264 = make_video(tmp_path / "web.mp4", codec="h264", frames=8)
    prores = make_video(tmp_path / "master.mov", codec="prores", frames=8)
    assert master_usable_as_proxy(h264)
    assert not master_usable_as_proxy(prores)
    assert not master_usable_as_proxy(tmp_path / "missing.mov")
    assert not master_usable_as_proxy(tmp_path)  # directory
