from __future__ import annotations

from pathlib import Path

import pytest

from unrender.lib.sequence import classify_master, detect_sequence


def _touch_frames(
    directory: Path, prefix: str, frames: list[int], *, padding: int = 4, ext: str = ".exr"
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        (directory / f"{prefix}{frame:0{padding}d}{ext}").write_bytes(b"x")


def test_detect_sequence_basic(tmp_path: Path) -> None:
    _touch_frames(tmp_path / "seq", "show.", [1001, 1002, 1003], padding=7)
    info = detect_sequence(tmp_path / "seq")
    assert info.prefix == "show."
    assert info.padding == 7
    assert info.ext == ".exr"
    assert (info.start_frame, info.end_frame, info.frame_count) == (1001, 1003, 3)
    assert info.missing_frames == ()
    assert info.pattern == "show.%07d.exr"
    assert info.frame_path(1002).name == "show.0001002.exr"


def test_detect_sequence_reports_gaps(tmp_path: Path) -> None:
    _touch_frames(tmp_path / "seq", "a_", [1, 2, 5, 6])
    info = detect_sequence(tmp_path / "seq")
    assert info.missing_frames == (3, 4)
    assert info.frame_count == 4


def test_detect_sequence_rejects_empty(tmp_path: Path) -> None:
    (tmp_path / "seq").mkdir()
    with pytest.raises(ValueError, match="no numbered image-sequence frames"):
        detect_sequence(tmp_path / "seq")


def test_detect_sequence_rejects_multiple_sequences(tmp_path: Path) -> None:
    _touch_frames(tmp_path / "seq", "a.", [1, 2])
    _touch_frames(tmp_path / "seq", "b.", [1, 2])
    with pytest.raises(ValueError, match="multiple image sequences"):
        detect_sequence(tmp_path / "seq")


def test_detect_sequence_rejects_mixed_padding(tmp_path: Path) -> None:
    seq = tmp_path / "seq"
    seq.mkdir()
    (seq / "a.001.exr").write_bytes(b"x")
    (seq / "a.0002.exr").write_bytes(b"x")
    with pytest.raises(ValueError, match="inconsistent frame-number padding"):
        detect_sequence(seq)


def test_detect_sequence_ignores_hidden_and_foreign_files(tmp_path: Path) -> None:
    _touch_frames(tmp_path / "seq", "a.", [1, 2])
    (tmp_path / "seq" / ".DS_Store").write_bytes(b"x")
    (tmp_path / "seq" / "notes.txt").write_text("hi")
    info = detect_sequence(tmp_path / "seq")
    assert info.frame_count == 2


def test_classify_master_video_file(tmp_path: Path) -> None:
    video = tmp_path / "master.mov"
    video.write_bytes(b"x")
    assert classify_master(video) == ("video", video)


def test_classify_master_sequence_dir_and_single_frame(tmp_path: Path) -> None:
    _touch_frames(tmp_path / "seq", "a.", [1, 2])
    kind, path = classify_master(tmp_path / "seq")
    assert (kind, path) == ("sequence", tmp_path / "seq")
    kind, path = classify_master(tmp_path / "seq" / "a.0001.exr")
    assert (kind, path) == ("sequence", tmp_path / "seq")


def test_classify_master_missing_or_unsupported(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        classify_master(tmp_path / "nope.mov")
    other = tmp_path / "master.wav"
    other.write_bytes(b"x")
    with pytest.raises(ValueError, match="unsupported master"):
        classify_master(other)
