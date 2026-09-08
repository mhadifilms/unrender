from __future__ import annotations

import pytest

from unrender.lib.timecode import (
    frames_to_seconds,
    frames_to_smpte,
    seconds_to_frames,
    smpte_to_frames,
    smpte_to_seconds,
)


def test_frames_round_trip_integer_rate() -> None:
    for frame in (0, 1, 23, 24, 86_399, 86_400):
        label = frames_to_smpte(frame, 24.0)
        assert smpte_to_frames(label, 24.0) == frame


def test_frames_round_trip_ntsc_non_drop() -> None:
    for frame in (0, 1, 1_000, 86_400):
        label = frames_to_smpte(frame, 23.976)
        assert smpte_to_frames(label, 23.976) == frame


@pytest.mark.parametrize("fps", [29.97, 59.94])
def test_frames_round_trip_drop_frame(fps: float) -> None:
    for frame in (0, 1, 1_799, 1_800, 17_982, 107_892, 1_000_000):
        label = frames_to_smpte(frame, fps, drop_frame=True)
        assert ";" in label
        assert smpte_to_frames(label, fps) == frame


def test_drop_frame_minute_boundary() -> None:
    # At 29.97 DF the label skips ;00 and ;01 after each non-tenth minute.
    assert frames_to_smpte(1_800, 29.97, drop_frame=True) == "00:01:00;02"
    assert smpte_to_frames("00:01:00;02", 29.97) == 1_800
    # Tenth minutes do not drop.
    assert frames_to_smpte(17_982, 29.97, drop_frame=True) == "00:10:00;00"


def test_smpte_to_frames_accepts_bare_seconds() -> None:
    assert smpte_to_frames("2.0", 24.0) == 48
    assert smpte_to_frames("0", 24.0) == 0


def test_smpte_to_frames_invalid() -> None:
    assert smpte_to_frames("", 24.0) is None
    assert smpte_to_frames("aa:bb:cc:dd", 24.0) is None
    assert smpte_to_frames("00:00:01", 24.0) is None
    assert smpte_to_frames("00:00:01:00", 0.0) is None


def test_seconds_frames_inverses() -> None:
    assert seconds_to_frames(2.5, 24.0) == 60
    assert frames_to_seconds(60, 24.0) == 2.5
    # NTSC true-rate math: one hour of 23.976 is 3600s * 23.976 frames.
    frame = seconds_to_frames(3600.0, 23.976)
    assert frame == round(3600 * 23.976)


def test_smpte_to_seconds_agrees_with_frames() -> None:
    fps = 23.976
    label = "01:00:00:00"
    frames = smpte_to_frames(label, fps)
    assert frames is not None
    assert smpte_to_seconds(label, fps) == pytest.approx(frames / fps)


def test_frames_to_smpte_rejects_bad_fps() -> None:
    with pytest.raises(ValueError):
        frames_to_smpte(10, 0.0)
    with pytest.raises(ValueError):
        seconds_to_frames(1.0, -1.0)
    with pytest.raises(ValueError):
        frames_to_seconds(1, 0.0)
