"""SMPTE timecode conversion, including NTSC and drop-frame rates."""

from __future__ import annotations

import pytest

from unrender.editorial.dialogue.transcript import seconds_from_timecode
from unrender.lib.timecode import smpte_to_seconds


def test_integer_rates_match_naive_math() -> None:
    assert smpte_to_seconds("00:00:01:12", 24.0) == pytest.approx(1.5)
    assert smpte_to_seconds("01:00:00:00", 24.0) == pytest.approx(3600.0)
    assert smpte_to_seconds("00:01:00:00", 25.0) == pytest.approx(60.0)


def test_plain_seconds_pass_through() -> None:
    assert smpte_to_seconds("12.5", 24.0) == pytest.approx(12.5)


def test_ntsc_non_drop_accounts_for_actual_rate() -> None:
    # One hour of NDF timecode at 29.97 is 108000 frames of real time,
    # not 3600 literal seconds (the naive math is 3.6s/hour fast).
    assert smpte_to_seconds("01:00:00:00", 29.97) == pytest.approx(108000 / 29.97)


def test_drop_frame_skips_two_frames_per_minute() -> None:
    # At 00:01:00;02 the label has skipped frame numbers ;00 and ;01,
    # so the absolute frame count is exactly 1800.
    assert smpte_to_seconds("00:01:00;02", 29.97) == pytest.approx(1800 / 29.97)
    # Every tenth minute does not drop: one hour drops 2 * 54 = 108 frames.
    assert smpte_to_seconds("01:00:00;00", 29.97) == pytest.approx((108000 - 108) / 29.97)


def test_drop_frame_at_5994_drops_four_frames() -> None:
    assert smpte_to_seconds("00:01:00;04", 59.94) == pytest.approx(3600 / 59.94)


def test_invalid_values_return_none() -> None:
    assert smpte_to_seconds("", 24.0) is None
    assert smpte_to_seconds("1:2:3", 24.0) is None
    assert smpte_to_seconds("aa:bb:cc:dd", 24.0) is None
    assert smpte_to_seconds("01:00:00:00", 0.0) is None


def test_transcript_timecode_delegates_to_shared_math() -> None:
    assert seconds_from_timecode("00:01:00;02", fps=29.97) == pytest.approx(1800 / 29.97)
    assert seconds_from_timecode("00:00:01:12", fps=24.0) == pytest.approx(1.5)
    assert seconds_from_timecode("not a timecode", fps=24.0) is None
