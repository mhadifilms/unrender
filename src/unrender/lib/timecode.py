"""SMPTE timecode conversion shared by shot manifests and transcripts."""

from __future__ import annotations

import re


def smpte_to_seconds(value: str, fps: float) -> float | None:
    """Convert SMPTE timecode (``HH:MM:SS:FF``, drop-frame with ``;``) to seconds.

    Timecode labels count nominal frames (e.g. 24 or 30 per second), so real
    elapsed time is the absolute frame number divided by the actual rate. For
    integer rates this matches naive HH*3600+... math; for NTSC rates (23.976,
    29.97, 59.94) the naive math drifts 3.6 seconds per hour. Drop-frame
    timecode (semicolon separators) additionally skips 2 frame numbers per
    minute (4 at 59.94) except every tenth minute.
    """
    text = str(value or "").strip()
    if not text:
        return None
    if ":" not in text and ";" not in text:
        try:
            return float(text)
        except ValueError:
            return None
    frames = smpte_to_frames(text, fps)
    if frames is None or fps <= 0:
        return None
    return frames / fps


def smpte_to_frames(value: str, fps: float) -> int | None:
    """Convert SMPTE timecode to an absolute frame number.

    Bare numbers (no ``:``/``;``) are treated as seconds and rounded to the
    nearest frame, so callers can accept either representation. Returns
    ``None`` on unparseable input, matching ``smpte_to_seconds``.
    """
    text = str(value or "").strip()
    if not text:
        return None
    if fps <= 0:
        return None
    if ":" not in text and ";" not in text:
        try:
            return round(float(text) * fps)
        except ValueError:
            return None
    drop_frame = ";" in text
    parts = re.split(r"[:;]", text)
    if len(parts) != 4:
        return None
    try:
        hours, minutes, seconds, frames = [int(part) for part in parts]
    except ValueError:
        return None
    nominal = round(fps)
    if nominal <= 0:
        return None
    frame_number = ((hours * 3600) + (minutes * 60) + seconds) * nominal + frames
    if drop_frame and nominal in (30, 60) and fps < nominal:
        dropped_per_minute = 2 * (nominal // 30)
        total_minutes = hours * 60 + minutes
        frame_number -= dropped_per_minute * (total_minutes - total_minutes // 10)
    return frame_number


def frames_to_smpte(frame: int, fps: float, *, drop_frame: bool = False) -> str:
    """Format an absolute frame number as SMPTE timecode.

    Inverse of :func:`smpte_to_frames`. Drop-frame output (``;`` separator)
    is only meaningful for 29.97/59.94; requesting it at other rates falls
    back to non-drop formatting.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive: {fps}")
    nominal = round(fps)
    frame = max(0, int(frame))
    separator = ":"
    if drop_frame and nominal in (30, 60) and fps < nominal:
        separator = ";"
        dropped = 2 * (nominal // 30)
        frames_per_minute = 60 * nominal - dropped
        frames_per_10min = 10 * frames_per_minute + dropped
        chunks, remainder = divmod(frame, frames_per_10min)
        frame += dropped * 9 * chunks
        if remainder > dropped:
            frame += dropped * ((remainder - dropped) // frames_per_minute)
    total_seconds, ff = divmod(frame, nominal)
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{ff:02d}"


def seconds_to_frames(seconds: float, fps: float) -> int:
    """Nearest frame number for a time in seconds at the true frame rate."""
    if fps <= 0:
        raise ValueError(f"fps must be positive: {fps}")
    return round(seconds * fps)


def frames_to_seconds(frame: int, fps: float) -> float:
    """Exact time in seconds of a frame number at the true frame rate."""
    if fps <= 0:
        raise ValueError(f"fps must be positive: {fps}")
    return frame / fps


_TIMECODE_RE = re.compile(r"^\d{1,2}[:;]\d{2}[:;]\d{2}[:;]\d{1,2}$")


def parse_time_value(value: str, fps: float) -> float | None:
    """Seconds for a strict SMPTE timecode cell, or ``None`` if not a timecode.

    Unlike :func:`smpte_to_seconds`, a bare number is *not* accepted: callers
    rely on the ``None`` result to tell timecode cells apart from other text.
    """
    text = str(value or "").strip()
    if not _TIMECODE_RE.match(text):
        return None
    return smpte_to_seconds(text, fps)
