"""Probe real media duration / start time for accurate timeline clips.

Uses ``ffprobe`` for general media (video and compressed audio) and the stdlib
``wave`` module for ``.wav`` files so the common stem case needs no subprocess.
All probing degrades gracefully: when ``ffprobe`` is unavailable or a file
cannot be read, :meth:`MediaProber.probe` returns ``None`` and the caller falls
back to timing derived from the run artifacts.
"""

from __future__ import annotations

import json
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path

from unrender.lib.audio import probe_duration_sec
from unrender.lib.proc import run_tool

FFPROBE_TIMEOUT_SEC = 30


@dataclass(frozen=True)
class MediaInfo:
    duration_sec: float
    start_sec: float = 0.0
    has_video: bool = False
    has_audio: bool = False


class MediaProber:
    """Caching probe for media duration and embedded start time."""

    def __init__(self, ffprobe: str = "ffprobe") -> None:
        self._ffprobe = ffprobe
        self._available = shutil.which(ffprobe) is not None
        self._cache: dict[str, MediaInfo | None] = {}

    @property
    def available(self) -> bool:
        return self._available

    def probe(self, path: Path) -> MediaInfo | None:
        key = str(path)
        if key not in self._cache:
            self._cache[key] = self._probe_uncached(path)
        return self._cache[key]

    def _probe_uncached(self, path: Path) -> MediaInfo | None:
        if not path.exists():
            return None
        if path.suffix.lower() == ".wav":
            info = _wav_info(path)
            if info is not None:
                return info
        if not self._available:
            return None
        return _ffprobe_info(self._ffprobe, path)


def _wav_info(path: Path) -> MediaInfo | None:
    try:
        duration = probe_duration_sec(path)
    except (wave.Error, OSError, EOFError):
        return None
    if duration <= 0:
        return None
    return MediaInfo(duration_sec=duration, has_audio=True)


def _ffprobe_info(ffprobe: str, path: Path) -> MediaInfo | None:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        stdout = run_tool(cmd, timeout=FFPROBE_TIMEOUT_SEC, capture=True)
    except RuntimeError:
        return None
    try:
        data = json.loads(stdout or "{}")
    except json.JSONDecodeError:
        return None

    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    has_video = any(stream.get("codec_type") == "video" for stream in streams)
    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)

    duration = _as_float(fmt.get("duration"))
    if duration is None:
        durations = [_as_float(stream.get("duration")) for stream in streams]
        valid = [value for value in durations if value is not None]
        duration = max(valid) if valid else None
    if duration is None or duration <= 0:
        return None

    start = _as_float(fmt.get("start_time"))
    if start is None:
        starts = [_as_float(stream.get("start_time")) for stream in streams]
        valid_starts = [value for value in starts if value is not None]
        start = min(valid_starts) if valid_starts else 0.0
    return MediaInfo(
        duration_sec=duration,
        start_sec=max(0.0, start),
        has_video=has_video,
        has_audio=has_audio,
    )


def _as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
