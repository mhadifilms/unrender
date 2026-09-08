"""ffprobe/ffmpeg wrappers for video masters, proxies, and shot cutting.

The audio-oriented helpers live in :mod:`unrender.lib.ffmpeg`; this module
adds the video side: stream probing (codec, frame rate, timecode, color) and
a runner whose failures carry enough stderr to act on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from unrender.lib.proc import run_tool

DEFAULT_VIDEO_TIMEOUT_SEC = 7200

# Every frame is a keyframe: cutting with -ss before -i and stream copy is
# frame accurate. Anything else needs a decode (re-encode) to cut cleanly.
INTRA_FRAME_CODECS = frozenset(
    {"prores", "dnxhd", "dnxhr", "mjpeg", "ffv1", "v210", "rawvideo", "cfhd", "jpeg2000"}
)

# Compressed delivery codecs a scene detector can chew through directly.
DETECT_SAFE_CODECS = frozenset(
    {"h264", "hevc", "vp8", "vp9", "av1", "mpeg4", "mpeg2video", "mpeg1video"}
)

HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67"})


@dataclass(frozen=True)
class VideoProbe:
    path: Path
    duration_sec: float
    fps: float
    r_frame_rate: str
    avg_frame_rate: str
    codec: str
    width: int
    height: int
    pix_fmt: str
    color_primaries: str | None
    color_transfer: str | None
    color_space: str | None
    start_timecode: str | None
    has_audio: bool
    audio_codec: str | None
    audio_channels: int
    audio_sample_rate: int | None

    @property
    def is_intra_frame(self) -> bool:
        return self.codec in INTRA_FRAME_CODECS

    @property
    def is_hdr(self) -> bool:
        return (self.color_transfer or "") in HDR_TRANSFERS


def run_video_tool(cmd: list[str], *, timeout: int = DEFAULT_VIDEO_TIMEOUT_SEC) -> str:
    """Run ffmpeg/ffprobe, returning stdout; failures include stderr detail."""
    return run_tool(cmd, timeout=timeout, capture=True)


def probe_video(path: Path, *, ffprobe: str = "ffprobe") -> VideoProbe:
    if not path.is_file():
        raise ValueError(f"not a video file: {path}")
    stdout = run_video_tool(
        [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        timeout=120,
    )
    data = json.loads(stdout or "{}")
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"no video stream found in {path}")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format") or {}

    r_rate = str(video.get("r_frame_rate") or "")
    fps = _parse_rate(r_rate)
    duration = _parse_float(fmt.get("duration")) or _parse_float(video.get("duration")) or 0.0
    if duration <= 0:
        nb_frames = _parse_float(video.get("nb_frames"))
        if nb_frames and fps > 0:
            duration = nb_frames / fps
    if duration <= 0 or fps <= 0:
        raise ValueError(f"could not determine duration/frame rate for {path}")

    return VideoProbe(
        path=path,
        duration_sec=duration,
        fps=fps,
        r_frame_rate=r_rate,
        avg_frame_rate=str(video.get("avg_frame_rate") or ""),
        codec=str(video.get("codec_name") or ""),
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        pix_fmt=str(video.get("pix_fmt") or ""),
        color_primaries=_optional_str(video.get("color_primaries")),
        color_transfer=_optional_str(video.get("color_transfer")),
        color_space=_optional_str(video.get("color_space")),
        start_timecode=_probe_timecode(fmt, streams),
        has_audio=audio is not None,
        audio_codec=_optional_str(audio.get("codec_name")) if audio else None,
        audio_channels=int(audio.get("channels") or 0) if audio else 0,
        audio_sample_rate=(
            int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None
        ),
    )


def count_video_frames(path: Path, *, ffprobe: str = "ffprobe") -> int:
    """Exact packet count of the first video stream (fast, no decode)."""
    stdout = run_video_tool(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(path),
        ],
        timeout=600,
    )
    text = stdout.strip()
    if not text:
        raise RuntimeError(f"could not count frames in {path}")
    # Some ffprobe builds append an empty CSV side-data column (for example
    # ``7127,``) even though only ``nb_read_packets`` was requested.  Treat
    # the first populated field as the packet count instead of requiring the
    # complete CSV row to be an integer.
    for field in text.replace("\r", "\n").replace(",", "\n").splitlines():
        value = field.strip()
        if value:
            try:
                return int(value)
            except ValueError:
                continue
    raise RuntimeError(f"could not parse frame count from ffprobe output: {text!r}")


def is_vfr(probe: VideoProbe) -> bool:
    r_rate = _parse_rate(probe.r_frame_rate)
    avg_rate = _parse_rate(probe.avg_frame_rate)
    if r_rate <= 0 or avg_rate <= 0:
        return False
    return abs(r_rate - avg_rate) > 0.01


def videotoolbox_available(*, ffmpeg: str = "ffmpeg") -> bool:
    try:
        stdout = run_video_tool([ffmpeg, "-hide_banner", "-hwaccels"], timeout=30)
    except RuntimeError:
        return False
    return "videotoolbox" in stdout.lower()


def _probe_timecode(fmt: dict, streams: list[dict]) -> str | None:
    for source in [fmt, *streams]:
        tags = source.get("tags") or {}
        for key, value in tags.items():
            if key.lower() == "timecode" and str(value).strip():
                return str(value).strip()
    return None


def _parse_rate(value: str) -> float:
    text = str(value or "").strip()
    if not text or text in ("0/0", "0"):
        return 0.0
    try:
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _parse_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def _optional_str(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None
