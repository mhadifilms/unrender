"""Image-sequence (EXR/DPX/...) detection and master-path classification.

A "master" in unrender is either a single video file or a directory holding a
numbered image sequence. Shot cutting and proxy creation branch on that
classification, so it lives here as the single source of truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

IMAGE_SEQUENCE_EXTS = frozenset({".exr", ".dpx", ".png", ".tif", ".tiff", ".jpg", ".jpeg"})
VIDEO_EXTS = frozenset(
    {".mov", ".mp4", ".m4v", ".mxf", ".mkv", ".avi", ".webm", ".mts", ".m2ts", ".wmv", ".mpg"}
)

_FRAME_RE = re.compile(r"^(?P<prefix>.*?)(?P<number>\d+)(?P<ext>\.[A-Za-z0-9]+)$")


@dataclass(frozen=True)
class SequenceInfo:
    directory: Path
    prefix: str
    padding: int
    ext: str
    start_frame: int
    end_frame: int  # inclusive last existing frame
    frame_count: int
    missing_frames: tuple[int, ...]

    @property
    def pattern(self) -> str:
        return f"{self.prefix}%0{self.padding}d{self.ext}"

    @property
    def ffmpeg_pattern(self) -> str:
        return str(self.directory / self.pattern)

    def frame_name(self, frame: int) -> str:
        return f"{self.prefix}{frame:0{self.padding}d}{self.ext}"

    def frame_path(self, frame: int) -> Path:
        return self.directory / self.frame_name(frame)


def detect_sequence(directory: Path) -> SequenceInfo:
    """Detect a single numbered image sequence in ``directory``.

    Raises ``ValueError`` with an actionable message when the directory holds
    no frames, mixes several sequences, or numbers frames inconsistently.
    Gaps do not raise here; they are reported in ``missing_frames`` so each
    caller can decide (proxy creation refuses, cutting can skip).
    """
    directory = directory.expanduser()
    if not directory.is_dir():
        raise ValueError(f"not a directory: {directory}")

    groups: dict[tuple[str, str], dict[int, int]] = {}
    for entry in directory.iterdir():
        if not entry.is_file() or entry.name.startswith("."):
            continue
        match = _FRAME_RE.match(entry.name)
        if not match or match.group("ext").lower() not in IMAGE_SEQUENCE_EXTS:
            continue
        key = (match.group("prefix"), match.group("ext").lower())
        number = match.group("number")
        groups.setdefault(key, {})[int(number)] = len(number)

    if not groups:
        raise ValueError(
            f"no numbered image-sequence frames found in {directory} "
            f"(looked for {', '.join(sorted(IMAGE_SEQUENCE_EXTS))})"
        )
    if len(groups) > 1:
        names = ", ".join(f"{prefix or '<bare>'}*{ext}" for prefix, ext in sorted(groups))
        raise ValueError(
            f"multiple image sequences found in {directory}: {names}. "
            "Point paths.master at a directory containing a single sequence."
        )

    (prefix, ext), frames = next(iter(groups.items()))
    paddings = set(frames.values())
    if len(paddings) > 1:
        raise ValueError(
            f"inconsistent frame-number padding in {directory} for sequence "
            f"{prefix}*{ext}: found digit widths {sorted(paddings)}"
        )
    padding = paddings.pop()
    numbers = sorted(frames)
    start, end = numbers[0], numbers[-1]
    present = set(numbers)
    missing = tuple(frame for frame in range(start, end + 1) if frame not in present)
    return SequenceInfo(
        directory=directory,
        prefix=prefix,
        padding=padding,
        ext=ext,
        start_frame=start,
        end_frame=end,
        frame_count=len(numbers),
        missing_frames=missing,
    )


def classify_master(path: Path) -> tuple[str, Path]:
    """Classify a master path as ``("video", file)`` or ``("sequence", dir)``.

    A single image-sequence frame is accepted and resolved to its directory.
    """
    path = path.expanduser()
    if not path.exists():
        raise ValueError(f"master path does not exist: {path}")
    if path.is_dir():
        detect_sequence(path)  # raise early with a precise message
        return "sequence", path
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTS:
        return "video", path
    if suffix in IMAGE_SEQUENCE_EXTS:
        return "sequence", path.parent
    raise ValueError(
        f"unsupported master: {path}. Expected a video file "
        f"({', '.join(sorted(VIDEO_EXTS))}) or an image-sequence directory."
    )
