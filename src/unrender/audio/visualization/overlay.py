"""Transparent overlay-frame renderer and optional MOV encoding."""

from __future__ import annotations

import math
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from unrender.audio.visualization.bundle import AnalysisBundle, load_analysis_bundle
from unrender.audio.visualization.mapping import VisualMapping, load_visual_mapping


def _rgba(color: str, alpha: int) -> tuple[int, int, int, int]:
    return (*ImageColor.getrgb(color)[:3], max(0, min(255, alpha)))


def _event_color(role: str, mapping: VisualMapping) -> str:
    role = role.lower()
    if any(value in role for value in ("dialog", "speech", "voice", "dx")):
        return mapping.colors["dialogue"]
    if any(value in role for value in ("music", "mx")):
        return mapping.colors["music"]
    return mapping.colors["fx"]


def render_overlay_sequence(
    bundle: AnalysisBundle | Mapping[str, Any] | str | Path,
    output_dir: str | Path,
    mapping: VisualMapping | Mapping[str, Any] | str | Path | None = None,
    *,
    fps: float | None = None,
    duration_sec: float | None = None,
    max_duration_sec: float | None = None,
    max_fps: float | None = None,
    max_frames: int | None = None,
    width: int | None = None,
    height: int | None = None,
) -> tuple[Path, ...]:
    """Render alpha-preserving PNG frames for compositing over picture."""

    loaded = load_analysis_bundle(bundle)
    visual = load_visual_mapping(mapping)
    requested_fps = float(fps or visual.geometry["overlay_fps"])
    fps_limit = float(max_fps or visual.geometry["overlay_max_fps"])
    actual_fps = min(max(0.01, requested_fps), max(0.01, fps_limit))
    requested_duration = float(duration_sec or loaded.duration_sec)
    duration_limit = float(max_duration_sec or visual.geometry["overlay_max_duration"])
    actual_duration = min(max(0.001, requested_duration), max(0.001, duration_limit))
    frame_count = max(1, math.ceil(actual_duration * actual_fps))
    if max_frames is not None:
        frame_count = min(frame_count, max(1, int(max_frames)))
    width = int(width or visual.geometry["overlay_width"])
    height = int(height or visual.geometry["overlay_height"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    events = loaded.events()
    font = ImageFont.load_default()
    frames: list[Path] = []
    line_y = max(20, height - 35)
    left = 24
    right = width - 24
    usable_width = max(1, right - left)

    for index in range(frame_count):
        time_sec = min(actual_duration, index / actual_fps)
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rounded_rectangle(
            (8, line_y - 17, width - 8, height - 8),
            radius=8,
            fill=(5, 10, 24, 155),
            outline=_rgba(visual.colors["grid"], 180),
        )
        draw.line((left, line_y, right, line_y), fill=(255, 255, 255, 150), width=2)
        for event in events:
            start = np.clip(float(event["start_sec"]) / loaded.duration_sec, 0.0, 1.0)
            end = np.clip(float(event["end_sec"]) / loaded.duration_sec, 0.0, 1.0)
            x0 = left + round(float(start) * usable_width)
            x1 = left + round(float(end) * usable_width)
            color = _event_color(str(event["role"]), visual)
            draw.rectangle((x0, line_y - 8, max(x0 + 2, x1), line_y + 8), fill=_rgba(color, 210))
        playhead_fraction = np.clip(time_sec / loaded.duration_sec, 0.0, 1.0)
        playhead = left + round(float(playhead_fraction) * usable_width)
        draw.line(
            (playhead, line_y - 16, playhead, line_y + 16),
            fill=_rgba(visual.colors["playhead"], 255),
            width=3,
        )
        active = [
            event
            for event in events
            if float(event["start_sec"]) <= time_sec <= float(event["end_sec"])
        ]
        label = " + ".join(str(event["label"]) for event in active[:3]) or "Audio analysis"
        draw.text((14, 10), label[:80], fill=_rgba(visual.colors["text"], 235), font=font)
        draw.text(
            (width - 94, 10),
            f"{time_sec:7.2f}s",
            fill=_rgba(visual.colors["text"], 220),
            font=font,
        )
        frame_path = output / f"frame_{index:06d}.png"
        image.save(frame_path, format="PNG")
        frames.append(frame_path)
    return tuple(frames)


def encode_overlay_mov(
    frame_dir: str | Path,
    output_path: str | Path,
    *,
    fps: float,
    ffmpeg: str = "ffmpeg",
) -> Path:
    """Encode ``frame_%06d.png`` as a QuickTime Animation MOV with alpha."""

    frame_dir = Path(frame_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        f"{float(fps):g}",
        "-i",
        str(frame_dir / "frame_%06d.png"),
        "-c:v",
        "qtrle",
        "-pix_fmt",
        "argb",
        str(output),
    ]
    try:
        subprocess.run(command, check=True, timeout=300)
    except FileNotFoundError as exc:
        raise RuntimeError(f"ffmpeg executable not found: {ffmpeg}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed to encode transparent overlay: {exc}") from exc
    return output
