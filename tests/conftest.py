from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from unrender.project import RunPaths

FFMPEG = shutil.which("ffmpeg")

requires_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not on PATH")


def make_video(
    path: Path,
    *,
    codec: str = "h264",
    frames: int = 48,
    fps: float = 24.0,
    size: str = "128x72",
    audio: bool = False,
    two_scenes: bool = False,
    timecode: str | None = None,
) -> Path:
    """Render a tiny synthetic video with ffmpeg for integration tests."""
    assert FFMPEG is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = frames / fps
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error"]
    if two_scenes:
        half = duration / 2
        cmd += ["-f", "lavfi", "-i", f"color=red:s={size}:r={fps}:d={half}"]
        cmd += ["-f", "lavfi", "-i", f"color=blue:s={size}:r={fps}:d={half}"]
        filter_parts = ["[0:v][1:v]concat=n=2:v=1[v]"]
        maps = ["-map", "[v]"]
    else:
        cmd += ["-f", "lavfi", "-i", f"testsrc2=s={size}:r={fps}:d={duration}"]
        filter_parts = []
        maps = ["-map", "0:v"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={duration}"]
        maps += ["-map", f"{2 if two_scenes else 1}:a"]
    if filter_parts:
        cmd += ["-filter_complex", ";".join(filter_parts)]
    cmd += maps
    if codec == "prores":
        cmd += ["-c:v", "prores_ks", "-profile:v", "0", "-pix_fmt", "yuv422p10le"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "pcm_s16le" if path.suffix == ".mov" else "aac"]
    cmd += ["-r", f"{fps}", "-frames:v", str(frames)]
    if timecode:
        cmd += ["-timecode", timecode]
    cmd.append(str(path))
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    return path


def make_png_sequence(
    directory: Path,
    *,
    frames: int = 24,
    start: int = 1,
    prefix: str = "seq.",
    padding: int = 4,
    fps: float = 24.0,
    size: str = "64x36",
) -> Path:
    assert FFMPEG is not None
    directory.mkdir(parents=True, exist_ok=True)
    pattern = str(directory / f"{prefix}%0{padding}d.png")
    cmd = [
        FFMPEG,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=s={size}:r={fps}:d={frames / fps}",
        "-frames:v",
        str(frames),
        "-start_number",
        str(start),
        pattern,
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    return directory


@pytest.fixture
def run_paths(tmp_path: Path) -> RunPaths:
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()
    return run


@pytest.fixture
def project_config_dir(tmp_path: Path) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


def write_project_config(
    path: Path,
    run_dir: Path,
    *,
    speakers: dict[str, object] | None = None,
    paths: dict[str, str] | None = None,
    audio: dict[str, object] | None = None,
    audio_assets: dict[str, object] | None = None,
    shots: dict[str, object] | None = None,
    stem_map: dict[str, str] | None = None,
) -> None:
    data: dict[str, object] = {
        "speakers": speakers
        or {
            "ALEX": {"aliases": []},
            "RENÉE": {"aliases": ["RENEE"]},
        },
        "paths": {"run_dir": str(run_dir), **(paths or {})},
    }
    if audio is not None:
        data["audio"] = audio
    if audio_assets is not None:
        data["audio_assets"] = audio_assets
    if shots is not None:
        data["shots"] = shots
    if stem_map is not None:
        data["stem_map"] = stem_map
    path.write_text(json.dumps(data), encoding="utf-8")


def write_wav(path: Path, samples: np.ndarray, *, sample_rate: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.astype(np.int16).tobytes())
