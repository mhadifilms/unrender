"""Robust audio loading for the spectral codec (native rate, 32-bit float).

Reads PCM WAV natively; float / high-rate WAV that the stdlib ``wave`` module
cannot parse, and any non-WAV (mp3/flac/m4a/...), fall back to an ffmpeg
float32 decode at the source's own sample rate and channel count so 32-bit
float / high-kHz material is never silently downgraded.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from unrender.lib.audio import read_wav_float
from unrender.lib.ffmpeg import decode_audio_f32le


def ffprobe_entry(source: Path, entry: str, *, ffmpeg: str = "ffmpeg") -> str:
    probe = ffmpeg.replace("ffmpeg", "ffprobe") if "ffmpeg" in ffmpeg else "ffprobe"
    cmd = [
        probe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        f"stream={entry}",
        "-of",
        "csv=p=0",
        str(source),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
        return out.splitlines()[0] if out else ""
    except (IndexError, FileNotFoundError):
        return ""


def load_audio_file(source: Path, *, ffmpeg: str = "ffmpeg") -> tuple[np.ndarray, int]:
    """Return ``(audio (N, C) float32, sample_rate)`` at the source's native rate."""
    source = Path(source)
    if source.suffix.lower() == ".wav":
        try:
            audio, rate, _ = read_wav_float(source)
            return audio.astype(np.float32), rate
        except Exception:
            # Float / WAVE_FORMAT_EXTENSIBLE files: fall back to the ffmpeg path.
            pass
    rate = int(ffprobe_entry(source, "sample_rate", ffmpeg=ffmpeg) or "44100")
    channels = max(1, min(int(ffprobe_entry(source, "channels", ffmpeg=ffmpeg) or "2"), 8))
    flat = decode_audio_f32le(source, ffmpeg=ffmpeg, sample_rate=rate, channels=channels)
    if flat is None:
        raise ValueError(f"could not decode audio from {source}")
    usable = (flat.size // channels) * channels
    return flat[:usable].reshape(-1, channels).astype(np.float32), rate
