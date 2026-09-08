from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """Read audio as float32 with shape (samples, channels)."""
    try:
        import soundfile as sf
    except ImportError:
        return _read_audio_scipy(path)
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def write_audio(path: Path, audio: np.ndarray, sample_rate: int, *, subtype: str = "FLOAT") -> None:
    try:
        import soundfile as sf
    except ImportError:
        _write_audio_scipy(path, audio, sample_rate, subtype=subtype)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), ensure_2d(audio), sample_rate, subtype=subtype)


def _read_audio_scipy(path: Path) -> tuple[np.ndarray, int]:
    from scipy.io import wavfile

    sample_rate, data = wavfile.read(path)
    arr = np.asarray(data)
    if np.issubdtype(arr.dtype, np.integer):
        max_value = float(np.iinfo(arr.dtype).max)
        arr = arr.astype(np.float32) / max_value
    else:
        arr = arr.astype(np.float32)
    return ensure_2d(arr), int(sample_rate)


def _write_audio_scipy(
    path: Path,
    audio: np.ndarray,
    sample_rate: int,
    *,
    subtype: str,
) -> None:
    from scipy.io import wavfile

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = ensure_2d(audio)
    if subtype.upper() in {"PCM_16", "PCM16"}:
        data = (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2")
    else:
        data = arr.astype(np.float32)
    wavfile.write(path, sample_rate, data)


def ensure_2d(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"expected 1D or 2D audio, got shape {arr.shape}")
    return arr


def to_mono(audio: np.ndarray) -> np.ndarray:
    arr = ensure_2d(audio)
    if arr.shape[1] == 1:
        return arr[:, 0]
    return arr.mean(axis=1)


def match_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    arr = ensure_2d(audio)
    if arr.shape[1] == channels:
        return arr
    if channels <= 0:
        raise ValueError("channels must be positive")
    if arr.shape[1] == 1:
        return np.repeat(arr, channels, axis=1)
    if channels == 1:
        return to_mono(arr)[:, None]
    raise ValueError(f"cannot map {arr.shape[1]} channels to {channels}")


def resample_to(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    arr = ensure_2d(audio)
    if source_sr <= 0 or target_sr <= 0:
        raise ValueError("sample rates must be positive")
    if source_sr == target_sr:
        return arr.astype(np.float32, copy=True)
    gcd = math.gcd(source_sr, target_sr)
    up = target_sr // gcd
    down = source_sr // gcd
    resampled = resample_poly(arr, up, down, axis=0)
    return np.asarray(resampled, dtype=np.float32)


def fit_length(audio: np.ndarray, sample_count: int) -> np.ndarray:
    arr = ensure_2d(audio)
    if len(arr) == sample_count:
        return arr
    if len(arr) > sample_count:
        return arr[:sample_count].copy()
    pad = np.zeros((sample_count - len(arr), arr.shape[1]), dtype=np.float32)
    return np.concatenate([arr, pad], axis=0)


def peak_normalize(audio: np.ndarray, *, ceiling: float = 0.98) -> np.ndarray:
    arr = ensure_2d(audio)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak <= 0.0 or peak <= ceiling:
        return arr.astype(np.float32, copy=True)
    return np.asarray(arr * (ceiling / peak), dtype=np.float32)


def rms(audio: np.ndarray, mask: np.ndarray | None = None) -> float:
    arr = ensure_2d(audio)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape[0] != arr.shape[0]:
            raise ValueError("mask length must match audio length")
        arr = arr[mask_arr]
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr), dtype=np.float64)))
