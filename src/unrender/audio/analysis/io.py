"""Bounded-memory audio decoding and atomic array storage."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from .models import AudioMetadata


def hash_file(path: Path | str, *, block_bytes: int = 1 << 20) -> str:
    """Return a stable full-file SHA-256 without reading the file at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def probe_audio(path: Path | str, *, ffmpeg: str = "ffmpeg") -> AudioMetadata:
    source = Path(path)
    if source.suffix.lower() in {".wav", ".wave"}:
        try:
            with wave.open(str(source), "rb") as handle:
                frames = handle.getnframes()
                rate = handle.getframerate()
                width = handle.getsampwidth()
                return AudioMetadata(
                    sample_rate=rate,
                    channels=handle.getnchannels(),
                    frames=frames,
                    duration_sec=frames / float(rate) if rate else 0.0,
                    codec=f"pcm_{width * 8}",
                    sample_format=f"s{width * 8}",
                )
        except (EOFError, wave.Error):
            pass

    command = [
        _ffprobe_executable(ffmpeg),
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,codec_name,sample_fmt,duration,nb_frames:format=duration",
        "-of",
        "json",
        str(source),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"could not probe audio file {source}") from exc
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise ValueError(f"no audio stream found in {source}")
    stream = streams[0]
    rate = int(stream.get("sample_rate") or 0)
    channels = int(stream.get("channels") or 0)
    duration = float(stream.get("duration") or payload.get("format", {}).get("duration") or 0.0)
    frames = int(stream.get("nb_frames") or round(duration * rate))
    if rate <= 0 or channels <= 0:
        raise ValueError(f"invalid audio stream metadata in {source}")
    return AudioMetadata(
        sample_rate=rate,
        channels=channels,
        frames=frames,
        duration_sec=duration or frames / float(rate),
        codec=str(stream.get("codec_name") or ""),
        sample_format=str(stream.get("sample_fmt") or ""),
    )


def iter_audio(
    path: Path | str,
    *,
    block_frames: int = 65_536,
    ffmpeg: str = "ffmpeg",
    metadata: AudioMetadata | None = None,
) -> Iterator[np.ndarray]:
    """Yield source-rate float32 blocks shaped ``(frames, channels)``.

    Integer PCM WAV files use the standard library and all other formats use
    an ffmpeg ``f32le`` stdout pipe. Neither path retains the complete decode.
    """
    if block_frames <= 0:
        raise ValueError("block_frames must be positive")
    source = Path(path)
    if source.suffix.lower() in {".wav", ".wave"}:
        try:
            yield from _iter_pcm_wav(source, block_frames)
            return
        except (EOFError, ValueError, wave.Error):
            # IEEE float, extensible WAV, and uncommon codecs go through ffmpeg.
            pass
    info = metadata or probe_audio(source, ffmpeg=ffmpeg)
    yield from _iter_ffmpeg(source, info, block_frames=block_frames, ffmpeg=ffmpeg)


def atomic_save_npy(path: Path | str, array: np.ndarray) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".npy",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, np.asarray(array), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return destination


def atomic_open_memmap(
    destination: Path | str,
    *,
    dtype: str | np.dtype[Any],
    shape: tuple[int, ...],
) -> tuple[np.memmap, Path]:
    """Create an NPY memmap temporary file to replace after successful writes."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".npy",
        dir=path.parent,
    )
    os.close(descriptor)
    mmap = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=shape)
    return mmap, Path(temporary)


def commit_memmap(mmap: np.memmap, temporary: Path, destination: Path | str) -> Path:
    path = Path(destination)
    mmap.flush()
    del mmap
    os.replace(temporary, path)
    return path


def _iter_pcm_wav(path: Path, block_frames: int) -> Iterator[np.ndarray]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        if channels <= 0 or width not in {1, 2, 3, 4}:
            raise ValueError("unsupported PCM WAV layout")
        while raw := handle.readframes(block_frames):
            values = _pcm_to_float(raw, width)
            usable = values.size - (values.size % channels)
            if usable:
                yield values[:usable].reshape(-1, channels)


def _pcm_to_float(raw: bytes, width: int) -> np.ndarray:
    if width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2_147_483_648.0
    bytes3 = np.frombuffer(raw, dtype=np.uint8)
    usable = bytes3.size - (bytes3.size % 3)
    triples = bytes3[:usable].reshape(-1, 3).astype(np.int32)
    values = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
    values = np.where(values & 0x800000, values - 0x1000000, values)
    return values.astype(np.float32) / 8_388_608.0


def _iter_ffmpeg(
    path: Path,
    metadata: AudioMetadata,
    *,
    block_frames: int,
    ffmpeg: str,
) -> Iterator[np.ndarray]:
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-ar",
        str(metadata.sample_rate),
        "-ac",
        str(metadata.channels),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    with tempfile.TemporaryFile() as errors:
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        except FileNotFoundError as exc:
            raise ValueError(f"ffmpeg is required to decode {path}") from exc
        assert process.stdout is not None
        frame_bytes = metadata.channels * 4
        pending = b""
        try:
            while raw := process.stdout.read(block_frames * frame_bytes):
                raw = pending + raw
                usable = len(raw) - (len(raw) % frame_bytes)
                if usable:
                    yield (
                        np.frombuffer(raw[:usable], dtype="<f4")
                        .reshape(-1, metadata.channels)
                        .copy()
                    )
                pending = raw[usable:]
            return_code = process.wait()
            if return_code:
                errors.seek(0)
                detail = errors.read().decode("utf-8", errors="replace").strip()
                raise ValueError(f"ffmpeg could not decode {path}: {detail}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def _ffprobe_executable(ffmpeg: str) -> str:
    executable = Path(ffmpeg)
    if executable.name.startswith("ffmpeg"):
        return str(executable.with_name(executable.name.replace("ffmpeg", "ffprobe", 1)))
    return "ffprobe"
