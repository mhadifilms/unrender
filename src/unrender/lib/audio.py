from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

_WAVE_FORMAT_PCM = 0x0001
_WAVE_FORMAT_IEEE_FLOAT = 0x0003
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE


def pcm_values(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if sample_width == 3:
        data = np.frombuffer(raw, dtype=np.uint8)
        if data.size % 3:
            data = data[: data.size - (data.size % 3)]
        triples = data.reshape(-1, 3).astype(np.int32)
        values = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        values = np.where(values & 0x800000, values | ~0xFFFFFF, values)
        return values.astype(np.float32)
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32)
    raise ValueError(f"unsupported WAV sample width: {sample_width}")


def full_scale(sample_width: int) -> float:
    if sample_width == 1:
        return 128.0
    if sample_width == 2:
        return 32768.0
    if sample_width == 3:
        return 8388608.0
    if sample_width == 4:
        return 2147483648.0
    raise ValueError(f"unsupported WAV sample width: {sample_width}")


def _parse_riff_chunks(file: BinaryIO, path: Path) -> tuple[bytes, int]:
    """Walk RIFF chunks to the data chunk; returns (fmt payload, data size)."""
    header = file.read(12)
    if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise wave.Error(f"file does not start with RIFF/WAVE header: {path}")
    fmt: bytes | None = None
    while True:
        chunk_header = file.read(8)
        if len(chunk_header) < 8:
            raise wave.Error(f"WAV file has no data chunk: {path}")
        chunk_id, size = struct.unpack("<4sI", chunk_header)
        if chunk_id == b"fmt ":
            fmt = file.read(size)
            if size % 2:
                file.seek(1, 1)
        elif chunk_id == b"data":
            if fmt is None:
                raise wave.Error(f"WAV data chunk precedes fmt chunk: {path}")
            return fmt, size
        else:
            file.seek(size + (size % 2), 1)


class WavReader:
    """Chunked WAV reader that normalizes integer PCM and IEEE-float samples.

    Integer PCM files go through the stdlib ``wave`` module exactly as before.
    IEEE-float files (format tag 3, e.g. float32 stems written by the BandIt
    separation backend) and WAVE_FORMAT_EXTENSIBLE files, which ``wave``
    rejects with ``unknown format``, are parsed directly from the RIFF chunks.
    ``read`` always yields interleaved float32 samples normalized to full
    scale so peak/RMS/dBFS math is identical across sample formats. Malformed
    files raise ``wave.Error`` so existing callers keep their stdlib
    exception contract.
    """

    def __init__(self, path: Path) -> None:
        self._wave: wave.Wave_read | None = None
        self._file: BinaryIO | None = None
        self._remaining = 0
        self.is_float = False
        try:
            self._wave = wave.open(str(path), "rb")  # noqa: SIM115 - closed by close()
        except wave.Error:
            self._open_riff(Path(path))
        else:
            self.channels = self._wave.getnchannels()
            self.sample_rate = self._wave.getframerate()
            self.frames = self._wave.getnframes()
            self.sample_width = self._wave.getsampwidth()

    def _open_riff(self, path: Path) -> None:
        file = path.open("rb")
        try:
            fmt, data_size = _parse_riff_chunks(file, path)
            if len(fmt) < 16:
                raise wave.Error(f"WAV fmt chunk is too short: {path}")
            tag, channels, rate, _byte_rate, _block_align, bits = struct.unpack_from("<HHIIHH", fmt)
            if tag == _WAVE_FORMAT_EXTENSIBLE:
                if len(fmt) < 26:
                    raise wave.Error(f"WAV extensible fmt chunk is too short: {path}")
                tag = struct.unpack_from("<H", fmt, 24)[0]
            if tag == _WAVE_FORMAT_IEEE_FLOAT:
                if bits not in (32, 64):
                    raise wave.Error(f"unsupported IEEE float bit depth: {bits}")
                self.is_float = True
            elif tag == _WAVE_FORMAT_PCM:
                if bits not in (8, 16, 24, 32):
                    raise wave.Error(f"unsupported PCM bit depth: {bits}")
            else:
                raise wave.Error(f"unknown format: {tag}")
            if channels <= 0 or rate <= 0:
                raise wave.Error(f"invalid WAV fmt fields in {path}")
            self.channels = int(channels)
            self.sample_rate = int(rate)
            self.sample_width = bits // 8
            frame_bytes = self.channels * self.sample_width
            # Placeholder/overstated data sizes (streamed writers) are clamped
            # to the bytes actually present in the file.
            data_start = file.tell()
            file.seek(0, 2)
            available = max(0, file.tell() - data_start)
            file.seek(data_start)
            self._remaining = min(int(data_size), available)
            self.frames = self._remaining // frame_bytes
        except BaseException:
            file.close()
            raise
        self._file = file

    @property
    def pcm_sample_width(self) -> int:
        """Integer sample width to use when re-encoding this audio as PCM."""
        return 4 if self.is_float else self.sample_width

    def read(self, frames: int) -> np.ndarray:
        """Return up to ``frames`` frames as interleaved normalized float32."""
        if frames <= 0:
            return np.empty(0, dtype=np.float32)
        if self._wave is not None:
            raw = self._wave.readframes(frames)
            if not raw:
                return np.empty(0, dtype=np.float32)
            return pcm_values(raw, self.sample_width) / full_scale(self.sample_width)
        assert self._file is not None
        want = min(frames * self.channels * self.sample_width, self._remaining)
        if want <= 0:
            return np.empty(0, dtype=np.float32)
        raw = self._file.read(want)
        self._remaining = self._remaining - len(raw) if raw else 0
        if not raw:
            return np.empty(0, dtype=np.float32)
        usable = len(raw) - (len(raw) % self.sample_width)
        if self.is_float:
            dtype = "<f4" if self.sample_width == 4 else "<f8"
            return np.frombuffer(raw[:usable], dtype=dtype).astype(np.float32)
        return pcm_values(raw[:usable], self.sample_width) / full_scale(self.sample_width)

    def close(self) -> None:
        if self._wave is not None:
            self._wave.close()
        if self._file is not None:
            self._file.close()

    def __enter__(self) -> WavReader:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def measure_peak_dbfs(path: Path) -> float:
    peak = 0.0
    with WavReader(path) as reader:
        while True:
            values = reader.read(65536)
            if not values.size:
                break
            peak = max(peak, float(np.max(np.abs(values))))
    if peak <= 0:
        return float("-inf")
    return 20.0 * math.log10(peak)


@dataclass(frozen=True)
class AudioActivity:
    peak_dbfs: float
    rms_dbfs: float
    voiced_ratio: float


def measure_activity(
    path: Path, *, threshold_db: float = -50.0, hop_sec: float = 0.02
) -> AudioActivity:
    with WavReader(path) as reader:
        sample_rate = reader.sample_rate
        channels = reader.channels
        hop_frames = max(1, int(sample_rate * hop_sec))
        samples_per_hop = hop_frames * max(1, channels)
        peak = 0.0
        squares = 0.0
        count = 0
        voiced = 0
        hops = 0
        carry = np.empty(0, dtype=np.float32)
        block_frames = hop_frames * 512
        while True:
            values = reader.read(block_frames)
            if not values.size:
                break
            peak = max(peak, float(np.max(np.abs(values))))
            squares += float(np.sum(np.square(values)))
            count += int(values.size)
            if carry.size:
                values = np.concatenate([carry, values])
            full = (values.size // samples_per_hop) * samples_per_hop
            carry = values[full:]
            if full:
                frames = values[:full].reshape(-1, samples_per_hop)
                mean_squares = np.mean(np.square(frames), axis=1)
                levels = np.full(mean_squares.shape, -np.inf)
                positive = mean_squares > 0
                levels[positive] = 10.0 * np.log10(mean_squares[positive])
                voiced += int(np.count_nonzero(levels > threshold_db))
                hops += int(levels.size)
        if carry.size:
            mean_square = float(np.mean(np.square(carry)))
            level = 10.0 * math.log10(mean_square) if mean_square > 0 else float("-inf")
            voiced += 1 if level > threshold_db else 0
            hops += 1
    peak_dbfs = 20.0 * math.log10(peak) if peak > 0 else float("-inf")
    rms = (squares / count) ** 0.5 if count > 0 and squares > 0 else 0.0
    rms_dbfs = 20.0 * math.log10(rms) if rms > 0 else float("-inf")
    voiced_ratio = voiced / hops if hops > 0 else 0.0
    return AudioActivity(peak_dbfs=peak_dbfs, rms_dbfs=rms_dbfs, voiced_ratio=voiced_ratio)


def probe_duration_sec(path: Path) -> float:
    with WavReader(path) as reader:
        frames = reader.frames
        rate = reader.sample_rate
    if rate <= 0:
        return 0.0
    return frames / float(rate)


def probe_rms_db(path: Path) -> float:
    with WavReader(path) as reader:
        squares = 0.0
        count = 0
        while True:
            values = reader.read(65536)
            if not values.size:
                break
            squares += float(np.sum(np.square(values)))
            count += int(values.size)
    if count <= 0 or squares <= 0:
        return float("-inf")
    rms = (squares / count) ** 0.5
    return 20.0 * float(np.log10(rms))


def mix_wav_files(sources: list[Path], target: Path) -> Path:
    """Sum WAV files sample-by-sample into a single integer-PCM stem.

    Used to reconstruct a full DX stem from separated per-speaker stems when
    no original dialogue stem exists: separation is additive, so summing the
    stems recovers the dialogue bed. Values are clipped at full scale. All
    sources must share sample rate, channel count, and effective sample width
    (an IEEE-float source counts as 32-bit). Float sources are mixed in the
    normalized domain and written out via the existing integer PCM path.
    """
    if not sources:
        raise ValueError("no stems provided to mix")
    readers = [WavReader(path) for path in sources]
    try:
        rates = {reader.sample_rate for reader in readers}
        widths = {reader.pcm_sample_width for reader in readers}
        channels = {reader.channels for reader in readers}
        if len(rates) != 1 or len(widths) != 1 or len(channels) != 1:
            raise ValueError(
                "stems disagree on format: "
                f"rates={sorted(rates)} widths={sorted(widths)} channels={sorted(channels)}"
            )
        rate = rates.pop()
        width = widths.pop()
        channel_count = channels.pop()
        total = max(reader.frames for reader in readers)
        scale = full_scale(width)
        target.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(target), "wb") as writer:
            writer.setnchannels(channel_count)
            writer.setsampwidth(width)
            writer.setframerate(rate)
            chunk = 262_144
            done = 0
            while done < total:
                count = min(chunk, total - done)
                mixed = np.zeros(count * channel_count, dtype=np.float64)
                for reader in readers:
                    values = reader.read(count)
                    if values.size:
                        mixed[: values.size] += values
                np.clip(mixed, -1.0, 1.0 - 1.0 / scale, out=mixed)
                writer.writeframes(_pcm_bytes(mixed * scale, width))
                done += count
    finally:
        for reader in readers:
            reader.close()
    return target


def _pcm_bytes(values: np.ndarray, sample_width: int) -> bytes:
    if sample_width == 1:
        return (values + 128.0).astype(np.uint8).tobytes()
    if sample_width == 2:
        return values.astype("<i2").tobytes()
    if sample_width == 3:
        clipped = np.clip(values, -8388608, 8388607).astype("<i4")
        raw = clipped.view(np.uint8).reshape(-1, 4)
        return raw[:, :3].reshape(-1).tobytes()
    if sample_width == 4:
        return values.astype("<i4").tobytes()
    raise ValueError(f"unsupported WAV sample width: {sample_width}")


def write_wav_mono(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    data = np.asarray(samples, dtype="float32").reshape(-1)
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def write_wav_float32(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write a 32-bit IEEE float WAV (no PCM quantization).

    The stdlib ``wave`` module only writes integer PCM, so float output goes
    through ``scipy.io.wavfile``. Used when audio quality must be preserved at
    full 32-bit float precision (e.g. the spectral codec's lossless mode).
    """
    from scipy.io import wavfile

    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), int(sample_rate), data)


def read_wav_float(path: Path) -> tuple[np.ndarray, int, int]:
    """Read a PCM or IEEE-float WAV as normalized float32 ``(N, ch)``.

    The returned width is the effective integer PCM width (float sources map
    to 4) so callers can hand it straight back to ``write_wav_float``.
    """
    with WavReader(path) as reader:
        channels = reader.channels
        rate = reader.sample_rate
        width = reader.pcm_sample_width
        values = reader.read(max(reader.frames, 0))
    if channels <= 0:
        raise ValueError(f"invalid channel count in {path}: {channels}")
    usable = (values.size // channels) * channels
    data = values[:usable].reshape(-1, channels).astype(np.float32)
    return data, rate, width


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Polyphase-resample ``audio`` (``(N,)`` or ``(N, ch)``) to ``target_sr``."""
    if orig_sr == target_sr or audio.size == 0:
        return np.asarray(audio, dtype=np.float32)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(orig_sr), int(target_sr))
    up, down = int(target_sr) // g, int(orig_sr) // g
    out = resample_poly(np.asarray(audio, dtype=np.float32), up, down, axis=0)
    return out.astype(np.float32)


def read_audio_any(path: Path, *, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Read any audio file to float32 ``(N, ch)`` at ``target_sr`` (or native).

    Tries the fast native WAV path (integer PCM and IEEE float) first, then
    falls back to ffmpeg for anything else (mp3, m4a, ogg, uncommon codecs),
    so the editing pipeline accepts arbitrary source formats.
    """
    try:
        data, rate, _ = read_wav_float(path)
    except (OSError, ValueError, EOFError, wave.Error):
        from unrender.lib.ffmpeg import decode_audio_f32le

        rate = int(target_sr or 48000)
        flat = decode_audio_f32le(Path(path), sample_rate=rate, channels=2)
        if flat is None:
            raise ValueError(f"could not decode audio: {path}") from None
        data = flat.reshape(-1, 2) if flat.size % 2 == 0 else flat.reshape(-1, 1)
    if target_sr is not None and rate != target_sr:
        data = resample_audio(data, rate, target_sr)
        rate = int(target_sr)
    return data.astype(np.float32), int(rate)


def write_wav_float(
    path: Path, audio: np.ndarray, sample_rate: int, *, sample_width: int = 3
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    scale = full_scale(sample_width)
    values = np.clip(data, -1.0, 1.0 - (1.0 / scale)) * scale
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(data.shape[1])
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        handle.writeframes(_pcm_bytes(values, sample_width))


def peak_limit(audio: np.ndarray, ceiling: float = 0.99) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > ceiling:
        return audio * (ceiling / peak)
    return audio


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio))) + 1e-12)


def rms_dbfs(audio: np.ndarray) -> float:
    return 20.0 * float(np.log10(rms(audio)))


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Collapse an ``(N, channels)`` or ``(N,)`` buffer to mono float32."""
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data.reshape(-1)


def frame_rms_db(
    mono: np.ndarray,
    *,
    sample_rate: int,
    hop_sec: float = 0.02,
) -> tuple[np.ndarray, float]:
    """Per-hop RMS levels in dBFS plus the actual hop duration."""
    flat = np.asarray(mono, dtype=np.float32).reshape(-1)
    if sample_rate <= 0 or flat.size == 0:
        return np.empty(0, dtype=np.float64), hop_sec
    hop = max(1, round(sample_rate * hop_sec))
    hop_duration = hop / float(sample_rate)
    usable = (flat.size // hop) * hop
    if usable == 0:
        mean_square = float(np.mean(np.square(flat)))
        level = 10.0 * np.log10(mean_square) if mean_square > 0 else -np.inf
        return np.asarray([level], dtype=np.float64), hop_duration
    frames = flat[:usable].reshape(-1, hop)
    mean_squares = np.mean(np.square(frames), axis=1)
    levels = np.full(mean_squares.shape, -np.inf)
    positive = mean_squares > 0
    levels[positive] = 10.0 * np.log10(mean_squares[positive])
    return levels, hop_duration


def speech_active_rms(reference: np.ndarray, fallback: np.ndarray) -> float:
    if reference.size:
        mono = reference.mean(axis=1) if reference.ndim == 2 else reference
        threshold = max(float(np.percentile(np.abs(mono), 75)) * 0.25, 1e-5)
        mask = np.abs(mono) >= threshold
        if np.any(mask):
            return rms(reference[mask])
    return rms(fallback)
