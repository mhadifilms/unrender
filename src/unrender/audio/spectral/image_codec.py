"""Core STFT phase-image codec.

The image is laid out as ``(channels * bins, frames)`` pixels, RGBA uint8:

* ``bins = n_fft // 2 + 1`` rows per channel, channels stacked top-to-bottom,
  with bin 0 (DC) at the top of each channel band.
* Each column is one STFT frame; ``frames`` columns cover the whole signal.
* **R (low) + G (high)** hold a 16-bit unsigned magnitude, normalized by a
  single per-image ``mag_scale`` so the loudest bin maps to full scale.
* **B** holds phase; in ``phase_bits=16`` mode the **A** channel carries the
  phase high byte, otherwise A is opaque (255) and phase is 8-bit.

Reconstruction rebuilds the complex spectrum, runs an inverse rFFT per frame,
and does a weighted overlap-add (divide by the summed squared window) so the
Hann analysis window is undone exactly except for magnitude/phase quantization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

METADATA_KEY = "unrender-spectral"
CODEC_VERSION = 1


VISUAL = "visual"
LOSSLESS = "lossless"


@dataclass(frozen=True)
class StftParams:
    """Analysis/synthesis parameters for the STFT phase-image codec.

    ``mode`` selects the pixel encoding:

    * ``"visual"`` -- 16-bit dB magnitude in R/G and quantized phase in B(/A).
      The image reads like a spectrogram and the round-trip is ~89 dB (16-bit
      PCM transparent), but it is lossy.
    * ``"lossless"`` -- the raw float32 real/imag STFT coefficients are packed
      into pixels (two RGBA pixels per bin/frame). Reconstruction is limited
      only by float32 STFT/ISTFT round-off (~140 dB), so 32-bit float / high
      sample-rate audio is preserved end to end. The image looks like noise.
    """

    n_fft: int = 1024
    hop: int = 256  # 75% overlap Hann satisfies the COLA constraint
    phase_bits: int = 16
    # Magnitude is stored as 16-bit dB below the per-image peak. A wide range
    # gives a low noise floor for loud bins; quiet leakage bins get finer
    # absolute resolution than a linear scale, cutting accumulated quantization
    # noise. ~120 dB comfortably exceeds 16-bit PCM's own floor.
    db_range: float = 120.0
    mode: str = VISUAL

    def validate(self) -> None:
        if self.n_fft < 8 or self.n_fft % 2 != 0:
            raise ValueError(f"n_fft must be an even integer >= 8, got {self.n_fft}")
        if not 1 <= self.hop <= self.n_fft:
            raise ValueError(f"hop must be in [1, n_fft], got {self.hop}")
        if self.phase_bits not in (8, 16):
            raise ValueError(f"phase_bits must be 8 or 16, got {self.phase_bits}")
        if self.db_range <= 0:
            raise ValueError(f"db_range must be positive, got {self.db_range}")
        if self.mode not in (VISUAL, LOSSLESS):
            raise ValueError(f"mode must be {VISUAL!r} or {LOSSLESS!r}, got {self.mode!r}")

    @property
    def bins(self) -> int:
        return self.n_fft // 2 + 1


@dataclass(frozen=True)
class SpectralAxes:
    """Maps between audio time/frequency and image pixel coordinates.

    This is what makes a spectral image *addressable* -- the audio analog of a
    shot manifest's timecodes. A UI, editor, or ML consumer uses it to translate
    a pixel column to a moment in the source audio and a pixel row to a
    frequency, without knowing how the image was packed.
    """

    sample_rate: int
    n_fft: int
    hop: int
    bins: int
    frames: int
    columns_per_frame: int  # 1 in visual mode, 2 in lossless mode
    duration_sec: float

    @property
    def freq_max_hz(self) -> float:
        return self.sample_rate / 2.0

    @property
    def hz_per_row(self) -> float:
        return self.freq_max_hz / max(1, self.bins - 1)

    @property
    def seconds_per_frame(self) -> float:
        return self.hop / self.sample_rate if self.sample_rate else 0.0

    def frame_time_sec(self, frame: int) -> float:
        """Center time (source seconds) of an STFT frame."""
        return (frame * self.hop - self.n_fft / 2.0) / self.sample_rate

    def time_to_frame(self, seconds: float) -> int:
        return round((seconds * self.sample_rate + self.n_fft / 2.0) / self.hop)

    def time_to_column(self, seconds: float) -> int:
        return self.time_to_frame(seconds) * self.columns_per_frame

    def to_dict(self) -> dict:
        return {
            "sample_rate": int(self.sample_rate),
            "n_fft": int(self.n_fft),
            "hop": int(self.hop),
            "bins": int(self.bins),
            "frames": int(self.frames),
            "columns_per_frame": int(self.columns_per_frame),
            "duration_sec": round(float(self.duration_sec), 6),
            "seconds_per_frame": round(float(self.seconds_per_frame), 9),
            "freq_max_hz": round(float(self.freq_max_hz), 4),
            "hz_per_row": round(float(self.hz_per_row), 6),
        }


@dataclass(frozen=True)
class SpectralImage:
    """An encoded image plus the metadata needed to decode it."""

    pixels: np.ndarray  # (H, W, 4) uint8
    sample_rate: int
    channels: int
    length: int  # original samples per channel
    params: StftParams
    mag_scale: float

    @property
    def axes(self) -> SpectralAxes:
        columns_per_frame = 2 if self.params.mode == LOSSLESS else 1
        frames = int(self.pixels.shape[1]) // columns_per_frame
        return SpectralAxes(
            sample_rate=int(self.sample_rate),
            n_fft=int(self.params.n_fft),
            hop=int(self.params.hop),
            bins=int(self.params.bins),
            frames=frames,
            columns_per_frame=columns_per_frame,
            duration_sec=self.length / self.sample_rate if self.sample_rate else 0.0,
        )

    def metadata(self) -> dict:
        return {
            "codec": METADATA_KEY,
            "version": CODEC_VERSION,
            "sample_rate": int(self.sample_rate),
            "channels": int(self.channels),
            "length": int(self.length),
            "n_fft": int(self.params.n_fft),
            "hop": int(self.params.hop),
            "phase_bits": int(self.params.phase_bits),
            "db_range": repr(float(self.params.db_range)),
            "mode": str(self.params.mode),
            "bins": int(self.params.bins),
            # repr keeps full float precision so decode recovers absolute level.
            "mag_scale": repr(float(self.mag_scale)),
            "axes": self.axes.to_dict(),
        }


def _hann(n_fft: int) -> np.ndarray:
    """Periodic Hann window (COLA-correct for overlap synthesis)."""
    k = np.arange(n_fft, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * k / n_fft)


def _framing(length: int, params: StftParams) -> tuple[int, int, int]:
    """Return ``(front_pad, n_frames, padded_len)`` for a signal of ``length``.

    The signal is padded by a full window on the front (and at least a full
    window on the back) so every original sample is spanned by the full set of
    overlapping Hann frames. Framing is a pure function of ``length`` and the
    params, so encode and decode agree without storing frame counts.
    """
    n_fft, hop = params.n_fft, params.hop
    front_pad = n_fft
    length = max(0, int(length))
    n_frames = int(np.ceil((front_pad + length) / hop)) + 1
    padded_len = (n_frames - 1) * hop + n_fft
    return front_pad, n_frames, padded_len


def _stft(mono: np.ndarray, params: StftParams) -> np.ndarray:
    """Complex STFT ``(frames, bins)`` of a mono float64 buffer."""
    n_fft, hop = params.n_fft, params.hop
    n = int(mono.shape[0])
    front_pad, _, padded_len = _framing(n, params)
    padded = np.zeros(padded_len, dtype=np.float64)
    padded[front_pad : front_pad + n] = mono
    frames = sliding_window_view(padded, n_fft)[::hop]
    windowed = frames * _hann(n_fft)
    return np.fft.rfft(windowed, axis=1)


def _istft(spectrum: np.ndarray, params: StftParams, length: int) -> np.ndarray:
    """Weighted overlap-add inverse of :func:`_stft`, trimmed to ``length``."""
    n_fft, hop = params.n_fft, params.hop
    window = _hann(n_fft)
    frames = np.fft.irfft(spectrum, n=n_fft, axis=1)
    front_pad, _, padded_len = _framing(length, params)
    signal = np.zeros(padded_len, dtype=np.float64)
    weights = np.zeros(padded_len, dtype=np.float64)
    win_sq = window * window
    for index in range(frames.shape[0]):
        start = index * hop
        if start + n_fft > padded_len:
            break
        signal[start : start + n_fft] += frames[index] * window
        weights[start : start + n_fft] += win_sq
    nonzero = weights > 1e-9
    signal[nonzero] /= weights[nonzero]
    return signal[front_pad : front_pad + length]


def _encode_magnitude(magnitude: np.ndarray, scale: float, db_range: float) -> np.ndarray:
    """Pack magnitudes into 16-bit dB-below-peak codes (0 == true silence).

    A logarithmic scale keeps the relative quantization error roughly constant
    across the whole dynamic range, so quiet spectral bins no longer accumulate
    the absolute-step noise a linear 16-bit scale would impose.
    """
    norm = magnitude / scale if scale > 0 else magnitude
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(np.maximum(norm, 1e-30))
    idx = (db + db_range) / db_range  # 0 at the floor, 1 at the peak
    idx = np.clip(idx, 0.0, 1.0)
    codes = np.rint(idx * 65534.0).astype(np.uint32) + 1  # reserve 0 for silence
    codes[norm <= 0.0] = 0
    codes[db <= -db_range] = 0
    return codes


def _decode_magnitude(codes: np.ndarray, scale: float, db_range: float) -> np.ndarray:
    """Inverse of :func:`_encode_magnitude`."""
    codes = codes.astype(np.float64)
    idx = (codes - 1.0) / 65534.0
    db = idx * db_range - db_range
    magnitude = np.power(10.0, db / 20.0) * scale
    magnitude[codes <= 0.0] = 0.0
    return magnitude


def encode_audio_to_image(
    audio: np.ndarray,
    sample_rate: int,
    *,
    params: StftParams | None = None,
) -> SpectralImage:
    """Encode an ``(N,)`` or ``(N, channels)`` float buffer to a spectral image."""
    params = params or StftParams()
    params.validate()
    data = np.asarray(audio, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError(f"audio must be 1-D or 2-D, got shape {data.shape}")
    length, channels = int(data.shape[0]), int(data.shape[1])
    bins = params.bins

    specs = [_stft(data[:, ch], params) for ch in range(channels)]
    n_frames = max((spec.shape[0] for spec in specs), default=1)

    if params.mode == LOSSLESS:
        pixels = _encode_lossless(specs, channels, bins, n_frames)
        mag_scale = 1.0
    else:
        pixels, mag_scale = _encode_visual(specs, params, channels, bins, n_frames)

    return SpectralImage(
        pixels=pixels,
        sample_rate=int(sample_rate),
        channels=channels,
        length=length,
        params=params,
        mag_scale=mag_scale,
    )


def _encode_visual(
    specs: list[np.ndarray], params: StftParams, channels: int, bins: int, n_frames: int
) -> tuple[np.ndarray, float]:
    magnitudes = np.zeros((channels, bins, n_frames), dtype=np.float64)
    phases = np.zeros((channels, bins, n_frames), dtype=np.float64)
    for ch, spec in enumerate(specs):
        frames = spec.shape[0]
        magnitudes[ch, :, :frames] = np.abs(spec).T
        phases[ch, :, :frames] = np.angle(spec).T

    mag_scale = float(magnitudes.max()) if magnitudes.size else 0.0
    if mag_scale <= 0.0:
        mag_scale = 1.0

    pixels = np.zeros((channels * bins, n_frames, 4), dtype=np.uint8)
    for ch in range(channels):
        top = ch * bins
        mag16 = _encode_magnitude(magnitudes[ch], mag_scale, params.db_range)
        block = pixels[top : top + bins]
        block[..., 0] = (mag16 & 0xFF).astype(np.uint8)
        block[..., 1] = ((mag16 >> 8) & 0xFF).astype(np.uint8)

        phase_norm = np.clip((phases[ch] + np.pi) / (2.0 * np.pi), 0.0, 1.0)
        if params.phase_bits == 16:
            phase16 = np.rint(phase_norm * 65535.0).astype(np.uint32)
            block[..., 2] = (phase16 & 0xFF).astype(np.uint8)
            block[..., 3] = ((phase16 >> 8) & 0xFF).astype(np.uint8)
        else:
            block[..., 2] = np.rint(phase_norm * 255.0).astype(np.uint8)
            block[..., 3] = 255
    return pixels, mag_scale


def _encode_lossless(
    specs: list[np.ndarray], channels: int, bins: int, n_frames: int
) -> np.ndarray:
    """Pack float32 real/imag coefficients into two RGBA pixels per bin/frame."""
    pixels = np.zeros((channels * bins, n_frames * 2, 4), dtype=np.uint8)
    for ch, spec in enumerate(specs):
        top = ch * bins
        frames = spec.shape[0]
        real = np.zeros((bins, n_frames), dtype="<f4")
        imag = np.zeros((bins, n_frames), dtype="<f4")
        real[:, :frames] = spec.real.T.astype("<f4")
        imag[:, :frames] = spec.imag.T.astype("<f4")
        real_bytes = real.view(np.uint8).reshape(bins, n_frames, 4)
        imag_bytes = imag.view(np.uint8).reshape(bins, n_frames, 4)
        block = pixels[top : top + bins]
        block[:, 0::2, :] = real_bytes
        block[:, 1::2, :] = imag_bytes
    return pixels


def decode_image_to_audio(image: SpectralImage) -> tuple[np.ndarray, int]:
    """Reconstruct ``(N, channels)`` float32 audio from a spectral image."""
    params = image.params
    params.validate()
    bins = params.bins
    pixels = np.asarray(image.pixels)
    if pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError(f"expected an (H, W, 4) RGBA image, got shape {pixels.shape}")
    channels = image.channels
    if pixels.shape[0] < channels * bins:
        raise ValueError(
            f"image height {pixels.shape[0]} too small for {channels} channels of {bins} bins"
        )

    out = np.zeros((image.length, channels), dtype=np.float32)
    for ch in range(channels):
        top = ch * bins
        block = pixels[top : top + bins]
        if params.mode == LOSSLESS:
            spectrum = _decode_lossless_block(block)
        else:
            spectrum = _decode_visual_block(block, params, image.mag_scale)
        signal = _istft(spectrum, params, image.length)
        out[: signal.shape[0], ch] = signal.astype(np.float32)
    return out, int(image.sample_rate)


def _decode_visual_block(block: np.ndarray, params: StftParams, mag_scale: float) -> np.ndarray:
    codes = block.astype(np.uint32)
    mag16 = codes[..., 0] | (codes[..., 1] << 8)
    magnitude = _decode_magnitude(mag16, mag_scale, params.db_range)
    if params.phase_bits == 16:
        phase16 = codes[..., 2] | (codes[..., 3] << 8)
        phase_norm = phase16.astype(np.float64) / 65535.0
    else:
        phase_norm = codes[..., 2].astype(np.float64) / 255.0
    phase = phase_norm * (2.0 * np.pi) - np.pi
    return (magnitude * np.exp(1j * phase)).T  # (frames, bins)


def _decode_lossless_block(block: np.ndarray) -> np.ndarray:
    real_bytes = np.ascontiguousarray(block[:, 0::2, :], dtype=np.uint8)
    imag_bytes = np.ascontiguousarray(block[:, 1::2, :], dtype=np.uint8)
    real = real_bytes.reshape(block.shape[0], -1, 4).view("<f4")[..., 0]
    imag = imag_bytes.reshape(block.shape[0], -1, 4).view("<f4")[..., 0]
    return (real.astype(np.float64) + 1j * imag.astype(np.float64)).T  # (frames, bins)


def write_spectral_image(path: Path, image: SpectralImage) -> Path:
    """Write the codec PNG with metadata embedded in a text chunk."""
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = PngInfo()
    info.add_text(METADATA_KEY, json.dumps(image.metadata()))
    Image.fromarray(image.pixels, mode="RGBA").save(path, format="PNG", pnginfo=info)
    return path


def read_spectral_image(path: Path) -> SpectralImage:
    """Read a codec PNG written by :func:`write_spectral_image`."""
    from PIL import Image

    path = Path(path)
    with Image.open(path) as handle:
        meta_raw = handle.text.get(METADATA_KEY) if hasattr(handle, "text") else None
        pixels = np.asarray(handle.convert("RGBA"), dtype=np.uint8)
    if not meta_raw:
        raise ValueError(
            f"{path} has no '{METADATA_KEY}' metadata; it was not written by this codec "
            "(or the metadata was stripped by a lossy re-save)"
        )
    meta = json.loads(meta_raw)
    params = StftParams(
        n_fft=int(meta["n_fft"]),
        hop=int(meta["hop"]),
        phase_bits=int(meta.get("phase_bits", 16)),
        db_range=float(meta.get("db_range", 120.0)),
        mode=str(meta.get("mode", VISUAL)),
    )
    return SpectralImage(
        pixels=pixels,
        sample_rate=int(meta["sample_rate"]),
        channels=int(meta["channels"]),
        length=int(meta["length"]),
        params=params,
        mag_scale=float(meta["mag_scale"]),
    )


def preview_array(image: SpectralImage, *, channel: int = 0) -> np.ndarray:
    """Return a ``(bins, frames)`` uint8 dB spectrogram (low frequencies at bottom).

    The codec PNG itself is a data container and looks like colored noise; this
    renders magnitude on the stored dB scale so an editor can eyeball
    separation quality (bleed, silences, cue boundaries) at a glance.
    """
    bins = image.params.bins
    top = channel * bins
    block = np.asarray(image.pixels[top : top + bins])
    if image.params.mode == LOSSLESS:
        spectrum = _decode_lossless_block(block)  # (frames, bins)
        magnitude = np.abs(spectrum).T  # (bins, frames)
        peak = magnitude.max()
        norm = magnitude / peak if peak > 0 else magnitude
        with np.errstate(divide="ignore"):
            db = 20.0 * np.log10(np.maximum(norm, 1e-30))
        gray = np.clip((db + image.params.db_range) / image.params.db_range, 0.0, 1.0)
    else:
        codes = block.astype(np.uint32)
        mag16 = codes[..., 0] | (codes[..., 1] << 8)
        # The stored code is already linear in dB below peak, so it maps
        # straight to a conventional dB spectrogram.
        gray = mag16.astype(np.float64) / 65535.0
    return np.rint(gray * 255.0).astype(np.uint8)[::-1]  # low frequencies at bottom


def write_preview_png(path: Path, image: SpectralImage) -> Path:
    """Write the :func:`preview_array` dB spectrogram (channel 0) to ``path``."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview_array(image), mode="L").save(path, format="PNG")
    return path


def roundtrip_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    """Fidelity metrics between an original and a reconstructed buffer."""
    ref = np.asarray(original, dtype=np.float64)
    est = np.asarray(reconstructed, dtype=np.float64)
    if ref.ndim == 1:
        ref = ref[:, None]
    if est.ndim == 1:
        est = est[:, None]
    n = min(ref.shape[0], est.shape[0])
    channels = min(ref.shape[1], est.shape[1])
    ref = ref[:n, :channels].reshape(-1)
    est = est[:n, :channels].reshape(-1)
    if ref.size == 0:
        return {"snr_db": float("nan"), "max_abs_error": float("nan"), "correlation": float("nan")}
    noise = est - ref
    signal_power = float(np.mean(ref**2))
    noise_power = float(np.mean(noise**2))
    if noise_power <= 0:
        snr = float("inf")
    elif signal_power <= 0:
        snr = float("-inf")
    else:
        snr = 10.0 * float(np.log10(signal_power / noise_power))
    denom = float(np.linalg.norm(ref) * np.linalg.norm(est))
    correlation = float(np.dot(ref, est) / denom) if denom > 0 else 0.0
    return {
        "snr_db": snr,
        "max_abs_error": float(np.max(np.abs(noise))),
        "correlation": correlation,
    }
