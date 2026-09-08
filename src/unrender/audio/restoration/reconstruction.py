from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import istft, stft

from unrender.audio.audio_io import (
    fit_length,
    match_channels,
    resample_to,
    to_mono,
)
from unrender.audio.restoration.upsamplers import (
    FLASHSR_MODEL_SHA256,
    REUSE_MODEL_REVISION,
    AudioUpsampler,
    UpsamplerSettings,
)

RestorerName = Literal["auto", "reuse", "flashsr", "resample"]


@dataclass(frozen=True)
class ReferenceReconstructionSettings:
    restorer: RestorerName = "auto"
    target_sample_rate: int = 48_000
    device: str = "auto"
    n_fft: int = 2048
    hop_length: int = 512
    mask_power: float = 1.5
    residual_floor: float = 0.02
    reuse_python: str | None = None
    reuse_model_dir: Path | None = None


class ReferenceSourceReconstructor:
    """Project restored source proposals through the original mixture spectrum.

    The neural restorer proposes source-specific high-band energy. Normalized
    ratio masks allocate the original complex stereo spectrum once, and an
    explicit residual receives all unallocated energy. This avoids adding the
    original high-passed mixture independently to every speaker stem.
    """

    def __init__(self, settings: ReferenceReconstructionSettings) -> None:
        self.settings = settings
        self._upsampler: AudioUpsampler | None = None
        self._resolved_restorer = ""

    def reconstruct(
        self,
        original: np.ndarray,
        sample_rate: int,
        estimates: list[np.ndarray],
        estimate_sample_rate: int,
        *,
        work_dir: Path,
    ) -> tuple[list[np.ndarray], np.ndarray, dict[str, object]]:
        mixture = np.asarray(original, dtype=np.float32)
        if mixture.ndim == 1:
            mixture = mixture[:, None]
        if not estimates:
            return (
                [],
                mixture.copy(),
                {
                    "backend": "original_reference_masks",
                    "restorer": self._restorer_name(),
                    "restorer_revision": self._restorer_revision(),
                    "residual_ratio": 1.0,
                },
            )
        if len(estimates) == 2 and estimate_sample_rate == 8_000:
            if self.settings.restorer == "auto" and not self._resolved_restorer:
                self._resolved_restorer = "reuse"
            return self._reconstruct_isolated_sources(
                mixture,
                sample_rate,
                estimates,
                estimate_sample_rate,
                work_dir=work_dir,
            )
        if len(estimates) == 2 and estimate_sample_rate == 16_000:
            return self._reconstruct_isolated_sources(
                mixture,
                sample_rate,
                estimates,
                estimate_sample_rate,
                work_dir=work_dir,
            )
        proposals = [
            fit_length(
                self._upsample(
                    estimate,
                    estimate_sample_rate,
                    sample_rate,
                    work_dir=work_dir / f"source_{index:02d}",
                ),
                len(mixture),
            )
            for index, estimate in enumerate(estimates)
        ]
        low_band_proposals = [
            fit_length(
                resample_to(
                    estimate,
                    estimate_sample_rate,
                    sample_rate,
                ),
                len(mixture),
            )
            for estimate in estimates
        ]
        # SSD validation favored direct separator low-band for two/four-way
        # models, while ratio-mask projection was more stable for three/five.
        preserve_low_band_hz = 0.475 * estimate_sample_rate if len(estimates) in {2, 4} else None
        sources, residual = project_original_spectrum(
            mixture,
            proposals,
            sample_rate,
            n_fft=self.settings.n_fft,
            hop_length=self.settings.hop_length,
            mask_power=self.settings.mask_power,
            residual_floor=self.settings.residual_floor,
            preserve_low_band_hz=preserve_low_band_hz,
            low_band_proposals=low_band_proposals,
        )
        source_sum = sum(sources, start=np.zeros_like(mixture))
        null = mixture - (source_sum + residual)
        mixture_energy = float(np.sum(np.square(mixture), dtype=np.float64))
        residual_energy = float(np.sum(np.square(residual), dtype=np.float64))
        return (
            sources,
            residual,
            {
                "backend": "original_reference_masks",
                "restorer": self._restorer_name(),
                "restorer_revision": self._restorer_revision(),
                "mask_power": self.settings.mask_power,
                "residual_floor": self.settings.residual_floor,
                "preserve_low_band_hz": preserve_low_band_hz,
                "residual_ratio": residual_energy / max(mixture_energy, 1e-12),
                "max_reconstruction_error": float(np.max(np.abs(null))) if null.size else 0.0,
            },
        )

    def _reconstruct_lowband_guided(
        self,
        mixture: np.ndarray,
        sample_rate: int,
        estimates: list[np.ndarray],
        estimate_sample_rate: int,
    ) -> tuple[list[np.ndarray], np.ndarray, dict[str, object]]:
        sources, residual, projection = project_lowband_guided_spectrum(
            mixture,
            estimates,
            estimate_sample_rate,
            sample_rate,
            n_fft=self.settings.n_fft,
            hop_length=self.settings.hop_length,
            mask_power=self.settings.mask_power,
            residual_floor=self.settings.residual_floor,
        )
        source_sum = sum(sources, start=np.zeros_like(mixture))
        null = mixture - (source_sum + residual)
        mixture_energy = float(np.sum(np.square(mixture), dtype=np.float64))
        residual_energy = float(np.sum(np.square(residual), dtype=np.float64))
        return (
            sources,
            residual,
            {
                "backend": "lowband_guided_highband_masks",
                "separator_sample_rate": estimate_sample_rate,
                "lowband_cutoff_hz": projection["lowband_cutoff_hz"],
                "crossfade_hz": projection["crossfade_hz"],
                "harmonic_divisors": projection["harmonic_divisors"],
                "mask_power": self.settings.mask_power,
                "residual_floor": self.settings.residual_floor,
                "residual_ratio": residual_energy / max(mixture_energy, 1e-12),
                "max_reconstruction_error": (float(np.max(np.abs(null))) if null.size else 0.0),
            },
        )

    def _reconstruct_isolated_sources(
        self,
        mixture: np.ndarray,
        sample_rate: int,
        estimates: list[np.ndarray],
        estimate_sample_rate: int,
        *,
        work_dir: Path,
    ) -> tuple[list[np.ndarray], np.ndarray, dict[str, object]]:
        restored: list[np.ndarray] = []
        source_level_gains: list[float] = []
        for index, estimate in enumerate(estimates):
            enhanced = fit_length(
                self._upsample(
                    estimate,
                    estimate_sample_rate,
                    sample_rate,
                    work_dir=work_dir / f"source_{index:02d}",
                ),
                len(mixture),
            )
            gated = _gate_from_raw_activity(
                to_mono(estimate),
                enhanced,
                estimate_sample_rate,
                sample_rate,
            )
            calibrated, level_gain = _match_restored_source_level(
                estimate,
                gated,
                estimate_sample_rate,
                sample_rate,
            )
            restored.append(match_channels(calibrated, mixture.shape[1]))
            source_level_gains.append(level_gain)
        restored, program_level_gain = _match_original_program_level(restored, mixture)
        source_sum = sum(restored, start=np.zeros_like(mixture))
        residual = np.asarray(mixture - source_sum, dtype=np.float32)
        null = mixture - (source_sum + residual)
        mixture_energy = float(np.sum(np.square(mixture), dtype=np.float64))
        residual_energy = float(np.sum(np.square(residual), dtype=np.float64))
        return (
            restored,
            residual,
            {
                "backend": "isolated_source_gated",
                "restorer": self._restorer_name(),
                "restorer_revision": self._restorer_revision(),
                "activity_gate": "source_relative_20db",
                "source_level_gains": source_level_gains,
                "program_level_gain": program_level_gain,
                "final_source_level_gains": [
                    round(value * program_level_gain, 8) for value in source_level_gains
                ],
                "source_level_reference": (
                    "separator_relative_lowband_rms_then_original_mix_program_energy"
                ),
                "residual_ratio": residual_energy / max(mixture_energy, 1e-12),
                "max_reconstruction_error": (float(np.max(np.abs(null))) if null.size else 0.0),
            },
        )

    def _upsample(
        self,
        estimate: np.ndarray,
        estimate_sample_rate: int,
        target_sample_rate: int,
        *,
        work_dir: Path,
    ) -> np.ndarray:
        proposal_rate = self.settings.target_sample_rate
        if self._upsampler is None or self._upsampler.settings.target_sample_rate != proposal_rate:
            backend = self._restorer_name()
            self._upsampler = AudioUpsampler(
                UpsamplerSettings(
                    backend=backend,
                    target_sample_rate=proposal_rate,
                    device=self.settings.device,
                    reuse_python=self.settings.reuse_python,
                    reuse_model_dir=self.settings.reuse_model_dir,
                )
            )
        restored, restored_rate = self._upsampler.upsample(
            estimate,
            estimate_sample_rate,
            work_dir=work_dir,
        )
        if restored_rate != target_sample_rate:
            restored = resample_to(restored, restored_rate, target_sample_rate)
        return restored

    def _restorer_name(self) -> Literal["reuse", "flashsr", "resample"]:
        if self._resolved_restorer:
            return self._resolved_restorer  # type: ignore[return-value]
        selected = self.settings.restorer
        if selected == "auto":
            selected = (
                "reuse"
                if self.settings.reuse_model_dir is not None
                or os.environ.get("UNRENDER_REUSE_MODEL_DIR")
                else "flashsr"
            )
        if selected not in {"reuse", "flashsr", "resample"}:
            raise ValueError(f"unsupported reference restorer: {selected}")
        self._resolved_restorer = selected
        return selected

    def _restorer_revision(self) -> str:
        selected = self._restorer_name()
        if selected == "reuse":
            return REUSE_MODEL_REVISION
        if selected == "flashsr":
            return FLASHSR_MODEL_SHA256
        return ""


def project_lowband_guided_spectrum(
    original: np.ndarray,
    estimates: list[np.ndarray],
    estimate_sample_rate: int,
    sample_rate: int,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
    mask_power: float = 1.5,
    residual_floor: float = 0.02,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, object]]:
    """Preserve separated low band and assign original high band by masks."""
    mixture = np.asarray(original, dtype=np.float32)
    if mixture.ndim == 1:
        mixture = mixture[:, None]
    if mixture.ndim != 2:
        raise ValueError(f"expected original shape (samples, channels), got {mixture.shape}")
    if not estimates:
        return (
            [],
            mixture.copy(),
            {
                "lowband_cutoff_hz": 0.0,
                "crossfade_hz": [0.0, 0.0],
                "harmonic_divisors": [],
            },
        )
    if estimate_sample_rate <= 0 or sample_rate <= 0:
        raise ValueError("sample rates must be positive")
    if mask_power <= 0:
        raise ValueError("mask_power must be positive")

    frame_length = min(max(16, n_fft), len(mixture))
    if frame_length < 2:
        return (
            [np.zeros_like(mixture) for _ in estimates],
            mixture.copy(),
            {
                "lowband_cutoff_hz": 0.0,
                "crossfade_hz": [0.0, 0.0],
                "harmonic_divisors": [],
            },
        )
    hop = min(max(1, hop_length), frame_length - 1)
    overlap = frame_length - hop
    frequencies = np.fft.rfftfreq(frame_length, 1.0 / sample_rate)

    mixture_stft = [
        _stft(mixture[:, channel], sample_rate, frame_length, overlap)
        for channel in range(mixture.shape[1])
    ]
    direct_audio = [
        fit_length(
            resample_to(estimate, estimate_sample_rate, sample_rate),
            len(mixture),
        )
        for estimate in estimates
    ]
    direct_stft = [
        _stft(
            to_mono(source),
            sample_rate,
            frame_length,
            overlap,
        )
        for source in direct_audio
    ]

    cutoff_hz = 0.475 * estimate_sample_rate
    lower_hz = 0.9 * cutoff_hz
    upper_hz = min(sample_rate / 2.0, 1.1 * cutoff_hz)
    low_bins = (frequencies >= 250.0) & (frequencies <= cutoff_hz)
    if not np.any(low_bins):
        raise ValueError("no usable low-band bins for high-band mask extrapolation")

    mixture_mono_stft = np.mean(np.stack(mixture_stft, axis=0), axis=0)
    direct_sum = sum(direct_stft, start=np.zeros_like(direct_stft[0]))
    direct_energy = float(np.sum(np.abs(direct_sum[low_bins, :]) ** 2, dtype=np.float64))
    mixture_energy = float(np.sum(np.abs(mixture_mono_stft[low_bins, :]) ** 2, dtype=np.float64))
    direct_gain = float(
        np.clip(
            np.sqrt(mixture_energy / max(direct_energy, 1e-12)),
            0.1,
            10.0,
        )
    )
    direct_stft = [values * direct_gain for values in direct_stft]
    source_magnitude = np.stack(
        [np.abs(values) for values in direct_stft],
        axis=0,
    )
    frame_activity = np.sqrt(np.mean(source_magnitude[:, low_bins, :] ** 2, axis=1))
    peak_activity = np.max(frame_activity, axis=1, keepdims=True)
    relative_db = 20.0 * np.log10(
        np.maximum(frame_activity, 1e-10) / np.maximum(peak_activity, 1e-10)
    )
    activity_gate = np.clip((relative_db + 45.0) / 10.0, 0.0, 1.0)
    activity_gate = uniform_filter1d(
        activity_gate,
        size=5,
        axis=-1,
        mode="nearest",
    )

    divisors = (2, 3, 4, 5, 6, 7, 8)
    score = np.zeros_like(source_magnitude, dtype=np.float32)
    low_weight = np.ones((len(frequencies), 1), dtype=np.float32)
    low_weight[frequencies >= upper_hz] = 0.0
    transition = (frequencies > lower_hz) & (frequencies < upper_hz)
    if upper_hz > lower_hz:
        low_weight[transition] = (
            1.0 - (frequencies[transition] - lower_hz) / (upper_hz - lower_hz)
        )[:, None]

    for frequency_index, frequency in enumerate(frequencies):
        if frequency <= cutoff_hz:
            score[:, frequency_index, :] = source_magnitude[:, frequency_index, :]
            continue
        harmonic = np.zeros_like(frame_activity, dtype=np.float32)
        harmonic_weight = 0.0
        for divisor in divisors:
            anchor_frequency = frequency / divisor
            if not 250.0 <= anchor_frequency <= cutoff_hz:
                continue
            anchor_index = int(np.argmin(np.abs(frequencies - anchor_frequency)))
            start = max(0, anchor_index - 1)
            end = min(len(frequencies), anchor_index + 2)
            weight = 1.0 / np.sqrt(divisor)
            harmonic += np.mean(source_magnitude[:, start:end, :], axis=1) * weight
            harmonic_weight += weight
        if harmonic_weight > 0:
            harmonic /= harmonic_weight
        score[:, frequency_index, :] = (harmonic + 0.15 * frame_activity) * activity_gate

    score = uniform_filter1d(score, size=3, axis=1, mode="nearest")
    score = uniform_filter1d(score, size=3, axis=2, mode="nearest")
    powered = np.power(np.maximum(score, 1e-10), mask_power)
    residual_weight = residual_floor * np.mean(powered, axis=0)
    denominator = np.sum(powered, axis=0) + residual_weight + 1e-10
    masks = powered / denominator[None, :, :]

    sources: list[np.ndarray] = []
    for source_index, mask in enumerate(masks):
        channels = [
            _istft(
                low_weight * direct_stft[source_index] + (1.0 - low_weight) * mask * channel_stft,
                sample_rate,
                frame_length,
                overlap,
                len(mixture),
            )
            for channel_stft in mixture_stft
        ]
        sources.append(np.stack(channels, axis=1).astype(np.float32))
    source_sum = sum(sources, start=np.zeros_like(mixture))
    residual = np.asarray(mixture - source_sum, dtype=np.float32)
    return (
        sources,
        residual,
        {
            "lowband_cutoff_hz": cutoff_hz,
            "crossfade_hz": [lower_hz, upper_hz],
            "harmonic_divisors": list(divisors),
        },
    )


def project_original_spectrum(
    original: np.ndarray,
    proposals: list[np.ndarray],
    sample_rate: int,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
    mask_power: float = 1.5,
    residual_floor: float = 0.02,
    preserve_low_band_hz: float | None = None,
    low_band_proposals: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    mixture = np.asarray(original, dtype=np.float32)
    if mixture.ndim == 1:
        mixture = mixture[:, None]
    if mixture.ndim != 2:
        raise ValueError(f"expected original shape (samples, channels), got {mixture.shape}")
    if not proposals:
        return [], mixture.copy()
    if mask_power <= 0:
        raise ValueError("mask_power must be positive")
    if residual_floor < 0:
        raise ValueError("residual_floor cannot be negative")
    frame_length = min(max(16, n_fft), max(16, len(mixture)))
    frame_length = min(frame_length, len(mixture))
    if frame_length < 2:
        return [np.zeros_like(mixture) for _ in proposals], mixture.copy()
    hop = min(max(1, hop_length), frame_length - 1)
    overlap = frame_length - hop

    mixture_stft = [
        _stft(mixture[:, channel], sample_rate, frame_length, overlap)
        for channel in range(mixture.shape[1])
    ]
    mixture_magnitude = np.mean(
        np.stack([np.abs(values) for values in mixture_stft], axis=0),
        axis=0,
    )
    proposal_stft = [
        _stft(
            to_mono(fit_length(proposal, len(mixture))),
            sample_rate,
            frame_length,
            overlap,
        )
        for proposal in proposals
    ]
    direct_proposals = low_band_proposals or proposals
    if len(direct_proposals) != len(proposals):
        raise ValueError("low_band_proposals must match proposals")
    direct_stft = [
        _stft(
            to_mono(fit_length(proposal, len(mixture))),
            sample_rate,
            frame_length,
            overlap,
        )
        for proposal in direct_proposals
    ]
    low_band_blend = _low_band_blend(
        frame_length,
        sample_rate,
        preserve_low_band_hz,
    )
    if np.any(low_band_blend > 0):
        mixture_mono_stft = np.mean(np.stack(mixture_stft, axis=0), axis=0)
        proposal_sum = sum(
            direct_stft,
            start=np.zeros_like(direct_stft[0]),
        )
        active = low_band_blend[:, 0] > 0.5
        proposal_energy = float(
            np.sum(
                np.square(np.abs(proposal_sum[active, :])),
                dtype=np.float64,
            )
        )
        mixture_energy = float(
            np.sum(
                np.square(np.abs(mixture_mono_stft[active, :])),
                dtype=np.float64,
            )
        )
        gain = float(
            np.clip(
                np.sqrt(mixture_energy / max(proposal_energy, 1e-12)),
                0.1,
                10.0,
            )
        )
        direct_stft = [values * gain for values in direct_stft]
    weights = np.stack(
        [np.power(np.maximum(np.abs(values), 1e-10), mask_power) for values in proposal_stft],
        axis=0,
    )
    residual_weight = residual_floor * np.power(
        np.maximum(mixture_magnitude, 1e-10),
        mask_power,
    )
    denominator = np.sum(weights, axis=0) + residual_weight + 1e-10
    masks = weights / denominator[None, :, :]

    sources: list[np.ndarray] = []
    for source_index, mask in enumerate(masks):
        channels = [
            _istft(
                low_band_blend * direct_stft[source_index] + (1.0 - low_band_blend) * mask * values,
                sample_rate,
                frame_length,
                overlap,
                len(mixture),
            )
            for values in mixture_stft
        ]
        sources.append(np.stack(channels, axis=1).astype(np.float32))
    source_sum = sum(sources, start=np.zeros_like(mixture))
    residual = np.asarray(mixture - source_sum, dtype=np.float32)
    return sources, residual


def _gate_from_raw_activity(
    raw: np.ndarray,
    restored: np.ndarray,
    raw_sample_rate: int,
    restored_sample_rate: int,
) -> np.ndarray:
    frame = max(1, round(0.02 * raw_sample_rate))
    frame_count = int(np.ceil(len(raw) / frame))
    padded = np.pad(raw, (0, frame_count * frame - len(raw)))
    frame_rms = np.sqrt(np.mean(padded.reshape(frame_count, frame) ** 2, axis=1))
    frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-8))
    reference_db = float(np.quantile(frame_db, 0.99))
    if reference_db <= -120.0:
        return np.zeros_like(restored, dtype=np.float32)
    active_db = min(-40.0, reference_db - 20.0)
    silent_db = active_db - 10.0
    frame_gain = np.clip((frame_db - silent_db) / (active_db - silent_db), 0.0, 1.0)
    frame_gain = np.convolve(
        frame_gain,
        np.ones(5, dtype=np.float32) / 5.0,
        mode="same",
    )
    frame_times = (np.arange(frame_count) + 0.5) * frame / raw_sample_rate
    sample_times = np.arange(len(restored)) / restored_sample_rate
    gain = np.interp(
        sample_times,
        frame_times,
        frame_gain,
        left=0.0,
        right=0.0,
    ).astype(np.float32)
    return np.asarray(restored * gain[:, None], dtype=np.float32)


def _match_restored_source_level(
    raw: np.ndarray,
    restored: np.ndarray,
    raw_sample_rate: int,
    restored_sample_rate: int,
) -> tuple[np.ndarray, float]:
    """Keep neural bandwidth restoration at the separator's original program level."""

    reference = to_mono(raw)
    candidate = to_mono(resample_to(restored, restored_sample_rate, raw_sample_rate))
    candidate = to_mono(fit_length(candidate[:, None], len(reference)))
    reference_rms = float(np.sqrt(np.mean(np.square(reference), dtype=np.float64)))
    candidate_rms = float(np.sqrt(np.mean(np.square(candidate), dtype=np.float64)))
    if reference_rms <= 1e-8 or candidate_rms <= 1e-8:
        return np.asarray(restored, dtype=np.float32), 1.0
    gain = float(np.clip(reference_rms / candidate_rms, 0.01, 10.0))
    return np.asarray(restored * gain, dtype=np.float32), round(gain, 8)


def _match_original_program_level(
    sources: list[np.ndarray],
    mixture: np.ndarray,
) -> tuple[list[np.ndarray], float]:
    """Apply one shared gain so isolated sources remain within original program level."""

    if not sources:
        return [], 1.0
    mixture_values = np.asarray(mixture, dtype=np.float64)
    mixture_rms = float(np.sqrt(np.mean(np.square(mixture_values))))
    source_energy = float(
        np.sqrt(
            sum(
                float(np.mean(np.square(np.asarray(source, dtype=np.float64))))
                for source in sources
            )
        )
    )
    source_peak_bound = sum(
        float(np.max(np.abs(source))) if np.asarray(source).size else 0.0 for source in sources
    )
    energy_gain = mixture_rms / source_energy if source_energy > 1e-8 else 1.0
    peak_gain = 0.98 / source_peak_bound if source_peak_bound > 0.98 else 1.0
    gain = float(np.clip(min(1.0, energy_gain, peak_gain), 0.0, 1.0))
    return [np.asarray(source * gain, dtype=np.float32) for source in sources], round(gain, 8)


def _low_band_blend(
    frame_length: int,
    sample_rate: int,
    cutoff_hz: float | None,
) -> np.ndarray:
    bins = frame_length // 2 + 1
    if cutoff_hz is None or cutoff_hz <= 0:
        return np.zeros((bins, 1), dtype=np.float32)
    frequencies = np.fft.rfftfreq(frame_length, 1.0 / sample_rate)
    lower = max(0.0, cutoff_hz * 0.9)
    upper = min(sample_rate / 2.0, cutoff_hz)
    if upper <= lower:
        weights = frequencies <= upper
        return weights.astype(np.float32)[:, None]
    weights = np.ones_like(frequencies, dtype=np.float32)
    weights[frequencies >= upper] = 0.0
    transition = (frequencies > lower) & (frequencies < upper)
    weights[transition] = 1.0 - (frequencies[transition] - lower) / (upper - lower)
    return weights[:, None]


def _stft(
    audio: np.ndarray,
    sample_rate: int,
    frame_length: int,
    overlap: int,
) -> np.ndarray:
    _frequencies, _times, values = stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=frame_length,
        noverlap=overlap,
        nfft=frame_length,
        boundary="zeros",
        padded=True,
    )
    return np.asarray(values)


def _istft(
    values: np.ndarray,
    sample_rate: int,
    frame_length: int,
    overlap: int,
    sample_count: int,
) -> np.ndarray:
    _times, audio = istft(
        values,
        fs=sample_rate,
        window="hann",
        nperseg=frame_length,
        noverlap=overlap,
        nfft=frame_length,
        input_onesided=True,
        boundary=True,
    )
    output = np.asarray(audio, dtype=np.float32).reshape(-1, 1)
    return fit_length(output, sample_count)[:, 0]
