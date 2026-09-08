"""Deterministic, bounded-memory audio feature extraction."""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import lfilter, resample_poly

from .io import atomic_save_npy, iter_audio
from .models import AnalysisSettings, AudioMetadata, FeatureDescriptor, Timebase

ANALYZER_VERSION = "1.0"
_EPSILON = 1.0e-12
_DB_FLOOR = -120.0


def analyze_audio_features(
    path: Path,
    *,
    artifact_id: str,
    metadata: AudioMetadata,
    settings: AnalysisSettings,
    analysis_dir: Path,
) -> list[FeatureDescriptor]:
    """Analyze one file without retaining feature-length decoded audio."""
    sample_rate = metadata.sample_rate
    channels = metadata.channels
    frame_samples = max(64, round(sample_rate * settings.frame_hop_sec))
    n_fft = 1 << max(9, (frame_samples - 1).bit_length())
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    window = np.hanning(frame_samples).astype(np.float32)
    window_power = max(float(np.sum(np.square(window), dtype=np.float64)), _EPSILON)
    mel_filters = _mel_filterbank(settings.mel_bands, n_fft=n_fft, sample_rate=sample_rate)
    bark_filters = _bark_filterbank(settings.bark_bands, n_fft=n_fft, sample_rate=sample_rate)
    dct_basis = _dct_basis(settings.mfcc_count, settings.mel_bands)
    k_weighting = _KWeighting(sample_rate, channels)

    series: dict[str, list[float]] = {
        name: []
        for name in (
            "rms_dbfs",
            "peak_dbfs",
            "true_peak_dbtp",
            "crest_factor",
            "spectral_centroid_hz",
            "spectral_rolloff_hz",
            "spectral_flatness",
            "spectral_flux",
            "harmonicity",
            "stereo_balance_db",
            "mid_side_width",
            "stereo_correlation",
            "fold_down_risk",
        )
    }
    weighted_energies: list[float] = []
    mel_rows: list[np.ndarray] = []
    mfcc_rows: list[np.ndarray] = []
    bark_rows: list[np.ndarray] = []
    contrast_rows: list[np.ndarray] = []
    previous_spectrum: np.ndarray | None = None
    pending = np.empty((0, channels), dtype=np.float32)
    weighted_pending = np.empty((0, channels), dtype=np.float64)
    total_sum_squares = 0.0
    total_values = 0
    overall_peak = 0.0
    overall_true_peak = 0.0

    def consume(frame: np.ndarray, weighted: np.ndarray) -> None:
        nonlocal overall_true_peak, previous_spectrum
        mono = np.mean(frame, axis=1, dtype=np.float64)
        rms = math.sqrt(float(np.mean(np.square(frame), dtype=np.float64)))
        peak = float(np.max(np.abs(frame))) if frame.size else 0.0
        oversampled = resample_poly(frame, 4, 1, axis=0) if frame.size else frame
        true_peak = float(np.max(np.abs(oversampled))) if oversampled.size else 0.0
        overall_true_peak = max(overall_true_peak, true_peak)
        series["rms_dbfs"].append(_power_db(rms * rms))
        series["peak_dbfs"].append(_amplitude_db(peak))
        series["true_peak_dbtp"].append(_amplitude_db(true_peak))
        series["crest_factor"].append(peak / max(rms, math.sqrt(_EPSILON)))

        padded_window = window[: len(mono)]
        spectrum = np.abs(np.fft.rfft(mono * padded_window, n=n_fft))
        power = np.square(spectrum, dtype=np.float64) / window_power
        power_sum = float(np.sum(power))
        centroid = float(np.dot(frequencies, power) / max(power_sum, _EPSILON))
        cumulative = np.cumsum(power)
        target = settings.rolloff_fraction * power_sum
        rolloff_index = int(np.searchsorted(cumulative, target)) if power_sum > 0 else 0
        rolloff_index = min(rolloff_index, frequencies.size - 1)
        flatness = float(
            np.exp(np.mean(np.log(np.maximum(power, _EPSILON))))
            / max(float(np.mean(power)), _EPSILON)
        )
        normalized = spectrum / max(float(np.linalg.norm(spectrum)), math.sqrt(_EPSILON))
        flux = (
            0.0
            if previous_spectrum is None
            else float(np.linalg.norm(np.maximum(normalized - previous_spectrum, 0.0)))
        )
        previous_spectrum = normalized
        series["spectral_centroid_hz"].append(centroid)
        series["spectral_rolloff_hz"].append(float(frequencies[rolloff_index]))
        series["spectral_flatness"].append(flatness)
        series["spectral_flux"].append(flux)
        series["harmonicity"].append(_harmonicity(mono, sample_rate))

        mel_power = np.maximum(mel_filters @ power, _EPSILON)
        mel_db = 10.0 * np.log10(mel_power)
        mel_rows.append(mel_db.astype(np.float32))
        mfcc_rows.append((dct_basis @ mel_db).astype(np.float32))
        bark_power = np.maximum(bark_filters @ power, _EPSILON)
        bark_rows.append((10.0 * np.log10(bark_power)).astype(np.float32))
        contrast_rows.append(_spectral_contrast(power, frequencies))

        weighted_energies.append(float(np.sum(np.mean(np.square(weighted), axis=0))))
        balance, width, correlation, fold_down = _stereo_features(frame)
        series["stereo_balance_db"].append(balance)
        series["mid_side_width"].append(width)
        series["stereo_correlation"].append(correlation)
        series["fold_down_risk"].append(fold_down)

    for block in iter_audio(
        path,
        block_frames=settings.decode_block_frames,
        ffmpeg=settings.ffmpeg,
        metadata=metadata,
    ):
        values = np.nan_to_num(np.asarray(block, dtype=np.float32), copy=False)
        if values.shape[1] != channels:
            raise ValueError(f"decoded channel count changed from {channels} to {values.shape[1]}")
        total_sum_squares += float(np.sum(np.square(values), dtype=np.float64))
        total_values += values.size
        if values.size:
            overall_peak = max(overall_peak, float(np.max(np.abs(values))))
        weighted_values = k_weighting.process(values)
        pending = np.concatenate((pending, values), axis=0)
        weighted_pending = np.concatenate((weighted_pending, weighted_values), axis=0)
        while len(pending) >= frame_samples:
            consume(pending[:frame_samples], weighted_pending[:frame_samples])
            pending = pending[frame_samples:]
            weighted_pending = weighted_pending[frame_samples:]
    if len(pending):
        consume(pending, weighted_pending)

    count = len(weighted_energies)
    timebase = Timebase(
        start_sec=0.0,
        hop_sec=settings.frame_hop_sec,
        frame_sec=settings.frame_hop_sec,
        count=count,
    )
    momentary_energy = _rolling_mean(weighted_energies, max(1, round(0.4 / timebase.hop_sec)))
    short_energy = _rolling_mean(weighted_energies, max(1, round(3.0 / timebase.hop_sec)))
    series["momentary_loudness_lufs"] = [_lufs(value) for value in momentary_energy]
    series["short_term_loudness_lufs"] = [_lufs(value) for value in short_energy]

    features: list[FeatureDescriptor] = []
    overall_rms = math.sqrt(total_sum_squares / max(total_values, 1))
    integrated_loudness = _integrated_loudness(momentary_energy)
    scalar_values: dict[str, tuple[float, str, dict[str, Any]]] = {
        "rms_dbfs": (_power_db(overall_rms * overall_rms), "dBFS", {}),
        "peak_dbfs": (_amplitude_db(overall_peak), "dBFS", {}),
        "true_peak_dbtp": (
            _amplitude_db(overall_true_peak),
            "dBTP",
            {"method": "4x polyphase oversampling", "estimate": True},
        ),
        "crest_factor": (
            overall_peak / max(overall_rms, math.sqrt(_EPSILON)),
            "ratio",
            {},
        ),
        "integrated_loudness": (
            integrated_loudness,
            "LUFS",
            {
                "standard": "ITU-R BS.1770-4",
                "method": "K-weighted 400 ms blocks with absolute and relative gating",
                "estimate": channels > 2,
            },
        ),
        "loudness_range": (
            _loudness_range(short_energy, momentary_energy),
            "LU",
            {
                "standard": "EBU Tech 3342",
                "method": "10th-to-95th percentile of gated short-term loudness",
                "estimate": True,
            },
        ),
        "peak_to_loudness_ratio": (
            _amplitude_db(overall_true_peak) - integrated_loudness,
            "dB",
            {"true_peak_method": "4x polyphase oversampling"},
        ),
    }
    for name, (value, unit, feature_metadata) in scalar_values.items():
        features.append(
            FeatureDescriptor(
                id=f"{artifact_id}.{name}.scalar",
                artifact_id=artifact_id,
                name=name,
                kind="scalar",
                unit=unit,
                value=float(value),
                metadata=feature_metadata,
            )
        )

    units = {
        "rms_dbfs": "dBFS",
        "peak_dbfs": "dBFS",
        "true_peak_dbtp": "dBTP",
        "crest_factor": "ratio",
        "momentary_loudness_lufs": "LUFS",
        "short_term_loudness_lufs": "LUFS",
        "spectral_centroid_hz": "Hz",
        "spectral_rolloff_hz": "Hz",
        "spectral_flatness": "ratio",
        "spectral_flux": "ratio",
        "harmonicity": "ratio",
        "stereo_balance_db": "dB",
        "mid_side_width": "ratio",
        "stereo_correlation": "correlation",
        "fold_down_risk": "probability",
    }
    series_metadata: dict[str, dict[str, Any]] = {
        "momentary_loudness_lufs": {
            "standard": "ITU-R BS.1770-4",
            "window_sec": 0.4,
            "estimate": channels > 2,
        },
        "short_term_loudness_lufs": {
            "standard": "ITU-R BS.1770-4",
            "window_sec": 3.0,
            "estimate": channels > 2,
        },
    }
    for name, values in series.items():
        features.append(
            _dense_feature(
                analysis_dir,
                artifact_id=artifact_id,
                name=name,
                kind="series",
                values=np.asarray(values, dtype=np.float32),
                unit=units[name],
                timebase=timebase,
                metadata=series_metadata.get(name),
            )
        )

    matrices: dict[str, tuple[np.ndarray, str, dict[str, Any]]] = {
        "mel_band_power": (
            _rows(mel_rows, settings.mel_bands),
            "dB",
            {"scale": "HTK mel", "filters": "triangular"},
        ),
        "mfcc": (
            _rows(mfcc_rows, settings.mfcc_count),
            "coefficient",
            {"source": "log mel-band power", "transform": "orthonormal DCT-II"},
        ),
        "bark_band_power": (
            _rows(bark_rows, settings.bark_bands),
            "dB",
            {"scale": "Bark-like Traunmuller approximation", "filters": "triangular"},
        ),
        "spectral_contrast": (_rows(contrast_rows, 6), "dB", {"bands": 6}),
    }
    for name, (values, unit, matrix_metadata) in matrices.items():
        features.append(
            _dense_feature(
                analysis_dir,
                artifact_id=artifact_id,
                name=name,
                kind="matrix",
                values=values,
                unit=unit,
                timebase=timebase,
                metadata=matrix_metadata,
            )
        )

    mfcc_matrix = matrices["mfcc"][0]
    embedding = (
        np.stack((np.mean(mfcc_matrix, axis=0), np.std(mfcc_matrix, axis=0)))
        if len(mfcc_matrix)
        else np.empty((0, settings.mfcc_count), dtype=np.float32)
    )
    features.append(
        _dense_feature(
            analysis_dir,
            artifact_id=artifact_id,
            name="acoustic_embedding",
            kind="embeddings",
            values=embedding,
            unit="coefficient",
            metadata={"rows": ["mean", "standard_deviation"] if len(embedding) else []},
        )
    )

    boundaries = _event_boundaries(
        np.asarray(series["spectral_flux"]),
        np.asarray(series["rms_dbfs"]),
        timebase.hop_sec,
    )
    features.append(
        FeatureDescriptor(
            id=f"{artifact_id}.event_boundaries.events",
            artifact_id=artifact_id,
            name="event_boundaries",
            kind="events",
            unit="seconds",
            value=boundaries,
            timebase=timebase,
            metadata={"method": "robust spectral-flux and level-change detector"},
        )
    )
    return features


def _dense_feature(
    analysis_dir: Path,
    *,
    artifact_id: str,
    name: str,
    kind: str,
    values: np.ndarray,
    unit: str,
    timebase: Timebase | None = None,
    metadata: dict[str, Any] | None = None,
) -> FeatureDescriptor:
    relative = Path("arrays") / artifact_id / f"{name}.npy"
    array = np.asarray(values, dtype=np.float32)
    atomic_save_npy(analysis_dir / relative, array)
    return FeatureDescriptor(
        id=f"{artifact_id}.{name}.{kind}",
        artifact_id=artifact_id,
        name=name,
        kind=kind,  # type: ignore[arg-type]
        unit=unit,
        path=relative.as_posix(),
        dtype=str(array.dtype),
        shape=tuple(array.shape),
        timebase=timebase,
        metadata=metadata or {},
    )


def _power_db(power: float) -> float:
    return max(_DB_FLOOR, 10.0 * math.log10(max(power, _EPSILON)))


def _amplitude_db(value: float) -> float:
    return max(_DB_FLOOR, 20.0 * math.log10(max(value, math.sqrt(_EPSILON))))


def _lufs(energy: float) -> float:
    return max(_DB_FLOOR, -0.691 + 10.0 * math.log10(max(energy, _EPSILON)))


def _rolling_mean(values: list[float], width: int) -> list[float]:
    if not values:
        return []
    queue: deque[float] = deque()
    total = 0.0
    result: list[float] = []
    for value in values:
        queue.append(value)
        total += value
        if len(queue) > width:
            total -= queue.popleft()
        result.append(total / len(queue))
    return result


def _integrated_loudness(block_energies: list[float]) -> float:
    if not block_energies:
        return _DB_FLOOR
    energies = np.asarray(block_energies, dtype=np.float64)
    absolute = energies[np.asarray([_lufs(value) >= -70.0 for value in energies])]
    if absolute.size == 0:
        return _DB_FLOOR
    relative_gate = _lufs(float(np.mean(absolute))) - 10.0
    gated = absolute[np.asarray([_lufs(value) >= relative_gate for value in absolute])]
    return _lufs(float(np.mean(gated))) if gated.size else _DB_FLOOR


def _loudness_range(short_energies: list[float], momentary_energies: list[float]) -> float:
    if not short_energies:
        return 0.0
    integrated = _integrated_loudness(momentary_energies)
    loudness = np.asarray([_lufs(value) for value in short_energies])
    gated = loudness[(loudness >= -70.0) & (loudness >= integrated - 20.0)]
    if gated.size < 2:
        return 0.0
    return float(np.percentile(gated, 95.0) - np.percentile(gated, 10.0))


def _mel_filterbank(bands: int, *, n_fft: int, sample_rate: int) -> np.ndarray:
    frequencies = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    mel_min = _hz_to_mel(20.0)
    mel_max = _hz_to_mel(sample_rate / 2.0)
    points = _mel_to_hz(np.linspace(mel_min, mel_max, bands + 2))
    filters = np.zeros((bands, frequencies.size), dtype=np.float64)
    for index in range(bands):
        left, center, right = points[index : index + 3]
        filters[index] = np.minimum(
            (frequencies - left) / max(center - left, _EPSILON),
            (right - frequencies) / max(right - center, _EPSILON),
        )
        filters[index] = np.maximum(filters[index], 0.0)
        filters[index] /= max(float(np.sum(filters[index])), _EPSILON)
    return filters


def _hz_to_mel(value: float | np.ndarray) -> float | np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(value) / 700.0)


def _mel_to_hz(value: np.ndarray) -> np.ndarray:
    return 700.0 * (np.power(10.0, value / 2595.0) - 1.0)


def _bark_filterbank(bands: int, *, n_fft: int, sample_rate: int) -> np.ndarray:
    frequencies = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    bark = 13.0 * np.arctan(0.00076 * frequencies) + 3.5 * np.arctan(
        np.square(frequencies / 7500.0)
    )
    points = np.linspace(0.0, float(bark[-1]), bands + 2)
    filters = np.zeros((bands, frequencies.size), dtype=np.float64)
    for index in range(bands):
        left, center, right = points[index : index + 3]
        filters[index] = np.maximum(
            0.0,
            np.minimum(
                (bark - left) / max(center - left, _EPSILON),
                (right - bark) / max(right - center, _EPSILON),
            ),
        )
        filters[index] /= max(float(np.sum(filters[index])), _EPSILON)
    return filters


def _dct_basis(coefficients: int, bands: int) -> np.ndarray:
    rows = np.arange(coefficients, dtype=np.float64)[:, None]
    columns = np.arange(bands, dtype=np.float64)[None, :]
    basis = np.cos(math.pi / bands * (columns + 0.5) * rows)
    basis[0] *= math.sqrt(1.0 / bands)
    if coefficients > 1:
        basis[1:] *= math.sqrt(2.0 / bands)
    return basis


def _spectral_contrast(power: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    edges = np.geomspace(100.0, max(101.0, frequencies[-1]), 7)
    result = np.zeros(6, dtype=np.float32)
    for index in range(6):
        values = power[(frequencies >= edges[index]) & (frequencies < edges[index + 1])]
        if values.size:
            high = float(np.percentile(values, 90.0))
            low = float(np.percentile(values, 10.0))
            result[index] = 10.0 * math.log10((high + _EPSILON) / (low + _EPSILON))
    return result


def _harmonicity(mono: np.ndarray, sample_rate: int) -> float:
    centered = np.asarray(mono, dtype=np.float64) - float(np.mean(mono))
    energy = float(np.dot(centered, centered))
    if energy <= _EPSILON or centered.size < 4:
        return 0.0
    size = 1 << (2 * centered.size - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=size)
    autocorrelation = np.fft.irfft(np.square(np.abs(spectrum)), n=size)[: centered.size]
    first = max(1, sample_rate // 1_000)
    last = min(centered.size, max(first + 1, sample_rate // 50))
    if last <= first:
        return 0.0
    return float(np.clip(np.max(autocorrelation[first:last]) / autocorrelation[0], 0.0, 1.0))


def _stereo_features(frame: np.ndarray) -> tuple[float, float, float, float]:
    if frame.shape[1] < 2:
        return 0.0, 0.0, 1.0, 0.0
    left = np.asarray(frame[:, 0], dtype=np.float64)
    right = np.asarray(frame[:, 1], dtype=np.float64)
    left_energy = float(np.mean(np.square(left)))
    right_energy = float(np.mean(np.square(right)))
    balance = 10.0 * math.log10(max(left_energy, _EPSILON) / max(right_energy, _EPSILON))
    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    mid_energy = float(np.mean(np.square(mid)))
    side_energy = float(np.mean(np.square(side)))
    width = math.sqrt(side_energy / max(mid_energy, _EPSILON))
    denominator = math.sqrt(left_energy * right_energy)
    correlation = float(np.mean(left * right)) / denominator if denominator > _EPSILON else 1.0
    stereo_energy = 0.5 * (left_energy + right_energy)
    fold_down = 1.0 - math.sqrt(mid_energy / max(stereo_energy, _EPSILON))
    return (
        float(np.clip(balance, -120.0, 120.0)),
        float(min(width, 1.0e6)),
        float(np.clip(correlation, -1.0, 1.0)),
        float(np.clip(fold_down, 0.0, 1.0)),
    )


def _event_boundaries(flux: np.ndarray, level: np.ndarray, hop_sec: float) -> list[dict[str, Any]]:
    if flux.size < 2:
        return []
    flux_median = float(np.median(flux))
    flux_mad = float(np.median(np.abs(flux - flux_median)))
    level_delta = np.abs(np.diff(level, prepend=level[0]))
    delta_median = float(np.median(level_delta))
    delta_mad = float(np.median(np.abs(level_delta - delta_median)))
    flux_limit = flux_median + 4.0 * max(flux_mad, 1.0e-4)
    delta_limit = delta_median + 4.0 * max(delta_mad, 0.5)
    indices = np.flatnonzero((flux > flux_limit) | (level_delta > delta_limit))
    events: list[dict[str, Any]] = []
    previous = -10
    for index in indices:
        if index - previous <= 1:
            continue
        events.append(
            {
                "time_sec": round(float(index * hop_sec), 6),
                "strength": round(
                    float(
                        max(
                            flux[index] / max(flux_limit, _EPSILON),
                            level_delta[index] / max(delta_limit, _EPSILON),
                        )
                    ),
                    6,
                ),
            }
        )
        previous = int(index)
    return events


def _rows(rows: list[np.ndarray], columns: int) -> np.ndarray:
    if not rows:
        return np.empty((0, columns), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32).reshape(-1, columns)


class _KWeighting:
    """Stateful BS.1770 pre-filter and high-shelf cascade."""

    def __init__(self, sample_rate: int, channels: int) -> None:
        self._coefficients = (
            _high_shelf_coefficients(sample_rate),
            _high_pass_coefficients(sample_rate),
        )
        self._states = [np.zeros((2, channels), dtype=np.float64) for _ in range(2)]

    def process(self, values: np.ndarray) -> np.ndarray:
        output = np.asarray(values, dtype=np.float64)
        for index, (numerator, denominator) in enumerate(self._coefficients):
            output, self._states[index] = lfilter(
                numerator,
                denominator,
                output,
                axis=0,
                zi=self._states[index],
            )
        return output


def _high_shelf_coefficients(sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    frequency = 1681.974450955533
    gain_db = 3.999843853973347
    quality = 0.7071752369554196
    k = math.tan(math.pi * frequency / sample_rate)
    vh = 10.0 ** (gain_db / 20.0)
    vb = vh**0.4996667741545416
    denominator = 1.0 + k / quality + k * k
    numerator = np.asarray(
        [
            (vh + vb * k / quality + k * k) / denominator,
            2.0 * (k * k - vh) / denominator,
            (vh - vb * k / quality + k * k) / denominator,
        ]
    )
    recursive = np.asarray(
        [
            1.0,
            2.0 * (k * k - 1.0) / denominator,
            (1.0 - k / quality + k * k) / denominator,
        ]
    )
    return numerator, recursive


def _high_pass_coefficients(sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    frequency = 38.13547087602444
    quality = 0.5003270373238773
    k = math.tan(math.pi * frequency / sample_rate)
    denominator = 1.0 + k / quality + k * k
    return (
        np.asarray([1.0, -2.0, 1.0]) / denominator,
        np.asarray(
            [
                1.0,
                2.0 * (k * k - 1.0) / denominator,
                (1.0 - k / quality + k * k) / denominator,
            ]
        ),
    )
