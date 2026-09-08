from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import ndimage

from ._core import (
    EPSILON,
    LoadedRole,
    RecipeProducts,
    align_to,
    combine_audio,
    dry_wet_mix,
    extract_events,
    find_role,
    numeric_analysis_series,
    peak_protect,
)
from .contracts import ParameterSpec, RecipeContract, RecipeSettings, RoleInputSpec
from .corrective import _istft, _stft, _usable_fft_size

TIMBRE_FILTER_MODULATION = RecipeContract(
    name="timbre_filter_modulation",
    category="creative",
    description="Drive a bounded spectral filter from the source's measured timbral centroid.",
    role_inputs=(RoleInputSpec("source", aliases=("target", "mx", "m")),),
    parameters={
        "base_cutoff_hz": ParameterSpec(6000.0, 100.0, 20000.0),
        "modulation_depth": ParameterSpec(0.55, 0.0, 1.0),
        "filter_order": ParameterSpec(4, 1, 12),
        "fft_size": ParameterSpec(1024, 64, 8192),
    },
    analysis_inputs=("timbre or spectral descriptors (optional)",),
)

GRANULAR_RESYNTHESIS = RecipeContract(
    name="granular_resynthesis",
    category="creative",
    description="Seeded overlap-add granular resynthesis guided by novelty and structure.",
    role_inputs=(RoleInputSpec("source", aliases=("target", "mx", "m")),),
    parameters={
        "grain_ms": ParameterSpec(80.0, 10.0, 500.0),
        "overlap": ParameterSpec(0.6, 0.1, 0.9),
        "jitter_ms": ParameterSpec(50.0, 0.0, 500.0),
        "reverse_probability": ParameterSpec(0.08, 0.0, 0.5),
    },
    analysis_inputs=("novelty", "structure"),
)

STEREO_SPATIAL_ANIMATION = RecipeContract(
    name="stereo_spatial_animation",
    category="creative",
    description="Apply bounded, seeded pan and width animation without changing layout.",
    role_inputs=(RoleInputSpec("source", aliases=("target", "mx", "m")),),
    parameters={
        "rate_hz": ParameterSpec(0.12, 0.01, 8.0),
        "pan_depth": ParameterSpec(0.45, 0.0, 1.0),
        "width_depth": ParameterSpec(0.35, 0.0, 1.0),
    },
)

EVENT_BEAT_MODULATION = RecipeContract(
    name="event_beat_modulation",
    category="creative",
    description=(
        "Trigger a click-safe modulation envelope from events or a deterministic beat grid."
    ),
    role_inputs=(RoleInputSpec("source", aliases=("target", "mx", "m")),),
    parameters={
        "depth": ParameterSpec(0.35, 0.0, 0.95),
        "attack_ms": ParameterSpec(5.0, 1.0, 100.0),
        "decay_ms": ParameterSpec(180.0, 5.0, 2000.0),
        "fallback_bpm": ParameterSpec(120.0, 20.0, 300.0),
    },
    analysis_inputs=("events", "beats"),
)

CROSS_STEM_MAPPING = RecipeContract(
    name="cross_stem_mapping",
    category="creative",
    description="Map a control stem envelope onto bounded target-stem texture enhancement.",
    role_inputs=(
        RoleInputSpec("control", aliases=("dx", "d")),
        RoleInputSpec("target", aliases=("mx", "m")),
    ),
    parameters={
        "amount": ParameterSpec(0.45, 0.0, 1.0),
        "envelope_ms": ParameterSpec(40.0, 5.0, 1000.0),
        "texture_cutoff_hz": ParameterSpec(1800.0, 100.0, 12000.0),
    },
    analysis_inputs=("cross-stem mapping hints (optional)",),
)


CREATIVE_CONTRACTS = {
    contract.name: contract
    for contract in (
        TIMBRE_FILTER_MODULATION,
        GRANULAR_RESYNTHESIS,
        STEREO_SPATIAL_ANIMATION,
        EVENT_BEAT_MODULATION,
        CROSS_STEM_MAPPING,
    )
}


def render_timbre_filter_modulation(
    roles: Mapping[str, LoadedRole],
    settings: RecipeSettings,
    parameters: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> RecipeProducts:
    source = find_role(roles, "source", "target", "mx", "m")
    assert source is not None
    fft_size = _usable_fft_size(int(parameters["fft_size"]), len(source.audio))
    frequencies, _, mono_stft = _stft(
        np.mean(source.audio, axis=1),
        source.sample_rate,
        fft_size,
    )
    magnitude = np.abs(mono_stft)
    centroid = np.sum(frequencies[:, None] * magnitude, axis=0) / (
        np.sum(magnitude, axis=0) + EPSILON
    )
    external_timbre = numeric_analysis_series(analysis, "timbre", "brightness", "spectral_centroid")
    if external_timbre is not None:
        source_axis = np.linspace(0.0, 1.0, len(external_timbre))
        target_axis = np.linspace(0.0, 1.0, len(centroid))
        external = np.interp(target_axis, source_axis, external_timbre)
        external -= np.min(external)
        external /= max(float(np.max(external)), EPSILON)
        centroid = 0.75 * centroid + 0.25 * external * (source.sample_rate / 2.0)
    normalized = np.clip(centroid / max(source.sample_rate / 2.0, 1.0), 0.0, 1.0)
    depth = float(parameters["modulation_depth"])
    cutoff = float(parameters["base_cutoff_hz"]) * (1.0 + depth * (2.0 * normalized - 1.0))
    cutoff = np.clip(cutoff, 40.0, source.sample_rate * 0.49)
    order = int(parameters["filter_order"])
    gain_mask = 1.0 / np.sqrt(
        1.0 + np.power(frequencies[:, None] / np.maximum(cutoff[None, :], 1.0), 2 * order)
    )
    wet_channels = []
    for channel in range(source.audio.shape[1]):
        _, _, spectrum = _stft(source.audio[:, channel], source.sample_rate, fft_size)
        wet_channels.append(
            _istft(
                spectrum * gain_mask,
                source.sample_rate,
                fft_size,
                len(source.audio),
            )
        )
    wet = np.stack(wet_channels, axis=1)
    rendered, limiter_gain = _creative_mix(source.audio, wet, settings)
    return RecipeProducts(
        outputs={"filtered": (rendered, source.sample_rate)},
        automation={
            "cutoff_hz": cutoff.astype(np.float32),
            "spectral_gain_mask": gain_mask.astype(np.float32),
        },
        report={
            "analysis_timbre_used": external_timbre is not None,
            "peak_protection_gain": limiter_gain,
            "latency_samples": 0,
            "tail_samples": 0,
        },
    )


def render_granular_resynthesis(
    roles: Mapping[str, LoadedRole],
    settings: RecipeSettings,
    parameters: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> RecipeProducts:
    source = find_role(roles, "source", "target", "mx", "m")
    assert source is not None
    audio = source.audio
    sample_count = len(audio)
    grain = max(
        2,
        min(
            sample_count,
            round(float(parameters["grain_ms"]) * source.sample_rate / 1000),
        ),
    )
    hop = max(1, round(grain * (1.0 - float(parameters["overlap"]))))
    jitter = round(float(parameters["jitter_ms"]) * source.sample_rate / 1000)
    rng = np.random.default_rng(settings.seed)
    novelty = numeric_analysis_series(analysis, "novelty", "novelty_curve", "onset_strength")
    structure = numeric_analysis_series(analysis, "structure", "structure_curve", "sections")
    novelty_normalized = _normalized_series(novelty)
    structure_normalized = _normalized_series(structure)
    wet = np.zeros_like(audio, dtype=np.float64)
    normalization = np.zeros(sample_count, dtype=np.float64)
    window = np.hanning(grain + 2)[1:-1]
    grain_map: list[tuple[int, int, int]] = []
    for output_start in range(0, sample_count, hop):
        usable = min(grain, sample_count - output_start)
        progress = output_start / max(sample_count - 1, 1)
        novelty_value = _series_value(novelty_normalized, progress, default=0.5)
        structure_value = _series_value(structure_normalized, progress, default=progress)
        guide = 0.8 * progress + 0.2 * structure_value
        center = round(guide * max(sample_count - grain, 0))
        local_jitter = round(jitter * (0.25 + 0.75 * novelty_value))
        offset = int(rng.integers(-local_jitter, local_jitter + 1)) if local_jitter else 0
        source_start = int(np.clip(center + offset, 0, max(sample_count - usable, 0)))
        chunk = audio[source_start : source_start + usable]
        reversed_grain = rng.random() < float(parameters["reverse_probability"]) * novelty_value
        if reversed_grain:
            chunk = chunk[::-1]
        grain_window = window[:usable]
        wet[output_start : output_start + usable] += chunk * grain_window[:, None]
        normalization[output_start : output_start + usable] += grain_window
        grain_map.append((output_start, source_start, -usable if reversed_grain else usable))
    valid = normalization > EPSILON
    wet[valid] /= normalization[valid, None]
    wet[~valid] = audio[~valid]
    rendered, limiter_gain = _creative_mix(audio, wet.astype(np.float32), settings)
    return RecipeProducts(
        outputs={"resynthesized": (rendered, source.sample_rate)},
        automation={"grain_map": np.asarray(grain_map, dtype=np.int64)},
        report={
            "seed": settings.seed,
            "novelty_used": novelty is not None,
            "structure_used": structure is not None,
            "grain_count": len(grain_map),
            "peak_protection_gain": limiter_gain,
            "latency_samples": 0,
            "tail_samples": 0,
        },
    )


def render_stereo_spatial_animation(
    roles: Mapping[str, LoadedRole],
    settings: RecipeSettings,
    parameters: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> RecipeProducts:
    del analysis
    source = find_role(roles, "source", "target", "mx", "m")
    assert source is not None
    rng = np.random.default_rng(settings.seed)
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    time = np.arange(len(source.audio), dtype=np.float64) / source.sample_rate
    oscillator = np.sin(2.0 * math.pi * float(parameters["rate_hz"]) * time + phase)
    pan = oscillator * float(parameters["pan_depth"])
    width = 1.0 + np.sin(
        2.0 * math.pi * float(parameters["rate_hz"]) * 0.5 * time + phase + math.pi / 2.0
    ) * float(parameters["width_depth"])
    wet = source.audio.astype(np.float64, copy=True)
    if source.audio.shape[1] >= 2:
        left = source.audio[:, 0]
        right = source.audio[:, 1]
        mid = (left + right) * 0.5
        side = (left - right) * 0.5 * width
        left_gain = np.sqrt(np.clip(1.0 - pan, 0.0, 2.0))
        right_gain = np.sqrt(np.clip(1.0 + pan, 0.0, 2.0))
        wet[:, 0] = (mid + side) * left_gain
        wet[:, 1] = (mid - side) * right_gain
    else:
        wet[:, 0] *= 1.0 - 0.1 * float(parameters["pan_depth"]) * (oscillator + 1.0)
    rendered, limiter_gain = _creative_mix(source.audio, wet.astype(np.float32), settings)
    return RecipeProducts(
        outputs={"animated": (rendered, source.sample_rate)},
        automation={"pan": pan.astype(np.float32), "width": width.astype(np.float32)},
        report={
            "seed": settings.seed,
            "initial_phase_radians": phase,
            "peak_protection_gain": limiter_gain,
            "latency_samples": 0,
            "tail_samples": 0,
        },
    )


def render_event_beat_modulation(
    roles: Mapping[str, LoadedRole],
    settings: RecipeSettings,
    parameters: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> RecipeProducts:
    source = find_role(roles, "source", "target", "mx", "m")
    assert source is not None
    events = extract_events(analysis)
    if events:
        event_samples = [
            round(start * source.sample_rate)
            for start, _ in events
            if start * source.sample_rate < len(source.audio)
        ]
        event_source = "analysis"
    else:
        interval = 60.0 * source.sample_rate / float(parameters["fallback_bpm"])
        event_samples = [round(value) for value in np.arange(0, len(source.audio), interval)]
        event_source = "fallback_beat_grid"
    decay_samples = max(
        1,
        round(float(parameters["decay_ms"]) * source.sample_rate / 1000.0),
    )
    attack_samples = max(
        2,
        round(float(parameters["attack_ms"]) * source.sample_rate / 1000.0),
    )
    envelope = np.zeros(len(source.audio), dtype=np.float64)
    rng = np.random.default_rng(settings.seed)
    for event_sample in event_samples:
        if not 0 <= event_sample < len(envelope):
            continue
        usable = min(attack_samples + decay_samples * 8, len(envelope) - event_sample)
        attack_length = min(attack_samples, usable)
        shape = np.empty(usable, dtype=np.float64)
        attack_phase = np.linspace(0.0, math.pi / 2.0, attack_length, endpoint=True)
        shape[:attack_length] = np.sin(attack_phase) ** 2
        if usable > attack_length:
            shape[attack_length:] = np.exp(
                -np.arange(usable - attack_length, dtype=np.float64) / decay_samples
            )
        strength = float(rng.uniform(0.85, 1.0))
        envelope[event_sample : event_sample + usable] = np.maximum(
            envelope[event_sample : event_sample + usable],
            strength * shape,
        )
    gain = 1.0 - float(parameters["depth"]) * envelope
    wet = source.audio * gain[:, None]
    rendered, limiter_gain = _creative_mix(source.audio, wet, settings)
    return RecipeProducts(
        outputs={"modulated": (rendered, source.sample_rate)},
        automation={
            "trigger_envelope": envelope.astype(np.float32),
            "gain": gain.astype(np.float32),
            "event_samples": np.asarray(event_samples, dtype=np.int64),
        },
        report={
            "seed": settings.seed,
            "event_source": event_source,
            "event_count": len(event_samples),
            "peak_protection_gain": limiter_gain,
            "latency_samples": 0,
            "tail_samples": 0,
        },
    )


def render_cross_stem_mapping(
    roles: Mapping[str, LoadedRole],
    settings: RecipeSettings,
    parameters: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> RecipeProducts:
    del analysis
    control = find_role(roles, "control", "dx", "d")
    target = find_role(roles, "target", "mx", "m")
    assert control is not None and target is not None
    mapped_control = align_to(
        control,
        target.sample_rate,
        len(target.audio),
        target.audio.shape[1],
    )
    mono_control = np.mean(mapped_control, axis=1)
    envelope_samples = max(
        1,
        round(float(parameters["envelope_ms"]) * target.sample_rate / 1000.0),
    )
    rms_envelope = np.sqrt(
        np.maximum(
            ndimage.uniform_filter1d(
                np.square(mono_control),
                size=envelope_samples,
                mode="nearest",
            ),
            0.0,
        )
    )
    low, high = np.percentile(rms_envelope, [10.0, 95.0])
    control_envelope = np.clip((rms_envelope - low) / max(high - low, EPSILON), 0.0, 1.0)
    cutoff = float(parameters["texture_cutoff_hz"])
    smoothing_samples = max(1.0, target.sample_rate / (2.0 * math.pi * cutoff))
    lowpass = ndimage.gaussian_filter1d(
        target.audio,
        sigma=smoothing_samples,
        axis=0,
        mode="nearest",
    )
    texture = target.audio - lowpass
    texture_gain = float(parameters["amount"]) * control_envelope
    wet = target.audio + texture * texture_gain[:, None]
    rendered, limiter_gain = _creative_mix(target.audio, wet, settings)
    mix, mix_rate = combine_audio(
        [(mapped_control, target.sample_rate), (rendered, target.sample_rate)]
    )
    mix, mix_limiter_gain = peak_protect(mix, settings.peak_ceiling_dbfs)
    return RecipeProducts(
        outputs={
            target.role: (rendered, target.sample_rate),
            "mix": (mix, mix_rate),
        },
        automation={
            "control_envelope": control_envelope.astype(np.float32),
            "texture_gain": texture_gain.astype(np.float32),
        },
        report={
            "control_role": control.role,
            "target_role": target.role,
            "peak_protection_gain": limiter_gain,
            "mix_peak_protection_gain": mix_limiter_gain,
            "latency_samples": 0,
            "tail_samples": 0,
        },
    )


CREATIVE_RENDERERS = {
    "timbre_filter_modulation": render_timbre_filter_modulation,
    "granular_resynthesis": render_granular_resynthesis,
    "stereo_spatial_animation": render_stereo_spatial_animation,
    "event_beat_modulation": render_event_beat_modulation,
    "cross_stem_mapping": render_cross_stem_mapping,
}


def _creative_mix(
    dry: np.ndarray,
    wet: np.ndarray,
    settings: RecipeSettings,
) -> tuple[np.ndarray, float]:
    rendered = dry_wet_mix(dry, wet, settings.dry_wet)
    if settings.dry_wet <= 0.0:
        return rendered, 1.0
    return peak_protect(rendered, settings.peak_ceiling_dbfs)


def _normalized_series(series: np.ndarray | None) -> np.ndarray | None:
    if series is None:
        return None
    minimum = float(np.min(series))
    span = float(np.max(series) - minimum)
    if span <= EPSILON:
        return np.full_like(series, 0.5)
    return np.asarray((series - minimum) / span, dtype=np.float32)


def _series_value(series: np.ndarray | None, progress: float, *, default: float) -> float:
    if series is None:
        return default
    return float(
        np.interp(
            progress,
            np.linspace(0.0, 1.0, len(series)),
            series,
        )
    )
