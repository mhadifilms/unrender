from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import ndimage, signal

from unrender.audio.audio_io import ensure_2d, fit_length, match_channels, resample_to

from ._core import (
    EPSILON,
    LoadedRole,
    RecipeProducts,
    align_to,
    combine_audio,
    db_to_gain,
    dry_wet_mix,
    extract_events,
    find_role,
    gain_to_db,
    peak_protect,
)
from .contracts import ParameterSpec, RecipeContract, RecipeSettings, RoleInputSpec

DIALOGUE_AWARE_DUCKING = RecipeContract(
    name="dialogue_aware_ducking",
    category="corrective",
    description="Smoothly duck music and effects under detected dialogue activity.",
    role_inputs=(
        RoleInputSpec("dx", description="Dialogue stem", aliases=("d",)),
        RoleInputSpec("mx", required=False, description="Music stem", aliases=("m",)),
        RoleInputSpec("fx", required=False, description="Effects stem", aliases=("e",)),
    ),
    parameters={
        "threshold_dbfs": ParameterSpec(-42.0, -80.0, -6.0),
        "depth_db": ParameterSpec(10.0, 0.0, 30.0),
        "attack_ms": ParameterSpec(30.0, 1.0, 500.0),
        "hold_ms": ParameterSpec(80.0, 0.0, 2000.0),
        "release_ms": ParameterSpec(250.0, 5.0, 5000.0),
        "activity_range_db": ParameterSpec(24.0, 3.0, 60.0),
    },
)

SPECTRAL_UNMASKING = RecipeContract(
    name="spectral_unmasking",
    category="corrective",
    description="Apply a bounded, smoothed STFT mask to background energy overlapping dialogue.",
    role_inputs=(
        RoleInputSpec("dx", description="Dialogue stem", aliases=("d",)),
        RoleInputSpec(
            "background",
            required=False,
            description="Background stem",
            aliases=("mx", "fx", "m", "e"),
        ),
    ),
    parameters={
        "max_reduction_db": ParameterSpec(8.0, 0.0, 18.0),
        "mask_power": ParameterSpec(1.0, 0.25, 4.0),
        "frequency_smoothing_bins": ParameterSpec(2.0, 0.0, 12.0),
        "time_smoothing_frames": ParameterSpec(2.0, 0.0, 12.0),
        "fft_size": ParameterSpec(1024, 64, 8192),
    },
)

DME_REBALANCE = RecipeContract(
    name="dme_rebalance",
    category="corrective",
    description="Rebalance dialogue, music, and effects with width and protected loudness control.",
    role_inputs=(
        RoleInputSpec("d", required=False, aliases=("dx",)),
        RoleInputSpec("m", required=False, aliases=("mx",)),
        RoleInputSpec("e", required=False, aliases=("fx",)),
    ),
    parameters={
        "dialogue_gain_db": ParameterSpec(0.0, -24.0, 12.0),
        "music_gain_db": ParameterSpec(-2.0, -24.0, 12.0),
        "effects_gain_db": ParameterSpec(-2.0, -24.0, 12.0),
        "stereo_width": ParameterSpec(1.0, 0.0, 2.0),
        "target_loudness_lufs": ParameterSpec(-18.0, -36.0, -8.0),
        "max_master_gain_db": ParameterSpec(12.0, 0.0, 24.0),
    },
)

PERSPECTIVE_MATCHING = RecipeContract(
    name="perspective_matching",
    category="corrective",
    description="Conservatively match source level, spectrum, and width to a reference.",
    role_inputs=(
        RoleInputSpec("source", required=False, aliases=("target", "dx")),
        RoleInputSpec("reference", description="Perspective reference"),
        RoleInputSpec("room_tone", required=False),
    ),
    parameters={
        "max_level_adjust_db": ParameterSpec(4.0, 0.0, 12.0),
        "max_eq_db": ParameterSpec(3.0, 0.0, 9.0),
        "max_width_change": ParameterSpec(0.25, 0.0, 0.75),
        "room_tone_blend": ParameterSpec(0.0, 0.0, 0.25),
        "fft_size": ParameterSpec(1024, 64, 8192),
    },
    analysis_inputs=("reference confidence (optional)",),
)

REVIEW_REGION_VARIANTS = RecipeContract(
    name="review_region_variants",
    category="corrective",
    description="Create click-safe review extracts plus an untouched bypass-safe variant.",
    role_inputs=(RoleInputSpec("source", aliases=("target", "dx", "mix")),),
    parameters={
        "padding_seconds": ParameterSpec(0.25, 0.0, 5.0),
        "fade_ms": ParameterSpec(10.0, 0.0, 250.0),
    },
    analysis_inputs=("events or review_regions",),
    preserves_duration=False,
)


CORRECTIVE_CONTRACTS = {
    contract.name: contract
    for contract in (
        DIALOGUE_AWARE_DUCKING,
        SPECTRAL_UNMASKING,
        DME_REBALANCE,
        PERSPECTIVE_MATCHING,
        REVIEW_REGION_VARIANTS,
    )
}


def render_dialogue_aware_ducking(
    roles: Mapping[str, LoadedRole],
    settings: RecipeSettings,
    parameters: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> RecipeProducts:
    del analysis
    dx = find_role(roles, "dx", "d")
    assert dx is not None
    targets = [role for role in (find_role(roles, "mx", "m"), find_role(roles, "fx", "e")) if role]
    if not targets:
        raise ValueError("dialogue_aware_ducking requires at least one MX or FX stem")
    products = RecipeProducts()
    products.outputs["dx"] = (dx.audio.copy(), dx.sample_rate)
    min_gains: dict[str, float] = {}
    mix_tracks: list[tuple[np.ndarray, int]] = [(dx.audio, dx.sample_rate)]
    for target in targets:
        detector = resample_to(dx.audio, dx.sample_rate, target.sample_rate)
        detector = fit_length(detector, len(target.audio))
        activity, gain = _ducking_envelope(
            detector,
            target.sample_rate,
            threshold_dbfs=float(parameters["threshold_dbfs"]),
            activity_range_db=float(parameters["activity_range_db"]),
            depth_db=float(parameters["depth_db"]),
            attack_ms=float(parameters["attack_ms"]),
            hold_ms=float(parameters["hold_ms"]),
            release_ms=float(parameters["release_ms"]),
        )
        wet = target.audio * gain[:, None]
        rendered = dry_wet_mix(target.audio, wet, settings.dry_wet)
        products.outputs[target.role] = (rendered, target.sample_rate)
        products.automation[f"{target.role}_gain"] = gain.astype(np.float32)
        products.automation[f"{target.role}_dx_activity"] = activity.astype(np.float32)
        min_gains[target.role] = float(np.min(gain)) if gain.size else 1.0
        mix_tracks.append((rendered, target.sample_rate))
    mixed, mix_rate = combine_audio(mix_tracks, sample_rate=dx.sample_rate)
    mixed, limiter_gain = peak_protect(mixed, settings.peak_ceiling_dbfs)
    products.outputs["mix"] = (mixed, mix_rate)
    products.metrics["ducking"] = {
        "minimum_gain": min_gains,
        "minimum_gain_db": {role: float(gain_to_db(value)) for role, value in min_gains.items()},
        "mix_peak_protection_gain": limiter_gain,
    }
    products.report = {
        "detector_role": dx.role,
        "ducked_roles": [target.role for target in targets],
        "lookahead_samples": 0,
    }
    return products


def render_spectral_unmasking(
    roles: Mapping[str, LoadedRole],
    settings: RecipeSettings,
    parameters: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> RecipeProducts:
    del analysis
    dx = find_role(roles, "dx", "d")
    background = find_role(roles, "background", "mx", "fx", "m", "e")
    assert dx is not None
    if background is None:
        raise ValueError("spectral_unmasking requires a background, MX, or FX stem")
    detector = align_to(
        dx,
        background.sample_rate,
        len(background.audio),
        background.audio.shape[1],
    )
    fft_size = _usable_fft_size(int(parameters["fft_size"]), len(background.audio))
    _, _, dx_stft = _stft(np.mean(detector, axis=1), background.sample_rate, fft_size)
    _, _, bg_stft = _stft(
        np.mean(background.audio, axis=1),
        background.sample_rate,
        fft_size,
    )
    dx_magnitude = np.abs(dx_stft)
    bg_magnitude = np.abs(bg_stft)
    overlap = dx_magnitude / (dx_magnitude + bg_magnitude + EPSILON)
    overlap = ndimage.gaussian_filter(
        overlap,
        sigma=(
            float(parameters["frequency_smoothing_bins"]),
            float(parameters["time_smoothing_frames"]),
        ),
        mode="nearest",
    )
    reduction_db = float(parameters["max_reduction_db"]) * np.power(
        np.clip(overlap, 0.0, 1.0),
        float(parameters["mask_power"]),
    )
    gain_mask = np.asarray(db_to_gain(-reduction_db), dtype=np.float32)
    wet_channels = []
    for channel in range(background.audio.shape[1]):
        _, _, spectrum = _stft(
            background.audio[:, channel],
            background.sample_rate,
            fft_size,
        )
        wet_channels.append(
            _istft(spectrum * gain_mask, background.sample_rate, fft_size, len(background.audio))
        )
    wet = np.stack(wet_channels, axis=1)
    rendered = dry_wet_mix(background.audio, wet, settings.dry_wet)
    effective_mask = 1.0 - settings.dry_wet * (1.0 - gain_mask)
    before_overlap = float(np.sum(np.minimum(dx_magnitude, bg_magnitude) ** 2))
    after_overlap = float(np.sum(np.minimum(dx_magnitude, bg_magnitude * effective_mask) ** 2))
    reduction = 10.0 * math.log10((before_overlap + EPSILON) / (after_overlap + EPSILON))
    mixed, mix_rate = combine_audio(
        [(detector, background.sample_rate), (rendered, background.sample_rate)]
    )
    mixed, limiter_gain = peak_protect(mixed, settings.peak_ceiling_dbfs)
    products = RecipeProducts(
        outputs={
            "dx": (dx.audio.copy(), dx.sample_rate),
            background.role: (rendered, background.sample_rate),
            "mix": (mixed, mix_rate),
        },
        automation={
            "spectral_gain_mask": gain_mask,
            "overlap_mask": np.asarray(overlap, dtype=np.float32),
        },
        metrics={
            "masking": {
                "overlap_energy_before": before_overlap,
                "overlap_energy_after": after_overlap,
                "reduction_db": reduction,
                "minimum_gain_db": float(gain_to_db(np.min(gain_mask))),
                "mix_peak_protection_gain": limiter_gain,
            }
        },
        report={
            "processed_role": background.role,
            "fft_size": fft_size,
            "bounded_reduction_db": float(parameters["max_reduction_db"]),
        },
    )
    return products


def render_dme_rebalance(
    roles: Mapping[str, LoadedRole],
    settings: RecipeSettings,
    parameters: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> RecipeProducts:
    del analysis
    assignments = (
        ("d", find_role(roles, "d", "dx"), float(parameters["dialogue_gain_db"])),
        ("m", find_role(roles, "m", "mx"), float(parameters["music_gain_db"])),
        ("e", find_role(roles, "e", "fx"), float(parameters["effects_gain_db"])),
    )
    present = [(name, role, gain_db) for name, role, gain_db in assignments if role is not None]
    if not present:
        raise ValueError("dme_rebalance requires at least one D, M, or E stem")
    width = float(parameters["stereo_width"])
    products = RecipeProducts()
    mix_tracks: list[tuple[np.ndarray, int]] = []
    gain_report: dict[str, float] = {}
    for name, role, gain_db in present:
        assert role is not None
        widened = _stereo_width(role.audio, width)
        wet = widened * float(db_to_gain(gain_db))
        rendered = dry_wet_mix(role.audio, wet, settings.dry_wet)
        products.outputs[name] = (rendered, role.sample_rate)
        products.automation[f"{name}_gain"] = np.full(
            len(role.audio),
            (1.0 - settings.dry_wet) + settings.dry_wet * float(db_to_gain(gain_db)),
            dtype=np.float32,
        )
        mix_tracks.append((rendered, role.sample_rate))
        gain_report[name] = gain_db
    mix, mix_rate = combine_audio(mix_tracks)
    rms = float(np.sqrt(np.mean(np.square(mix), dtype=np.float64))) if mix.size else 0.0
    current_loudness = float(gain_to_db(rms) - 0.691)
    requested_gain_db = float(parameters["target_loudness_lufs"]) - current_loudness
    master_gain_db = float(
        np.clip(
            requested_gain_db,
            -float(parameters["max_master_gain_db"]),
            float(parameters["max_master_gain_db"]),
        )
    )
    master_gain = float(db_to_gain(master_gain_db)) if rms > EPSILON else 1.0
    mix *= master_gain
    mix, limiter_gain = peak_protect(mix, settings.peak_ceiling_dbfs)
    products.outputs["mix"] = (mix, mix_rate)
    products.automation["master_gain"] = np.full(
        len(mix),
        master_gain * limiter_gain,
        dtype=np.float32,
    )
    products.metrics["rebalance"] = {
        "stem_gain_db": gain_report,
        "master_gain_db": master_gain_db,
        "peak_protection_gain": limiter_gain,
        "target_loudness_lufs": float(parameters["target_loudness_lufs"]),
    }
    products.report = {
        "stereo_width": width,
        "master_gain_limited": not math.isclose(requested_gain_db, master_gain_db),
    }
    return products


def render_perspective_matching(
    roles: Mapping[str, LoadedRole],
    settings: RecipeSettings,
    parameters: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> RecipeProducts:
    source = find_role(roles, "source", "target", "dx")
    reference = find_role(roles, "reference")
    assert reference is not None
    if source is None:
        raise ValueError("perspective_matching requires a source or target role")
    reference_audio = resample_to(reference.audio, reference.sample_rate, source.sample_rate)
    fft_size = _usable_fft_size(
        int(parameters["fft_size"]),
        min(len(source.audio), len(reference_audio)),
    )
    source_mono = np.mean(source.audio, axis=1)
    reference_mono = np.mean(reference_audio, axis=1)
    source_rms = float(np.sqrt(np.mean(np.square(source_mono), dtype=np.float64)))
    reference_rms = float(np.sqrt(np.mean(np.square(reference_mono), dtype=np.float64)))
    raw_level_db = float(gain_to_db(reference_rms / max(source_rms, EPSILON)))
    level_db = float(
        np.clip(
            raw_level_db,
            -float(parameters["max_level_adjust_db"]),
            float(parameters["max_level_adjust_db"]),
        )
    )
    _, _, source_stft = _stft(source_mono, source.sample_rate, fft_size)
    _, _, reference_stft = _stft(reference_mono, source.sample_rate, fft_size)
    source_profile = np.mean(np.abs(source_stft), axis=1)
    reference_profile = np.mean(np.abs(reference_stft), axis=1)
    eq_db = np.asarray(
        gain_to_db((reference_profile + EPSILON) / (source_profile + EPSILON)),
        dtype=np.float64,
    )
    eq_db -= float(np.median(eq_db))
    eq_db = ndimage.gaussian_filter1d(eq_db, sigma=2.0, mode="nearest")
    eq_db = np.clip(eq_db, -float(parameters["max_eq_db"]), float(parameters["max_eq_db"]))
    eq_gain = np.asarray(db_to_gain(eq_db), dtype=np.float32)
    matched_channels = []
    for channel in range(source.audio.shape[1]):
        _, _, spectrum = _stft(source.audio[:, channel], source.sample_rate, fft_size)
        matched_channels.append(
            _istft(
                spectrum * eq_gain[:, None],
                source.sample_rate,
                fft_size,
                len(source.audio),
            )
        )
    wet = np.stack(matched_channels, axis=1) * float(db_to_gain(level_db))
    width_factor = _reference_width_factor(
        source.audio,
        reference_audio,
        max_change=float(parameters["max_width_change"]),
    )
    wet = _stereo_width(wet, width_factor)
    room_tone = find_role(roles, "room_tone")
    room_blend = float(parameters["room_tone_blend"])
    if room_tone is not None and room_blend > 0.0:
        room = resample_to(room_tone.audio, room_tone.sample_rate, source.sample_rate)
        room = _tile_to_length(match_channels(room, source.audio.shape[1]), len(source.audio))
        room_rms = float(np.sqrt(np.mean(np.square(room), dtype=np.float64)))
        wet_rms = float(np.sqrt(np.mean(np.square(wet), dtype=np.float64)))
        room = room * (wet_rms / max(room_rms, EPSILON))
        wet = wet * (1.0 - room_blend) + room * room_blend
    rendered = dry_wet_mix(source.audio, wet, settings.dry_wet)
    if settings.dry_wet > 0.0:
        rendered, limiter_gain = peak_protect(rendered, settings.peak_ceiling_dbfs)
    else:
        limiter_gain = 1.0
    reasons = []
    if len(reference.audio) / reference.sample_rate < 1.0:
        reasons.append("reference_shorter_than_one_second")
    if reference_rms < 1e-4:
        reasons.append("reference_near_silent")
    if source.audio.shape[1] != reference.audio.shape[1]:
        reasons.append("channel_layout_mismatch")
    if not math.isclose(raw_level_db, level_db):
        reasons.append("level_adjustment_bounded")
    analysis_confidence = analysis.get("perspective_confidence", 1.0)
    try:
        confidence = float(np.clip(float(analysis_confidence), 0.0, 1.0))
    except (TypeError, ValueError):
        confidence = 0.5
        reasons.append("invalid_analysis_confidence")
    uncertainty = float(np.clip(0.15 * len(reasons) + (1.0 - confidence) * 0.5, 0.0, 1.0))
    report = {
        "level_adjustment_db": level_db,
        "raw_level_difference_db": raw_level_db,
        "eq_range_db": [float(np.min(eq_db)), float(np.max(eq_db))],
        "stereo_width_factor": width_factor,
        "room_tone_blend": room_blend if room_tone is not None else 0.0,
        "peak_protection_gain": limiter_gain,
        "uncertainty": {"score": uncertainty, "reasons": reasons},
    }
    return RecipeProducts(
        outputs={"matched": (rendered, source.sample_rate)},
        automation={
            "eq_gain": eq_gain,
            "level_gain": np.full(len(source.audio), db_to_gain(level_db), dtype=np.float32),
            "width": np.full(len(source.audio), width_factor, dtype=np.float32),
        },
        report=report,
        metrics={"perspective": report},
    )


def render_review_region_variants(
    roles: Mapping[str, LoadedRole],
    settings: RecipeSettings,
    parameters: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> RecipeProducts:
    del settings
    source = find_role(roles, "source", "target", "dx", "mix")
    assert source is not None
    events = extract_events(analysis)
    padding = float(parameters["padding_seconds"])
    fade_samples = round(float(parameters["fade_ms"]) * source.sample_rate / 1000.0)
    products = RecipeProducts(outputs={"bypass": (source.audio.copy(), source.sample_rate)})
    regions: list[tuple[int, int]] = []
    for index, (start_seconds, end_seconds) in enumerate(events, start=1):
        start = max(0, math.floor((start_seconds - padding) * source.sample_rate))
        end = min(
            len(source.audio),
            math.ceil((end_seconds + padding) * source.sample_rate),
        )
        if end <= start:
            continue
        extract = source.audio[start:end].copy()
        _fade_edges(extract, fade_samples)
        products.outputs[f"review_region_{index:03d}"] = (extract, source.sample_rate)
        regions.append((start, end))
    products.automation["review_regions_samples"] = np.asarray(
        regions,
        dtype=np.int64,
    ).reshape(-1, 2)
    products.report = {
        "event_count": len(events),
        "extract_count": len(regions),
        "bypass_is_sample_identical": True,
        "padding_seconds": padding,
    }
    return products


CORRECTIVE_RENDERERS = {
    "dialogue_aware_ducking": render_dialogue_aware_ducking,
    "spectral_unmasking": render_spectral_unmasking,
    "dme_rebalance": render_dme_rebalance,
    "perspective_matching": render_perspective_matching,
    "review_region_variants": render_review_region_variants,
}


def _ducking_envelope(
    detector: np.ndarray,
    sample_rate: int,
    *,
    threshold_dbfs: float,
    activity_range_db: float,
    depth_db: float,
    attack_ms: float,
    hold_ms: float,
    release_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    mono = np.mean(ensure_2d(detector), axis=1)
    if mono.size == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    window = max(1, round(sample_rate * 0.02))
    hop = max(1, round(sample_rate * 0.005))
    power = ndimage.uniform_filter1d(np.square(mono), size=window, mode="nearest")
    frame_rms = np.sqrt(np.maximum(power[::hop], 0.0))
    frame_db = gain_to_db(frame_rms)
    frame_activity = np.clip((frame_db - threshold_dbfs) / activity_range_db, 0.0, 1.0)
    hold_frames = max(1, math.ceil(hold_ms * sample_rate / (1000.0 * hop)))
    if hold_frames > 1:
        frame_activity = ndimage.maximum_filter1d(
            frame_activity,
            size=hold_frames,
            origin=-(hold_frames // 2),
            mode="nearest",
        )
    desired = np.asarray(db_to_gain(-depth_db * frame_activity), dtype=np.float64)
    smoothed = np.empty_like(desired)
    smoothed[0] = desired[0]
    attack_coefficient = math.exp(-hop / max(attack_ms * sample_rate / 1000.0, 1.0))
    release_coefficient = math.exp(-hop / max(release_ms * sample_rate / 1000.0, 1.0))
    for index in range(1, len(desired)):
        coefficient = (
            attack_coefficient if desired[index] < smoothed[index - 1] else release_coefficient
        )
        smoothed[index] = coefficient * smoothed[index - 1] + (1.0 - coefficient) * desired[index]
    sample_positions = np.arange(len(mono), dtype=np.float64)
    frame_positions = np.minimum(np.arange(len(desired)) * hop, len(mono) - 1)
    gain = np.interp(sample_positions, frame_positions, smoothed)
    activity = np.interp(sample_positions, frame_positions, frame_activity)
    minimum = float(db_to_gain(-depth_db))
    return (
        np.asarray(np.clip(activity, 0.0, 1.0), dtype=np.float32),
        np.asarray(np.clip(gain, minimum, 1.0), dtype=np.float32),
    )


def _usable_fft_size(requested: int, sample_count: int) -> int:
    if sample_count <= 1:
        return 1
    usable = min(requested, sample_count)
    return max(2, 2 ** math.floor(math.log2(usable)))


def _stft(
    audio: np.ndarray,
    sample_rate: int,
    fft_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if fft_size <= 1:
        value = np.asarray(audio, dtype=np.float32).reshape(1, -1)
        return np.array([0.0]), np.arange(value.shape[1]), value.astype(np.complex64)
    return signal.stft(
        np.asarray(audio, dtype=np.float32),
        fs=sample_rate,
        window="hann",
        nperseg=fft_size,
        noverlap=fft_size * 3 // 4,
        nfft=fft_size,
        boundary="zeros",
        padded=True,
    )


def _istft(
    spectrum: np.ndarray,
    sample_rate: int,
    fft_size: int,
    sample_count: int,
) -> np.ndarray:
    if fft_size <= 1:
        return fit_length(np.real(spectrum).reshape(-1, 1), sample_count)[:, 0]
    _, audio = signal.istft(
        spectrum,
        fs=sample_rate,
        window="hann",
        nperseg=fft_size,
        noverlap=fft_size * 3 // 4,
        nfft=fft_size,
        input_onesided=True,
        boundary=True,
    )
    return fit_length(np.asarray(audio, dtype=np.float32)[:, None], sample_count)[:, 0]


def _stereo_width(audio: np.ndarray, width: float) -> np.ndarray:
    arr = ensure_2d(audio)
    if arr.shape[1] != 2:
        return arr.astype(np.float32, copy=True)
    mid = (arr[:, 0] + arr[:, 1]) * 0.5
    side = (arr[:, 0] - arr[:, 1]) * 0.5 * width
    return np.stack((mid + side, mid - side), axis=1).astype(np.float32)


def _reference_width_factor(
    source: np.ndarray,
    reference: np.ndarray,
    *,
    max_change: float,
) -> float:
    source_arr = ensure_2d(source)
    reference_arr = ensure_2d(reference)
    if source_arr.shape[1] != 2 or reference_arr.shape[1] != 2:
        return 1.0

    def ratio(audio: np.ndarray) -> float:
        mid = (audio[:, 0] + audio[:, 1]) * 0.5
        side = (audio[:, 0] - audio[:, 1]) * 0.5
        mid_rms = np.sqrt(np.mean(np.square(mid), dtype=np.float64))
        side_rms = np.sqrt(np.mean(np.square(side), dtype=np.float64))
        return float(side_rms / max(mid_rms, EPSILON))

    raw = ratio(reference_arr) / max(ratio(source_arr), EPSILON)
    return float(np.clip(raw, 1.0 - max_change, 1.0 + max_change))


def _tile_to_length(audio: np.ndarray, sample_count: int) -> np.ndarray:
    arr = ensure_2d(audio)
    if len(arr) == 0:
        return np.zeros((sample_count, arr.shape[1]), dtype=np.float32)
    repeats = math.ceil(sample_count / len(arr))
    return np.tile(arr, (repeats, 1))[:sample_count].astype(np.float32)


def _fade_edges(audio: np.ndarray, fade_samples: int) -> None:
    usable = min(fade_samples, len(audio) // 2)
    if usable <= 0:
        return
    phase = np.linspace(0.0, math.pi / 2.0, usable, endpoint=True)
    fade = np.sin(phase) ** 2
    audio[:usable] *= fade[:, None]
    audio[-usable:] *= fade[::-1, None]
