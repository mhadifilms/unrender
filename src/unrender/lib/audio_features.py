"""Dependency-light DSP helpers shared across the M&E (``mne``) stages.

Everything here operates on numpy arrays read via :mod:`unrender.lib.audio`
(plain PCM WAV, no soundfile), so the module imports and runs without any of
the optional heavy backends (transformers/torch/soundfile). The spectral
fingerprints, activity gating, equal-power masks, and granular loop synthesis
are reused by FX classification, music cue detection, and room-tone harvesting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unrender.lib.audio import (
    frame_rms_db as _shared_frame_rms_db,
)
from unrender.lib.audio import to_mono as _shared_to_mono


@dataclass(frozen=True)
class Segment:
    """A half-open ``[start_sec, end_sec)`` region on a stem's own timeline."""

    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


def to_mono(audio: np.ndarray) -> np.ndarray:
    return _shared_to_mono(audio)


def to_channels(mono: np.ndarray, channels: int) -> np.ndarray:
    """Broadcast a mono buffer to an ``(N, channels)`` interleaved buffer."""
    flat = np.asarray(mono, dtype=np.float32).reshape(-1)
    count = max(1, int(channels))
    return np.repeat(flat[:, None], count, axis=1)


def resample_mono(mono: np.ndarray, *, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a mono buffer to ``target_sr`` with a polyphase filter.

    Models such as CLAP expect a fixed input rate (48 kHz), while separated
    stems are usually 44.1/48 kHz; resampling here keeps the classification
    frontend correct regardless of the stem's native rate.
    """
    flat = np.asarray(mono, dtype=np.float32).reshape(-1)
    if orig_sr <= 0 or target_sr <= 0 or orig_sr == target_sr or flat.size == 0:
        return flat
    from math import gcd

    from scipy.signal import resample_poly

    divisor = gcd(int(orig_sr), int(target_sr))
    up = int(target_sr) // divisor
    down = int(orig_sr) // divisor
    return np.asarray(resample_poly(flat, up, down), dtype=np.float32)


def split_windows(clip: np.ndarray, *, sample_rate: int, window_sec: float) -> list[np.ndarray]:
    """Chop a clip into ``window_sec`` windows for chunked classification.

    A single event can run for minutes, but a tagger only sees a bounded
    context, so long events are windowed and their per-window scores averaged
    rather than classifying (and silently truncating) the whole event once.
    """
    flat = np.asarray(clip, dtype=np.float32).reshape(-1)
    span = max(1, round(window_sec * sample_rate))
    if flat.size <= span:
        return [flat]
    windows: list[np.ndarray] = []
    for offset in range(0, flat.size, span):
        window = flat[offset : offset + span]
        # Keep the trailing remainder only when it carries enough context.
        if window.size >= span // 2:
            windows.append(window)
    return windows or [flat]


def frame_rms_db(
    mono: np.ndarray, *, sample_rate: int, hop_sec: float = 0.02
) -> tuple[np.ndarray, float]:
    return _shared_frame_rms_db(
        mono,
        sample_rate=sample_rate,
        hop_sec=hop_sec,
    )


def activity_segments(
    mono: np.ndarray,
    *,
    sample_rate: int,
    threshold_db: float,
    hop_sec: float = 0.02,
    merge_gap_sec: float = 0.0,
    min_duration_sec: float = 0.0,
) -> list[Segment]:
    """Contiguous regions whose per-hop RMS stays above ``threshold_db``.

    Adjacent regions separated by less than ``merge_gap_sec`` are merged and
    regions shorter than ``min_duration_sec`` are dropped, mirroring the
    activity gate used for dialogue islands.
    """
    levels, hop_dur = frame_rms_db(mono, sample_rate=sample_rate, hop_sec=hop_sec)
    if levels.size == 0:
        return []
    active = levels > threshold_db
    raw: list[Segment] = []
    start_index: int | None = None
    for index, flag in enumerate(active):
        if flag and start_index is None:
            start_index = index
        elif not flag and start_index is not None:
            raw.append(Segment(start_index * hop_dur, index * hop_dur))
            start_index = None
    if start_index is not None:
        raw.append(Segment(start_index * hop_dur, active.size * hop_dur))
    return merge_segments(raw, merge_gap_sec=merge_gap_sec, min_duration_sec=min_duration_sec)


def gap_segments(
    islands: list[Segment], *, total_sec: float, min_gap_sec: float = 0.0
) -> list[Segment]:
    """Invert active islands into the silent gaps between and around them."""
    gaps: list[Segment] = []
    cursor = 0.0
    for island in sorted(islands, key=lambda seg: seg.start_sec):
        if island.start_sec - cursor >= min_gap_sec and island.start_sec > cursor:
            gaps.append(Segment(cursor, island.start_sec))
        cursor = max(cursor, island.end_sec)
    if total_sec - cursor >= min_gap_sec and total_sec > cursor:
        gaps.append(Segment(cursor, total_sec))
    return [gap for gap in gaps if gap.duration_sec >= min_gap_sec]


def merge_segments(
    segments: list[Segment], *, merge_gap_sec: float = 0.0, min_duration_sec: float = 0.0
) -> list[Segment]:
    ordered = sorted(segments, key=lambda seg: (seg.start_sec, seg.end_sec))
    merged: list[Segment] = []
    for segment in ordered:
        if merged and segment.start_sec - merged[-1].end_sec <= merge_gap_sec:
            merged[-1] = Segment(merged[-1].start_sec, max(merged[-1].end_sec, segment.end_sec))
        else:
            merged.append(segment)
    return [segment for segment in merged if segment.duration_sec >= min_duration_sec]


def inset_gaps(gaps: list[Segment], *, handle: float, min_gap_sec: float) -> list[Segment]:
    """Shrink each gap by ``handle`` seconds per side, dropping any that collapse.

    Insetting keeps word tails, breaths, and reverb decay out of a harvested
    noise plate; gaps that fall below ``min_gap_sec`` after the trim carry too
    little usable tone and are dropped.
    """
    if handle <= 0:
        return gaps
    inset: list[Segment] = []
    for gap in gaps:
        start = gap.start_sec + handle
        end = gap.end_sec - handle
        if end - start >= max(0.0, min_gap_sec):
            inset.append(Segment(start, end))
    return inset


def as_float(value: float | str | None, default: float) -> float:
    """Coerce a CLI/config value to ``float``, falling back on empty/None.

    Shared by the ``settings_from_values`` factories so the M&E subcommands
    parse their numeric flags identically.
    """
    if value is None or value == "":
        return default
    return float(value)


def equal_power_ramp(length: int) -> np.ndarray:
    """A ``0 -> 1`` equal-power (sine-law) fade-in of ``length`` samples."""
    if length <= 0:
        return np.empty(0, dtype=np.float32)
    theta = np.linspace(0.0, np.pi / 2.0, length, endpoint=False, dtype=np.float32)
    return np.sin(theta).astype(np.float32)


def equal_power_mask(
    sample_count: int,
    *,
    sample_rate: int,
    segments: list[Segment],
    crossfade_ms: float,
) -> np.ndarray:
    """A ``[0, 1]`` gain envelope that is 1 inside ``segments``.

    Segment edges use equal-power (sine-law) crossfades so a stem split into
    complementary category tracks sums back to roughly constant power, matching
    the click-free muting used for diarized speaker stems.
    """
    mask = np.zeros(sample_count, dtype=np.float32)
    fade = max(0, round(sample_rate * crossfade_ms / 1000.0))
    for segment in segments:
        start = min(sample_count, max(0, round(segment.start_sec * sample_rate)))
        end = min(sample_count, max(start, round(segment.end_sec * sample_rate)))
        if end <= start:
            continue
        window = np.ones(end - start, dtype=np.float32)
        local_fade = min(fade, (end - start) // 2)
        if local_fade > 0:
            ramp = equal_power_ramp(local_fade)
            window[:local_fade] = ramp
            window[-local_fade:] = ramp[::-1]
        mask[start:end] = np.maximum(mask[start:end], window)
    return mask


def _spectrogram(
    mono: np.ndarray, *, sample_rate: int, window_sec: float, hop_sec: float
) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude STFT ``(frames, bins)`` with the center time of each frame."""
    win = max(16, round(sample_rate * window_sec))
    hop = max(1, round(sample_rate * hop_sec))
    if mono.size < win:
        mono = np.pad(mono, (0, win - mono.size))
    n_frames = 1 + (mono.size - win) // hop
    window = np.hanning(win).astype(np.float32)
    frames = np.empty((n_frames, win // 2 + 1), dtype=np.float32)
    times = np.empty(n_frames, dtype=np.float64)
    for index in range(n_frames):
        offset = index * hop
        chunk = mono[offset : offset + win] * window
        frames[index] = np.abs(np.fft.rfft(chunk)).astype(np.float32)
        times[index] = (offset + win / 2.0) / sample_rate
    return frames, times


def _log_mel_bands(magnitudes: np.ndarray, *, n_bands: int) -> np.ndarray:
    """Fold linear FFT magnitudes into ``n_bands`` log-spaced band energies."""
    bins = magnitudes.shape[-1]
    edges = np.unique(np.round(np.logspace(0, np.log10(bins), n_bands + 1)).astype(int))
    edges = np.clip(edges, 1, bins)
    if edges.size < 2:
        edges = np.asarray([1, bins])
    out = np.zeros((magnitudes.shape[0], edges.size - 1), dtype=np.float32)
    for index in range(edges.size - 1):
        lo, hi = edges[index], max(edges[index] + 1, edges[index + 1])
        out[:, index] = np.mean(np.square(magnitudes[:, lo:hi]), axis=1)
    return np.log1p(out)


def log_mel_frames(
    mono: np.ndarray,
    *,
    sample_rate: int,
    window_sec: float = 0.05,
    hop_sec: float = 0.025,
    n_bands: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame log-mel band energies plus each frame's center time."""
    magnitudes, times = _spectrogram(
        mono, sample_rate=sample_rate, window_sec=window_sec, hop_sec=hop_sec
    )
    return _log_mel_bands(magnitudes, n_bands=n_bands), times


def spectral_fingerprint(mono: np.ndarray, *, sample_rate: int, n_bands: int = 16) -> np.ndarray:
    """A single L2-normalized log-mel band-energy vector describing a clip."""
    frames, _ = log_mel_frames(mono, sample_rate=sample_rate, n_bands=n_bands)
    if frames.size == 0:
        return np.zeros(n_bands, dtype=np.float32)
    vector = frames.mean(axis=0)
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return vector
    return (vector / norm).astype(np.float32)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in ``[0, 2]`` (0 == identical direction)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)


def spectral_change_points(
    mono: np.ndarray,
    *,
    sample_rate: int,
    window_sec: float,
    threshold: float,
) -> list[float]:
    """Times (seconds, clip-relative) where the spectrum changes abruptly.

    Adjacent log-mel frames are compared by cosine distance; local maxima above
    ``threshold`` mark butt-joined cue boundaries hidden inside one silent-gap
    region.
    """
    frames, times = log_mel_frames(
        mono, sample_rate=sample_rate, window_sec=window_sec, hop_sec=window_sec
    )
    if frames.shape[0] < 3:
        return []
    distances = np.asarray(
        [cosine_distance(frames[i - 1], frames[i]) for i in range(1, frames.shape[0])]
    )
    points: list[float] = []
    for index in range(1, distances.size - 1):
        value = distances[index]
        if value >= threshold and value >= distances[index - 1] and value >= distances[index + 1]:
            points.append(float(times[index + 1]))
    return points


def granular_loop(
    donor: np.ndarray,
    *,
    loop_samples: int,
    grain_samples: int,
    crossfade_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthesize a seamless loop by overlap-adding random donor grains.

    Grains are drawn at random from the donor material and overlap-added with
    equal-power fades (Ambience-Match style), so grain joins carry no amplitude
    discontinuity and the returned loop crossfades its own tail into its head.
    """
    donor = to_mono(donor)
    if donor.size == 0 or loop_samples <= 0:
        return np.zeros(max(0, loop_samples), dtype=np.float32)
    grain = int(min(grain_samples, donor.size))
    grain = max(2, grain)
    crossfade = int(min(crossfade_samples, grain // 2))
    crossfade = max(1, crossfade)
    hop = max(1, grain - crossfade)
    ramp = equal_power_ramp(crossfade)
    window = np.ones(grain, dtype=np.float32)
    window[:crossfade] = ramp
    window[-crossfade:] = ramp[::-1]

    total = loop_samples + grain
    buffer = np.zeros(total, dtype=np.float32)
    position = 0
    max_start = max(1, donor.size - grain)
    while position < loop_samples:
        start = int(rng.integers(0, max_start))
        buffer[position : position + grain] += donor[start : start + grain] * window
        position += hop

    loop = buffer[:loop_samples].copy()
    # Wrap the overflow tail back over the head so the loop point is seamless.
    tail = buffer[loop_samples : loop_samples + crossfade]
    if tail.size:
        loop[: tail.size] += tail
    return loop.astype(np.float32)


def tile_to_length(loop: np.ndarray, length: int) -> np.ndarray:
    """Repeat ``loop`` to fill ``length`` samples (empty loop -> silence)."""
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    if loop.size == 0:
        return np.zeros(length, dtype=np.float32)
    repeats = int(np.ceil(length / loop.size))
    return np.tile(loop, repeats)[:length].astype(np.float32)
