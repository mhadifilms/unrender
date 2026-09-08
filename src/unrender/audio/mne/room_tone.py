"""Room tone: harvest speech-free gaps from DX and synthesize seamless beds.

Room tone is the acoustic floor of the recording space, so it is extracted from
the *dialogue* stem (not FX): the gaps between spoken lines are the cleanest
donor material for the space. Gaps are harvested by inverting
:func:`unrender.editorial.dialogue.transcription.detect_speech_islands`, filtered to
reject digital silence and transient-containing gaps, grouped by scene (or by
spectral fingerprint when no ``scenes.json`` exists), and each group is turned
into a seamless loop by granular concatenation with equal-power crossfades
(Ambience-Match style). A full-length RT bed lays each group's tone across its
span.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from unrender.editorial.dialogue.transcription import detect_speech_islands
from unrender.lib.audio import probe_duration_sec, read_wav_float, write_wav_float
from unrender.lib.audio_features import (
    Segment,
    as_float,
    gap_segments,
    granular_loop,
    inset_gaps,
    spectral_fingerprint,
    tile_to_length,
    to_channels,
    to_mono,
)
from unrender.lib.fingerprints import input_fingerprints, warn_if_inputs_changed
from unrender.manifests import read_json, write_json
from unrender.project import RunPaths


@dataclass(frozen=True)
class RoomToneSettings:
    min_gap_sec: float = 0.4
    threshold_db: float = -45.0
    digital_silence_db: float = -80.0
    transient_ratio: float = 6.0
    # Inset each harvested gap by this handle on both sides so breath tails and
    # reverb decay bleeding out of adjacent speech do not pollute donor tone.
    edge_handle_sec: float = 0.15
    grain_ms: float = 250.0
    crossfade_ms: float = 50.0
    loop_sec: float = 4.0
    cluster_distance: float = 0.3
    seed: int = 0


@dataclass
class DonorGap:
    start_sec: float
    end_sec: float
    rms_db: float
    crest_factor: float
    fingerprint: np.ndarray


@dataclass
class ToneGroup:
    group_id: str
    gap_indices: list[int] = field(default_factory=list)
    loop_path: Path | None = None
    spectral_profile: list[float] = field(default_factory=list)


def extract_room_tone(
    *,
    run: RunPaths,
    dx_stem: str | Path,
    settings: RoomToneSettings | None = None,
    scenes_path: Path | None = None,
    prefix: str,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Harvest donor gaps and render per-scene loops plus a full-length bed."""
    run.ensure()
    active = settings or RoomToneSettings()
    source = Path(dx_stem).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"DX stem not found: {source}")
    output_json = run.mne_room_tone_json
    inputs = {"dx_stem": source}

    if output_json.exists() and not force:
        data = read_json(output_json)
        warn_if_inputs_changed(data, inputs, output_json)
        print(f"Room tone already exists, skipping: {output_json}", flush=True)
        return data

    if dry_run:
        print(f"DRY RUN: would harvest room-tone gaps from DX stem: {source}", flush=True)
        print(f"DRY RUN: would write RT bed and per-scene loops to {run.mne_room_tone_dir}")
        return {}

    total_sec = probe_duration_sec(source)
    islands = [
        Segment(island.start_sec, island.end_sec)
        for island in detect_speech_islands(source, threshold_db=active.threshold_db)
    ]
    raw_gaps = gap_segments(islands, total_sec=total_sec, min_gap_sec=active.min_gap_sec)
    raw_gaps = inset_gaps(raw_gaps, handle=active.edge_handle_sec, min_gap_sec=active.min_gap_sec)

    audio, sample_rate, sample_width = read_wav_float(source)
    channels = audio.shape[1] if audio.ndim == 2 else 1
    mono = to_mono(audio)
    gaps, rejected = _accept_gaps(mono, sample_rate, raw_gaps, active)

    scenes = _load_scenes(scenes_path)
    groups = _group_gaps(gaps, scenes=scenes, settings=active)

    rng = np.random.default_rng(active.seed)
    grain = max(2, int(active.grain_ms / 1000.0 * sample_rate))
    crossfade = max(1, int(active.crossfade_ms / 1000.0 * sample_rate))
    loop_samples = max(grain * 2, int(active.loop_sec * sample_rate))
    global_donor = _concat_gaps(mono, sample_rate, gaps)
    global_loop = granular_loop(
        global_donor,
        loop_samples=loop_samples,
        grain_samples=grain,
        crossfade_samples=crossfade,
        rng=rng,
    )

    loops: dict[str, np.ndarray] = {}
    for group in groups:
        donor = _concat_gaps(mono, sample_rate, [gaps[i] for i in group.gap_indices])
        if donor.size < grain:
            donor = global_donor
        loop = granular_loop(
            donor,
            loop_samples=loop_samples,
            grain_samples=grain,
            crossfade_samples=crossfade,
            rng=rng,
        )
        loops[group.group_id] = loop
        group.spectral_profile = [
            round(float(value), 5) for value in spectral_fingerprint(loop, sample_rate=sample_rate)
        ]
        loop_path = run.mne_room_tone_dir / f"scene_{group.group_id}_tone.wav"
        write_wav_float(
            loop_path, to_channels(loop, channels), sample_rate, sample_width=sample_width
        )
        group.loop_path = loop_path

    total_samples = max(1, int(total_sec * sample_rate))
    spans = _tone_spans(
        groups, gaps, scenes=scenes, total_samples=total_samples, sample_rate=sample_rate
    )
    # A scene with no usable donor gaps (e.g. wall-to-wall dialogue) still needs
    # a fill, so fall back to the global tone rather than leaving silence.
    bed = _render_bed(spans, loops, total_samples=total_samples, fallback_loop=global_loop)
    rt_stem = run.mne_room_tone_dir / f"{prefix}_RT_stem.wav"
    write_wav_float(rt_stem, to_channels(bed, channels), sample_rate, sample_width=sample_width)

    payload: dict[str, Any] = {
        "version": "1.0",
        "source": str(source),
        "format": "wav",
        "output_dir": str(run.mne_room_tone_dir),
        "duration_sec": round(total_sec, 3),
        "grouping": "scene" if scenes else "spectral_cluster",
        "rt_stem": str(rt_stem),
        "inputs": input_fingerprints(inputs),
        "donor_gaps": [
            {
                "start_sec": round(gap.start_sec, 3),
                "end_sec": round(gap.end_sec, 3),
                "rms_db": round(gap.rms_db, 2),
                "crest_factor": round(gap.crest_factor, 3),
            }
            for gap in gaps
        ],
        "rejected_gaps": rejected,
        "groups": [
            {
                "group_id": group.group_id,
                "gap_count": len(group.gap_indices),
                "loop_path": str(group.loop_path) if group.loop_path else "",
                "spectral_profile": group.spectral_profile,
            }
            for group in groups
        ],
        "spans": [
            {
                "start_sec": round(start / sample_rate, 3),
                "end_sec": round(end / sample_rate, 3),
                "group_id": gid,
            }
            for start, end, gid in spans
        ],
    }
    write_json(output_json, payload)
    print(
        f"Room tone saved: {output_json} ({len(gaps)} donor gap(s), {len(groups)} group(s))",
        flush=True,
    )
    return payload


def _accept_gaps(
    mono: np.ndarray,
    sample_rate: int,
    gaps: list[Segment],
    settings: RoomToneSettings,
) -> tuple[list[DonorGap], list[dict[str, Any]]]:
    accepted: list[DonorGap] = []
    rejected: list[dict[str, Any]] = []
    for gap in gaps:
        clip = mono[int(gap.start_sec * sample_rate) : int(gap.end_sec * sample_rate)]
        if clip.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(clip))))
        peak = float(np.max(np.abs(clip)))
        rms_db = 20.0 * math.log10(rms) if rms > 0 else float("-inf")
        crest = peak / rms if rms > 0 else float("inf")
        reason = ""
        if rms_db <= settings.digital_silence_db:
            reason = "digital_silence"
        elif crest > settings.transient_ratio:
            reason = "transient"
        if reason:
            rejected.append(
                {
                    "start_sec": round(gap.start_sec, 3),
                    "end_sec": round(gap.end_sec, 3),
                    "reason": reason,
                }
            )
            continue
        accepted.append(
            DonorGap(
                start_sec=gap.start_sec,
                end_sec=gap.end_sec,
                rms_db=rms_db,
                crest_factor=crest,
                fingerprint=spectral_fingerprint(clip, sample_rate=sample_rate),
            )
        )
    return accepted, rejected


def _concat_gaps(mono: np.ndarray, sample_rate: int, gaps: list[DonorGap]) -> np.ndarray:
    parts = [
        mono[int(gap.start_sec * sample_rate) : int(gap.end_sec * sample_rate)] for gap in gaps
    ]
    parts = [part for part in parts if part.size]
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


def _load_scenes(scenes_path: Path | None) -> list[dict[str, Any]]:
    if scenes_path is None or not scenes_path.exists():
        return []
    from unrender.editorial.scenes.manifest import load_scene_manifest

    try:
        return load_scene_manifest(scenes_path)
    except ValueError:
        return []


def _group_gaps(
    gaps: list[DonorGap], *, scenes: list[dict[str, Any]], settings: RoomToneSettings
) -> list[ToneGroup]:
    if not gaps:
        return []
    if scenes:
        from unrender.editorial.scenes.manifest import scene_for_time

        by_scene: dict[str, ToneGroup] = {}
        for index, gap in enumerate(gaps):
            center = (gap.start_sec + gap.end_sec) / 2.0
            scene_id = scene_for_time(scenes, center) or "000"
            group = by_scene.setdefault(scene_id, ToneGroup(group_id=scene_id))
            group.gap_indices.append(index)
        return [by_scene[key] for key in sorted(by_scene)]

    labels = _cluster_labels(gaps, distance=settings.cluster_distance)
    by_label: dict[int, ToneGroup] = {}
    for index, label in enumerate(labels):
        group = by_label.setdefault(int(label), ToneGroup(group_id=f"{int(label):02d}"))
        group.gap_indices.append(index)
    return [by_label[key] for key in sorted(by_label)]


def _cluster_labels(gaps: list[DonorGap], *, distance: float) -> list[int]:
    if len(gaps) == 1:
        return [0]
    from sklearn.cluster import AgglomerativeClustering

    matrix = np.vstack([gap.fingerprint for gap in gaps])
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance,
        metric="cosine",
        linkage="average",
    )
    return [int(label) for label in model.fit_predict(matrix)]


def _tone_spans(
    groups: list[ToneGroup],
    gaps: list[DonorGap],
    *,
    scenes: list[dict[str, Any]],
    total_samples: int,
    sample_rate: int,
) -> list[tuple[int, int, str]]:
    """Full-timeline ``(start_sample, end_sample, group_id)`` coverage."""
    if not groups:
        return []

    def to_sample(seconds: float) -> int:
        return max(0, min(total_samples, round(seconds * sample_rate)))

    if scenes:
        spans = [
            (
                to_sample(float(scene["start_sec"])),
                to_sample(float(scene["end_sec"])),
                str(scene["scene_id"]),
            )
            for scene in scenes
        ]
        return [(start, end, gid) for start, end, gid in spans if end > start]

    # Voronoi-in-time: each gap owns the region up to the midpoint with its
    # temporal neighbour; adjacent same-cluster regions merge into one span.
    group_of_gap: dict[int, str] = {}
    for group in groups:
        for index in group.gap_indices:
            group_of_gap[index] = group.group_id
    order = sorted(range(len(gaps)), key=lambda i: gaps[i].start_sec)
    centers = [(gaps[i].start_sec + gaps[i].end_sec) / 2.0 for i in order]
    boundaries = [0.0]
    for i in range(len(centers) - 1):
        boundaries.append((centers[i] + centers[i + 1]) / 2.0)
    boundaries.append(total_samples / float(sample_rate))
    merged: list[tuple[int, int, str]] = []
    for position, gap_index in enumerate(order):
        start = to_sample(boundaries[position])
        end = to_sample(boundaries[position + 1])
        gid = group_of_gap[gap_index]
        if end <= start:
            continue
        if merged and merged[-1][2] == gid:
            merged[-1] = (merged[-1][0], end, gid)
        else:
            merged.append((start, end, gid))
    return merged


def _render_bed(
    spans: list[tuple[int, int, str]],
    loops: dict[str, np.ndarray],
    *,
    total_samples: int,
    fallback_loop: np.ndarray | None = None,
) -> np.ndarray:
    bed = np.zeros(total_samples, dtype=np.float32)
    for start, end, gid in spans:
        if end <= start:
            continue
        loop = loops.get(gid)
        if loop is None or loop.size == 0:
            loop = fallback_loop
        if loop is None or loop.size == 0:
            continue
        bed[start:end] = tile_to_length(loop, end - start)
    return bed


def settings_from_values(
    *,
    min_gap_sec: float | str | None = None,
    threshold_db: float | str | None = None,
    digital_silence_db: float | str | None = None,
    transient_ratio: float | str | None = None,
    edge_handle_sec: float | str | None = None,
    grain_ms: float | str | None = None,
    crossfade_ms: float | str | None = None,
    loop_sec: float | str | None = None,
    cluster_distance: float | str | None = None,
    seed: int | str | None = None,
) -> RoomToneSettings:
    base = RoomToneSettings()
    return RoomToneSettings(
        min_gap_sec=as_float(min_gap_sec, base.min_gap_sec),
        threshold_db=as_float(threshold_db, base.threshold_db),
        digital_silence_db=as_float(digital_silence_db, base.digital_silence_db),
        transient_ratio=as_float(transient_ratio, base.transient_ratio),
        edge_handle_sec=as_float(edge_handle_sec, base.edge_handle_sec),
        grain_ms=as_float(grain_ms, base.grain_ms),
        crossfade_ms=as_float(crossfade_ms, base.crossfade_ms),
        loop_sec=as_float(loop_sec, base.loop_sec),
        cluster_distance=as_float(cluster_distance, base.cluster_distance),
        seed=base.seed if seed is None or seed == "" else int(seed),
    )
