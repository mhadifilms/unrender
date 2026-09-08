"""Cross-artifact analysis, recurrence, and dialogue-line signatures."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from unrender.project import RunPaths

from .io import atomic_open_memmap, atomic_save_npy, commit_memmap
from .models import (
    AnalysisSettings,
    AudioArtifact,
    FeatureDescriptor,
    Timebase,
)

POSTPROCESSOR_VERSION = "1.0"
_EPSILON = 1.0e-12


def postprocess_audio_features(
    *,
    run: RunPaths,
    artifacts: list[AudioArtifact],
    features: list[FeatureDescriptor],
    settings: AnalysisSettings,
    analysis_dir: Path,
) -> list[FeatureDescriptor]:
    output: list[FeatureDescriptor] = []
    output.extend(
        _source_interactions(
            artifacts=artifacts,
            features=features,
            analysis_dir=analysis_dir,
        )
    )
    recurrence, motif_events = _recurrence_features(
        artifacts=artifacts,
        features=features,
        settings=settings,
        analysis_dir=analysis_dir,
    )
    output.extend(recurrence)
    if motif_events:
        output.append(
            FeatureDescriptor(
                id="cross.repeated_motifs.events",
                name="repeated_motifs",
                kind="events",
                unit="seconds",
                value=motif_events,
                metadata={"method": "off-diagonal cosine self-similarity peaks"},
            )
        )
    output.extend(
        _dialogue_line_features(
            run=run,
            artifacts=artifacts,
            features=features,
            analysis_dir=analysis_dir,
        )
    )
    return output


def compute_masking_risk(
    dialogue_power: np.ndarray | list[float],
    background_power: np.ndarray | list[float],
) -> np.ndarray:
    """Return a bounded risk that rises monotonically with background power."""
    dialogue = np.maximum(np.asarray(dialogue_power, dtype=np.float64), 0.0)
    background = np.maximum(np.asarray(background_power, dtype=np.float64), 0.0)
    dialogue, background = np.broadcast_arrays(dialogue, background)
    risk = background / (dialogue + background + _EPSILON)
    return np.asarray(np.clip(risk, 0.0, 1.0), dtype=np.float32)


def compute_self_similarity(
    embeddings: np.ndarray,
    output_path: Path | str,
    *,
    block_size: int = 256,
) -> Path:
    """Write a cosine similarity matrix blockwise through an atomic NPY memmap."""
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    count = values.shape[0]
    destination = Path(output_path)
    mmap, temporary = atomic_open_memmap(destination, dtype=np.float32, shape=(count, count))
    try:
        norms = np.linalg.norm(values, axis=1)
        for row_start in range(0, count, block_size):
            row_end = min(count, row_start + block_size)
            rows = values[row_start:row_end]
            row_norms = norms[row_start:row_end, None]
            for column_start in range(0, count, block_size):
                column_end = min(count, column_start + block_size)
                columns = values[column_start:column_end]
                denominator = row_norms * norms[column_start:column_end][None, :]
                block = rows @ columns.T
                np.divide(block, denominator, out=block, where=denominator > _EPSILON)
                block[denominator <= _EPSILON] = 0.0
                mmap[row_start:row_end, column_start:column_end] = np.clip(block, -1.0, 1.0)
        return commit_memmap(mmap, temporary, destination)
    except BaseException:
        del mmap
        temporary.unlink(missing_ok=True)
        raise


def find_repeated_motifs(
    similarity: np.ndarray,
    *,
    hop_sec: float,
    threshold: float = 0.88,
    min_separation_sec: float = 1.0,
    max_candidates: int = 64,
) -> list[dict[str, Any]]:
    """Find deterministic, well-separated off-diagonal recurrence peaks."""
    matrix = np.asarray(similarity)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("similarity must be a square matrix")
    separation = max(1, math.ceil(min_separation_sec / hop_sec))
    candidates: list[tuple[float, int, int]] = []
    for first in range(matrix.shape[0]):
        start = first + separation
        if start >= matrix.shape[1]:
            continue
        row = matrix[first, start:]
        if row.size == 0:
            continue
        relative = int(np.argmax(row))
        score = float(row[relative])
        if score >= threshold:
            candidates.append((score, first, start + relative))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    accepted: list[tuple[float, int, int]] = []
    for candidate in candidates:
        _, first, second = candidate
        if any(
            abs(first - previous_first) <= 1 and abs(second - previous_second) <= 1
            for _, previous_first, previous_second in accepted
        ):
            continue
        accepted.append(candidate)
        if len(accepted) >= max_candidates:
            break
    return [
        {
            "first_start_sec": round(first * hop_sec, 6),
            "second_start_sec": round(second * hop_sec, 6),
            "duration_sec": round(hop_sec, 6),
            "similarity": round(score, 6),
        }
        for score, first, second in accepted
    ]


def _source_interactions(
    *,
    artifacts: list[AudioArtifact],
    features: list[FeatureDescriptor],
    analysis_dir: Path,
) -> list[FeatureDescriptor]:
    rms_by_artifact = {
        feature.artifact_id: feature
        for feature in features
        if feature.name == "rms_dbfs"
        and feature.kind == "series"
        and feature.artifact_id is not None
    }
    bark_by_artifact = {
        feature.artifact_id: feature
        for feature in features
        if feature.name == "bark_band_power"
        and feature.kind == "matrix"
        and feature.artifact_id is not None
    }
    grouped: dict[str, list[np.ndarray]] = {"D": [], "M": [], "E": []}
    grouped_bands: dict[str, list[np.ndarray]] = {"D": [], "M": [], "E": []}
    timebase: Timebase | None = None
    for artifact in artifacts:
        descriptor = rms_by_artifact.get(artifact.id)
        group = _dme_group(artifact.role)
        if descriptor is None or group is None:
            continue
        values = _load_array(analysis_dir, descriptor)
        power = np.power(10.0, np.asarray(values, dtype=np.float64) / 10.0)
        if group == "ME":
            grouped["M"].append(power * 0.5)
            grouped["E"].append(power * 0.5)
        else:
            grouped[group].append(power)
        bark_descriptor = bark_by_artifact.get(artifact.id)
        if bark_descriptor is not None:
            bark_db = _load_array(analysis_dir, bark_descriptor)
            bark_power = np.power(10.0, np.asarray(bark_db, dtype=np.float64) / 10.0)
            if group == "ME":
                grouped_bands["M"].append(bark_power * 0.5)
                grouped_bands["E"].append(bark_power * 0.5)
            else:
                grouped_bands[group].append(bark_power)
        timebase = timebase or descriptor.timebase
    count = max((len(item) for values in grouped.values() for item in values), default=0)
    if count == 0 or timebase is None:
        return []
    powers = np.zeros((count, 3), dtype=np.float64)
    for column, role in enumerate(("D", "M", "E")):
        for values in grouped[role]:
            powers[: len(values), column] += values
    total = np.sum(powers, axis=1, keepdims=True)
    dominance = np.divide(
        powers,
        total,
        out=np.zeros_like(powers),
        where=total > _EPSILON,
    ).astype(np.float32)
    risk = compute_masking_risk(powers[:, 0], powers[:, 1] + powers[:, 2])
    common_timebase = Timebase(
        start_sec=timebase.start_sec,
        hop_sec=timebase.hop_sec,
        frame_sec=timebase.frame_sec,
        count=count,
    )
    output = [
        _save_dense(
            analysis_dir,
            relative=Path("arrays") / "cross" / "source_dominance.npy",
            feature_id="cross.source_dominance.matrix",
            name="source_dominance",
            kind="matrix",
            values=dominance,
            unit="proportion",
            timebase=common_timebase,
            metadata={"columns": ["D", "M", "E"]},
        ),
        _save_dense(
            analysis_dir,
            relative=Path("arrays") / "cross" / "dialogue_masking_risk.npy",
            feature_id="cross.dialogue_masking_risk.series",
            name="dialogue_masking_risk",
            kind="series",
            values=risk,
            unit="probability",
            timebase=common_timebase,
            metadata={
                "method": "background_power / (dialogue_power + background_power)",
                "estimate": True,
            },
        ),
    ]
    band_shapes = [
        item.shape
        for values in grouped_bands.values()
        for item in values
        if item.ndim == 2 and item.size
    ]
    if band_shapes:
        band_count = min(shape[1] for shape in band_shapes)
        frame_count = max(shape[0] for shape in band_shapes)
        band_powers = np.zeros((frame_count, band_count, 3), dtype=np.float64)
        for column, role in enumerate(("D", "M", "E")):
            for values in grouped_bands[role]:
                usable = values[:frame_count, :band_count]
                band_powers[: len(usable), :, column] += usable
        band_total = np.sum(band_powers, axis=2, keepdims=True)
        band_dominance = np.divide(
            band_powers,
            band_total,
            out=np.zeros_like(band_powers),
            where=band_total > _EPSILON,
        ).astype(np.float32)
        dialogue = band_powers[:, :, 0]
        background = band_powers[:, :, 1] + band_powers[:, :, 2]
        band_masking = compute_masking_risk(dialogue, background)
        band_timebase = Timebase(
            start_sec=timebase.start_sec,
            hop_sec=timebase.hop_sec,
            frame_sec=timebase.frame_sec,
            count=frame_count,
        )
        output.extend(
            [
                _save_dense(
                    analysis_dir,
                    relative=Path("arrays") / "cross" / "source_dominance_by_band.npy",
                    feature_id="cross.source_dominance_by_band.matrix",
                    name="source_dominance_by_band",
                    kind="matrix",
                    values=band_dominance,
                    unit="proportion",
                    timebase=band_timebase,
                    metadata={
                        "axes": ["time", "bark_band", "source"],
                        "sources": ["D", "M", "E"],
                    },
                ),
                _save_dense(
                    analysis_dir,
                    relative=Path("arrays") / "cross" / "dialogue_masking_by_band.npy",
                    feature_id="cross.dialogue_masking_by_band.matrix",
                    name="dialogue_masking_by_band",
                    kind="matrix",
                    values=band_masking,
                    unit="probability",
                    timebase=band_timebase,
                    metadata={
                        "axes": ["time", "bark_band"],
                        "estimate": True,
                    },
                ),
            ]
        )
    return output


def _recurrence_features(
    *,
    artifacts: list[AudioArtifact],
    features: list[FeatureDescriptor],
    settings: AnalysisSettings,
    analysis_dir: Path,
) -> tuple[list[FeatureDescriptor], list[dict[str, Any]]]:
    artifact = _reference_artifact(artifacts)
    if artifact is None:
        return [], []
    descriptor = next(
        (
            feature
            for feature in features
            if feature.artifact_id == artifact.id
            and feature.name == "mfcc"
            and feature.path is not None
        ),
        None,
    )
    if descriptor is None or descriptor.timebase is None:
        return [], []
    mfcc = _load_array(analysis_dir, descriptor)
    output: list[FeatureDescriptor] = []
    motifs: list[dict[str, Any]] = []
    for requested_scale in settings.similarity_scales_sec:
        factor = max(1, round(requested_scale / descriptor.timebase.hop_sec))
        downsampled_count = math.ceil(len(mfcc) / factor) if len(mfcc) else 0
        if downsampled_count > settings.max_similarity_frames:
            factor *= math.ceil(downsampled_count / settings.max_similarity_frames)
        embeddings = _aggregate_rows(mfcc, factor)
        actual_hop = factor * descriptor.timebase.hop_sec
        label = f"{actual_hop:g}s".replace(".", "p")
        relative = Path("arrays") / artifact.id / f"self_similarity_{label}.npy"
        compute_self_similarity(
            embeddings,
            analysis_dir / relative,
            block_size=settings.similarity_block_frames,
        )
        timebase = Timebase(0.0, actual_hop, actual_hop, len(embeddings))
        output.append(
            FeatureDescriptor(
                id=f"{artifact.id}.self_similarity_{label}.matrix",
                artifact_id=artifact.id,
                name=f"self_similarity_{label}",
                kind="matrix",
                unit="cosine_similarity",
                path=relative.as_posix(),
                dtype="float32",
                shape=(len(embeddings), len(embeddings)),
                timebase=timebase,
                metadata={
                    "requested_scale_sec": requested_scale,
                    "computed_blockwise": True,
                    "storage": "npy_memmap",
                },
            )
        )
        similarity = np.load(analysis_dir / relative, mmap_mode="r", allow_pickle=False)
        scale_motifs = find_repeated_motifs(
            similarity,
            hop_sec=actual_hop,
            threshold=settings.motif_threshold,
            min_separation_sec=max(actual_hop * 2.0, requested_scale),
        )
        for motif in scale_motifs:
            motif["scale_sec"] = round(actual_hop, 6)
            motif["artifact_id"] = artifact.id
        motifs.extend(scale_motifs)
    return output, motifs


def _dialogue_line_features(
    *,
    run: RunPaths,
    artifacts: list[AudioArtifact],
    features: list[FeatureDescriptor],
    analysis_dir: Path,
) -> list[FeatureDescriptor]:
    if not run.dialogue_lines_json.is_file():
        return []
    try:
        with run.dialogue_lines_json.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return []
    rows = payload.get("lines", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    artifact = next(
        (item for item in artifacts if item.role == "dialogue"),
        _reference_artifact(artifacts),
    )
    if artifact is None:
        return []
    descriptors = {
        feature.name: feature
        for feature in features
        if feature.artifact_id == artifact.id and feature.path is not None
    }
    rms_descriptor = descriptors.get("rms_dbfs")
    centroid_descriptor = descriptors.get("spectral_centroid_hz")
    mfcc_descriptor = descriptors.get("mfcc")
    if (
        rms_descriptor is None
        or centroid_descriptor is None
        or mfcc_descriptor is None
        or rms_descriptor.timebase is None
    ):
        return []
    rms = _load_array(analysis_dir, rms_descriptor)
    centroid = _load_array(analysis_dir, centroid_descriptor)
    mfcc = _load_array(analysis_dir, mfcc_descriptor)
    hop = rms_descriptor.timebase.hop_sec
    signatures: list[np.ndarray] = []
    line_ids: list[str] = []
    speakers: list[str] = []
    intervals: list[tuple[float, float]] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            continue
        start = float(row.get("start_sec") or row.get("start") or 0.0)
        end = float(row.get("end_sec") or row.get("end") or start)
        first = max(0, math.floor(start / hop))
        last = min(len(rms), max(first + 1, math.ceil(end / hop)))
        if first >= len(rms) or last <= first:
            continue
        signature = np.concatenate(
            (
                np.asarray(
                    [
                        float(np.mean(rms[first:last])),
                        float(np.mean(centroid[first:last])),
                    ],
                    dtype=np.float32,
                ),
                np.asarray(np.mean(mfcc[first:last], axis=0), dtype=np.float32),
            )
        )
        signatures.append(signature)
        line_ids.append(str(row.get("line_id") or row.get("id") or index))
        speakers.append(
            str(row.get("speaker") or row.get("diarized_speaker") or row.get("source_group") or "")
        )
        intervals.append((start, end))
    if not signatures:
        return []
    matrix = np.stack(signatures).astype(np.float32)
    outliers = _line_outliers(matrix, line_ids, speakers, intervals)
    return [
        _save_dense(
            analysis_dir,
            relative=Path("arrays") / "cross" / "dialogue_line_signatures.npy",
            feature_id="cross.dialogue_line_signatures.embeddings",
            name="dialogue_line_signatures",
            kind="embeddings",
            values=matrix,
            unit="mixed",
            metadata={
                "line_ids": line_ids,
                "speakers": speakers,
                "columns": ["rms_dbfs", "spectral_centroid_hz", "mfcc..."],
            },
        ),
        FeatureDescriptor(
            id="cross.dialogue_line_outliers.events",
            name="dialogue_line_outliers",
            kind="events",
            unit="robust_z",
            value=outliers,
            metadata={"method": "speaker-conditioned robust multivariate distance"},
        ),
    ]


def _line_outliers(
    matrix: np.ndarray,
    line_ids: list[str],
    speakers: list[str],
    intervals: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for index, row in enumerate(matrix):
        peers = np.asarray(
            [
                candidate
                for candidate_index, candidate in enumerate(matrix)
                if candidate_index != index
                and (not speakers[index] or speakers[candidate_index] == speakers[index])
            ]
        )
        if len(peers) < 2:
            peers = np.delete(matrix, index, axis=0)
        if len(peers) < 2:
            continue
        median = np.median(peers, axis=0)
        scale = 1.4826 * np.median(np.abs(peers - median), axis=0)
        valid = scale > 1.0e-6
        score = (
            float(np.sqrt(np.mean(np.square((row[valid] - median[valid]) / scale[valid]))))
            if np.any(valid)
            else 0.0
        )
        if score >= 3.5:
            events.append(
                {
                    "line_id": line_ids[index],
                    "speaker": speakers[index],
                    "start_sec": round(intervals[index][0], 6),
                    "end_sec": round(intervals[index][1], 6),
                    "score": round(score, 6),
                }
            )
    return events


def _aggregate_rows(values: np.ndarray, factor: int) -> np.ndarray:
    if len(values) == 0:
        return np.empty((0, values.shape[1]), dtype=np.float32)
    count = math.ceil(len(values) / factor)
    output = np.empty((count, values.shape[1]), dtype=np.float32)
    for index in range(count):
        output[index] = np.mean(values[index * factor : (index + 1) * factor], axis=0)
    return output


def _load_array(analysis_dir: Path, descriptor: FeatureDescriptor) -> np.ndarray:
    if descriptor.path is None:
        raise ValueError(f"feature {descriptor.id} has no dense-array path")
    return np.load(analysis_dir / descriptor.path, mmap_mode="r", allow_pickle=False)


def _save_dense(
    analysis_dir: Path,
    *,
    relative: Path,
    feature_id: str,
    name: str,
    kind: str,
    values: np.ndarray,
    unit: str,
    timebase: Timebase | None = None,
    metadata: dict[str, Any] | None = None,
) -> FeatureDescriptor:
    array = np.asarray(values, dtype=np.float32)
    atomic_save_npy(analysis_dir / relative, array)
    return FeatureDescriptor(
        id=feature_id,
        name=name,
        kind=kind,  # type: ignore[arg-type]
        unit=unit,
        path=relative.as_posix(),
        dtype=str(array.dtype),
        shape=tuple(array.shape),
        timebase=timebase,
        metadata=metadata or {},
    )


def _reference_artifact(artifacts: list[AudioArtifact]) -> AudioArtifact | None:
    priorities = ("source_mix", "dialogue", "music_effects", "explicit")
    for role in priorities:
        match = next((artifact for artifact in artifacts if artifact.role == role), None)
        if match is not None:
            return match
    return artifacts[0] if artifacts else None


def _dme_group(role: str) -> str | None:
    normalized = role.lower()
    if normalized in {"d", "dx", "dialogue"} or normalized.startswith(("speaker:", "dialogue_")):
        return "D"
    if normalized in {"m", "mx", "music"} or normalized.startswith(
        ("music:", "music_cue:", "music_instrument:")
    ):
        return "M"
    if normalized in {"e", "fx", "sfx", "effects"} or normalized.startswith("effects:"):
        return "E"
    if normalized in {"me", "m&e", "music_effects"}:
        return "ME"
    return None
