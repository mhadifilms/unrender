"""Execution bridge from optional providers into canonical feature descriptors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..io import atomic_save_npy, iter_audio
from ..models import AnalysisSettings, AudioArtifact, FeatureDescriptor, Timebase
from .beat_this import BeatThisProvider
from .brouhaha import BrouhahaProvider
from .clap import ClapProvider
from .contract import CapabilityState, ProviderPolicy, ProviderResult
from .whisperx import WhisperXProvider


def run_installed_providers(
    *,
    artifacts: list[AudioArtifact],
    settings: AnalysisSettings,
    analysis_dir: Path,
    policy: ProviderPolicy,
) -> tuple[list[FeatureDescriptor], list[dict[str, Any]]]:
    """Run available providers on role-appropriate artifacts.

    MFA is intentionally preflight-only here because it requires a
    language-specific dictionary, acoustic model, and trusted transcript. Its
    adapter remains available for explicit use.
    """
    selected = (
        set(settings.providers)
        if settings.providers
        else {
            "whisperx",
            "brouhaha",
            "clap",
            "beat_this",
        }
    )
    features: list[FeatureDescriptor] = []
    results: list[dict[str, Any]] = []
    dialogue = _artifact(artifacts, "dialogue", "dx", "speaker")
    music = _artifact(artifacts, "music", "mx", "cue")
    semantic = _artifact(artifacts, "effects", "fx", "ambience", "mix")

    if "whisperx" in selected and dialogue is not None:
        whisper_provider = WhisperXProvider(
            whisper_model=settings.whisper_model,
            device=settings.device,
            compute_type=settings.compute_type,
        )
        if _available(whisper_provider.preflight(policy)):
            result = whisper_provider.align(dialogue.path, language=settings.language)
            results.append(result.as_dict())
            features.extend(_whisper_features(result, dialogue))

    if "brouhaha" in selected and dialogue is not None:
        brouhaha_provider = BrouhahaProvider(
            enabled="brouhaha" in settings.provider_opt_ins,
        )
        if _available(brouhaha_provider.preflight(policy)):
            result = brouhaha_provider.analyze(dialogue.path, policy=policy)
            results.append(result.as_dict())
            features.extend(_brouhaha_features(result, dialogue, analysis_dir))

    if "beat_this" in selected and music is not None:
        beat_provider = BeatThisProvider(device=settings.device)
        if _available(beat_provider.preflight(policy)):
            result = beat_provider.analyze(music.path, policy=policy)
            results.append(result.as_dict())
            features.extend(_music_features(result, music))

    if "clap" in selected and semantic is not None:
        clap_provider = ClapProvider(device=settings.device)
        if _available(clap_provider.preflight(policy)):
            result = _run_clap_windows(clap_provider, semantic, settings)
            results.append(result.as_dict())
            features.extend(_clap_features(result, semantic, analysis_dir))
    return features, results


def _artifact(artifacts: list[AudioArtifact], *roles: str) -> AudioArtifact | None:
    wanted = {role.lower() for role in roles}
    return next((item for item in artifacts if item.role.lower() in wanted), None)


def _available(statuses: list[Any]) -> bool:
    return bool(statuses) and all(status.state is CapabilityState.AVAILABLE for status in statuses)


def _whisper_features(
    result: ProviderResult,
    artifact: AudioArtifact,
) -> list[FeatureDescriptor]:
    if not result.complete:
        return []
    output: list[FeatureDescriptor] = []
    for name in ("words", "characters"):
        values = result.data.get(name)
        if isinstance(values, list):
            output.append(
                FeatureDescriptor(
                    id=f"{artifact.id}.model_{name}.events",
                    artifact_id=artifact.id,
                    name=f"model_{name}",
                    kind="events",
                    unit="seconds",
                    value=values,
                    metadata=result.metadata,
                )
            )
    return output


def _brouhaha_features(
    result: ProviderResult,
    artifact: AudioArtifact,
    analysis_dir: Path,
) -> list[FeatureDescriptor]:
    if not result.complete:
        return []
    raw_series = result.data.get("series")
    if not isinstance(raw_series, dict):
        return []
    output: list[FeatureDescriptor] = []
    for name, records in raw_series.items():
        if not isinstance(records, list) or not records:
            continue
        value_key = next(
            (key for key in records[0] if key not in {"time", "time_sec", "start_sec", "end_sec"}),
            None,
        )
        if value_key is None:
            continue
        values = np.asarray(
            [float(record.get(value_key, 0.0)) for record in records],
            dtype=np.float32,
        )
        times = [float(record.get("time_sec", 0.0)) for record in records]
        hop = times[1] - times[0] if len(times) > 1 else 0.02
        output.append(
            _save_array(
                analysis_dir,
                artifact,
                name,
                values,
                kind="series",
                unit="probability" if name == "vad" else "dB",
                timebase=Timebase(times[0] if times else 0.0, hop, hop, len(values)),
                metadata=result.metadata,
            )
        )
    return output


def _music_features(
    result: ProviderResult,
    artifact: AudioArtifact,
) -> list[FeatureDescriptor]:
    if not result.complete:
        return []
    output: list[FeatureDescriptor] = []
    for name in ("beats", "downbeats"):
        values = result.data.get(name)
        if isinstance(values, list):
            output.append(
                FeatureDescriptor(
                    id=f"{artifact.id}.{name}.events",
                    artifact_id=artifact.id,
                    name=name,
                    kind="events",
                    unit="seconds",
                    value=[
                        {
                            "time_sec": float(value),
                            "start_sec": float(value),
                            "end_sec": float(value),
                            "type": name[:-1],
                            "label": name[:-1],
                        }
                        for value in values
                    ],
                    metadata=result.metadata,
                )
            )
    chords = result.data.get("chord_events")
    if isinstance(chords, list):
        output.append(
            FeatureDescriptor(
                id=f"{artifact.id}.chords.events",
                artifact_id=artifact.id,
                name="chords",
                kind="events",
                unit="seconds",
                value=chords,
                metadata=result.metadata,
            )
        )
    key = result.data.get("key")
    if key is not None:
        output.append(
            FeatureDescriptor(
                id=f"{artifact.id}.musical_key.scalar",
                artifact_id=artifact.id,
                name="musical_key",
                kind="scalar",
                value=key,
                metadata=result.metadata,
            )
        )
    return output


def _run_clap_windows(
    provider: ClapProvider,
    artifact: AudioArtifact,
    settings: AnalysisSettings,
) -> ProviderResult:
    window_frames = max(1, round(artifact.audio.sample_rate * settings.clap_window_sec))
    clips: list[np.ndarray] = []
    times: list[tuple[float, float]] = []
    pending = np.empty((0, artifact.audio.channels), dtype=np.float32)
    cursor = 0
    for block in iter_audio(
        Path(artifact.path),
        block_frames=settings.decode_block_frames,
        ffmpeg=settings.ffmpeg,
        metadata=artifact.audio,
    ):
        pending = np.concatenate((pending, block), axis=0)
        while len(pending) >= window_frames:
            clips.append(np.mean(pending[:window_frames], axis=1))
            start = cursor / artifact.audio.sample_rate
            times.append((start, start + settings.clap_window_sec))
            pending = pending[window_frames:]
            cursor += window_frames
    if len(pending) >= artifact.audio.sample_rate:
        clips.append(np.mean(pending, axis=1))
        start = cursor / artifact.audio.sample_rate
        times.append((start, start + len(pending) / artifact.audio.sample_rate))
    return provider.embed(clips, sample_rate=artifact.audio.sample_rate, times=times)


def _clap_features(
    result: ProviderResult,
    artifact: AudioArtifact,
    analysis_dir: Path,
) -> list[FeatureDescriptor]:
    if not result.complete:
        return []
    records = result.data.get("embeddings")
    if not isinstance(records, list) or not records:
        return []
    vectors = np.stack([np.asarray(record["embedding"], dtype=np.float16) for record in records])
    descriptor = _save_array(
        analysis_dir,
        artifact,
        "clap_embeddings",
        vectors,
        kind="embeddings",
        unit="cosine_embedding",
        metadata={
            **result.metadata,
            "times": [[float(record["start_sec"]), float(record["end_sec"])] for record in records],
        },
    )
    return [descriptor]


def _save_array(
    analysis_dir: Path,
    artifact: AudioArtifact,
    name: str,
    values: np.ndarray,
    *,
    kind: str,
    unit: str,
    timebase: Timebase | None = None,
    metadata: dict[str, Any] | None = None,
) -> FeatureDescriptor:
    relative = Path("arrays") / "models" / artifact.id / f"{name}.npy"
    array = np.asarray(values)
    atomic_save_npy(analysis_dir / relative, array)
    return FeatureDescriptor(
        id=f"{artifact.id}.{name}.{kind}",
        artifact_id=artifact.id,
        name=name,
        kind=kind,  # type: ignore[arg-type]
        unit=unit,
        path=relative.as_posix(),
        dtype=str(array.dtype),
        shape=tuple(array.shape),
        timebase=timebase,
        metadata=metadata or {},
    )


__all__ = ["run_installed_providers"]
