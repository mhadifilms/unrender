"""Beat This adapter with an explicit librosa analysis fallback."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np

from .contract import (
    CapabilityState,
    LicenseMetadata,
    ModelProvenance,
    ProviderPolicy,
    ProviderProvenance,
    ProviderResult,
    module_available,
    package_provenance,
    statuses_for_capabilities,
)

_PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
CHORD_VOCABULARY = (
    *tuple(f"{name}:maj" for name in _PITCH_NAMES),
    *tuple(f"{name}:min" for name in _PITCH_NAMES),
    "N",
)


class BeatThisProvider:
    provider_id = "beat_this"
    capabilities = ("beats", "downbeats", "key", "chords")
    license_metadata = LicenseMetadata(
        license_id="MIT (code and published weights); librosa fallback ISC",
        source="https://github.com/CPJKU/beat_this",
        permissive=True,
    )

    def __init__(
        self,
        *,
        model_id: str = "CPJKU/beat_this",
        model_revision: str | None = None,
        checkpoint_path: str = "final0",
        device: str = "cpu",
        allow_fallback: bool = True,
    ) -> None:
        self.model_id = model_id
        self.model_revision = model_revision
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.allow_fallback = allow_fallback

    @property
    def provenance(self) -> ProviderProvenance:
        packages = [package_provenance("beat-this")]
        if self.allow_fallback:
            packages.append(package_provenance("librosa"))
        return ProviderProvenance(
            packages=tuple(packages),
            models=(
                ModelProvenance(
                    model_id=self.model_id,
                    revision=self.model_revision,
                    source="beat_this",
                ),
            ),
        )

    def preflight(self, policy: ProviderPolicy | None = None):
        active_policy = policy or ProviderPolicy()
        beat_available = module_available("beat_this")
        fallback_available = self.allow_fallback and module_available("librosa")
        blocked = active_policy.block_reason(self.provider_id, self.license_metadata)
        if blocked and fallback_available:
            state = CapabilityState.AVAILABLE
            reason = f"{blocked}; permissive librosa fallback is available"
            backend = "librosa_fallback"
        elif blocked:
            state, reason, backend = CapabilityState.BLOCKED_BY_POLICY, blocked, None
        elif beat_available:
            state = CapabilityState.AVAILABLE
            reason = "Beat This is installed; model loading is deferred until analysis"
            backend = "beat_this"
        elif fallback_available:
            state = CapabilityState.AVAILABLE
            reason = "Beat This is absent; librosa fallback is available"
            backend = "librosa_fallback"
        else:
            state = CapabilityState.UNAVAILABLE
            reason = "neither Beat This nor the librosa fallback is installed"
            backend = None
        return statuses_for_capabilities(
            provider_id=self.provider_id,
            capabilities=self.capabilities,
            state=state,
            reason=reason,
            provenance=self.provenance,
            license_metadata=self.license_metadata,
            backend=backend,
        )

    def analyze(
        self,
        audio_path: str | Path,
        *,
        policy: ProviderPolicy | None = None,
    ) -> ProviderResult:
        status = self.preflight(policy)[0]
        if status.state is not CapabilityState.AVAILABLE:
            return ProviderResult(
                provider_id=self.provider_id,
                capability="music_timing_harmony",
                state=status.state,
                provenance=self.provenance,
                error=status.reason,
            )

        fallback_reason: str | None = None
        if status.backend == "beat_this":
            try:
                beat_data = self._run_beat_this(Path(audio_path))
                beat_data.setdefault("key", None)
                beat_data.setdefault("chord_events", [])
                beat_data.setdefault("chord_vocabulary", list(CHORD_VOCABULARY))
                return ProviderResult(
                    provider_id=self.provider_id,
                    capability="music_timing_harmony",
                    state=CapabilityState.COMPLETE,
                    data=beat_data,
                    metadata={"backend": "beat_this", "uncertainty": beat_data.get("uncertainty")},
                    provenance=self.provenance,
                )
            except Exception as exc:
                fallback_reason = str(exc)
                if not self.allow_fallback:
                    return self._failure(f"Beat This analysis failed: {exc}")

        try:
            fallback = _run_librosa_fallback(Path(audio_path))
        except Exception as exc:
            detail = f" after Beat This failed ({fallback_reason})" if fallback_reason else ""
            return self._failure(f"librosa fallback failed{detail}: {exc}")
        return ProviderResult(
            provider_id=self.provider_id,
            capability="music_timing_harmony",
            state=CapabilityState.COMPLETE,
            data=fallback,
            metadata={
                "backend": "librosa_fallback",
                "fallback_reason": fallback_reason,
                "uncertainty": fallback["uncertainty"],
            },
            provenance=self.provenance,
        )

    def _run_beat_this(self, audio_path: Path) -> dict[str, Any]:
        inference = importlib.import_module("beat_this.inference")
        predictor_class = getattr(inference, "File2Beats", None)
        if predictor_class is None:
            raise RuntimeError("installed Beat This package does not expose File2Beats")
        predictor = predictor_class(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            dbn=False,
        )
        output = predictor(str(audio_path))
        if isinstance(output, dict):
            beats = output.get("beats", [])
            downbeats = output.get("downbeats", [])
        elif isinstance(output, (tuple, list)) and len(output) == 2:
            beats, downbeats = output
        else:
            beats, downbeats = output, []
        return {
            "beats": [float(value) for value in np.asarray(beats).reshape(-1)],
            "downbeats": [float(value) for value in np.asarray(downbeats).reshape(-1)],
            "backend": "beat_this",
            "uncertainty": {
                "beats": "model_output_no_calibrated_probability",
                "downbeats": "model_output_no_calibrated_probability",
            },
        }

    def _failure(self, error: str) -> ProviderResult:
        return ProviderResult(
            provider_id=self.provider_id,
            capability="music_timing_harmony",
            state=CapabilityState.FAILED,
            provenance=self.provenance,
            error=error,
        )


def _run_librosa_fallback(audio_path: Path) -> dict[str, Any]:
    librosa = importlib.import_module("librosa")
    audio, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sample_rate)
    beat_times = np.asarray(
        librosa.frames_to_time(beat_frames, sr=sample_rate), dtype=np.float64
    ).reshape(-1)
    downbeat_times = beat_times[::4]
    chroma = np.asarray(librosa.feature.chroma_cqt(y=audio, sr=sample_rate), dtype=np.float32)
    frame_times = np.asarray(
        librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sample_rate), dtype=np.float64
    )
    key = _estimate_key(chroma)
    chord_events = _estimate_chords(chroma, frame_times)
    return {
        "beats": [float(value) for value in beat_times],
        "downbeats": [float(value) for value in downbeat_times],
        "tempo_bpm": float(np.asarray(tempo).reshape(-1)[0]),
        "key": key,
        "chord_events": chord_events,
        "chord_vocabulary": list(CHORD_VOCABULARY),
        "backend": "librosa_fallback",
        "uncertainty": {
            "beats": "signal-processing estimate",
            "downbeats": "assumed 4/4 phase from first detected beat",
            "key": float(key["uncertainty"]),
            "chords": "chroma-template estimate; no learned chord model",
        },
    }


def _estimate_key(chroma: np.ndarray) -> dict[str, Any]:
    if chroma.size == 0 or chroma.shape[1] == 0:
        return {"label": "unknown", "confidence": 0.0, "uncertainty": 1.0}
    profile = np.mean(chroma, axis=1)
    major = np.asarray(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        dtype=np.float32,
    )
    minor = np.asarray(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
        dtype=np.float32,
    )
    scores: list[tuple[str, float]] = []
    for root, name in enumerate(_PITCH_NAMES):
        scores.append((f"{name} major", _cosine(profile, np.roll(major, root))))
        scores.append((f"{name} minor", _cosine(profile, np.roll(minor, root))))
    scores.sort(key=lambda item: item[1], reverse=True)
    best_label, best_score = scores[0]
    second_score = scores[1][1]
    confidence = float(np.clip((best_score - second_score + 1.0) / 2.0, 0.0, 1.0))
    return {
        "label": best_label,
        "confidence": confidence,
        "uncertainty": 1.0 - confidence,
        "backend": "cqt_chroma_template",
    }


def _estimate_chords(chroma: np.ndarray, frame_times: np.ndarray) -> list[dict[str, Any]]:
    if chroma.size == 0 or chroma.shape[1] == 0:
        return []
    templates = np.zeros((24, 12), dtype=np.float32)
    for root in range(12):
        templates[root, [root, (root + 4) % 12, (root + 7) % 12]] = 1.0
        templates[12 + root, [root, (root + 3) % 12, (root + 7) % 12]] = 1.0
    frame_labels: list[str] = []
    frame_confidences: list[float] = []
    energy_threshold = max(float(np.percentile(np.sum(chroma, axis=0), 10)) * 0.25, 1e-6)
    for frame in chroma.T:
        if float(np.sum(frame)) <= energy_threshold:
            frame_labels.append("N")
            frame_confidences.append(0.0)
            continue
        scores = np.asarray([_cosine(frame, template) for template in templates])
        best = int(np.argmax(scores))
        frame_labels.append(CHORD_VOCABULARY[best])
        frame_confidences.append(float(np.clip(scores[best], 0.0, 1.0)))
    hop = float(np.median(np.diff(frame_times))) if frame_times.size > 1 else 0.0
    events: list[dict[str, Any]] = []
    start = 0
    for index in range(1, len(frame_labels) + 1):
        if index < len(frame_labels) and frame_labels[index] == frame_labels[start]:
            continue
        confidence = float(np.mean(frame_confidences[start:index]))
        end_sec = float(frame_times[index - 1] + hop)
        events.append(
            {
                "start_sec": float(frame_times[start]),
                "end_sec": end_sec,
                "label": frame_labels[start],
                "confidence": confidence,
                "uncertainty": 1.0 - confidence,
                "backend": "cqt_chroma_template",
            }
        )
        start = index
    return events


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= np.finfo(np.float32).eps:
        return 0.0
    return float(np.dot(left, right) / denominator)
