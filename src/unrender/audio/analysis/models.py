"""Typed contracts for the canonical ``audio_analysis.json`` bundle."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = "1.0"

FeatureKind = Literal["scalar", "series", "events", "matrix", "embeddings", "image"]
AnalyzerState = Literal["completed", "cached", "failed", "skipped"]
AnalysisProfile = Literal["core", "auto", "full", "quick", "standard", "deep"]


@dataclass(frozen=True)
class AudioMetadata:
    sample_rate: int
    channels: int
    frames: int
    duration_sec: float
    codec: str
    sample_format: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AudioMetadata:
        return cls(
            sample_rate=int(value["sample_rate"]),
            channels=int(value["channels"]),
            frames=int(value["frames"]),
            duration_sec=float(value["duration_sec"]),
            codec=str(value.get("codec", "")),
            sample_format=str(value.get("sample_format", "")),
        )


@dataclass(frozen=True)
class AudioArtifact:
    """A role-bearing analyzed audio input."""

    id: str
    role: str
    path: str
    hash: str
    audio: AudioMetadata
    discovered_from: str = "explicit"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AudioArtifact:
        return cls(
            id=str(value["id"]),
            role=str(value["role"]),
            path=str(value["path"]),
            hash=str(value["hash"]),
            audio=AudioMetadata.from_dict(value["audio"]),
            discovered_from=str(value.get("discovered_from", "explicit")),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class Timebase:
    start_sec: float
    hop_sec: float
    frame_sec: float
    count: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Timebase:
        return cls(
            start_sec=float(value.get("start_sec", 0.0)),
            hop_sec=float(value["hop_sec"]),
            frame_sec=float(value.get("frame_sec", value["hop_sec"])),
            count=int(value["count"]),
        )


@dataclass(frozen=True)
class FeatureDescriptor:
    """A small inline feature or a reference to an atomic dense array."""

    id: str
    name: str
    kind: FeatureKind
    artifact_id: str | None = None
    unit: str = ""
    value: Any = None
    path: str | None = None
    dtype: str | None = None
    shape: tuple[int, ...] = ()
    timebase: Timebase | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FeatureDescriptor:
        raw_timebase = value.get("timebase")
        return cls(
            id=str(value["id"]),
            name=str(value["name"]),
            kind=str(value["kind"]),  # type: ignore[arg-type]
            artifact_id=str(value["artifact_id"]) if value.get("artifact_id") else None,
            unit=str(value.get("unit", "")),
            value=value.get("value"),
            path=str(value["path"]) if value.get("path") else None,
            dtype=str(value["dtype"]) if value.get("dtype") else None,
            shape=tuple(int(item) for item in value.get("shape", ())),
            timebase=(
                Timebase.from_dict(raw_timebase) if isinstance(raw_timebase, Mapping) else None
            ),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class AnalyzerProvenance:
    analyzer: str
    version: str
    cache_key: str
    input_hashes: tuple[str, ...]
    config: dict[str, Any] = field(default_factory=dict)
    implementation: str = "unrender.audio.analysis"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AnalyzerProvenance:
        return cls(
            analyzer=str(value["analyzer"]),
            version=str(value["version"]),
            cache_key=str(value["cache_key"]),
            input_hashes=tuple(str(item) for item in value.get("input_hashes", ())),
            config=dict(value.get("config", {})),
            implementation=str(value.get("implementation", "unrender.audio.analysis")),
        )


@dataclass(frozen=True)
class AnalyzerStatus:
    id: str
    status: AnalyzerState
    provenance: AnalyzerProvenance
    feature_ids: tuple[str, ...] = ()
    error: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AnalyzerStatus:
        return cls(
            id=str(value["id"]),
            status=str(value["status"]),  # type: ignore[arg-type]
            provenance=AnalyzerProvenance.from_dict(value["provenance"]),
            feature_ids=tuple(str(item) for item in value.get("feature_ids", ())),
            error=str(value["error"]) if value.get("error") else None,
        )


@dataclass(frozen=True)
class AudioAnalysisBundle:
    artifacts: tuple[AudioArtifact, ...] = ()
    features: tuple[FeatureDescriptor, ...] = ()
    analyzers: tuple[AnalyzerStatus, ...] = ()
    schema_version: str = SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AudioAnalysisBundle:
        return cls(
            schema_version=str(value.get("schema_version", SCHEMA_VERSION)),
            artifacts=tuple(AudioArtifact.from_dict(item) for item in value.get("artifacts", ())),
            features=tuple(FeatureDescriptor.from_dict(item) for item in value.get("features", ())),
            analyzers=tuple(AnalyzerStatus.from_dict(item) for item in value.get("analyzers", ())),
            metadata=dict(value.get("metadata", {})),
        )

    def write(self, path: Path | str) -> Path:
        destination = Path(path)
        _atomic_json(destination, self.to_dict())
        return destination

    @classmethod
    def read(cls, path: Path | str) -> AudioAnalysisBundle:
        with Path(path).open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError("audio analysis bundle must be a JSON object")
        return cls.from_dict(value)

    def feature(self, name: str, artifact_id: str | None = None) -> FeatureDescriptor | None:
        return next(
            (
                feature
                for feature in self.features
                if feature.name == name
                and (artifact_id is None or feature.artifact_id == artifact_id)
            ),
            None,
        )


@dataclass(frozen=True)
class AnalysisSettings:
    """Deterministic analysis controls.

    ``analysis_dir`` is intentionally optional so callers can use a run's
    conventional ``audio/analysis`` directory without changing ``RunPaths``.
    """

    profile: AnalysisProfile = "core"
    analysis_dir: Path | str | None = None
    frame_hop_sec: float = 0.1
    decode_block_frames: int = 65_536
    mel_bands: int = 32
    mfcc_count: int = 13
    bark_bands: int = 24
    rolloff_fraction: float = 0.85
    similarity_scales_sec: tuple[float, ...] = (1.0, 4.0, 16.0)
    max_similarity_frames: int = 2_048
    similarity_block_frames: int = 256
    motif_threshold: float = 0.88
    ffmpeg: str = "ffmpeg"
    providers: tuple[str, ...] = ()
    permissive_only: bool = True
    allow_gated: bool = False
    provider_opt_ins: tuple[str, ...] = ()
    device: str = "cpu"
    language: str | None = None
    whisper_model: str = "large-v3"
    compute_type: str = "int8"
    clap_window_sec: float = 10.0

    def __post_init__(self) -> None:
        if self.frame_hop_sec <= 0.0:
            raise ValueError("frame_hop_sec must be positive")
        if self.decode_block_frames <= 0:
            raise ValueError("decode_block_frames must be positive")
        if min(self.mel_bands, self.mfcc_count, self.bark_bands) <= 0:
            raise ValueError("feature band and coefficient counts must be positive")
        if not 0.0 < self.rolloff_fraction < 1.0:
            raise ValueError("rolloff_fraction must be between zero and one")
        if not self.similarity_scales_sec or any(
            scale <= 0.0 for scale in self.similarity_scales_sec
        ):
            raise ValueError("similarity_scales_sec must contain positive scales")
        if min(self.max_similarity_frames, self.similarity_block_frames) <= 0:
            raise ValueError("similarity frame limits must be positive")
        if not 0.0 <= self.motif_threshold <= 1.0:
            raise ValueError("motif_threshold must be between zero and one")
        if self.clap_window_sec <= 0.0:
            raise ValueError("clap_window_sec must be positive")

    @classmethod
    def for_profile(
        cls,
        profile: AnalysisProfile,
        *,
        analysis_dir: Path | str | None = None,
    ) -> AnalysisSettings:
        if profile == "quick":
            return cls(
                profile=profile,
                analysis_dir=analysis_dir,
                frame_hop_sec=0.2,
                mel_bands=20,
                mfcc_count=10,
                similarity_scales_sec=(2.0, 8.0),
                max_similarity_frames=1_024,
            )
        if profile == "deep":
            return cls(
                profile=profile,
                analysis_dir=analysis_dir,
                frame_hop_sec=0.05,
                mel_bands=48,
                mfcc_count=20,
                similarity_scales_sec=(0.5, 2.0, 8.0, 32.0),
                max_similarity_frames=4_096,
            )
        return cls(profile=profile, analysis_dir=analysis_dir)

    def cache_config(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("analysis_dir")
        value.pop("ffmpeg")
        return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
