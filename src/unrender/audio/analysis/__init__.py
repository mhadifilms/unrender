"""Canonical deterministic audio analysis."""

from __future__ import annotations

from pathlib import Path

from .discovery import AudioInputs, DiscoveredAudio, discover_audio_inputs
from .io import hash_file, iter_audio, probe_audio
from .models import (
    AnalysisProfile,
    AnalysisSettings,
    AnalyzerProvenance,
    AnalyzerStatus,
    AudioAnalysisBundle,
    AudioArtifact,
    AudioMetadata,
    FeatureDescriptor,
    FeatureKind,
    Timebase,
)
from .pipeline import analyze_audio_run, load_audio_analysis
from .postprocess import (
    compute_masking_risk,
    compute_self_similarity,
    find_repeated_motifs,
)


def profile(
    name: AnalysisProfile = "core",
    *,
    analysis_dir: Path | str | None = None,
) -> AnalysisSettings:
    """Create settings for a named deterministic analysis profile."""
    return AnalysisSettings.for_profile(name, analysis_dir=analysis_dir)


__all__ = [
    "AnalysisProfile",
    "AnalysisSettings",
    "AnalyzerProvenance",
    "AnalyzerStatus",
    "AudioAnalysisBundle",
    "AudioArtifact",
    "AudioInputs",
    "AudioMetadata",
    "DiscoveredAudio",
    "FeatureDescriptor",
    "FeatureKind",
    "Timebase",
    "analyze_audio_run",
    "compute_masking_risk",
    "compute_self_similarity",
    "discover_audio_inputs",
    "find_repeated_motifs",
    "hash_file",
    "iter_audio",
    "load_audio_analysis",
    "probe_audio",
    "profile",
]
