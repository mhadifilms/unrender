"""Canonical audio-analysis orchestration and analyzer-level caching."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from unrender.project import RunPaths

from .discovery import AudioInputs, discover_audio_inputs
from .features import ANALYZER_VERSION, analyze_audio_features
from .io import hash_file, probe_audio
from .models import (
    AnalysisSettings,
    AnalyzerProvenance,
    AnalyzerStatus,
    AudioAnalysisBundle,
    AudioArtifact,
    FeatureDescriptor,
    _atomic_json,
)
from .postprocess import (
    POSTPROCESSOR_VERSION,
    postprocess_audio_features,
)
from .providers import Profile, ProviderPolicy, default_registry
from .providers.execution import run_installed_providers


def analyze_audio_run(
    run: RunPaths | Path | str,
    inputs: AudioInputs = None,
    settings: AnalysisSettings | None = None,
    force: bool = False,
) -> AudioAnalysisBundle:
    """Analyze discovered run audio and atomically write ``audio_analysis.json``."""
    run_paths = run if isinstance(run, RunPaths) else RunPaths.from_path(run)
    active = settings or AnalysisSettings()
    analysis_dir = (
        Path(active.analysis_dir).expanduser().resolve()
        if active.analysis_dir is not None
        else run_paths.audio_analysis_dir
    )
    analysis_dir.mkdir(parents=True, exist_ok=True)
    provider_preflight: dict[str, Any] | None = None
    provider_policy = ProviderPolicy(
        permissive_only=active.permissive_only,
        allow_gated=active.allow_gated,
        explicit_opt_ins=frozenset(active.provider_opt_ins),
    )
    if active.profile in ("auto", "full"):
        provider_profile = Profile(active.profile)
        report = default_registry(enabled_gated=active.provider_opt_ins).preflight(
            provider_profile,
            configured_providers=active.providers or None,
            policy=provider_policy,
            raise_on_failure=provider_profile is Profile.FULL,
        )
        provider_preflight = {
            "profile": report.profile.value,
            "strict": report.strict,
            "ok": report.ok,
            "configured_providers": list(report.configured_providers),
            "statuses": [status.as_dict() for status in report.statuses],
        }

    artifacts: list[AudioArtifact] = []
    features: list[FeatureDescriptor] = []
    statuses: list[AnalyzerStatus] = []
    core_keys: list[str] = []

    for discovered in discover_audio_inputs(run_paths, inputs):
        try:
            digest = hash_file(discovered.path)
            metadata = probe_audio(discovered.path, ffmpeg=active.ffmpeg)
        except Exception as exc:
            provenance = AnalyzerProvenance(
                analyzer="audio_probe",
                version="1.0",
                cache_key=_cache_key({"path": str(discovered.path), "version": "1.0"}),
                input_hashes=(),
                config={},
            )
            statuses.append(
                AnalyzerStatus(
                    id=f"probe:{discovered.id}",
                    status="failed",
                    provenance=provenance,
                    error=_error_text(exc),
                )
            )
            continue

        artifact = AudioArtifact(
            id=discovered.id,
            role=discovered.role,
            path=str(discovered.path),
            hash=digest,
            audio=metadata,
            discovered_from=discovered.discovered_from,
            metadata=discovered.metadata,
        )
        artifacts.append(artifact)
        config = _canonical(active.cache_config())
        cache_key = _cache_key(
            {
                "analyzer": "deterministic_features",
                "version": ANALYZER_VERSION,
                "input_hash": digest,
                "config": config,
            }
        )
        core_keys.append(cache_key)
        provenance = AnalyzerProvenance(
            analyzer="deterministic_features",
            version=ANALYZER_VERSION,
            cache_key=cache_key,
            input_hashes=(digest,),
            config=config,
        )
        cache_path = analysis_dir / "cache" / "features" / f"{artifact.id}.json"
        cached = None if force else _load_cached_features(cache_path, cache_key, analysis_dir)
        if cached is not None:
            features.extend(cached)
            statuses.append(
                AnalyzerStatus(
                    id=f"features:{artifact.id}",
                    status="cached",
                    provenance=provenance,
                    feature_ids=tuple(item.id for item in cached),
                )
            )
            continue
        try:
            generated = analyze_audio_features(
                discovered.path,
                artifact_id=artifact.id,
                metadata=metadata,
                settings=active,
                analysis_dir=analysis_dir,
            )
            features.extend(generated)
            _write_feature_cache(cache_path, cache_key, generated)
            statuses.append(
                AnalyzerStatus(
                    id=f"features:{artifact.id}",
                    status="completed",
                    provenance=provenance,
                    feature_ids=tuple(item.id for item in generated),
                )
            )
        except Exception as exc:
            statuses.append(
                AnalyzerStatus(
                    id=f"features:{artifact.id}",
                    status="failed",
                    provenance=provenance,
                    error=_error_text(exc),
                )
            )

    dialogue_hash = (
        hash_file(run_paths.dialogue_lines_json) if run_paths.dialogue_lines_json.is_file() else ""
    )
    post_config = _canonical(active.cache_config())
    post_key = _cache_key(
        {
            "analyzer": "postprocess",
            "version": POSTPROCESSOR_VERSION,
            "core_keys": core_keys,
            "dialogue_lines_hash": dialogue_hash,
            "config": post_config,
        }
    )
    post_provenance = AnalyzerProvenance(
        analyzer="postprocess",
        version=POSTPROCESSOR_VERSION,
        cache_key=post_key,
        input_hashes=tuple(artifact.hash for artifact in artifacts)
        + ((dialogue_hash,) if dialogue_hash else ()),
        config=post_config,
    )
    post_cache_path = analysis_dir / "cache" / "postprocess.json"
    cached_post = None if force else _load_cached_features(post_cache_path, post_key, analysis_dir)
    if cached_post is not None:
        features.extend(cached_post)
        statuses.append(
            AnalyzerStatus(
                id="postprocess",
                status="cached",
                provenance=post_provenance,
                feature_ids=tuple(item.id for item in cached_post),
            )
        )
    elif not artifacts:
        statuses.append(
            AnalyzerStatus(
                id="postprocess",
                status="skipped",
                provenance=post_provenance,
                error="no probeable audio artifacts",
            )
        )
    else:
        try:
            generated_post = postprocess_audio_features(
                run=run_paths,
                artifacts=artifacts,
                features=features,
                settings=active,
                analysis_dir=analysis_dir,
            )
            features.extend(generated_post)
            _write_feature_cache(post_cache_path, post_key, generated_post)
            statuses.append(
                AnalyzerStatus(
                    id="postprocess",
                    status="completed",
                    provenance=post_provenance,
                    feature_ids=tuple(item.id for item in generated_post),
                )
            )
        except Exception as exc:
            statuses.append(
                AnalyzerStatus(
                    id="postprocess",
                    status="failed",
                    provenance=post_provenance,
                    error=_error_text(exc),
                )
            )

    provider_results: list[dict[str, Any]] = []
    if active.profile in ("auto", "full") and artifacts:
        try:
            model_features, provider_results = run_installed_providers(
                artifacts=artifacts,
                settings=active,
                analysis_dir=analysis_dir,
                policy=provider_policy,
            )
            features.extend(model_features)
        except Exception as exc:
            provider_results.append(
                {
                    "provider_id": "execution",
                    "capability": "model_providers",
                    "state": "failed",
                    "error": _error_text(exc),
                }
            )

    bundle = AudioAnalysisBundle(
        artifacts=tuple(artifacts),
        features=tuple(features),
        analyzers=tuple(statuses),
        metadata={
            "profile": active.profile,
            "analysis_dir": str(analysis_dir),
            "array_storage": "relative_npy",
            "deterministic": True,
            "provider_preflight": provider_preflight,
            "provider_results": provider_results,
        },
    )
    bundle.write(
        analysis_dir / "audio_analysis.json"
        if active.analysis_dir is not None
        else run_paths.audio_analysis_json
    )
    return bundle


def load_audio_analysis(path_or_run: RunPaths | Path | str) -> AudioAnalysisBundle:
    """Load a bundle from its JSON path, analysis directory, or run root."""
    candidates: tuple[Path, ...]
    if isinstance(path_or_run, RunPaths):
        candidates = (
            path_or_run.audio_analysis_json,
            path_or_run.root / "audio_analysis.json",
        )
    else:
        path = Path(path_or_run).expanduser()
        if path.is_file():
            return AudioAnalysisBundle.read(path)
        candidates = (
            path / "audio_analysis.json",
            path / "analysis" / "audio" / "audio_analysis.json",
            path / "audio" / "analysis" / "audio_analysis.json",
        )
    bundle_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if bundle_path is None:
        raise FileNotFoundError(f"audio_analysis.json not found under {path_or_run}")
    return AudioAnalysisBundle.read(bundle_path)


def _write_feature_cache(
    path: Path,
    cache_key: str,
    features: list[FeatureDescriptor],
) -> None:
    _atomic_json(
        path,
        {
            "cache_key": cache_key,
            "features": [asdict(feature) for feature in features],
        },
    )


def _load_cached_features(
    path: Path,
    cache_key: str,
    analysis_dir: Path,
) -> list[FeatureDescriptor] | None:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or payload.get("cache_key") != cache_key:
            return None
        raw_features = payload.get("features")
        if not isinstance(raw_features, list):
            return None
        features = [FeatureDescriptor.from_dict(item) for item in raw_features]
        if any(
            feature.path is not None and not (analysis_dir / feature.path).is_file()
            for feature in features
        ):
            return None
        return features
    except (OSError, TypeError, ValueError):
        return None


def _cache_key(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _error_text(exc: Exception) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__
