from __future__ import annotations

import json
import time
import wave
from pathlib import Path

import numpy as np

from unrender.audio.analysis import (
    AnalysisSettings,
    AudioAnalysisBundle,
    analyze_audio_run,
    compute_masking_risk,
    compute_self_similarity,
    discover_audio_inputs,
    find_repeated_motifs,
    load_audio_analysis,
)
from unrender.project import RunPaths


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int = 8_000) -> Path:
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(values, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(values.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return path


def _tone(frequency: float, duration: float = 0.8, sample_rate: int = 8_000) -> np.ndarray:
    times = np.arange(round(duration * sample_rate)) / sample_rate
    return (0.25 * np.sin(2.0 * np.pi * frequency * times)).astype(np.float32)


def _settings(path: Path) -> AnalysisSettings:
    return AnalysisSettings(
        analysis_dir=path,
        frame_hop_sec=0.05,
        mel_bands=16,
        mfcc_count=8,
        bark_bands=12,
        similarity_scales_sec=(0.2,),
        max_similarity_frames=64,
        similarity_block_frames=8,
    )


def _dense(bundle: AudioAnalysisBundle, name: str, analysis_dir: Path) -> np.ndarray:
    descriptor = next(feature for feature in bundle.features if feature.name == name)
    assert descriptor.path is not None
    return np.load(analysis_dir / descriptor.path, allow_pickle=False)


def test_bundle_schema_round_trip(tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()
    source = _write_wav(tmp_path / "tone.wav", _tone(440.0))
    analysis_dir = tmp_path / "analysis"

    bundle = analyze_audio_run(
        run,
        {"dialogue": source},
        _settings(analysis_dir),
    )
    loaded = load_audio_analysis(analysis_dir)
    direct = AudioAnalysisBundle.from_dict(bundle.to_dict())

    assert loaded == bundle == direct
    assert loaded.schema_version == "1.0"
    assert loaded.artifacts[0].hash
    assert {feature.kind for feature in loaded.features} >= {
        "scalar",
        "series",
        "events",
        "matrix",
        "embeddings",
    }


def test_discovery_prefers_manifest_roles_and_deduplicates(tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()
    stem = _write_wav(run.source_stems_dir / "unexpected-name.wav", _tone(220.0))
    run.source_separation_json.write_text(
        json.dumps({"stems": {"dialogue": str(stem)}}),
        encoding="utf-8",
    )

    discovered = discover_audio_inputs(run, [f"wrong-role={stem}"])

    assert len(discovered) == 1
    assert discovered[0].role == "dialogue"
    assert discovered[0].discovered_from.startswith("manifest:")


def test_feature_properties_for_synthetic_tone(tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()
    source = _write_wav(tmp_path / "tone.wav", _tone(440.0))
    analysis_dir = tmp_path / "analysis"

    bundle = analyze_audio_run(run, {"dialogue": source}, _settings(analysis_dir))
    centroid = _dense(bundle, "spectral_centroid_hz", analysis_dir)
    flatness = _dense(bundle, "spectral_flatness", analysis_dir)
    mel = _dense(bundle, "mel_band_power", analysis_dir)
    mfcc = _dense(bundle, "mfcc", analysis_dir)

    assert np.all(np.isfinite(centroid))
    assert 350.0 < float(np.median(centroid)) < 550.0
    assert float(np.median(flatness)) < 0.1
    assert mel.shape[1] == 16
    assert mfcc.shape[1] == 8
    integrated = next(
        feature for feature in bundle.features if feature.name == "integrated_loudness"
    )
    assert integrated.unit == "LUFS"
    assert np.isfinite(integrated.value)
    true_peak = next(feature for feature in bundle.features if feature.name == "true_peak_dbtp")
    plr = next(feature for feature in bundle.features if feature.name == "peak_to_loudness_ratio")
    assert true_peak.unit == "dBTP"
    assert np.isfinite(true_peak.value)
    assert plr.value > 0.0


def test_stereo_fold_down_risk_detects_antiphase(tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()
    tone = _tone(330.0)
    in_phase = _write_wav(tmp_path / "in.wav", np.column_stack((tone, tone)))
    antiphase = _write_wav(tmp_path / "anti.wav", np.column_stack((tone, -tone)))
    analysis_dir = tmp_path / "analysis"

    bundle = analyze_audio_run(
        run,
        {"in_phase": in_phase, "antiphase": antiphase},
        _settings(analysis_dir),
    )
    artifact_by_role = {artifact.role: artifact.id for artifact in bundle.artifacts}

    def values(name: str, role: str) -> np.ndarray:
        descriptor = next(
            feature
            for feature in bundle.features
            if feature.name == name and feature.artifact_id == artifact_by_role[role]
        )
        assert descriptor.path is not None
        return np.load(analysis_dir / descriptor.path, allow_pickle=False)

    assert float(np.mean(values("fold_down_risk", "antiphase"))) > 0.95
    assert float(np.mean(values("fold_down_risk", "in_phase"))) < 0.05
    assert float(np.mean(values("stereo_correlation", "antiphase"))) < -0.95


def test_masking_risk_is_monotonic() -> None:
    risks = compute_masking_risk(
        np.ones(5, dtype=np.float32),
        np.asarray([0.0, 0.1, 1.0, 10.0, 100.0], dtype=np.float32),
    )

    assert np.all(np.diff(risks) > 0.0)
    assert 0.0 <= risks[0] < risks[-1] <= 1.0


def test_dme_band_territory_and_masking_are_materialized(tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()
    analysis_dir = tmp_path / "analysis"
    inputs = {
        "dialogue": _write_wav(tmp_path / "dx.wav", _tone(300.0)),
        "music": _write_wav(tmp_path / "mx.wav", _tone(900.0)),
        "effects": _write_wav(tmp_path / "fx.wav", _tone(1800.0)),
    }

    bundle = analyze_audio_run(run, inputs, _settings(analysis_dir))
    territory = _dense(bundle, "source_dominance_by_band", analysis_dir)
    masking = _dense(bundle, "dialogue_masking_by_band", analysis_dir)

    assert territory.ndim == 3
    assert territory.shape[2] == 3
    assert masking.shape == territory.shape[:2]
    assert np.all((territory >= 0.0) & (territory <= 1.0))
    assert np.all((masking >= 0.0) & (masking <= 1.0))


def test_a_b_a_self_similarity_and_motif(tmp_path: Path) -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [1.0, 0.0],
            [0.9, 0.1],
        ],
        dtype=np.float32,
    )
    path = compute_self_similarity(embeddings, tmp_path / "similarity.npy", block_size=2)
    similarity = np.load(path, allow_pickle=False)
    motifs = find_repeated_motifs(
        similarity,
        hop_sec=1.0,
        threshold=0.95,
        min_separation_sec=3.0,
    )

    assert similarity[0, 4] > similarity[0, 2]
    assert any(
        item["first_start_sec"] == 0.0 and item["second_start_sec"] == 4.0 for item in motifs
    )


def test_cache_reuse_and_source_untouched(tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()
    source = _write_wav(tmp_path / "tone.wav", _tone(440.0))
    original = source.read_bytes()
    analysis_dir = tmp_path / "analysis"
    settings = _settings(analysis_dir)

    first = analyze_audio_run(run, {"dialogue": source}, settings)
    descriptor = next(feature for feature in first.features if feature.name == "mfcc")
    assert descriptor.path is not None
    array_path = analysis_dir / descriptor.path
    first_mtime = array_path.stat().st_mtime_ns
    time.sleep(0.002)
    second = analyze_audio_run(run, {"dialogue": source}, settings)

    assert array_path.stat().st_mtime_ns == first_mtime
    assert source.read_bytes() == original
    assert {status.status for status in second.analyzers} == {"cached"}
