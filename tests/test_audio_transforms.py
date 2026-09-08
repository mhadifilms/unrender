from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from unrender.audio.audio_io import read_audio
from unrender.audio.transforms import RecipeSettings, list_recipes, run_recipe


def _write(path: Path, audio: np.ndarray, sample_rate: int = 8000) -> Path:
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    wavfile.write(path, sample_rate, data)
    return path


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tone(
    frequency: float,
    *,
    seconds: float = 1.0,
    sample_rate: int = 8000,
    amplitude: float = 0.2,
    channels: int = 2,
) -> np.ndarray:
    time = np.arange(int(seconds * sample_rate)) / sample_rate
    mono = amplitude * np.sin(2.0 * np.pi * frequency * time)
    return np.repeat(mono[:, None], channels, axis=1).astype(np.float32)


def test_ducking_is_non_destructive_and_writes_smooth_bounded_automation(
    tmp_path: Path,
) -> None:
    sample_rate = 8000
    dialogue = np.zeros((sample_rate, 1), dtype=np.float32)
    dialogue[2000:5000] = 0.5
    dx = _write(tmp_path / "dx.wav", dialogue, sample_rate)
    mx = _write(tmp_path / "mx.wav", _tone(220.0, channels=2), sample_rate)
    original_hashes = (_hash(dx), _hash(mx))

    result = run_recipe(
        "dialogue-aware-ducking",
        {"dx": dx, "mx": mx},
        tmp_path / "ducked",
        params={"depth_db": 12.0, "attack_ms": 30.0, "release_ms": 200.0},
    )

    assert (_hash(dx), _hash(mx)) == original_hashes
    gain = np.load(result.automation["mx_gain"])
    assert float(np.min(gain)) >= 10.0 ** (-12.0 / 20.0) - 1e-6
    assert float(np.max(gain)) <= 1.0
    assert float(np.max(np.abs(np.diff(gain)))) < 0.05
    assert {"dx", "mx", "mix"} <= set(result.outputs)
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["provenance"]["deterministic"] is True
    assert manifest["inputs"]["dx"]["sha256"] == original_hashes[0]
    assert manifest["automation"]["mx_gain"]["path"].endswith(".npy")


def test_spectral_unmasking_reduces_overlap_and_preserves_layout(tmp_path: Path) -> None:
    dx_audio = _tone(500.0, amplitude=0.3, channels=1)
    background_audio = _tone(500.0, amplitude=0.3, channels=2) + _tone(
        1500.0,
        amplitude=0.05,
        channels=2,
    )
    dx = _write(tmp_path / "dx.wav", dx_audio)
    background = _write(tmp_path / "background.wav", background_audio)

    result = run_recipe(
        "spectral_unmasking",
        {"dx": dx, "background": background},
        tmp_path / "unmasked",
        params={"max_reduction_db": 12.0},
    )

    rendered, rate = read_audio(result.outputs["background"])
    assert rate == 8000
    assert rendered.shape == background_audio.shape
    assert np.sqrt(np.mean(rendered**2)) < np.sqrt(np.mean(background_audio**2))
    assert (
        result.metrics["masking"]["overlap_energy_after"]
        < result.metrics["masking"]["overlap_energy_before"]
    )
    mask = np.load(result.automation["spectral_gain_mask"])
    assert np.min(mask) >= 10.0 ** (-12.0 / 20.0) - 1e-6
    assert np.max(mask) <= 1.0


def test_dme_rebalance_enforces_peak_ceiling(tmp_path: Path) -> None:
    roles = {
        "d": _write(tmp_path / "d.wav", _tone(180.0, amplitude=0.8)),
        "m": _write(tmp_path / "m.wav", _tone(360.0, amplitude=0.8)),
        "e": _write(tmp_path / "e.wav", _tone(720.0, amplitude=0.8)),
    }
    result = run_recipe(
        "d/m/e rebalance",
        roles,
        tmp_path / "rebalance",
        settings=RecipeSettings(peak_ceiling_dbfs=-3.0),
        params={"d_gain_db": 6.0, "m_gain_db": 6.0, "e_gain_db": 6.0},
    )

    mix, _ = read_audio(result.outputs["mix"])
    assert np.max(np.abs(mix)) <= 10.0 ** (-3.0 / 20.0) + 1e-6
    assert result.metrics["rebalance"]["peak_protection_gain"] < 1.0


@pytest.mark.parametrize(
    ("recipe", "roles", "output_name"),
    [
        ("timbre_filter_modulation", ("source",), "filtered"),
        ("granular_resynthesis", ("source",), "resynthesized"),
        ("stereo_spatial_animation", ("source",), "animated"),
        ("event_beat_modulation", ("source",), "modulated"),
        ("cross_stem_mapping", ("control", "target"), "target"),
    ],
)
def test_creative_dry_identity_duration_layout_and_automation(
    tmp_path: Path,
    recipe: str,
    roles: tuple[str, ...],
    output_name: str,
) -> None:
    source_audio = _tone(330.0, seconds=0.5, channels=2)
    source = _write(tmp_path / f"{recipe}_source.wav", source_audio)
    role_paths = dict.fromkeys(roles, source)
    result = run_recipe(
        recipe,
        role_paths,
        tmp_path / recipe,
        settings={"dry_wet": 0.0, "seed": 73},
        analysis_bundle={"events": [{"start": 0.1}], "novelty": [0.0, 1.0, 0.3]},
    )

    rendered, rate = read_audio(result.outputs[output_name])
    assert rate == 8000
    assert rendered.shape == source_audio.shape
    np.testing.assert_array_equal(rendered, source_audio)
    assert result.automation
    assert all(path.suffix == ".npy" and path.is_file() for path in result.automation.values())
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["latency_samples"] == 0
    assert manifest["tail_samples"] == 0


def test_seeded_granular_is_deterministic(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    audio = rng.normal(0.0, 0.1, (4000, 2)).astype(np.float32)
    source = _write(tmp_path / "source.wav", audio)
    settings = {"seed": 991, "dry_wet": 1.0}
    analysis = {"novelty": [0.0, 1.0, 0.2, 0.8], "structure": [0.0, 0.3, 1.0]}

    first = run_recipe(
        "granular",
        {"source": source},
        tmp_path / "first",
        settings=settings,
        analysis_bundle=analysis,
    )
    second = run_recipe(
        "granular",
        {"source": source},
        tmp_path / "second",
        settings=settings,
        analysis_bundle=analysis,
    )

    first_audio, _ = read_audio(first.outputs["resynthesized"])
    second_audio, _ = read_audio(second.outputs["resynthesized"])
    np.testing.assert_array_equal(first_audio, second_audio)
    np.testing.assert_array_equal(
        np.load(first.automation["grain_map"]),
        np.load(second.automation["grain_map"]),
    )
    assert first.provenance["fingerprint"] == second.provenance["fingerprint"]


def test_perspective_matching_is_conservative_and_reports_uncertainty(tmp_path: Path) -> None:
    source_audio = _tone(300.0, seconds=1.0, amplitude=0.1, channels=2)
    reference_audio = _tone(900.0, seconds=0.5, amplitude=0.3, channels=1)
    source = _write(tmp_path / "perspective_source.wav", source_audio)
    reference = _write(tmp_path / "perspective_reference.wav", reference_audio)
    source_hash = _hash(source)

    result = run_recipe(
        "perspective_matching",
        {"source": source, "reference": reference},
        tmp_path / "perspective",
        params={"max_level_adjust_db": 3.0, "max_eq_db": 2.0},
    )

    matched, rate = read_audio(result.outputs["matched"])
    assert rate == 8000
    assert matched.shape == source_audio.shape
    assert _hash(source) == source_hash
    assert abs(result.report["level_adjustment_db"]) <= 3.0
    assert result.report["eq_range_db"][0] >= -2.0
    assert result.report["eq_range_db"][1] <= 2.0
    assert 0.0 <= result.report["uncertainty"]["score"] <= 1.0


def test_review_regions_include_untouched_bypass_and_click_safe_extracts(
    tmp_path: Path,
) -> None:
    audio = _tone(440.0, seconds=1.0, channels=1)
    source = _write(tmp_path / "source.wav", audio)
    result = run_recipe(
        "review_regions",
        {"source": source},
        tmp_path / "review",
        params={"padding_seconds": 0.05, "fade_ms": 10.0},
        analysis_bundle={"review_regions": [{"start": 0.25, "end": 0.5}]},
    )

    bypass, _ = read_audio(result.outputs["bypass"])
    extract, _ = read_audio(result.outputs["review_region_001"])
    np.testing.assert_array_equal(bypass, audio)
    assert abs(float(extract[0, 0])) < 1e-7
    assert abs(float(extract[-1, 0])) < 1e-7
    assert result.report["bypass_is_sample_identical"] is True
    assert "review_region_variants" in list_recipes(category="corrective")
