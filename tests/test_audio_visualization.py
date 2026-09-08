from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from unrender.audio.visualization import (
    RendererRegistry,
    VisualMapping,
    available_renderers,
    encode_overlay_mov,
    render_audio_analysis,
    render_overlay_sequence,
)


def _synthetic_bundle(tmp_path: Path) -> Path:
    frame_count = 32
    phase = np.linspace(0.0, np.pi * 2.0, frame_count)
    arrays = {
        "roles.npy": np.column_stack(
            (
                np.clip(np.sin(phase), 0, 1),
                np.clip(np.cos(phase), 0, 1),
                np.linspace(0.1, 0.8, frame_count),
            )
        ),
        "timbre.npy": np.column_stack(
            (
                np.linspace(0.0, 1.0, frame_count),
                np.linspace(1.0, 0.0, frame_count),
                np.full(frame_count, 0.35),
            )
        ),
        "stereo.npy": np.column_stack((np.sin(phase), np.abs(np.cos(phase)))),
        "masking.npy": np.abs(np.sin(phase * 1.7)),
        "loudness.npy": -30.0 + np.sin(phase) * 8.0,
        "dynamics.npy": 5.0 + np.cos(phase) * 3.0,
        "dominance.npy": np.column_stack(
            (
                np.linspace(0.8, 0.1, frame_count),
                np.linspace(0.1, 0.8, frame_count),
                np.full(frame_count, 0.2),
            )
        ),
        "bands.npy": np.abs(np.random.default_rng(4).normal(size=(frame_count, 6))),
    }
    embedding = np.column_stack((np.sin(phase), np.cos(phase), np.sin(phase * 0.5)))
    arrays["similarity.npy"] = embedding @ embedding.T
    for name, array in arrays.items():
        np.save(tmp_path / name, array.astype(np.float32))
    bundle = {
        "version": "1.0",
        "title": "</script><script>window.injected=true</script>",
        "duration_sec": 4.0,
        "source_names": ["Dialogue", "Music", "Effects"],
        "features": {
            "role_activity": {"path": "roles.npy"},
            "timbral_color": {"path": "timbre.npy"},
            "stereo_space": {"path": "stereo.npy"},
            "masking": {"path": "masking.npy"},
            "loudness": {"path": "loudness.npy"},
            "dynamics": {"path": "dynamics.npy"},
            "source_dominance": {"path": "dominance.npy"},
            "band_power": {"path": "bands.npy"},
        },
        "structure": {
            "self_similarity": {"path": "similarity.npy"},
            "boundaries": [1.0, 2.5],
            "motifs": [
                {"start_sec": 0.4, "end_sec": 1.1, "label": "A"},
                {"start_sec": 2.4, "end_sec": 3.2, "label": "A'"},
            ],
        },
        "events": [
            {"start_sec": 0.1, "end_sec": 1.2, "role": "dialogue", "label": "Line"},
            {"start_sec": 0.8, "end_sec": 3.8, "role": "music", "label": "Score"},
            {"start_sec": 2.2, "end_sec": 2.7, "role": "fx", "label": "Door"},
        ],
    }
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    return path


def _small_mapping() -> VisualMapping:
    return VisualMapping.from_dict(
        {
            "version": 1,
            "geometry": {
                "timeline_width": 420,
                "timeline_height": 300,
                "structure_size": 240,
                "territory_width": 360,
                "territory_height": 220,
                "poster_width": 520,
                "poster_height": 440,
                "overlay_width": 320,
                "overlay_height": 72,
                "overlay_fps": 2,
                "overlay_max_fps": 4,
                "overlay_max_duration": 1,
                "margin": 10,
            },
        }
    )


def test_mapping_roundtrip_and_safe_defaults(tmp_path: Path) -> None:
    mapping = VisualMapping.from_dict(
        {
            "version": 1,
            "colors": {"dialogue": "#123456", "music": "not-a-color"},
            "geometry": {"timeline_width": 333, "timeline_height": -1},
            "layers": {"events": False, "masking": "no"},
            "rules": {"events": {"opacity": 5, "palette": ["#abcdef", "bad"]}},
        }
    )
    assert mapping.colors["dialogue"] == "#123456"
    assert mapping.colors["music"] != "not-a-color"
    assert mapping.geometry["timeline_width"] == 333
    assert mapping.geometry["timeline_height"] > 0
    assert mapping.layers["events"] is False
    assert mapping.layers["masking"] is True
    assert mapping.rules["events"].opacity == 1.0
    assert mapping.rules["events"].palette == ("#abcdef",)

    mapping_path = mapping.save(tmp_path / "mapping.json")
    assert VisualMapping.load(mapping_path).to_dict() == mapping.to_dict()
    with pytest.raises(ValueError, match="newer"):
        VisualMapping.from_dict({"version": 999})


def test_dispatcher_renders_static_views_and_html(tmp_path: Path) -> None:
    bundle_path = _synthetic_bundle(tmp_path)
    output = tmp_path / "renders"
    results = render_audio_analysis(
        bundle_path,
        ["timeline", "structure", "territory", "poster", "html"],
        output,
        _small_mapping(),
        source_paths=[tmp_path / "mix.wav"],
        alternate_paths=[tmp_path / "alternate.wav"],
    )
    expected_sizes = {
        "timeline_atlas": (420, 300),
        "structure_map": (240, 240),
        "mix_territory": (360, 220),
        "poster": (520, 440),
    }
    for name, size in expected_sizes.items():
        assert results[name].is_file()
        with Image.open(results[name]) as image:
            assert image.size == size
            assert image.mode == "RGBA"

    document = results["html"].read_text(encoding="utf-8")
    assert '<audio id="audio-player" controls' in document
    assert 'id="scrubber"' in document
    assert 'id="zoom"' in document
    assert 'class="layer-toggle"' in document
    assert 'id="source-select"' in document
    assert 'id="a-button"' in document and 'id="b-button"' in document
    assert 'id="json-file"' in document
    assert "</script><script>window.injected" not in document
    assert "\\u003c/script\\u003e" in document


def test_registry_contract() -> None:
    registry = RendererRegistry()

    @registry.register("custom")
    def custom(context):
        return context.output_dir / "custom.txt"

    assert registry.get("CUSTOM") is custom
    assert registry.names() == ("custom",)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("custom", custom)
    assert {
        "timeline_atlas",
        "structure_map",
        "mix_territory",
        "poster",
        "html",
        "overlay",
        "spectral",
    }.issubset(available_renderers())


def test_overlay_sequence_has_transparency_and_optional_mov(tmp_path: Path) -> None:
    bundle_path = _synthetic_bundle(tmp_path)
    frame_dir = tmp_path / "frames"
    frames = render_overlay_sequence(
        bundle_path,
        frame_dir,
        _small_mapping(),
        fps=2,
        duration_sec=1,
        max_frames=2,
    )
    assert len(frames) == 2
    with Image.open(frames[0]) as image:
        assert image.size == (320, 72)
        assert image.mode == "RGBA"
        alpha_min, alpha_max = image.getchannel("A").getextrema()
        assert alpha_min == 0
        assert alpha_max > 0

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not installed")
    mov = encode_overlay_mov(frame_dir, tmp_path / "overlay.mov", fps=2, ffmpeg=ffmpeg)
    assert mov.is_file()
    assert mov.stat().st_size > 0
