from __future__ import annotations

import builtins
import json
from pathlib import Path

import numpy as np
import pytest

from unrender.editorial.scenes import (
    assign_scenes,
    dp_persistence_scores,
    fuse_features,
    load_scene_manifest,
    novelty_scores,
    scene_for_time,
    scene_manifest_entries,
    write_scenes_manifest,
)
from unrender.editorial.scenes.features import dialogue_spans, keyframe_indices
from unrender.editorial.scenes.group import FUSION_FEATURES, FUSION_WEIGHTS


def _three_scene_embeddings(rng: np.random.Generator) -> np.ndarray:
    """Three visually distinct scenes of 6 shots each (unit vectors)."""
    centers = np.eye(3, 16)
    rows = []
    for center in centers:
        for _ in range(6):
            vec = center + 0.05 * rng.standard_normal(16)
            rows.append(vec / np.linalg.norm(vec))
    return np.asarray(rows, dtype=np.float32)


def test_novelty_scores_peak_at_scene_changes() -> None:
    emb = _three_scene_embeddings(np.random.default_rng(0))
    scores = novelty_scores(emb)
    assert {int(np.argsort(scores)[-1]), int(np.argsort(scores)[-2])} == {5, 11}
    assert scores[-1] == 0.0  # final position gets no score


def test_dp_persistence_scores_cut_at_scene_changes() -> None:
    emb = _three_scene_embeddings(np.random.default_rng(1))
    scores = dp_persistence_scores(emb)
    assert scores[5] > 0.8
    assert scores[11] > 0.8
    assert scores[2] < 0.3


def test_fuse_features_selects_weight_variant_by_availability() -> None:
    emb = _three_scene_embeddings(np.random.default_rng(2))
    audio = np.abs(np.random.default_rng(3).standard_normal((18, 8))).astype(np.float32)
    spans = np.zeros(18, dtype=np.float32)
    assert fuse_features(clip_embeddings=emb, dino_embeddings=emb).variant == "visual_only"
    assert (
        fuse_features(clip_embeddings=emb, dino_embeddings=emb, audio_features=audio).variant
        == "no_dialogue"
    )
    assert (
        fuse_features(clip_embeddings=emb, dino_embeddings=emb, dialogue_spans=spans).variant
        == "no_audio"
    )
    full = fuse_features(
        clip_embeddings=emb, dino_embeddings=emb, audio_features=audio, dialogue_spans=spans
    )
    assert full.variant == "full"
    assert full.scores.shape == (18,)
    assert np.all((full.scores >= 0.0) & (full.scores <= 1.0))
    # boundaries should outrank intra-scene cuts even with noisy extras
    assert full.scores[5] > full.scores[2]


def test_fusion_weight_variants_match_feature_arity() -> None:
    lengths = {"full": 5, "no_audio": 4, "no_dialogue": 4, "visual_only": 3}
    assert set(FUSION_WEIGHTS) == set(lengths)
    for name, expected in lengths.items():
        assert len(FUSION_WEIGHTS[name]) == expected + 1  # + bias
    assert FUSION_FEATURES == ("clip_dp", "clip_nov", "dino_nov", "audio_nov", "dlg_span")


def test_assign_scenes_thresholds_and_final_shot() -> None:
    ids = [f"{i:03d}" for i in range(6)]
    scores = np.array([0.1, 0.9, 0.2, 0.2, 0.95, 0.99])
    assignment = assign_scenes(ids, scores, threshold=0.6)
    assert assignment == {"000": 1, "001": 1, "002": 2, "003": 2, "004": 2, "005": 3}
    with pytest.raises(ValueError):
        assign_scenes(ids[:3], scores)


def test_keyframe_indices_skip_cut_adjacent_frames() -> None:
    indices = keyframe_indices(100, 200)  # end-exclusive
    assert indices[0] >= 102
    assert indices[-1] <= 197
    assert keyframe_indices(10, 12) == [10, 11]  # tiny shot uses what it has
    assert keyframe_indices(10, 11) == [10]


def test_dialogue_spans_mark_lines_crossing_cuts() -> None:
    shots = [
        {"start_sec": 0.0, "end_sec": 4.0},
        {"start_sec": 4.0, "end_sec": 8.0},
        {"start_sec": 8.0, "end_sec": 12.0},
    ]
    lines = [
        {"start_sec": 3.0, "end_sec": 5.0},  # crosses the first cut
        {"start_sec": 8.5, "end_sec": 9.5},  # inside the last shot
    ]
    spans = dialogue_spans(lines, shots)
    assert spans is not None
    assert spans.tolist() == [1.0, 0.0, 0.0]
    assert dialogue_spans([], shots) is None


def test_scene_manifest_round_trip_and_lookup(tmp_path: Path) -> None:
    shots = [
        {
            "shot_id": "001",
            "start_frame": 0,
            "end_frame": 48,
            "start_sec": 0.0,
            "end_sec": 2.0,
            "start_tc": "00:00:00:00",
            "end_tc": "00:00:02:00",
        },
        {
            "shot_id": "002",
            "start_frame": 48,
            "end_frame": 96,
            "start_sec": 2.0,
            "end_sec": 4.0,
            "start_tc": "00:00:02:00",
            "end_tc": "00:00:04:00",
        },
        {
            "shot_id": "003",
            "start_frame": 96,
            "end_frame": 144,
            "start_sec": 4.0,
            "end_sec": 6.0,
            "start_tc": "00:00:04:00",
            "end_tc": "00:00:06:00",
        },
    ]
    assignment = {"001": 1, "002": 1, "003": 2}
    entries = scene_manifest_entries(shots, assignment, [0.1, 0.9, 1.0])
    assert [entry["scene_id"] for entry in entries] == ["001", "002"]
    assert entries[0]["shot_ids"] == ["001", "002"]
    assert entries[0]["end_sec"] == 4.0
    assert entries[0]["boundary_score"] == 0.9

    manifest = tmp_path / "scenes.json"
    write_scenes_manifest(
        manifest,
        entries=entries,
        shots_manifest=tmp_path / "shots.json",
        engine="fusion",
        variant="full",
        threshold=0.6,
    )
    loaded = load_scene_manifest(manifest)
    assert len(loaded) == 2
    assert scene_for_time(loaded, 1.0) == "001"
    assert scene_for_time(loaded, 4.5) == "002"
    assert scene_for_time(loaded, 99.0) == "002"  # clamps past the end
    data = json.loads(manifest.read_text())
    assert data["engine"] == "fusion"


def test_shot_embeddings_report_missing_scenes_extra(monkeypatch) -> None:
    from unrender.editorial.scenes import features

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in ("torch", "transformers"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ImportError, match=r"unrender\[scenes\]"):
        features.shot_embeddings(Path("missing.mov"), [])


def _write_shots_manifest(path: Path, proxy: Path, count: int) -> None:
    shots = []
    for index in range(count):
        shots.append(
            {
                "shot_id": f"{index + 1:03d}",
                "start_frame": index * 24,
                "end_frame": (index + 1) * 24,
                "start_sec": float(index),
                "end_sec": float(index + 1),
                "start_tc": f"00:00:{index:02d}:00",
                "end_tc": f"00:00:{index + 1:02d}:00",
            }
        )
    path.write_text(json.dumps({"proxy": str(proxy), "shots": shots}), encoding="utf-8")


def test_scenes_group_cli_writes_manifest_and_skips(monkeypatch, tmp_path: Path) -> None:
    from unrender.cli import main
    from unrender.editorial.scenes import features as features_mod

    run_dir = tmp_path / "run"
    proxy = tmp_path / "proxy.mov"
    proxy.write_bytes(b"\x00")
    shots_json = tmp_path / "shots.json"
    _write_shots_manifest(shots_json, proxy, count=4)

    # Two visually distinct halves: a clean boundary after shot 2.
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    monkeypatch.setattr(features_mod, "shot_embeddings", lambda *a, **k: embeddings)

    argv = [
        "scenes",
        "group",
        "--shots",
        str(shots_json),
        "--video",
        str(proxy),
        "--out",
        str(run_dir),
        "--no-audio",
        "--csv",
    ]
    assert main(argv) == 0

    manifest = run_dir / "scenes.json"
    data = json.loads(manifest.read_text())
    assert data["engine"] == "fusion"
    assert data["variant"] == "visual_only"
    assert [s["scene_id"] for s in data["scenes"]] == ["001", "002"]
    assert data["scenes"][0]["shot_ids"] == ["001", "002"]
    assert data["scenes"][1]["shot_ids"] == ["003", "004"]
    assert manifest.with_suffix(".csv").exists()
    assert "inputs" in data

    # Re-running without --force skips (detection function must not run again).
    def _boom(*_a, **_k):
        raise AssertionError("scene grouping re-ran despite existing manifest")

    monkeypatch.setattr(features_mod, "shot_embeddings", _boom)
    assert main(argv) == 0

    # --force regroups.
    monkeypatch.setattr(features_mod, "shot_embeddings", lambda *a, **k: embeddings)
    assert main([*argv, "--force"]) == 0
