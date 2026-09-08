from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unrender.analysis.identity.voice import (
    ContinuityPolicy,
    _attach_adjacent_ungrouped,
    _cluster_embeddings,
    _patch_hf_hub_use_auth_token,
    _patch_numpy_legacy_aliases,
    _pretrained_auth_kwargs,
    _promote_ungrouped_to_singletons,
    _torch_load_legacy_checkpoints,
    build_voice_db,
)
from unrender.manifests import VoiceInput


def test_hf_hub_patch_translates_pyannote_use_auth_token(monkeypatch) -> None:
    calls = {}

    def hf_hub_download(*, repo_id: str, token: str | None = None) -> str:
        calls["repo_id"] = repo_id
        calls["token"] = token
        return "downloaded"

    file_download = SimpleNamespace(hf_hub_download=hf_hub_download)
    huggingface_hub = SimpleNamespace(
        hf_hub_download=hf_hub_download,
        file_download=file_download,
    )
    pyannote_pipeline = SimpleNamespace(hf_hub_download=hf_hub_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.file_download", file_download)
    monkeypatch.setitem(sys.modules, "pyannote.audio.core.pipeline", pyannote_pipeline)

    _patch_hf_hub_use_auth_token()

    assert (
        file_download.hf_hub_download(repo_id="pyannote/embedding", use_auth_token="hf_test")
        == "downloaded"
    )
    assert (
        pyannote_pipeline.hf_hub_download(
            repo_id="pyannote/pipeline",
            use_auth_token="hf_test",
        )
        == "downloaded"
    )
    assert calls == {"repo_id": "pyannote/pipeline", "token": "hf_test"}


def test_torch_load_legacy_checkpoints_sets_weights_only_false() -> None:
    calls = {}

    def load(path: str, *, weights_only: bool = True) -> str:
        calls["path"] = path
        calls["weights_only"] = weights_only
        return "loaded"

    torch = SimpleNamespace(load=load)

    with _torch_load_legacy_checkpoints(torch):
        assert torch.load("checkpoint.ckpt", weights_only=True) == "loaded"

    assert calls == {"path": "checkpoint.ckpt", "weights_only": False}
    assert torch.load is load


def test_pretrained_auth_kwargs_supports_pyannote_fork_and_modern_api() -> None:
    def legacy(model: str, *, use_auth_token: str | None = None) -> None:
        pass

    def modern(model: str, *, token: str | None = None) -> None:
        pass

    assert _pretrained_auth_kwargs(legacy, "hf_test") == {"use_auth_token": "hf_test"}
    assert _pretrained_auth_kwargs(modern, "hf_test") == {"token": "hf_test"}


def test_numpy_legacy_alias_patch_supports_pyannote_31(monkeypatch) -> None:
    monkeypatch.delattr(np, "NaN", raising=False)
    monkeypatch.delattr(np, "NAN", raising=False)

    _patch_numpy_legacy_aliases()

    assert np.NaN is np.nan
    assert np.NAN is np.nan


def test_build_voice_db_rejects_full_stem_inputs(tmp_path: Path) -> None:
    full_stem = tmp_path / "speaker_01_stem.wav"
    full_stem.write_bytes(b"wav")

    with pytest.raises(ValueError, match="dialogue-line clips"):
        build_voice_db(
            clips=[VoiceInput(path=full_stem, clip_id="speaker_01")],
            voice_db_path=tmp_path / "voice_db.json",
            review_dir=tmp_path / "review",
            backend="pyannote",
        )


def test_build_voice_db_promotes_outliers_to_singleton_clusters(tmp_path: Path) -> None:
    clips = [
        _voice_input(tmp_path, "DL_000001", "speaker_01", (1.0, 0.0), start=1.0, end=2.0),
        _voice_input(tmp_path, "DL_000002", "speaker_01", (0.99, 0.01), start=3.0, end=4.0),
        _voice_input(tmp_path, "DL_000003", "speaker_02", (0.0, 1.0), start=20.0, end=21.0),
    ]

    data = build_voice_db(
        clips=clips,
        voice_db_path=tmp_path / "voice_db.json",
        review_dir=tmp_path / "review",
        backend="pyannote",
        threshold=0.05,
        min_cluster_size=2,
        continuity_policy=ContinuityPolicy(max_gap_sec=10.0),
        force=True,
    )

    assert data["ungrouped"] == []
    assert data["continuity_policy"]["max_gap_sec"] == 10.0
    assert data["singleton_clusters"] == 1
    assert [cluster["cluster_kind"] for cluster in data["clusters"]] == ["cluster", "singleton"]
    singleton = data["clusters"][1]
    assert singleton["singleton"] is True
    assert singleton["clips"][0]["singleton_cluster"] is True


def test_voice_cluster_embeddings_can_use_target_cluster_count() -> None:
    rng = np.random.default_rng(123)
    a = [tuple((np.asarray([1.0, 0.0]) + rng.normal(0, 0.01, 2)).tolist()) for _ in range(10)]
    b = [tuple((np.asarray([0.0, 1.0]) + rng.normal(0, 0.01, 2)).tolist()) for _ in range(8)]
    labels = _cluster_embeddings(
        a + b,
        threshold=0.01,
        min_cluster_size=1,
        target_clusters=2,
    )

    assert set(labels) == {0, 1}


def test_adjacent_ungrouped_clip_attaches_to_same_source_cluster() -> None:
    by_cluster = {
        0: [
            _embedded_clip(
                "DL_000001",
                "speaker_01",
                [1.0, 0.0],
                start=10.0,
                end=12.0,
            )
        ],
        1: [
            _embedded_clip(
                "DL_000002",
                "speaker_02",
                [0.0, 1.0],
                start=20.0,
                end=22.0,
            )
        ],
    }
    ungrouped = [
        _embedded_clip(
            "DL_000003",
            "speaker_01",
            [0.98, 0.02],
            start=13.0,
            end=13.5,
        )
    ]

    attached = _attach_adjacent_ungrouped(by_cluster, ungrouped)

    assert attached == 1
    assert ungrouped == []
    assert [clip["line_id"] for clip in by_cluster[0]] == ["DL_000001", "DL_000003"]
    assert by_cluster[0][1]["continuity_attached_to"] == 0


def test_adjacent_ungrouped_clip_stays_out_when_other_cluster_is_more_similar() -> None:
    by_cluster = {
        0: [_embedded_clip("DL_000001", "speaker_01", [1.0, 0.0], start=10.0, end=12.0)],
        1: [_embedded_clip("DL_000002", "speaker_02", [0.0, 1.0], start=20.0, end=22.0)],
    }
    ungrouped = [_embedded_clip("DL_000003", "speaker_01", [0.1, 0.9], start=13.0, end=13.5)]

    attached = _attach_adjacent_ungrouped(by_cluster, ungrouped)

    assert attached == 0
    assert [clip["line_id"] for clip in ungrouped] == ["DL_000003"]


def test_close_short_fragment_can_attach_with_weak_embedding_margin() -> None:
    by_cluster = {
        0: [_embedded_clip("DL_000001", "speaker_01", [1.0, 0.0], start=10.0, end=12.0)],
        1: [_embedded_clip("DL_000002", "speaker_02", [0.0, 1.0], start=20.0, end=22.0)],
    }
    ungrouped = [
        # Slightly prefers the competing cluster, but it is a sub-second
        # fragment from the same local source immediately after cluster 0.
        _embedded_clip("DL_000003", "speaker_01", [0.68, 0.73], start=12.4, end=12.9)
    ]

    attached = _attach_adjacent_ungrouped(by_cluster, ungrouped)

    assert attached == 1
    assert ungrouped == []
    assert by_cluster[0][1]["line_id"] == "DL_000003"


def test_ultra_short_interjection_attaches_despite_unreliable_embedding() -> None:
    by_cluster = {
        0: [_embedded_clip("DL_000001", "speaker_01", [1.0, 0.0], start=10.0, end=12.0)],
        1: [_embedded_clip("DL_000002", "speaker_02", [0.0, 1.0], start=20.0, end=22.0)],
    }
    ungrouped = [_embedded_clip("DL_000003", "speaker_01", [0.45, 0.75], start=15.0, end=15.3)]

    attached = _attach_adjacent_ungrouped(by_cluster, ungrouped)

    assert attached == 1
    assert ungrouped == []
    assert by_cluster[0][1]["line_id"] == "DL_000003"


def test_remaining_ungrouped_clips_become_singleton_clusters() -> None:
    by_cluster = {0: [_embedded_clip("DL_000001", "speaker_01", [1.0, 0.0], start=10.0, end=12.0)]}
    ungrouped = [
        _embedded_clip("DL_000002", "speaker_02", [0.0, 1.0], start=20.0, end=22.0),
        _embedded_clip("DL_000003", "speaker_03", [0.5, 0.5], start=30.0, end=31.0),
    ]

    promoted = _promote_ungrouped_to_singletons(by_cluster, ungrouped)

    assert promoted == 2
    assert ungrouped == []
    assert by_cluster[1][0]["line_id"] == "DL_000002"
    assert by_cluster[1][0]["singleton_cluster"] is True
    assert by_cluster[2][0]["line_id"] == "DL_000003"


def _embedded_clip(
    line_id: str,
    source_group: str,
    embedding: list[float],
    *,
    start: float,
    end: float,
) -> dict:
    return {
        "path": Path(f"/tmp/{line_id}_{source_group}.wav"),
        "clip_id": f"{line_id}_{source_group}",
        "clip_type": "dialogue_line",
        "line_id": line_id,
        "source_group": source_group,
        "clip_start_sec": start,
        "clip_end_sec": end,
        "embedding": tuple(embedding),
    }


def _voice_input(
    tmp_path: Path,
    line_id: str,
    source_group: str,
    embedding: tuple[float, ...],
    *,
    start: float,
    end: float,
) -> VoiceInput:
    path = tmp_path / f"{line_id}_{source_group}_stem.wav"
    path.write_bytes(b"audio")
    return VoiceInput(
        path=path,
        clip_id=f"{line_id}_{source_group}",
        clip_type="dialogue_line",
        line_id=line_id,
        source_group=source_group,
        clip_start_sec=start,
        clip_end_sec=end,
        embedding=embedding,
    )
