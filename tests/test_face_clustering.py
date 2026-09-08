"""Face embedding clustering, exemplar selection, and crop encoding."""

from __future__ import annotations

import numpy as np

from unrender.analysis.identity.face import (
    _decode_crop,
    _encode_crop,
    _select_exemplars,
    _weighted_centroid,
    cluster_embeddings,
)


def _group(base: list[float], count: int, seed: int, scale: float = 0.03) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    vecs = []
    for _ in range(count):
        vec = np.asarray(base, dtype=np.float32) + rng.normal(0.0, scale, len(base)).astype(
            np.float32
        )
        vecs.append(vec / np.linalg.norm(vec))
    return vecs


def test_cluster_embeddings_separates_identities() -> None:
    alex = _group([1.0, 0.0, 0.0], 8, seed=1)
    jordan = _group([0.0, 1.0, 0.0], 6, seed=2)

    labels = cluster_embeddings(alex + jordan, threshold=0.5, min_cluster_size=5)

    assert set(labels) == {0, 1}
    # Cluster 0 is the largest group.
    assert labels[:8] == [0] * 8
    assert labels[8:] == [1] * 6


def test_cluster_embeddings_can_use_target_cluster_count() -> None:
    alex = _group([1.0, 0.0, 0.0], 8, seed=11)
    jordan = _group([0.0, 1.0, 0.0], 6, seed=12)
    noise = _group([0.0, 0.0, 1.0], 3, seed=13)

    labels = cluster_embeddings(
        alex + jordan + noise,
        threshold=0.5,
        min_cluster_size=1,
        target_clusters=2,
    )

    assert set(labels) == {0, 1}


def test_small_clusters_are_reabsorbed_not_dropped() -> None:
    # 8 tight ALEX faces plus 2 slightly-off ALEX faces that agglomerative
    # clustering may isolate; they must rejoin the big cluster, not vanish.
    alex = _group([1.0, 0.0, 0.0], 8, seed=3, scale=0.01)
    drifted = _group([0.95, 0.2, 0.0], 2, seed=4, scale=0.01)

    labels = cluster_embeddings(alex + drifted, threshold=0.25, min_cluster_size=5)

    assert labels[:8] == [0] * 8
    assert labels[8:] == [0, 0]


def test_isolated_faces_become_noise() -> None:
    alex = _group([1.0, 0.0, 0.0], 6, seed=5, scale=0.01)
    stranger = _group([0.0, 0.0, 1.0], 1, seed=6, scale=0.01)

    labels = cluster_embeddings(alex + stranger, threshold=0.2, min_cluster_size=5)

    assert labels[:6] == [0] * 6
    assert labels[6] == -1


def test_cluster_embeddings_edge_cases() -> None:
    assert cluster_embeddings([]) == []
    single = _group([1.0, 0.0, 0.0], 1, seed=7)
    assert cluster_embeddings(single, min_cluster_size=1) == [0]
    # Nothing reaches min_cluster_size: everything is noise.
    assert cluster_embeddings(single, min_cluster_size=5) == [-1]


def test_select_exemplars_spreads_over_time() -> None:
    members = [{"ts": float(i), "confidence": 0.9, "quality": 50.0 + (i % 3)} for i in range(100)]
    exemplars = _select_exemplars(members, 10)

    assert len(exemplars) == 10
    timestamps = [det["ts"] for det in members]
    picked = [det["ts"] for det in exemplars]
    # Picks span most of the timeline instead of bunching at one end.
    assert max(picked) - min(picked) > (max(timestamps) - min(timestamps)) * 0.5
    assert picked == sorted(picked)


def test_crop_encoding_roundtrip_and_bounds() -> None:
    frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
    encoded = _encode_crop(frame, (600, 300, 400, 400))

    decoded = _decode_crop(encoded)
    assert decoded is not None
    assert max(decoded.shape[:2]) <= 200
    # Box partially outside the frame still encodes.
    assert _decode_crop(_encode_crop(frame, (1200, 650, 400, 400))) is not None
    assert _decode_crop(b"") is None


def test_onnx_providers_prefer_cuda_and_always_keep_cpu(monkeypatch) -> None:
    import sys
    import types

    from unrender.analysis.identity.face import _onnx_providers

    fake = types.SimpleNamespace(
        get_available_providers=lambda: [
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)

    monkeypatch.delenv("UNRENDER_FACE_COREML", raising=False)
    assert _onnx_providers() == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    monkeypatch.setenv("UNRENDER_FACE_COREML", "1")
    assert _onnx_providers() == [
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_weighted_centroid_prefers_confident_faces() -> None:
    members = [
        {"embedding": np.asarray([1.0, 0.0], dtype=np.float32), "confidence": 0.99},
        {"embedding": np.asarray([0.0, 1.0], dtype=np.float32), "confidence": 0.05},
    ]
    centroid = _weighted_centroid(members)
    assert centroid[0] > 0.9
    assert abs(float(np.linalg.norm(centroid)) - 1.0) < 1e-5
