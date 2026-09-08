from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from unrender.audio.embeddings import (
    PyannoteVoiceEmbedder as _PyannoteEmbedder,
)
from unrender.audio.embeddings import (
    SpeechBrainEcapaEmbedder as _SpeechBrainEcapaEmbedder,
)
from unrender.audio.embeddings import (
    VoiceEmbedder as _VoiceEmbedder,
)
from unrender.audio.embeddings import (
    build_voice_embedder,
)
from unrender.audio.embeddings import (
    huggingface_token as _hf_token,
)
from unrender.audio.embeddings import (
    patch_hf_hub_use_auth_token as _patch_hf_hub_use_auth_token,
)
from unrender.audio.embeddings import (
    patch_numpy_legacy_aliases as _patch_numpy_legacy_aliases,
)
from unrender.audio.embeddings import (
    preload_audio as _preload_audio,
)
from unrender.audio.embeddings import (
    pretrained_auth_kwargs as _pretrained_auth_kwargs,
)
from unrender.audio.embeddings import (
    torch_device as _torch_device,
)
from unrender.audio.embeddings import (
    torch_load_legacy_checkpoints as _torch_load_legacy_checkpoints,
)
from unrender.lib.fingerprints import input_fingerprints, warn_if_inputs_changed
from unrender.manifests import VoiceInput, read_json, write_json
from unrender.speakers import load_labeled_voice_centroids, load_speaker_db

__all__ = [
    "ContinuityPolicy",
    "_PyannoteEmbedder",
    "_SpeechBrainEcapaEmbedder",
    "_VoiceEmbedder",
    "_hf_token",
    "_patch_hf_hub_use_auth_token",
    "_patch_numpy_legacy_aliases",
    "_preload_audio",
    "_pretrained_auth_kwargs",
    "_torch_device",
    "_torch_load_legacy_checkpoints",
    "build_voice_db",
    "embed_voice_inputs",
    "match_voice_db",
]


@dataclass(frozen=True)
class ContinuityPolicy:
    max_gap_sec: float = 45.0
    min_similarity: float = 0.35
    min_margin: float = 0.02
    short_max_duration_sec: float = 1.25
    short_max_gap_sec: float = 8.0
    short_similarity_slack: float = 0.15
    short_margin_slack: float = 0.10
    interjection_max_duration_sec: float = 0.75
    interjection_max_gap_sec: float = 20.0
    interjection_min_similarity: float = 0.0
    interjection_margin_slack: float = 0.35

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class ContinuityCandidate:
    cluster_id: int
    similarity: float
    distance_sec: float
    duration_sec: float
    other_best_similarity: float
    candidate_count: int


@dataclass(frozen=True)
class ContinuityDecision:
    cluster_id: int
    mode: str
    similarity: float
    distance_sec: float
    duration_sec: float
    margin_to_other: float


def build_voice_db(
    *,
    clips: list[VoiceInput],
    voice_db_path: Path,
    review_dir: Path,
    backend: str = "pyannote",
    threshold: float = 0.35,
    min_cluster_size: int = 2,
    target_clusters: int | None = None,
    samples_per_cluster: int = 4,
    continuity_policy: ContinuityPolicy | None = None,
    force: bool = False,
) -> dict[str, Any]:
    continuity_policy = continuity_policy or ContinuityPolicy()
    inputs = {"clips": [clip.path for clip in clips]}
    if voice_db_path.exists() and not force:
        print(f"Voice DB already exists, skipping: {voice_db_path}", flush=True)
        data = read_json(voice_db_path)
        warn_if_inputs_changed(data, inputs, voice_db_path)
        return data
    if not clips:
        raise ValueError("no voice clips provided")
    full_stem_inputs = [
        str(clip.path)
        for clip in clips
        if clip.clip_type != "dialogue_line" or not str(clip.line_id or "").strip()
    ]
    if full_stem_inputs:
        sample = ", ".join(full_stem_inputs[:5])
        raise ValueError(
            "voice DB must be built from dialogue-line clips, not full stems. "
            "Run `unrender audio transcribe-lines` and `unrender audio build-clips` first. "
            f"Invalid clip input(s): {sample}"
        )
    review_dir.mkdir(parents=True, exist_ok=True)
    embedder = _voice_embedder(backend)
    embedded = [_embed_input(clip, embedder=embedder) for clip in clips]
    labels = _cluster_embeddings(
        [clip["embedding"] for clip in embedded],
        threshold=threshold,
        min_cluster_size=min_cluster_size,
        target_clusters=target_clusters,
    )

    by_cluster: dict[int, list[dict[str, Any]]] = {}
    ungrouped: list[dict[str, Any]] = []
    for clip, label in zip(embedded, labels, strict=True):
        if label < 0:
            ungrouped.append(clip)
        else:
            by_cluster.setdefault(label, []).append(clip)
    continuity_attached = _attach_adjacent_ungrouped(
        by_cluster,
        ungrouped,
        policy=continuity_policy,
    )
    singleton_clusters = _promote_ungrouped_to_singletons(by_cluster, ungrouped)

    clusters: list[dict[str, Any]] = []
    for cluster_id in sorted(by_cluster):
        members = by_cluster[cluster_id]
        cluster_review_dir = review_dir / f"cluster_{cluster_id:03d}"
        cluster_review_dir.mkdir(parents=True, exist_ok=True)
        samples: list[str] = []
        for member in members[:samples_per_cluster]:
            destination = cluster_review_dir / member["path"].name
            if destination.exists() and not force:
                samples.append(str(destination))
                continue
            shutil.copy2(member["path"], destination)
            samples.append(str(destination))
        clusters.append(
            {
                "cluster_id": int(cluster_id),
                "name": "",
                "centroid": _centroid([member["embedding"] for member in members]),
                "sample_count": len(members),
                "samples": samples,
                "clips": [_clip_json(member) for member in members],
                "review_dir": str(cluster_review_dir),
                "cluster_kind": _cluster_kind(members),
                "singleton": any(bool(member.get("singleton_cluster")) for member in members),
                "source_groups": sorted(
                    {str(member.get("source_group") or "") for member in members} - {""}
                ),
            }
        )
        print(
            f"  voice cluster {cluster_id}: {len(members)} clips -> {cluster_review_dir}",
            flush=True,
        )

    data = {
        "version": "1.0",
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z",
        "backend": backend,
        "cluster_threshold": threshold,
        "min_cluster_size": min_cluster_size,
        "target_clusters": target_clusters,
        "continuity_policy": continuity_policy.as_dict(),
        "continuity_attached": continuity_attached,
        "singleton_clusters": singleton_clusters,
        "inputs": input_fingerprints(inputs),
        "clusters": clusters,
        "ungrouped": [_clip_json(clip) for clip in ungrouped],
    }
    write_json(voice_db_path, data)
    print(f"Voice DB saved: {voice_db_path}", flush=True)
    return data


def match_voice_db(
    *,
    clips: list[VoiceInput],
    speaker_db_path: Path,
    output_json: Path,
    backend: str = "pyannote",
    sim_threshold: float = 0.65,
    min_margin: float = 0.0,
) -> list[dict[str, Any]]:
    """Match each clip against labeled voice centroids independently.

    A clip is accepted when its best similarity clears ``sim_threshold`` AND
    the gap to the second-best speaker clears ``min_margin`` — a confidently
    ambiguous match (two speakers scoring nearly the same) is worse than an
    abstention the editor can review.
    """
    identity_db = load_speaker_db(speaker_db_path)
    names, raw_centroids = load_labeled_voice_centroids(identity_db)
    if not names:
        raise ValueError(f"speaker DB has no labeled voice clusters: {speaker_db_path}")
    centroids = _normalize(np.asarray(raw_centroids, dtype=np.float32))
    embedder = _voice_embedder(backend)
    matches: list[dict[str, Any]] = []
    for index, clip in enumerate(clips, 1):
        embedded = _embed_input(clip, embedder=embedder)
        emb = _normalize(np.asarray([embedded["embedding"]], dtype=np.float32))[0]
        sims = centroids @ emb
        order = np.argsort(sims)[::-1]
        best = int(order[0])
        score = float(sims[best])
        # Cosine similarity is bounded below by -1, so a single-speaker DB
        # always yields a comfortably large margin.
        second = float(sims[int(order[1])]) if len(order) > 1 else -1.0
        margin = score - second
        accepted = [names[best]] if score >= sim_threshold and margin >= min_margin else []
        top = [{"speaker": names[int(i)], "score": float(sims[int(i)])} for i in order[:2]]
        entry = {
            "clip_id": clip.clip_id,
            "clip_type": clip.clip_type,
            "line_id": clip.line_id,
            "shot_id": clip.shot_id,
            "clip": str(clip.path),
            "source_group": clip.source_group,
            "source_stem": clip.source_stem,
            "start_sec": clip.start_sec,
            "end_sec": clip.end_sec,
            "clip_start_sec": clip.clip_start_sec,
            "clip_end_sec": clip.clip_end_sec,
            "text": clip.text,
            "matches": top,
            "margin": margin,
            "accepted": accepted,
        }
        if accepted:
            entry["proposed_value"] = ", ".join(accepted)
        matches.append(entry)
        print(
            f"  [{index}/{len(clips)}] {clip.path.name} best={names[best]} "
            f"score={score:.3f} margin={margin:.3f}",
            flush=True,
        )

    write_json(
        output_json,
        {
            "version": "1.0",
            "speaker_db": str(speaker_db_path),
            "sim_threshold": sim_threshold,
            "min_margin": min_margin,
            "shots": matches,
        },
    )
    print(f"Voice matches saved: {output_json}", flush=True)
    return matches


def embed_voice_inputs(
    clips: list[VoiceInput], *, backend: str = "pyannote"
) -> list[dict[str, Any]]:
    """Embed clips, loading the backend only if some clip lacks an embedding."""
    embedder = _voice_embedder(backend) if any(c.embedding is None for c in clips) else None
    return [_embed_input(clip, embedder=embedder) for clip in clips]


def _embed_input(clip: VoiceInput, *, embedder: _VoiceEmbedder | None) -> dict[str, Any]:
    if clip.embedding is not None:
        embedding = clip.embedding
    else:
        if not clip.path.exists():
            raise FileNotFoundError(f"voice clip not found: {clip.path}")
        if embedder is None:
            raise ValueError("voice embedding backend is required when clips lack embeddings")
        embedding = embedder.embed(clip.path)
    return {
        "path": clip.path,
        "clip_id": clip.clip_id,
        "clip_type": clip.clip_type,
        "line_id": clip.line_id,
        "shot_id": clip.shot_id,
        "source_group": clip.source_group,
        "source_stem": clip.source_stem,
        "start_sec": clip.start_sec,
        "end_sec": clip.end_sec,
        "clip_start_sec": clip.clip_start_sec,
        "clip_end_sec": clip.clip_end_sec,
        "text": clip.text,
        "embedding": tuple(float(value) for value in embedding),
    }


def _voice_embedder(backend: str) -> _VoiceEmbedder | None:
    return build_voice_embedder(backend)


def _cluster_embeddings(
    embeddings: list[tuple[float, ...]],
    *,
    threshold: float,
    min_cluster_size: int,
    target_clusters: int | None = None,
) -> list[int]:
    if not embeddings:
        return []
    matrix = _normalize(np.asarray(embeddings, dtype=np.float32))
    if target_clusters is not None and target_clusters > 0:
        try:
            from sklearn.cluster import AgglomerativeClustering
        except ImportError as exc:
            raise ImportError(
                "voice clustering requires scikit-learn: pip install .[voice]"
            ) from exc
        labels = AgglomerativeClustering(
            n_clusters=min(target_clusters, len(matrix)),
            metric="cosine",
            linkage="average",
        ).fit_predict(matrix)
        return [int(label) for label in labels.tolist()]
    if len(embeddings) < min_cluster_size:
        return [-1 for _ in embeddings]
    try:
        from sklearn.cluster import DBSCAN
        from sklearn.metrics.pairwise import cosine_distances
    except ImportError as exc:
        raise ImportError("voice clustering requires scikit-learn: pip install .[voice]") from exc
    distances = cosine_distances(matrix)
    labels = DBSCAN(
        eps=threshold,
        min_samples=min_cluster_size,
        metric="precomputed",
    ).fit_predict(distances)
    return [int(label) for label in labels.tolist()]


def _attach_adjacent_ungrouped(
    by_cluster: dict[int, list[dict[str, Any]]],
    ungrouped: list[dict[str, Any]],
    *,
    policy: ContinuityPolicy | None = None,
) -> int:
    """Fold short outlier clips into nearby same-source voice clusters.

    Source-separated channels are only local identities, but a short clip from
    the same local source as its neighbors is usually the same speaker when its
    embedding also prefers that source's cluster over the other voice clusters.
    """
    if not by_cluster or not ungrouped:
        return 0
    policy = policy or ContinuityPolicy()
    attached = 0
    remaining: list[dict[str, Any]] = []
    for clip in ungrouped:
        decision = _continuity_decision(
            clip,
            by_cluster,
            policy=policy,
        )
        if decision is None:
            remaining.append(clip)
            continue
        clip["continuity_attached_to"] = int(decision.cluster_id)
        clip["continuity_mode"] = decision.mode
        clip["continuity_similarity"] = round(decision.similarity, 6)
        clip["continuity_distance_sec"] = round(decision.distance_sec, 6)
        clip["continuity_margin_to_other"] = round(decision.margin_to_other, 6)
        by_cluster[decision.cluster_id].append(clip)
        attached += 1
    ungrouped[:] = remaining
    return attached


def _promote_ungrouped_to_singletons(
    by_cluster: dict[int, list[dict[str, Any]]],
    ungrouped: list[dict[str, Any]],
) -> int:
    """Make every remaining outlier reviewable instead of hiding it as noise."""
    if not ungrouped:
        return 0
    next_cluster_id = (max(by_cluster) + 1) if by_cluster else 0
    for clip in ungrouped:
        clip["singleton_cluster"] = True
        by_cluster[next_cluster_id] = [clip]
        next_cluster_id += 1
    count = len(ungrouped)
    ungrouped.clear()
    return count


def _continuity_decision(
    clip: dict[str, Any],
    by_cluster: dict[int, list[dict[str, Any]]],
    *,
    policy: ContinuityPolicy,
) -> ContinuityDecision | None:
    candidates = _continuity_candidates(clip, by_cluster, policy=policy)
    if not candidates:
        return None
    candidate = candidates[0]
    mode = _continuity_mode(candidate, policy=policy)
    if mode is None:
        return None
    return ContinuityDecision(
        cluster_id=candidate.cluster_id,
        mode=mode,
        similarity=candidate.similarity,
        distance_sec=candidate.distance_sec,
        duration_sec=candidate.duration_sec,
        margin_to_other=candidate.similarity - candidate.other_best_similarity,
    )


def _continuity_candidates(
    clip: dict[str, Any],
    by_cluster: dict[int, list[dict[str, Any]]],
    *,
    policy: ContinuityPolicy,
) -> list[ContinuityCandidate]:
    source_group = str(clip.get("source_group") or "")
    if not source_group:
        return []
    emb = np.asarray(clip["embedding"], dtype=np.float32)
    norm = float(np.linalg.norm(emb))
    if norm <= 0:
        return []
    emb = emb / norm
    start, end = _clip_bounds(clip)
    if start is None or end is None:
        return []
    duration = end - start
    raw: list[tuple[int, float, float, bool]] = []
    other_best = float("-inf")
    for cluster_id, members in by_cluster.items():
        centroid = _normalize(np.asarray([_centroid([m["embedding"] for m in members])]))[0]
        similarity = float(emb @ centroid)
        distance = _temporal_distance_to_members(clip, members)
        same_source = _dominant_source_group(members) == source_group
        if same_source and distance <= policy.max_gap_sec:
            raw.append((cluster_id, similarity, distance, same_source))
        else:
            other_best = max(other_best, similarity)
    candidates = [
        ContinuityCandidate(
            cluster_id=cluster_id,
            similarity=similarity,
            distance_sec=distance,
            duration_sec=duration,
            other_best_similarity=other_best,
            candidate_count=len(raw),
        )
        for cluster_id, similarity, distance, _same_source in raw
    ]
    candidates.sort(key=lambda item: (item.similarity, -item.distance_sec), reverse=True)
    return candidates


def _continuity_mode(candidate: ContinuityCandidate, *, policy: ContinuityPolicy) -> str | None:
    if (
        candidate.similarity >= policy.min_similarity
        and candidate.similarity >= candidate.other_best_similarity + policy.min_margin
    ):
        return "standard"
    if (
        candidate.candidate_count == 1
        and candidate.duration_sec <= policy.short_max_duration_sec
        and candidate.distance_sec <= policy.short_max_gap_sec
        and candidate.similarity >= policy.min_similarity - policy.short_similarity_slack
        and candidate.similarity >= candidate.other_best_similarity - policy.short_margin_slack
    ):
        return "short_fragment"
    if (
        candidate.candidate_count == 1
        and candidate.duration_sec <= policy.interjection_max_duration_sec
        and candidate.distance_sec <= policy.interjection_max_gap_sec
        and candidate.similarity >= policy.interjection_min_similarity
        and candidate.similarity
        >= candidate.other_best_similarity - policy.interjection_margin_slack
    ):
        return "interjection"
    return None


def _cluster_kind(members: list[dict[str, Any]]) -> str:
    if any(bool(member.get("singleton_cluster")) for member in members):
        return "singleton"
    if any(str(member.get("continuity_mode") or "") for member in members):
        return "continuity"
    return "cluster"


def _dominant_source_group(members: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for member in members:
        group = str(member.get("source_group") or "")
        if group:
            counts[group] = counts.get(group, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda item: item[1])[0]


def _temporal_distance_to_members(clip: dict[str, Any], members: list[dict[str, Any]]) -> float:
    start, end = _clip_bounds(clip)
    if start is None or end is None:
        return float("inf")
    return min(_interval_distance(start, end, *_clip_bounds(member)) for member in members)


def _clip_bounds(clip: dict[str, Any]) -> tuple[float | None, float | None]:
    start = clip.get("clip_start_sec")
    end = clip.get("clip_end_sec")
    if start is None or end is None:
        start = clip.get("start_sec")
        end = clip.get("end_sec")
    if start is None or end is None:
        return None, None
    return float(start), float(end)


def _interval_distance(
    start: float,
    end: float,
    other_start: float | None,
    other_end: float | None,
) -> float:
    if other_start is None or other_end is None:
        return float("inf")
    return max(other_start - end, start - other_end, 0.0)


def _centroid(embeddings: list[tuple[float, ...]]) -> list[float]:
    matrix = _normalize(np.asarray(embeddings, dtype=np.float32))
    center = matrix.mean(axis=0)
    norm = float(np.linalg.norm(center))
    if norm > 0:
        center = center / norm
    return [float(value) for value in center.tolist()]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def _clip_json(clip: dict[str, Any]) -> dict[str, Any]:
    out = {
        "path": str(clip["path"]),
        "clip_id": clip.get("clip_id", ""),
        "clip_type": clip.get("clip_type", ""),
        "line_id": clip.get("line_id", ""),
        "shot_id": clip.get("shot_id", ""),
        "source_group": clip.get("source_group", ""),
        "source_stem": clip.get("source_stem", ""),
        "start_sec": clip.get("start_sec"),
        "end_sec": clip.get("end_sec"),
        "clip_start_sec": clip.get("clip_start_sec"),
        "clip_end_sec": clip.get("clip_end_sec"),
        "text": clip.get("text", ""),
    }
    for key in (
        "continuity_attached_to",
        "continuity_mode",
        "continuity_similarity",
        "continuity_distance_sec",
        "continuity_margin_to_other",
        "singleton_cluster",
    ):
        if key in clip:
            out[key] = clip[key]
    return out
