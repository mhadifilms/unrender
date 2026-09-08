"""Cluster-then-assign dialogue clip resolution.

Per-clip speaker matching makes thousands of independent decisions from
roughly one second of audio each; on short lines those decisions degrade and
the errors scatter randomly across the timeline. This resolver instead pools
evidence globally:

1. One representative clip per dialogue line (the most energetic stem slice)
   is embedded and all representatives are clustered — every line by the same
   voice lands in the same cluster.
2. Clusters are jointly assigned to the labeled speaker DB with the Hungarian
   algorithm, so two clusters cannot claim the same speaker while another
   goes unused. Every line inherits its cluster's speaker.
3. A long, confidently-matched clip may override its cluster label — the
   guard for rare speakers whose few lines get folded into a bigger cluster.
4. The stem file for each line is chosen by fusing identity similarity with
   relative energy across the line's stem slices: bleed carries the right
   voice at the wrong level, so energy breaks the tie.

A wrong cluster assignment is systematic — relabeling one cluster fixes every
affected line at once — which is also why cluster ids are recorded in the
output plan.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from unrender.analysis.identity.voice import embed_voice_inputs
from unrender.editorial.dialogue.plan_writers import _write_clip_plan_csv
from unrender.lib.audio import measure_peak_dbfs
from unrender.manifests import VoiceInput, write_json
from unrender.speakers import load_labeled_voice_centroids, load_speaker_db

DEFAULT_OVERRIDE_MIN_SEC = 1.5
DEFAULT_OVERRIDE_MARGIN = 0.15
# Stem slices this far below the line's loudest slice cannot win energy-fused
# stem selection (weight <= 0.18 even at perfect similarity), so embedding
# them is pure cost. On five-stem projects this skips most inference calls.
EMBED_WINDOW_DB = 30.0


def resolve_voice_clips_clustered(
    *,
    clips: list[VoiceInput],
    speaker_db_path: Path,
    output_json: Path,
    output_csv: Path,
    backend: str = "pyannote",
    n_clusters: int | None = None,
    override_min_sec: float = DEFAULT_OVERRIDE_MIN_SEC,
    override_margin: float = DEFAULT_OVERRIDE_MARGIN,
) -> list[dict[str, Any]]:
    """Resolve dialogue clips to speakers via global clustering."""
    if not clips:
        raise ValueError("no voice clips provided")
    identity_db = load_speaker_db(speaker_db_path)
    names, raw_centroids = load_labeled_voice_centroids(identity_db)
    if not names:
        raise ValueError(f"speaker DB has no labeled voice clusters: {speaker_db_path}")
    centroids = _normalize(np.asarray(raw_centroids, dtype=np.float32))

    to_embed, peaks, skipped = _energy_gate(clips)
    if skipped:
        print(f"  skipping {skipped} near-silent stem slice(s) before embedding", flush=True)
    embedded = embed_voice_inputs(to_embed, backend=backend)
    for clip, entry in zip(to_embed, embedded, strict=True):
        entry["peak_dbfs"] = peaks[id(clip)]
    units = _group_units(embedded)
    representatives = {unit_id: _representative(members) for unit_id, members in units.items()}
    unit_ids = sorted(representatives)
    rep_matrix = _normalize(
        np.asarray([representatives[uid]["embedding"] for uid in unit_ids], dtype=np.float32)
    )

    labels = _cluster(rep_matrix, n_clusters=n_clusters or len(names))
    assignment, cluster_info = _assign_clusters(labels, rep_matrix, centroids, names)

    plan: list[dict[str, Any]] = []
    overrides = 0
    for row, unit_id in enumerate(unit_ids):
        cluster_id = int(labels[row])
        speaker = assignment[cluster_id]
        source = "cluster"
        emb = rep_matrix[row]
        override = _clip_override(
            representatives[unit_id],
            emb,
            centroids,
            names,
            override_min_sec=override_min_sec,
            override_margin=override_margin,
        )
        if override is not None and override != speaker:
            speaker = override
            source = "clip-override"
            overrides += 1
        chosen, fused_score = _select_stem_clip(units[unit_id], speaker, centroids, names)
        info = cluster_info[cluster_id]
        plan.append(
            {
                "clip_id": str(chosen.get("clip_id") or ""),
                "clip_type": str(chosen.get("clip_type") or ""),
                "line_id": str(chosen.get("line_id") or ""),
                "shot_id": str(chosen.get("shot_id") or ""),
                "speaker": speaker,
                "stem_path": str(chosen["path"]),
                "source_stem": str(chosen.get("source_stem") or ""),
                "source_group": str(chosen.get("source_group") or ""),
                "start_sec": chosen.get("start_sec"),
                "end_sec": chosen.get("end_sec"),
                "clip_start_sec": chosen.get("clip_start_sec"),
                "clip_end_sec": chosen.get("clip_end_sec"),
                "score": fused_score,
                "margin": info["margin"],
                "cluster_id": cluster_id,
                "status": "matched",
                "resolution_source": source,
                "text": str(chosen.get("text") or ""),
            }
        )
    plan.sort(key=lambda entry: (entry["line_id"], entry["shot_id"], entry["clip_id"]))

    write_json(
        output_json,
        {
            "version": "1.0",
            "resolver": "cluster",
            "speaker_db": str(speaker_db_path),
            "n_clusters": len(cluster_info),
            "override_min_sec": override_min_sec,
            "override_margin": override_margin,
            "clusters": [cluster_info[cid] for cid in sorted(cluster_info)],
            "clips": plan,
        },
    )
    _write_clip_plan_csv(output_csv, plan)
    print(
        f"Clip stem plan saved: {output_json} ({len(plan)} line(s), "
        f"{len(cluster_info)} cluster(s), {overrides} clip override(s))",
        flush=True,
    )
    return plan


def _energy_gate(
    clips: list[VoiceInput],
) -> tuple[list[VoiceInput], dict[int, float], int]:
    """Drop clips too quiet (relative to their line) to affect any decision.

    Only clips that would need a fresh embedding are gated; clips that already
    carry an embedding are free to keep. Clips with unreadable peaks are kept,
    preserving the existing missing-file error behavior downstream.
    """
    peaks: dict[int, float] = {}
    by_unit: dict[str, list[VoiceInput]] = {}
    for clip in clips:
        unit_id = str(clip.line_id or clip.shot_id or clip.clip_id or "")
        by_unit.setdefault(unit_id, []).append(clip)
        try:
            peaks[id(clip)] = measure_peak_dbfs(clip.path) if clip.path.exists() else float("-inf")
        except (OSError, ValueError, EOFError):
            peaks[id(clip)] = float("-inf")

    kept: list[VoiceInput] = []
    skipped = 0
    for members in by_unit.values():
        loudest = max(peaks[id(clip)] for clip in members)
        for clip in members:
            peak = peaks[id(clip)]
            if (
                clip.embedding is None
                and not math.isinf(loudest)
                and not math.isinf(peak)
                and peak < loudest - EMBED_WINDOW_DB
            ):
                skipped += 1
                continue
            kept.append(clip)
    return kept, peaks, skipped


def _group_units(embedded: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group per-stem clips by the dialogue line (or shot) they slice."""
    units: dict[str, list[dict[str, Any]]] = {}
    for clip in embedded:
        unit_id = str(clip.get("line_id") or clip.get("shot_id") or clip.get("clip_id") or "")
        if not unit_id:
            raise ValueError(f"voice clip is missing line_id/shot_id/clip_id: {clip.get('path')}")
        units.setdefault(unit_id, []).append(clip)
    return units


def _representative(members: list[dict[str, Any]]) -> dict[str, Any]:
    """The most energetic stem slice carries the line's actual voice."""
    return max(members, key=lambda clip: _peak_dbfs(clip))


def _peak_dbfs(clip: dict[str, Any]) -> float:
    if "peak_dbfs" not in clip:
        path = Path(str(clip["path"]))
        try:
            clip["peak_dbfs"] = measure_peak_dbfs(path) if path.exists() else float("-inf")
        except (OSError, ValueError, EOFError):
            clip["peak_dbfs"] = float("-inf")
    return float(clip["peak_dbfs"])


def _cluster(matrix: np.ndarray, *, n_clusters: int) -> list[int]:
    count = matrix.shape[0]
    k = max(1, min(n_clusters, count))
    if count == 1 or k == 1:
        return [0] * count
    from sklearn.cluster import AgglomerativeClustering

    labels = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(
        matrix
    )
    return [int(label) for label in labels]


def _assign_clusters(
    labels: list[int],
    rep_matrix: np.ndarray,
    centroids: np.ndarray,
    names: list[str],
) -> tuple[dict[int, str], dict[int, dict[str, Any]]]:
    from scipy.optimize import linear_sum_assignment

    cluster_ids = sorted(set(labels))
    label_array = np.asarray(labels)
    cluster_centroids = _normalize(
        np.stack([rep_matrix[label_array == cid].mean(axis=0) for cid in cluster_ids])
    )
    similarity = cluster_centroids @ centroids.T  # clusters x speakers

    assignment: dict[int, str] = {}
    rows, cols = linear_sum_assignment(-similarity)
    for row, col in zip(rows, cols, strict=True):
        assignment[cluster_ids[int(row)]] = names[int(col)]
    # More clusters than labeled speakers: leftovers take their best match.
    for index, cid in enumerate(cluster_ids):
        if cid not in assignment:
            assignment[cid] = names[int(np.argmax(similarity[index]))]

    info: dict[int, dict[str, Any]] = {}
    for index, cid in enumerate(cluster_ids):
        sims = similarity[index]
        assigned = assignment[cid]
        assigned_sim = float(sims[names.index(assigned)])
        others = [float(s) for name, s in zip(names, sims, strict=True) if name != assigned]
        margin = assigned_sim - max(others) if others else assigned_sim + 1.0
        info[cid] = {
            "cluster_id": cid,
            "speaker": assigned,
            "similarity": assigned_sim,
            "margin": margin,
            "size": int(np.sum(label_array == cid)),
        }
    return assignment, info


def _clip_override(
    representative: dict[str, Any],
    embedding: np.ndarray,
    centroids: np.ndarray,
    names: list[str],
    *,
    override_min_sec: float,
    override_margin: float,
) -> str | None:
    """Long, unambiguous clips may out-vote their cluster.

    Short clips give noisy embeddings, so only clips of at least
    ``override_min_sec`` whose best speaker beats the runner-up by
    ``override_margin`` are trusted over the pooled cluster evidence.
    """
    duration = _clip_duration(representative)
    if duration is None or duration < override_min_sec:
        return None
    sims = centroids @ embedding
    order = np.argsort(sims)[::-1]
    best = float(sims[int(order[0])])
    second = float(sims[int(order[1])]) if len(order) > 1 else -1.0
    if best - second < override_margin:
        return None
    return names[int(order[0])]


def _clip_duration(clip: dict[str, Any]) -> float | None:
    start = clip.get("clip_start_sec")
    end = clip.get("clip_end_sec")
    if start is None or end is None:
        start = clip.get("start_sec")
        end = clip.get("end_sec")
    if start is None or end is None:
        return None
    return float(end) - float(start)


def _select_stem_clip(
    members: list[dict[str, Any]],
    speaker: str,
    centroids: np.ndarray,
    names: list[str],
) -> tuple[dict[str, Any], float]:
    """Fuse identity similarity with relative energy to pick the stem file.

    A speaker's bleed into the wrong stem embeds as the same voice at a lower
    level, so identity alone cannot distinguish the true stem; relative peak
    energy across the line's slices breaks the tie.
    """
    target = centroids[names.index(speaker)]
    peaks = [_peak_dbfs(clip) for clip in members]
    finite = [peak for peak in peaks if not math.isinf(peak)]
    max_peak = max(finite) if finite else None

    best_clip = members[0]
    best_score = float("-inf")
    for clip, peak in zip(members, peaks, strict=True):
        emb = np.asarray(clip["embedding"], dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        similarity = float((emb / norm) @ target) if norm > 0 else 0.0
        if max_peak is None or math.isinf(peak):
            energy_weight = 1.0
        else:
            # dB below the loudest slice, mapped to (0, 1]; sqrt keeps the
            # weight mild so identity still dominates near-equal levels.
            energy_weight = math.sqrt(10.0 ** ((peak - max_peak) / 20.0))
        score = similarity * energy_weight
        if score > best_score:
            best_score = score
            best_clip = clip
    return best_clip, best_score


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms
