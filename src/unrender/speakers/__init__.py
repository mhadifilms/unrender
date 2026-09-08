from __future__ import annotations

from unrender.speakers.registry import (
    allow_custom_speaker_labels,
    canonical_speaker_name,
    configured_speakers,
    empty_speaker_db,
    load_labeled_face_centroids,
    load_labeled_voice_centroids,
    load_speaker_db,
    resolve_speaker_label,
    save_speaker_db,
    seed_config_speakers,
    speaker_key,
    upsert_face_clusters,
    upsert_voice_clusters,
)

__all__ = [
    "allow_custom_speaker_labels",
    "canonical_speaker_name",
    "configured_speakers",
    "empty_speaker_db",
    "load_labeled_face_centroids",
    "load_labeled_voice_centroids",
    "load_speaker_db",
    "resolve_speaker_label",
    "save_speaker_db",
    "seed_config_speakers",
    "speaker_key",
    "upsert_face_clusters",
    "upsert_voice_clusters",
]
