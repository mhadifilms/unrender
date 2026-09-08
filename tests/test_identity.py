from __future__ import annotations

from unrender.speakers import (
    empty_speaker_db,
    resolve_speaker_label,
    seed_config_speakers,
    speaker_key,
    upsert_face_clusters,
    upsert_voice_clusters,
)


def test_speaker_key_strips_accents_and_punctuation() -> None:
    assert speaker_key("RENÉE!") == "RENEE"


def test_alias_resolution_and_cluster_links() -> None:
    config = {"speakers": {"RENÉE": {"aliases": ["RENEE"]}, "ALEX": {"aliases": []}}}
    db = empty_speaker_db()
    seed_config_speakers(db, config)

    assert resolve_speaker_label("RENEE", db, allow_custom=False) == "RENÉE"

    upsert_face_clusters(
        db,
        [{"cluster_id": 7, "name": "RENEE", "centroid": [1.0, 0.0], "face_count": 3}],
    )
    upsert_voice_clusters(
        db,
        [{"cluster_id": 2, "name": "ALEX", "centroid": [0.0, 1.0], "sample_count": 4}],
    )

    assert db["speakers"]["RENEE"]["name"] == "RENÉE"
    assert db["speakers"]["RENEE"]["face_clusters"] == [7]
    assert db["speakers"]["ALEX"]["voice_clusters"] == [2]


def test_custom_labels_rejected_when_speaker_set_is_configured() -> None:
    db = empty_speaker_db()
    seed_config_speakers(db, {"speakers": ["ALEX"]})

    assert resolve_speaker_label("RANDOM", db, allow_custom=False) == ""
    assert "RANDOM" not in db["speakers"]
