from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from unrender.analysis.identity.face import detect_speakers_for_shot, match_shots
from unrender.analysis.identity.resolution import (
    FaceDbVote,
    VoiceDbMatch,
    resolve_speakers,
)
from unrender.analysis.identity.voice import match_voice_db
from unrender.manifests import ShotRecord, VoiceInput, read_json


def test_voice_match_uses_labeled_voice_centroid(tmp_path: Path) -> None:
    speaker_db = tmp_path / "speaker_db.json"
    speaker_db.write_text(
        json.dumps(
            {
                "version": "1.0",
                "speakers": {},
                "face_clusters": [],
                "voice_clusters": [
                    {"cluster_id": 2, "speaker": "ALEX", "name": "ALEX", "centroid": [1.0, 0.0]}
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "voice_matches.json"

    matches = match_voice_db(
        clips=[
            VoiceInput(
                path=tmp_path / "001_speaker_01.wav",
                shot_id="001",
                embedding=(0.99, 0.01),
            )
        ],
        speaker_db_path=speaker_db,
        output_json=output,
    )

    assert matches[0]["accepted"] == ["ALEX"]
    assert read_json(output)["shots"][0]["proposed_value"] == "ALEX"


def test_resolver_prefers_face_then_voice_then_existing() -> None:
    rows = [
        ShotRecord(shot_id="001", video_path=Path("/tmp/001.mov")),
        ShotRecord(shot_id="002", video_path=Path("/tmp/002.mov")),
        ShotRecord(shot_id="003", video_path=Path("/tmp/003.mov"), existing_speaker="JORDAN"),
    ]

    resolutions = resolve_speakers(
        rows,
        face_db_votes={"001": FaceDbVote(shot_id="001", accepted=["ALEX"])},
        voice_db_matches={"002": VoiceDbMatch(shot_id="002", accepted=["RENEE"])},
        config={"speakers": {"RENÉE": {"aliases": ["RENEE"]}, "ALEX": {}, "JORDAN": {}}},
    )

    assert [(r.shot_id, r.source, r.proposed_value) for r in resolutions] == [
        ("001", "face-db", "ALEX"),
        ("002", "voice-db", "RENÉE"),
        ("003", "existing", "JORDAN"),
    ]


def test_match_shots_writes_json_and_csv(monkeypatch, tmp_path: Path) -> None:
    speaker_db = tmp_path / "speaker_db.json"
    speaker_db.write_text(
        json.dumps(
            {
                "version": "1.0",
                "speakers": {},
                "face_clusters": [
                    {"cluster_id": 7, "speaker": "ALEX", "name": "ALEX", "centroid": [1.0, 0.0]}
                ],
                "voice_clusters": [],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("unrender.analysis.identity.face.FaceEngine", lambda: object())
    monkeypatch.setattr(
        "unrender.analysis.identity.face.detect_speakers_for_shot",
        lambda *args, **kwargs: (Counter({"ALEX": 2}), 2),
    )

    matches = match_shots(
        shots=[ShotRecord(shot_id="001", video_path=Path("/tmp/001.mov"))],
        speaker_db_path=speaker_db,
        output_json=tmp_path / "matches.json",
        output_csv=tmp_path / "matches.csv",
        min_votes=2,
    )

    assert matches[0]["proposed_value"] == "ALEX"
    assert "ALEX" in (tmp_path / "matches.csv").read_text(encoding="utf-8")


class _FakeEngine:
    """Two faces: ALEX-like at the origin, JORDAN-like at (100, 100)."""

    def analyze(self, frame):
        from unrender.analysis.identity.face import FaceObservation

        return [
            FaceObservation(
                box=(0, 0, 10, 10),
                confidence=0.9,
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            ),
            FaceObservation(
                box=(100, 100, 10, 10),
                confidence=0.9,
                embedding=np.asarray([0.0, 1.0], dtype=np.float32),
            ),
        ]


def test_detect_speakers_for_shot_can_target_specific_face(monkeypatch) -> None:
    monkeypatch.setattr(
        "unrender.analysis.identity.face.sample_video_frames",
        lambda path, n_samples: [np.zeros((10, 10, 3), dtype=np.uint8)],
    )

    votes, faces = detect_speakers_for_shot(
        Path("/tmp/shot.mov"),
        _FakeEngine(),
        ["ALEX", "JORDAN"],
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        n_samples=1,
        sim_threshold=0.1,
        min_confidence=0.5,
        target_box=(98, 98, 15, 15),
    )

    assert faces == 1
    assert votes == Counter({"JORDAN": 1})


def test_detect_speakers_abstains_on_ambiguous_faces(monkeypatch) -> None:
    from unrender.analysis.identity.face import FaceObservation

    class AmbiguousEngine:
        def analyze(self, frame):
            return [
                FaceObservation(
                    box=(0, 0, 50, 50),
                    confidence=0.9,
                    embedding=np.asarray([0.707, 0.707], dtype=np.float32),
                )
            ]

    monkeypatch.setattr(
        "unrender.analysis.identity.face.sample_video_frames",
        lambda path, n_samples: [np.zeros((10, 10, 3), dtype=np.uint8)],
    )

    votes, faces = detect_speakers_for_shot(
        Path("/tmp/shot.mov"),
        AmbiguousEngine(),
        ["ALEX", "JORDAN"],
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        n_samples=1,
        sim_threshold=0.1,
        min_confidence=0.5,
        min_margin=0.05,
    )

    # Equidistant between both enrolled faces: counted but no vote cast.
    assert faces == 1
    assert votes == Counter()
