from __future__ import annotations

import json
from pathlib import Path

from unrender.analysis.identity.face import build_face_db
from unrender.analysis.identity.voice import build_voice_db


def test_face_build_skips_existing_db_before_media_or_model_checks(tmp_path: Path) -> None:
    face_db = tmp_path / "face_db.json"
    face_db.write_text(json.dumps({"version": "1.0", "clusters": []}), encoding="utf-8")

    data = build_face_db(
        video_path=tmp_path / "missing.mov",
        face_db_path=face_db,
        review_dir=tmp_path / "review",
    )

    assert data["version"] == "1.0"


def test_voice_build_skips_existing_db_before_clip_or_model_checks(tmp_path: Path) -> None:
    voice_db = tmp_path / "voice_db.json"
    voice_db.write_text(json.dumps({"version": "1.0", "clusters": []}), encoding="utf-8")

    data = build_voice_db(
        clips=[],
        voice_db_path=voice_db,
        review_dir=tmp_path / "review",
    )

    assert data["version"] == "1.0"
