from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from unrender.manifests import ShotRecord, read_json, write_json
from unrender.speakers import (
    canonical_speaker_name,
    configured_speakers,
    resolve_speaker_label,
)


@dataclass(frozen=True)
class FaceDbVote:
    shot_id: str
    accepted: list[str]
    raw_votes: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class VoiceDbMatch:
    shot_id: str
    accepted: list[str]
    raw_matches: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SpeakerResolution:
    shot_id: str
    proposed_value: str
    source: str
    existing_value: str = ""
    notes: str = ""


def load_face_db_votes(path: Path) -> dict[str, FaceDbVote]:
    if not path.exists():
        return {}
    data = read_json(path)
    votes: dict[str, FaceDbVote] = {}
    for entry in data.get("shots") or []:
        shot_id = str(entry.get("shot_id") or "").strip()
        accepted = _speaker_list(entry.get("accepted") or entry.get("proposed_value") or "")
        if shot_id and accepted:
            votes[shot_id] = FaceDbVote(
                shot_id=shot_id,
                accepted=accepted,
                raw_votes={str(k): int(v) for k, v in (entry.get("votes") or {}).items()},
            )
    return votes


def load_voice_db_matches(path: Path) -> dict[str, VoiceDbMatch]:
    if not path.exists():
        return {}
    data = read_json(path)
    matches: dict[str, VoiceDbMatch] = {}
    if data.get("clusters"):
        for cluster in data.get("clusters") or []:
            name = canonical_speaker_name(str(cluster.get("speaker") or cluster.get("name") or ""))
            if not name:
                continue
            for clip in cluster.get("clips") or []:
                shot_id = str(clip.get("shot_id") or "").strip()
                if not shot_id:
                    continue
                existing = matches.setdefault(
                    shot_id,
                    VoiceDbMatch(shot_id=shot_id, accepted=[], raw_matches=[]),
                )
                existing.accepted.append(name)
                existing.raw_matches.append({"speaker": name, "source": "voice-db-cluster"})
        return matches

    for entry in data.get("shots") or []:
        shot_id = str(entry.get("shot_id") or "").strip()
        accepted = _speaker_list(entry.get("accepted") or entry.get("proposed_value") or "")
        if shot_id and accepted:
            matches[shot_id] = VoiceDbMatch(
                shot_id=shot_id,
                accepted=accepted,
                raw_matches=list(entry.get("matches") or []),
            )
    return matches


def resolve_speakers(
    shots: Sequence[ShotRecord | Mapping[str, Any]],
    *,
    face_db_votes: dict[str, FaceDbVote] | None = None,
    voice_db_matches: dict[str, VoiceDbMatch] | None = None,
    config: Mapping[str, Any] | None = None,
    only_missing: bool = True,
) -> list[SpeakerResolution]:
    face_db_votes = face_db_votes or {}
    voice_db_matches = voice_db_matches or {}
    configured = configured_speakers(config)
    resolutions: list[SpeakerResolution] = []
    identity_db: dict[str, Any] = {"speakers": {}}
    for speaker in configured.values():
        identity_db["speakers"][speaker["keys"][0]] = {
            "name": speaker["name"],
            "aliases": speaker["aliases"],
            "keys": speaker["keys"],
        }

    for shot in shots:
        shot_id, existing = _shot_id_and_existing(shot)
        if not shot_id:
            continue
        if only_missing and existing:
            resolutions.append(
                SpeakerResolution(
                    shot_id=shot_id,
                    proposed_value=existing,
                    source="existing",
                    existing_value=existing,
                )
            )
            continue
        if shot_id in face_db_votes:
            value = _canonical_join(face_db_votes[shot_id].accepted, identity_db, config)
            if value:
                resolutions.append(
                    SpeakerResolution(
                        shot_id=shot_id,
                        proposed_value=value,
                        source="face-db",
                        existing_value=existing,
                    )
                )
                continue
        if shot_id in voice_db_matches:
            value = _canonical_join(voice_db_matches[shot_id].accepted, identity_db, config)
            if value:
                resolutions.append(
                    SpeakerResolution(
                        shot_id=shot_id,
                        proposed_value=value,
                        source="voice-db",
                        existing_value=existing,
                    )
                )
                continue
        resolutions.append(
            SpeakerResolution(
                shot_id=shot_id,
                proposed_value="",
                source="unresolved",
                existing_value=existing,
            )
        )
    return resolutions


def write_resolution_plan(path: Path, resolutions: list[SpeakerResolution]) -> None:
    write_json(
        path,
        [
            {
                "shot_id": resolution.shot_id,
                "proposed_value": resolution.proposed_value,
                "source": resolution.source,
                "existing_value": resolution.existing_value,
                "notes": resolution.notes,
            }
            for resolution in resolutions
        ],
    )


def _canonical_join(
    speakers: list[str],
    identity_db: dict[str, Any],
    config: Mapping[str, Any] | None,
) -> str:
    out: list[str] = []
    allow_custom = not configured_speakers(config)
    for speaker in speakers:
        canonical = resolve_speaker_label(speaker, identity_db, allow_custom=allow_custom)
        if canonical and canonical not in out:
            out.append(canonical)
    return ", ".join(out)


def _shot_id_and_existing(shot: ShotRecord | Mapping[str, Any]) -> tuple[str, str]:
    if isinstance(shot, ShotRecord):
        return shot.shot_id, shot.existing_speaker
    return (
        str(shot.get("shot_id") or shot.get("id") or "").strip(),
        str(shot.get("existing_speaker") or shot.get("speaker") or "").strip(),
    )


def _speaker_list(value: Any) -> list[str]:
    if isinstance(value, list):
        raw = value
    else:
        raw = str(value or "").replace(";", ",").split(",")
    return [canonical_speaker_name(str(item)) for item in raw if canonical_speaker_name(str(item))]
