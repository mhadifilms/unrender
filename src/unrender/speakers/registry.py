from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from unrender.manifests import read_json, write_json
from unrender.workflows import (
    MANUAL_WORKFLOW,
    WorkflowProvenance,
)
from unrender.workflows import (
    workflow_provenance as validate_workflow_provenance,
)


def speaker_key(name: str) -> str:
    text = str(name or "").strip().upper()
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return "".join(ch for ch in ascii_text if ch.isalnum())


def canonical_speaker_name(name: str) -> str:
    return str(name or "").strip().upper()


def configured_speakers(config: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not config:
        return {}
    raw = config.get("speakers") or config.get("prepare", {}).get("speakers")
    if isinstance(raw, Mapping):
        speakers = {}
        for key, value in raw.items():
            if isinstance(value, Mapping):
                name = canonical_speaker_name(str(value.get("name") or key))
                aliases = [canonical_speaker_name(alias) for alias in value.get("aliases") or []]
                keys = {speaker_key(name), *[speaker_key(alias) for alias in aliases]}
            else:
                name = canonical_speaker_name(str(key))
                aliases = []
                keys = {speaker_key(name)}
            keys.discard("")
            speakers[speaker_key(name)] = {
                "name": name,
                "aliases": sorted(set(aliases)),
                "keys": sorted(keys),
                "custom": bool(isinstance(value, Mapping) and value.get("custom", False)),
            }
        return speakers
    if isinstance(raw, list):
        return {
            speaker_key(str(name)): {
                "name": canonical_speaker_name(str(name)),
                "aliases": [],
                "keys": [speaker_key(str(name))],
                "custom": False,
            }
            for name in raw
            if speaker_key(str(name))
        }
    return {}


def empty_speaker_db() -> dict[str, Any]:
    return {"version": "1.0", "speakers": {}, "face_clusters": [], "voice_clusters": []}


def load_speaker_db(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_speaker_db()
    data = read_json(path)
    db = empty_speaker_db()
    if isinstance(data, Mapping):
        db.update({key: value for key, value in data.items() if key in db})
        raw_workflow = data.get("workflow_provenance")
        if raw_workflow not in (None, ""):
            db["workflow_provenance"] = validate_workflow_provenance(str(raw_workflow))
    db["speakers"] = dict(db.get("speakers") or {})
    db["face_clusters"] = list(db.get("face_clusters") or [])
    db["voice_clusters"] = list(db.get("voice_clusters") or [])
    return db


def save_speaker_db(
    path: Path,
    db: Mapping[str, Any],
    *,
    workflow_provenance: WorkflowProvenance = MANUAL_WORKFLOW,
) -> None:
    payload = dict(db)
    payload["workflow_provenance"] = validate_workflow_provenance(workflow_provenance)
    if isinstance(db, dict):
        db["workflow_provenance"] = payload["workflow_provenance"]
    write_json(path, payload)


def seed_config_speakers(db: dict[str, Any], config: Mapping[str, Any] | None) -> None:
    speakers = db.setdefault("speakers", {})
    for key, meta in configured_speakers(config).items():
        existing = speakers.get(key, {})
        speakers[key] = {
            **existing,
            "name": meta["name"],
            "aliases": sorted(set(existing.get("aliases") or []) | set(meta.get("aliases") or [])),
            "keys": sorted(set(existing.get("keys") or []) | set(meta.get("keys") or [])),
            "custom": bool(existing.get("custom", False) or meta.get("custom", False)),
            "face_clusters": sorted(set(existing.get("face_clusters") or [])),
            "voice_clusters": sorted(set(existing.get("voice_clusters") or [])),
        }


def allow_custom_speaker_labels(config: Mapping[str, Any] | None) -> bool:
    return not bool(configured_speakers(config))


def resolve_speaker_label(
    label: str,
    db: dict[str, Any],
    *,
    allow_custom: bool = True,
) -> str:
    text = canonical_speaker_name(label)
    key = speaker_key(text)
    if not key:
        return ""
    speakers = db.setdefault("speakers", {})
    for speaker_id, meta in speakers.items():
        keys = {
            speaker_key(meta.get("name", "")),
            *[speaker_key(alias) for alias in meta.get("aliases") or []],
            *[speaker_key(k) for k in meta.get("keys") or []],
            speaker_key(speaker_id),
        }
        if key in keys:
            return canonical_speaker_name(str(meta.get("name") or speaker_id))
    if not allow_custom:
        return ""
    speakers[key] = {
        "name": text,
        "aliases": [],
        "keys": [key],
        "custom": True,
        "face_clusters": [],
        "voice_clusters": [],
    }
    return text


def upsert_face_clusters(
    db: dict[str, Any],
    clusters: Iterable[Mapping[str, Any]],
    *,
    allow_custom: bool = True,
) -> None:
    db["face_clusters"] = [_identity_cluster(cluster) for cluster in clusters]
    _link_named_clusters(db, db["face_clusters"], "face_clusters", allow_custom=allow_custom)


def upsert_voice_clusters(
    db: dict[str, Any],
    clusters: Iterable[Mapping[str, Any]],
    *,
    allow_custom: bool = True,
) -> None:
    db["voice_clusters"] = [_identity_cluster(cluster) for cluster in clusters]
    _link_named_clusters(db, db["voice_clusters"], "voice_clusters", allow_custom=allow_custom)


def load_labeled_face_centroids(db: Mapping[str, Any]) -> tuple[list[str], list[list[float]]]:
    return _load_labeled_centroids(db.get("face_clusters") or [])


def load_labeled_voice_centroids(db: Mapping[str, Any]) -> tuple[list[str], list[list[float]]]:
    return _load_labeled_centroids(db.get("voice_clusters") or [])


def _identity_cluster(cluster: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(cluster)
    out.pop("faces", None)
    out.pop("clips", None)
    return out


def _link_named_clusters(
    db: dict[str, Any],
    clusters: list[dict[str, Any]],
    field: str,
    *,
    allow_custom: bool,
) -> None:
    for cluster in clusters:
        name = canonical_speaker_name(str(cluster.get("speaker") or cluster.get("name") or ""))
        if not name:
            continue
        canonical = resolve_speaker_label(name, db, allow_custom=allow_custom)
        if not canonical:
            cluster["speaker"] = ""
            cluster["name"] = ""
            continue
        cluster["speaker"] = canonical
        cluster["name"] = canonical
        _link_cluster(db, speaker=canonical, cluster_id=int(cluster["cluster_id"]), field=field)


def _link_cluster(db: dict[str, Any], *, speaker: str, cluster_id: int, field: str) -> None:
    key = speaker_key(speaker)
    if not key:
        return
    meta = db.setdefault("speakers", {}).setdefault(
        key,
        {
            "name": canonical_speaker_name(speaker),
            "aliases": [],
            "keys": [key],
            "custom": True,
            "face_clusters": [],
            "voice_clusters": [],
        },
    )
    values = {int(v) for v in meta.get(field) or []}
    values.add(int(cluster_id))
    meta[field] = sorted(values)


def _load_labeled_centroids(
    clusters: Iterable[Mapping[str, Any]],
) -> tuple[list[str], list[list[float]]]:
    names: list[str] = []
    centroids: list[list[float]] = []
    for cluster in clusters:
        name = canonical_speaker_name(str(cluster.get("speaker") or cluster.get("name") or ""))
        centroid = cluster.get("centroid")
        if name and isinstance(centroid, list):
            names.append(name)
            centroids.append(centroid)
    return names, centroids
