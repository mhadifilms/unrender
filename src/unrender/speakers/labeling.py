from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Any

from unrender.lib.csv_io import write_rows
from unrender.manifests import read_json, write_json
from unrender.project import RunPaths
from unrender.speakers import (
    configured_speakers,
    load_speaker_db,
    resolve_speaker_label,
    save_speaker_db,
    seed_config_speakers,
    upsert_face_clusters,
    upsert_voice_clusters,
)

LABEL_FIELDS = [
    "cluster_ref",
    "type",
    "cluster_id",
    "current_name",
    "speaker",
    "review_path",
    "evidence_count",
]


def write_label_template(run: RunPaths, labels_path: Path | None = None) -> Path:
    run.ensure()
    output = labels_path or run.labels_csv
    rows: list[dict[str, Any]] = []
    if run.face_db.exists():
        face_db = read_json(run.face_db)
        for cluster in face_db.get("clusters") or []:
            rows.append(
                {
                    "cluster_ref": f"face:{cluster['cluster_id']}",
                    "type": "face",
                    "cluster_id": cluster["cluster_id"],
                    "current_name": cluster.get("name", ""),
                    "speaker": cluster.get("speaker") or cluster.get("name") or "",
                    "review_path": cluster.get("grid_path", ""),
                    "evidence_count": cluster.get("face_count", ""),
                }
            )
    if run.voice_db.exists():
        voice_db = read_json(run.voice_db)
        for cluster in voice_db.get("clusters") or []:
            rows.append(
                {
                    "cluster_ref": f"voice:{cluster['cluster_id']}",
                    "type": "voice",
                    "cluster_id": cluster["cluster_id"],
                    "current_name": cluster.get("name", ""),
                    "speaker": cluster.get("speaker") or cluster.get("name") or "",
                    "review_path": cluster.get("review_dir", ""),
                    "evidence_count": cluster.get("sample_count", ""),
                }
            )
    write_rows(output, rows, LABEL_FIELDS)
    print(f"Label template saved: {output}", flush=True)
    return output


def apply_labels(
    run: RunPaths,
    labels_path: Path,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    if not labels_path.exists():
        raise FileNotFoundError(f"labels CSV not found: {labels_path}")
    _require_configured_speakers(config)
    face_db = read_json(run.face_db) if run.face_db.exists() else {"clusters": []}
    voice_db = read_json(run.voice_db) if run.voice_db.exists() else {"clusters": []}
    speaker_db = load_speaker_db(run.speaker_db)
    seed_config_speakers(speaker_db, config)

    face_clusters = {
        int(cluster["cluster_id"]): cluster for cluster in face_db.get("clusters") or []
    }
    voice_clusters = {
        int(cluster["cluster_id"]): cluster for cluster in voice_db.get("clusters") or []
    }
    applied = 0
    skipped = 0
    with labels_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cluster_type = str(row.get("type") or "").strip().lower()
            cluster_id = int(str(row.get("cluster_id") or "0"))
            label = str(row.get("speaker") or "").strip()
            if not label:
                skipped += 1
                continue
            canonical = resolve_speaker_label(label, speaker_db, allow_custom=False)
            if not canonical:
                skipped += 1
                continue
            if cluster_type == "face" and cluster_id in face_clusters:
                face_clusters[cluster_id]["name"] = canonical
                face_clusters[cluster_id]["speaker"] = canonical
                applied += 1
            elif cluster_type == "voice" and cluster_id in voice_clusters:
                voice_clusters[cluster_id]["name"] = canonical
                voice_clusters[cluster_id]["speaker"] = canonical
                applied += 1
            else:
                skipped += 1

    upsert_face_clusters(speaker_db, face_db.get("clusters") or [], allow_custom=False)
    upsert_voice_clusters(speaker_db, voice_db.get("clusters") or [], allow_custom=False)
    if run.face_db.exists():
        write_json(run.face_db, face_db)
    if run.voice_db.exists():
        write_json(run.voice_db, voice_db)
    save_speaker_db(run.speaker_db, speaker_db)
    print(f"Labels applied: {applied}; skipped: {skipped}; speaker DB saved: {run.speaker_db}")
    return speaker_db


def label_interactive(
    run: RunPaths,
    *,
    config: dict[str, Any],
    cluster_type: str = "all",
    relabel_all: bool = False,
    label_skipped: bool = False,
    preview: bool = False,
) -> dict[str, Any]:
    _require_configured_speakers(config)
    face_db = read_json(run.face_db) if run.face_db.exists() else {"clusters": []}
    voice_db = read_json(run.voice_db) if run.voice_db.exists() else {"clusters": []}
    speaker_db = load_speaker_db(run.speaker_db)
    seed_config_speakers(speaker_db, config)
    speakers = [meta["name"] for meta in configured_speakers(config).values()]
    speakers = sorted(set(speakers))

    if cluster_type in {"all", "face"}:
        _label_clusters_interactively(
            face_db.get("clusters") or [],
            kind="face",
            speakers=speakers,
            speaker_db=speaker_db,
            relabel_all=relabel_all,
            label_skipped=label_skipped,
            preview=preview,
        )
    if cluster_type in {"all", "voice"}:
        _label_clusters_interactively(
            voice_db.get("clusters") or [],
            kind="voice",
            speakers=speakers,
            speaker_db=speaker_db,
            relabel_all=relabel_all,
            label_skipped=label_skipped,
            preview=preview,
        )

    upsert_face_clusters(speaker_db, face_db.get("clusters") or [], allow_custom=False)
    upsert_voice_clusters(speaker_db, voice_db.get("clusters") or [], allow_custom=False)
    if run.face_db.exists():
        write_json(run.face_db, face_db)
    if run.voice_db.exists():
        write_json(run.voice_db, voice_db)
    save_speaker_db(run.speaker_db, speaker_db)
    print(f"Speaker DB saved: {run.speaker_db}")
    return speaker_db


def _label_clusters_interactively(
    clusters: list[dict[str, Any]],
    *,
    kind: str,
    speakers: list[str],
    speaker_db: dict[str, Any],
    relabel_all: bool,
    label_skipped: bool,
    preview: bool,
) -> None:
    def evidence_count(cluster: dict[str, Any]) -> int:
        return int(cluster.get("face_count") or cluster.get("sample_count") or 0)

    targets = [
        cluster
        for cluster in sorted(clusters, key=lambda c: -evidence_count(c))
        if relabel_all
        or (label_skipped and cluster.get("skipped"))
        or (
            not cluster.get("skipped")
            and not str(cluster.get("speaker") or cluster.get("name") or "").strip()
        )
    ]
    if not targets:
        print(f"No {kind} clusters need labels.")
        return

    print(
        f"\nLabeling {len(targets)} {kind} cluster(s). "
        "Choose a configured speaker or Enter twice to skip."
    )
    for cluster in targets:
        evidence = cluster.get("face_count") or cluster.get("sample_count") or 0
        review_path = cluster.get("grid_path") if kind == "face" else cluster.get("review_dir")
        print(f"\n{kind}:{cluster['cluster_id']} evidence={evidence}")
        if review_path:
            print(f"review: {review_path}")
            if preview:
                _open_review_path(Path(str(review_path)))
        print("available speakers:")
        for index, speaker in enumerate(speakers, 1):
            print(f"  {index}. {speaker}")
        while True:
            choice = input("speaker number/name (Enter to skip): ").strip()
            if not choice:
                confirm = input(
                    f"confirm skip {kind}:{cluster['cluster_id']} (Enter=yes): "
                ).strip()
                if confirm:
                    choice = confirm
                else:
                    cluster["skipped"] = True
                    cluster.pop("name", None)
                    cluster.pop("speaker", None)
                    print(f"  skipped {kind}:{cluster['cluster_id']}")
                    break
            else:
                cluster.pop("skipped", None)
            if not choice:
                continue
            label = _choice_to_speaker(choice, speakers)
            canonical = resolve_speaker_label(label, speaker_db, allow_custom=False)
            if not canonical:
                print(
                    f"  invalid: {choice!r} is not in the configured speaker set; "
                    "choose one of the listed options or Enter twice to skip"
                )
                continue
            cluster["name"] = canonical
            cluster["speaker"] = canonical
            print(f"  mapped {kind}:{cluster['cluster_id']} -> {canonical}")
            break


def _choice_to_speaker(choice: str, speakers: list[str]) -> str:
    if choice.isdigit():
        index = int(choice)
        if 1 <= index <= len(speakers):
            return speakers[index - 1]
    return choice


def _existing_canonical_label(
    cluster: dict[str, Any],
    speaker_db: dict[str, Any],
) -> str:
    label = str(cluster.get("speaker") or cluster.get("name") or "").strip()
    if not label:
        return ""
    return resolve_speaker_label(label, speaker_db, allow_custom=False)


def _open_review_path(path: Path) -> None:
    if not path.exists():
        print(f"  preview skipped: missing {path}", flush=True)
        return
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", "-a", "Preview", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except OSError as exc:
        print(f"  preview failed: {exc}", flush=True)


def _require_configured_speakers(config: dict[str, Any]) -> None:
    if not configured_speakers(config):
        raise ValueError("labeling requires a config with a non-empty 'speakers' set")
