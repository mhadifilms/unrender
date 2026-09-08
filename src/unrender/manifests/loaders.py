from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path
from typing import Any

from unrender.lib.csv_io import write_rows
from unrender.lib.json_io import read_json as _read_json
from unrender.lib.json_io import write_json as _write_json
from unrender.lib.names import source_group_from_text
from unrender.manifests.models import DialogueLineRecord, ShotRecord, VoiceInput


def read_json(path: Path) -> Any:
    return _read_json(path)


def write_json(path: Path, data: Any) -> None:
    _write_json(path, data)


def load_shot_manifest(path: Path) -> list[ShotRecord]:
    if not path.exists():
        raise FileNotFoundError(f"shot manifest not found: {path}")
    if path.suffix.lower() == ".json":
        raw = read_json(path)
        rows = raw.get("shots", raw) if isinstance(raw, dict) else raw
        if not isinstance(rows, list):
            raise ValueError("shot JSON must be a list or object with a 'shots' list")
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

    records: list[ShotRecord] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"shot row {index} is not an object")
        shot_id = str(row.get("shot_id") or row.get("id") or "").strip()
        video_text = str(
            row.get("video_path") or row.get("path") or row.get("file_path") or ""
        ).strip()
        if not shot_id:
            raise ValueError(f"shot row {index} is missing shot_id")
        if not video_text:
            raise ValueError(f"shot row {index} is missing video_path/path/file_path")
        video_path = Path(video_text).expanduser()
        records.append(
            ShotRecord(
                shot_id=shot_id,
                video_path=video_path,
                existing_speaker=str(
                    row.get("existing_speaker") or row.get("speaker") or ""
                ).strip(),
                start_sec=_optional_float(row.get("start_sec")),
                end_sec=_optional_float(row.get("end_sec")),
                target_face_box=_optional_box(row),
            )
        )
    return records


def load_dialogue_lines(path: Path) -> list[DialogueLineRecord]:
    if not path.exists():
        raise FileNotFoundError(f"dialogue lines not found: {path}")
    if path.suffix.lower() == ".json":
        raw = read_json(path)
        rows = raw.get("lines", raw) if isinstance(raw, dict) else raw
        if not isinstance(rows, list):
            raise ValueError("dialogue line JSON must be a list or object with a 'lines' list")
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

    records: list[DialogueLineRecord] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"dialogue line row {index} is not an object")
        line_id = str(row.get("line_id") or row.get("id") or "").strip()
        if not line_id:
            raise ValueError(f"dialogue line row {index} is missing line_id")
        records.append(
            DialogueLineRecord(
                line_id=line_id,
                start_sec=float(row.get("start_sec") or row.get("start") or 0.0),
                end_sec=float(row.get("end_sec") or row.get("end") or 0.0),
                cut_start_sec=_optional_float(row.get("cut_start_sec")),
                cut_end_sec=_optional_float(row.get("cut_end_sec")),
                text=str(row.get("text") or "").strip(),
                diarized_speaker=str(row.get("diarized_speaker") or "").strip(),
                source_group=str(
                    row.get("source_group") or row.get("diarized_speaker") or ""
                ).strip(),
                speaker=str(row.get("speaker") or row.get("char") or "").strip(),
                transcription_source=str(row.get("transcription_source") or "").strip(),
                review_required=_truthy(row.get("review_required")),
            )
        )
    return records


def write_dialogue_lines_csv(path: Path, lines: list[dict[str, Any]]) -> None:
    fields = [
        "line_id",
        "start_sec",
        "end_sec",
        "cut_start_sec",
        "cut_end_sec",
        "diarized_speaker",
        "source_group",
        "speaker",
        "transcription_source",
        "review_required",
        "text",
    ]
    write_rows(path, lines, fields)


def load_voice_inputs(spec: str) -> list[VoiceInput]:
    candidate = Path(spec).expanduser()
    if candidate.exists() and candidate.suffix.lower() in {".csv", ".json"}:
        return _load_voice_manifest(candidate)

    paths = [Path(p).expanduser() for p in sorted(glob.glob(spec))]
    if not paths:
        raise FileNotFoundError(f"no voice clips matched: {spec}")
    inputs: list[VoiceInput] = []
    for path in paths:
        inputs.append(
            VoiceInput(
                path=path,
                clip_id=path.stem,
                shot_id=_infer_shot_id(path),
                source_group=_infer_source_group(path),
            )
        )
    return inputs


def write_shot_matches_csv(path: Path, matches: list[dict[str, Any]]) -> None:
    fields = ["shot_id", "proposed_value", "accepted", "faces_detected", "votes", "video_path"]
    rows = []
    for match in matches:
        accepted = match.get("accepted") or []
        rows.append(
            {
                "shot_id": match.get("shot_id", ""),
                "proposed_value": match.get("proposed_value", ""),
                "accepted": ", ".join(accepted),
                "faces_detected": match.get("faces_detected", 0),
                "votes": json.dumps(match.get("votes") or {}, sort_keys=True),
                "video_path": match.get("video_path", ""),
            }
        )
    write_rows(path, rows, fields)


def _load_voice_manifest(path: Path) -> list[VoiceInput]:
    if path.suffix.lower() == ".json":
        raw = read_json(path)
        rows = raw.get("clips", raw) if isinstance(raw, dict) else raw
        if not isinstance(rows, list):
            raise ValueError("voice JSON must be a list or object with a 'clips' list")
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

    inputs: list[VoiceInput] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"voice row {index} is not an object")
        if "kept" in row and not _truthy(row.get("kept")):
            continue
        path_text = str(row.get("path") or row.get("clip_path") or "").strip()
        if not path_text:
            raise ValueError(f"voice row {index} is missing path/clip_path")
        embedding = _parse_embedding(row.get("embedding"))
        inputs.append(
            VoiceInput(
                path=Path(path_text).expanduser(),
                clip_id=str(row.get("clip_id") or "").strip(),
                clip_type=str(row.get("clip_type") or "").strip(),
                line_id=str(row.get("line_id") or "").strip(),
                shot_id=str(row.get("shot_id") or "").strip(),
                source_group=str(row.get("source_group") or row.get("speaker_group") or "").strip(),
                source_stem=str(row.get("source_stem") or "").strip(),
                start_sec=_optional_float(row.get("start_sec")),
                end_sec=_optional_float(row.get("end_sec")),
                clip_start_sec=_optional_float(row.get("clip_start_sec")),
                clip_end_sec=_optional_float(row.get("clip_end_sec")),
                text=str(row.get("text") or "").strip(),
                embedding=embedding,
            )
        )
    return inputs


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"", "0", "false", "no", "n"}


def _parse_embedding(value: Any) -> tuple[float, ...] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("embedding must be a JSON list of floats")
    return tuple(float(v) for v in value)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_box(row: dict[str, Any]) -> tuple[int, int, int, int] | None:
    value = row.get("target_face_box") or row.get("face_box") or row.get("bbox")
    if value not in (None, ""):
        if isinstance(value, str):
            parsed = json.loads(value)
        else:
            parsed = value
        if not isinstance(parsed, list) or len(parsed) != 4:
            raise ValueError("target_face_box must be a JSON list [x, y, width, height]")
        return tuple(int(float(str(item))) for item in parsed)  # type: ignore[return-value]

    names = [
        ("target_x", "target_y", "target_w", "target_h"),
        ("face_x", "face_y", "face_w", "face_h"),
    ]
    for keys in names:
        raw = [row.get(key) for key in keys]
        if any(item not in (None, "") for item in raw):
            if not all(item not in (None, "") for item in raw):
                raise ValueError(f"incomplete target face box columns: {', '.join(keys)}")
            return tuple(int(float(str(item))) for item in raw)  # type: ignore[return-value]
    return None


def _infer_shot_id(path: Path) -> str:

    speaker_match = re.match(r"(.+?)[_-]speaker[_-]?\d+", path.stem, flags=re.IGNORECASE)
    if speaker_match:
        return speaker_match.group(1).rstrip("_-")
    parent_name = path.parent.name
    if parent_name and path.stem.upper().startswith(f"{parent_name.upper()}_"):
        return parent_name
    for part in (
        path.stem,
        path.parent.name,
        path.parent.parent.name if path.parent.parent else "",
    ):
        chunks = [chunk for chunk in part.replace("-", "_").split("_") if chunk]
        for chunk in chunks:
            if chunk.isdigit():
                return chunk
    return ""


def _infer_source_group(path: Path) -> str:
    group = source_group_from_text(path.stem)
    if group:
        return group
    stem = re.sub(r"[_-]?stem$", "", path.stem, flags=re.IGNORECASE)
    shot = _infer_shot_id(path)
    if shot and stem.upper().startswith(f"{shot.upper()}_"):
        stem = stem[len(shot) + 1 :]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem.strip())
    return safe.strip("._")
