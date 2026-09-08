from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unrender.lib.timecode import parse_time_value
from unrender.speakers import canonical_speaker_name, speaker_key

SKIP_CHARACTERS = {"NARRATIVE TITLE", "MAIN TITLE", "LAST FRAME OF PICTURE"}
ONSCREEN_PREFIXES = ("ON-SCREEN TEXT", "ON-SCEEN TEXT")
SKIP_DIALOGUE_RE = re.compile(
    r"^<(walla|laughing walla|sings|sing|indistinct)[^>]*>\s*(\(|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TranscriptLine:
    start_sec: float
    end_sec: float
    speaker: str
    text: str
    line_id: str = ""
    start_tc: str = ""
    end_tc: str = ""


def parse_transcript(
    path: Path,
    *,
    fps: float = 24.0,
    default_duration_sec: float = 3.0,
    min_duration_sec: float = 0.3,
    handle_sec: float = 0.18,
    include_onscreen_text: bool = False,
) -> list[dict[str, Any]]:
    rows = _load_rows(path)
    raw_lines = _rows_to_lines(rows, fps=fps, include_onscreen_text=include_onscreen_text)
    lines = _infer_missing_windows(
        raw_lines,
        default_duration_sec=default_duration_sec,
        min_duration_sec=min_duration_sec,
    )
    out: list[dict[str, Any]] = []
    for index, line in enumerate(lines, 1):
        line_id = line.line_id or f"DL_{index:06d}"
        out.append(
            {
                "line_id": line_id,
                "start_sec": round(line.start_sec, 3),
                "end_sec": round(line.end_sec, 3),
                "cut_start_sec": round(max(0.0, line.start_sec - handle_sec), 3),
                "cut_end_sec": round(max(line.start_sec, line.end_sec + handle_sec), 3),
                "diarized_speaker": "",
                "speaker": line.speaker,
                "text": line.text,
                "start_tc": line.start_tc,
                "end_tc": line.end_tc,
                "source": "transcript",
            }
        )
    return out


def extract_speakers_from_transcript(path: Path, *, fps: float = 24.0) -> list[str]:
    speakers: list[str] = []
    seen: set[str] = set()
    for line in parse_transcript(path, fps=fps):
        speaker = canonical_speaker_name(str(line.get("speaker") or ""))
        key = speaker_key(speaker)
        if not key or key in seen:
            continue
        seen.add(key)
        speakers.append(speaker)
    return speakers


def seconds_from_timecode(value: str, *, fps: float) -> float | None:
    # Strict timecode-only parse (bare numbers return None) so callers can tell
    # timecode cells apart from other text; drop-frame and NTSC handled upstream.
    return parse_time_value(value, fps)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"transcript not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("lines", data) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise ValueError("transcript JSON must be a list or object with a 'lines' list")
        return [dict(row) for row in rows if isinstance(row, dict)]
    if suffix == ".xlsx":
        return _load_xlsx_rows(path)
    if suffix == ".docx":
        return _load_docx_rows(path)
    raise ValueError(f"unsupported transcript format: {path.suffix}")


def _load_xlsx_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import openpyxl
    except ImportError as exc:
        raise ImportError("XLSX transcripts require openpyxl: pip install '.[scripts]'") from exc
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook.active
    rows = list(sheet.iter_rows(values_only=True))
    if not rows:
        return []
    header_index = _header_row_index(rows)
    headers = [str(cell or "").strip() for cell in rows[header_index]]
    out: list[dict[str, Any]] = []
    for raw in rows[header_index + 1 :]:
        if not any(cell not in (None, "") for cell in raw):
            continue
        out.append(
            {headers[index]: value for index, value in enumerate(raw) if index < len(headers)}
        )
    return out


def _load_docx_rows(path: Path) -> list[dict[str, Any]]:
    try:
        from docx import Document
    except ImportError as exc:
        raise ImportError("DOCX transcripts require python-docx: pip install '.[scripts]'") from exc
    doc = Document(str(path))
    if not doc.tables:
        return []
    table = max(doc.tables, key=lambda item: len(item.rows))
    if not table.rows:
        return []
    headers = [cell.text.strip() for cell in table.rows[0].cells]
    out: list[dict[str, Any]] = []
    for row in table.rows[1:]:
        cells = [cell.text.strip() for cell in row.cells]
        if not any(cells):
            continue
        out.append(
            {headers[index]: value for index, value in enumerate(cells) if index < len(headers)}
        )
    return out


def _rows_to_lines(
    rows: list[dict[str, Any]],
    *,
    fps: float,
    include_onscreen_text: bool,
) -> list[TranscriptLine]:
    lines: list[TranscriptLine] = []
    for index, row in enumerate(rows, 1):
        columns = {_normalize_header(key): key for key in row}
        start_value = _value(row, columns, START_COLUMNS)
        end_value = _value(row, columns, END_COLUMNS)
        speaker = canonical_speaker_name(_value(row, columns, SPEAKER_COLUMNS))
        text = _clean_text(_value(row, columns, TEXT_COLUMNS))
        if not speaker or not text:
            continue
        if speaker.upper() in SKIP_CHARACTERS:
            continue
        if not include_onscreen_text and any(
            speaker.upper().startswith(p) for p in ONSCREEN_PREFIXES
        ):
            continue
        if SKIP_DIALOGUE_RE.match(text):
            continue
        start_sec = _seconds_value(start_value, fps=fps)
        if start_sec is None:
            continue
        end_sec = _seconds_value(end_value, fps=fps)
        line_id = str(_value(row, columns, ID_COLUMNS) or "").strip()
        lines.append(
            TranscriptLine(
                line_id=line_id or f"DL_{index:06d}",
                start_sec=start_sec,
                end_sec=end_sec if end_sec is not None else start_sec,
                speaker=speaker,
                text=text,
                start_tc=str(start_value or "").strip(),
                end_tc=str(end_value or "").strip(),
            )
        )
    return sorted(lines, key=lambda line: line.start_sec)


def _infer_missing_windows(
    lines: list[TranscriptLine],
    *,
    default_duration_sec: float,
    min_duration_sec: float,
) -> list[TranscriptLine]:
    out: list[TranscriptLine] = []
    for index, line in enumerate(lines):
        if line.end_sec > line.start_sec:
            end_sec = line.end_sec
        else:
            next_start = (
                lines[index + 1].start_sec
                if index + 1 < len(lines)
                else line.start_sec + default_duration_sec
            )
            end_sec = min(next_start, line.start_sec + default_duration_sec)
        if end_sec - line.start_sec < min_duration_sec:
            end_sec = line.start_sec + min_duration_sec
        out.append(
            TranscriptLine(
                line_id=line.line_id,
                start_sec=line.start_sec,
                end_sec=end_sec,
                speaker=line.speaker,
                text=line.text,
                start_tc=line.start_tc,
                end_tc=line.end_tc,
            )
        )
    return out


def _seconds_value(value: Any, *, fps: float) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return seconds_from_timecode(text, fps=fps)


def _clean_text(value: Any) -> str:
    text = str(value or "").replace("\n", " ")
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"Change Note:.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(' ,."')


def _value(row: dict[str, Any], columns: dict[str, str], candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        key = columns.get(candidate)
        if key is not None:
            value = row.get(key)
            if value not in (None, ""):
                return str(value).strip()
    return ""


def _normalize_header(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return text.strip("_")


def _header_row_index(rows: list[tuple[Any, ...]]) -> int:
    for index, row in enumerate(rows[:10]):
        normalized = {_normalize_header(cell) for cell in row}
        if normalized.intersection(SPEAKER_COLUMNS) and normalized.intersection(TEXT_COLUMNS):
            return index
    return 0


ID_COLUMNS = ("line_id", "id", "cue_id", "number", "no")
START_COLUMNS = (
    "start_sec",
    "start",
    "in",
    "in_sec",
    "time_sec",
    "time",
    "start_tc",
    "start_timecode",
    "start_time_code",
    "in_timecode",
    "in_time_code",
    "in_timecode",
    "tc",
    "timecode",
)
END_COLUMNS = (
    "end_sec",
    "end",
    "out",
    "out_sec",
    "end_tc",
    "end_timecode",
    "end_time_code",
    "out_timecode",
    "out_time_code",
)
SPEAKER_COLUMNS = (
    "speaker",
    "speaker_name",
    "character",
    "character_name",
    "source",
    "name",
    "on_screen_speaker",
    "onscreen_speaker",
)
TEXT_COLUMNS = (
    "dialogue",
    "dialog",
    "line",
    "text",
    "english",
    "english_dialogue",
)
