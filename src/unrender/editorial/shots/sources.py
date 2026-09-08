"""Shot-list ingestion: JSON manifests, CSV, plain text, and CMX 3600 EDL.

Every format is normalized to frame-exact ``CutRange`` records (end
exclusive) on the master timeline, where frame 0 is the first frame of the
master. Absolute timecode labels (e.g. shows starting at 01:00:00:00) are
shifted by a resolved timecode origin; values already expressed as seconds
or frames are taken as timeline-relative and never shifted.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unrender.lib.json_io import read_json
from unrender.lib.timecode import seconds_to_frames, smpte_to_frames

_TC_CHARS = (":", ";")


@dataclass(frozen=True)
class CutRange:
    shot_id: str
    start_frame: int
    end_frame: int  # exclusive
    label: str = ""

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True)
class CutList:
    ranges: tuple[CutRange, ...]
    origin_frame: int
    origin_source: str
    source_kind: str


def resolve_tc_origin(
    tc_origin: str | None,
    *,
    fps: float,
    probed_tc: str | None = None,
    auto_candidate_frames: int | None = None,
) -> tuple[int, str]:
    """Resolve the timecode label of master frame 0 to a frame number.

    ``auto_candidate_frames`` is the format-specific automatic guess (for EDL:
    the minimum record-in across events). Returns ``(frame, description)``.
    """
    value = (tc_origin or "").strip()
    if value.lower() == "zero":
        return 0, "explicit zero"
    if value and value.lower() != "auto":
        frames = smpte_to_frames(value, fps)
        if frames is None:
            raise ValueError(f"invalid --tc-origin timecode: {tc_origin!r}")
        return frames, f"explicit {value}"
    if value.lower() == "auto" and auto_candidate_frames is not None:
        return auto_candidate_frames, "auto (earliest event)"
    if not value and auto_candidate_frames is not None:
        return auto_candidate_frames, "auto (earliest event)"
    if probed_tc:
        frames = smpte_to_frames(probed_tc, fps)
        if frames is not None:
            return frames, f"master start timecode {probed_tc}"
    return 0, "default 00:00:00:00"


def load_cut_ranges(
    path: Path,
    *,
    fps: float,
    tc_origin: str | None = None,
    probed_tc: str | None = None,
    duration_frames: int | None = None,
    padding: int = 3,
    edl_source_tc: bool = False,
) -> CutList:
    """Load and validate shot ranges from a JSON/CSV/TXT/EDL file."""
    if not path.exists():
        raise FileNotFoundError(f"shot list not found: {path}")
    if fps <= 0:
        raise ValueError(f"fps must be positive to interpret timecodes: {fps}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        rows, kind, auto_origin = _rows_from_json(path), "json", None
    elif suffix == ".csv":
        rows, kind, auto_origin = _rows_from_csv(path), "csv", None
    elif suffix == ".txt":
        rows, kind, auto_origin = _rows_from_txt(path), "txt", None
    elif suffix == ".edl":
        rows, kind = _rows_from_edl(path, source_tc=edl_source_tc)
        auto_origin = _min_tc_frames(rows, fps)
    else:
        raise ValueError(f"unsupported shot list format {suffix!r}: {path} (json/csv/txt/edl)")

    if not rows:
        raise ValueError(f"no shots found in {path}")

    origin_frame, origin_source = resolve_tc_origin(
        tc_origin, fps=fps, probed_tc=probed_tc, auto_candidate_frames=auto_origin
    )
    print(f"Timecode origin: frame {origin_frame} ({origin_source})", flush=True)

    ranges = _build_ranges(
        rows,
        path=path,
        fps=fps,
        origin_frame=origin_frame,
        duration_frames=duration_frames,
        padding=padding,
    )
    return CutList(
        ranges=tuple(ranges),
        origin_frame=origin_frame,
        origin_source=origin_source,
        source_kind=kind,
    )


# ---------------------------------------------------------------------------
# Row extraction (one dict per shot, values still raw strings/numbers)


def _rows_from_json(path: Path) -> list[dict[str, Any]]:
    raw = read_json(path)
    rows = raw.get("shots", raw) if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError(f"shot JSON must be a list or an object with a 'shots' list: {path}")
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"shot row {index} is not an object: {path}")
        out.append(row)
    return out


_CSV_ID_KEYS = ("shot_id", "id", "shot", "event")
_CSV_START_KEYS = ("start_frame", "start_sec", "start_tc", "start", "src_in", "in")
_CSV_END_KEYS = ("end_frame", "end_sec", "end_tc", "end", "src_out", "out")


def _rows_from_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV shot list has no header row: {path}")
        raw_rows = list(reader)
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        row = {_normalize_key(key): value for key, value in raw.items() if key is not None}
        if not any(str(value or "").strip() for value in row.values()):
            continue
        start_key = next((key for key in _CSV_START_KEYS if key in row), None)
        end_key = next((key for key in _CSV_END_KEYS if key in row), None)
        if start_key is None or end_key is None:
            raise ValueError(
                f"CSV shot list {path} needs start/end columns "
                f"(accepted: {', '.join(_CSV_START_KEYS)} / {', '.join(_CSV_END_KEYS)})"
            )
        entry: dict[str, Any] = {start_key: row[start_key], end_key: row[end_key]}
        id_key = next((key for key in _CSV_ID_KEYS if key in row), None)
        if id_key is not None:
            entry["shot_id"] = row[id_key]
        rows.append(entry)
    return rows


def _rows_from_txt(path: Path) -> list[dict[str, Any]]:
    lines: list[tuple[int, list[str]]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        text = raw.split("#", 1)[0].strip()
        if not text:
            continue
        tokens = [token for token in re.split(r"[,\s]+", text) if token]
        lines.append((number, tokens))
    if not lines:
        return []
    counts = {len(tokens) for _, tokens in lines}
    if len(counts) > 1 or counts - {1, 2, 3}:
        expected = len(lines[0][1])
        bad = next((n, t) for n, t in lines if len(t) not in (1, 2, 3) or len(t) != expected)
        raise ValueError(
            f"TXT shot list {path} must use one format throughout: either one cut "
            f"timecode per line, 'start end', or 'shot_id start end' "
            f"(line {bad[0]}: {' '.join(bad[1])!r})"
        )
    count = counts.pop()
    if count == 1:
        return [{"cut_point": tokens[0]} for _, tokens in lines]
    if count == 2:
        return [{"start_tc": tokens[0], "end_tc": tokens[1]} for _, tokens in lines]
    return [
        {"shot_id": tokens[0], "start_tc": tokens[1], "end_tc": tokens[2]} for _, tokens in lines
    ]


_EDL_TC = r"\d{2}[:;]\d{2}[:;]\d{2}[:;]\d{2}"
_EDL_EVENT_RE = re.compile(
    rf"^(?P<event>\d+)\s+(?P<reel>\S+)\s+(?P<track>\S+)\s+(?P<transition>\S+)(?P<extra>\s+\S+)??"
    rf"\s+(?P<src_in>{_EDL_TC})\s+(?P<src_out>{_EDL_TC})"
    rf"\s+(?P<rec_in>{_EDL_TC})\s+(?P<rec_out>{_EDL_TC})\s*$"
)
_EDL_CLIP_RE = re.compile(r"^\*\s*FROM\s+CLIP\s+NAME:\s*(?P<name>.+)$", re.IGNORECASE)


def _rows_from_edl(path: Path, *, source_tc: bool) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        clip = _EDL_CLIP_RE.match(line)
        if clip and rows:
            rows[-1].setdefault("label", clip.group("name").strip())
            continue
        match = _EDL_EVENT_RE.match(line)
        if not match:
            continue
        track = match.group("track").upper()
        if "V" not in track and track != "B":
            continue  # audio-only event
        transition = match.group("transition").upper()
        if transition != "C":
            print(
                f"  WARNING: EDL event {match.group('event')} uses transition "
                f"{transition!r}; cutting at the event in-point.",
                flush=True,
            )
        in_key, out_key = ("src_in", "src_out") if source_tc else ("rec_in", "rec_out")
        rows.append(
            {
                "shot_id": match.group("event"),
                "start_tc": match.group(in_key),
                "end_tc": match.group(out_key),
            }
        )
    return rows, "edl"


# ---------------------------------------------------------------------------
# Normalization to CutRange


def _build_ranges(
    rows: list[dict[str, Any]],
    *,
    path: Path,
    fps: float,
    origin_frame: int,
    duration_frames: int | None,
    padding: int,
) -> list[CutRange]:
    if any("cut_point" in row for row in rows):
        ranges = _ranges_from_cut_points(
            rows, path=path, fps=fps, origin_frame=origin_frame, duration_frames=duration_frames
        )
    else:
        ranges = []
        for index, row in enumerate(rows, 1):
            start = _frame_value(row, "start", fps=fps, origin_frame=origin_frame)
            end = _frame_value(row, "end", fps=fps, origin_frame=origin_frame)
            if start is None or end is None:
                raise ValueError(f"shot row {index} in {path} is missing start/end values")
            shot_id = str(row.get("shot_id") or "").strip()
            ranges.append(
                CutRange(
                    shot_id=shot_id,
                    start_frame=start,
                    end_frame=end,
                    label=str(row.get("label") or "").strip(),
                )
            )

    ranges.sort(key=lambda item: (item.start_frame, item.end_frame))
    ranges = [
        (
            range_
            if range_.shot_id
            else CutRange(
                shot_id=f"{index:0{padding}d}",
                start_frame=range_.start_frame,
                end_frame=range_.end_frame,
                label=range_.label,
            )
        )
        for index, range_ in enumerate(ranges, 1)
    ]
    _validate_ranges(ranges, path=path, duration_frames=duration_frames, padding=padding)
    return [
        CutRange(
            shot_id=_pad_shot_id(range_.shot_id, padding),
            start_frame=range_.start_frame,
            end_frame=range_.end_frame,
            label=range_.label,
        )
        for range_ in ranges
    ]


def _ranges_from_cut_points(
    rows: list[dict[str, Any]],
    *,
    path: Path,
    fps: float,
    origin_frame: int,
    duration_frames: int | None,
) -> list[CutRange]:
    if duration_frames is None:
        raise ValueError(
            f"cut-point list {path} needs the master duration; provide the master "
            "so segments can be built"
        )
    cuts: list[int] = []
    for row in rows:
        frame = _to_frames(row["cut_point"], fps=fps, origin_frame=origin_frame)
        if frame is None:
            raise ValueError(f"invalid cut point in {path}: {row['cut_point']!r}")
        if 0 < frame < duration_frames:
            cuts.append(frame)
    edges = [0, *sorted(set(cuts)), duration_frames]
    return [
        CutRange(shot_id="", start_frame=edges[i], end_frame=edges[i + 1])
        for i in range(len(edges) - 1)
        if edges[i + 1] > edges[i]
    ]


def _validate_ranges(
    ranges: list[CutRange],
    *,
    path: Path,
    duration_frames: int | None,
    padding: int,
) -> None:
    seen: dict[str, int] = {}
    for range_ in ranges:
        key = _pad_shot_id(range_.shot_id, padding)
        seen[key] = seen.get(key, 0) + 1
    duplicates = sorted(key for key, count in seen.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate shot_id(s) in {path}: {', '.join(duplicates)}")

    for index, range_ in enumerate(ranges):
        if range_.start_frame < 0:
            raise ValueError(
                f"shot {range_.shot_id} starts before the master ({range_.start_frame}); "
                "check --tc-origin / shots.timecode_start"
            )
        if range_.end_frame <= range_.start_frame:
            raise ValueError(
                f"shot {range_.shot_id} has non-positive duration "
                f"({range_.start_frame}..{range_.end_frame})"
            )
        if duration_frames is not None:
            if range_.start_frame >= duration_frames:
                raise ValueError(
                    f"shot {range_.shot_id} starts at frame {range_.start_frame}, beyond the "
                    f"master ({duration_frames} frames); check --tc-origin"
                )
            if range_.end_frame > duration_frames:
                over = range_.end_frame - duration_frames
                if over > 1:
                    raise ValueError(
                        f"shot {range_.shot_id} ends {over} frames past the master "
                        f"({duration_frames} frames); check --tc-origin"
                    )
                print(
                    f"  WARNING: shot {range_.shot_id} clamped {over} frame(s) "
                    "to the master duration.",
                    flush=True,
                )
                ranges[index] = CutRange(
                    shot_id=range_.shot_id,
                    start_frame=range_.start_frame,
                    end_frame=duration_frames,
                    label=range_.label,
                )
        if index > 0:
            previous = ranges[index - 1]
            if range_.start_frame < previous.end_frame:
                print(
                    f"  WARNING: shot {range_.shot_id} overlaps {previous.shot_id} "
                    f"by {previous.end_frame - range_.start_frame} frame(s).",
                    flush=True,
                )
            elif range_.start_frame > previous.end_frame:
                print(
                    f"  WARNING: {range_.start_frame - previous.end_frame} frame(s) "
                    f"uncovered between shots {previous.shot_id} and {range_.shot_id}.",
                    flush=True,
                )


def _frame_value(
    row: dict[str, Any],
    which: str,
    *,
    fps: float,
    origin_frame: int,
) -> int | None:
    frame = row.get(f"{which}_frame")
    if frame not in (None, ""):
        return int(float(str(frame)))
    seconds = row.get(f"{which}_sec")
    if seconds not in (None, ""):
        return seconds_to_frames(float(str(seconds)), fps)
    for key in (f"{which}_tc", which, f"src_{'in' if which == 'start' else 'out'}"):
        value = row.get(key)
        if value not in (None, ""):
            return _to_frames(value, fps=fps, origin_frame=origin_frame)
    return None


def _to_frames(value: Any, *, fps: float, origin_frame: int) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    frames = smpte_to_frames(text, fps)
    if frames is None:
        raise ValueError(f"invalid timecode or seconds value: {text!r}")
    if any(char in text for char in _TC_CHARS):
        return frames - origin_frame
    return frames  # bare seconds are timeline-relative already


def _min_tc_frames(rows: list[dict[str, Any]], fps: float) -> int | None:
    frames = [
        smpte_to_frames(str(row.get("start_tc") or ""), fps) for row in rows if row.get("start_tc")
    ]
    values = [frame for frame in frames if frame is not None]
    return min(values) if values else None


def _pad_shot_id(shot_id: str, padding: int) -> str:
    text = shot_id.strip()
    if text.isdigit():
        return f"{int(text):0{padding}d}"
    return text


def _normalize_key(key: str) -> str:
    return re.sub(r"\s+", "_", str(key or "").strip().lower())
