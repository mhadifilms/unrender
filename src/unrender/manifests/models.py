from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ShotRecord:
    shot_id: str
    video_path: Path
    existing_speaker: str = ""
    start_sec: float | None = None
    end_sec: float | None = None
    target_face_box: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class DialogueLineRecord:
    line_id: str
    start_sec: float
    end_sec: float
    text: str = ""
    diarized_speaker: str = ""
    source_group: str = ""
    speaker: str = ""
    cut_start_sec: float | None = None
    cut_end_sec: float | None = None
    transcription_source: str = ""
    review_required: bool = False


@dataclass(frozen=True)
class ShotDialogueRecord:
    shot_id: str
    line_id: str
    start_sec: float
    end_sec: float
    overlap_start_sec: float
    overlap_end_sec: float
    overlap_ratio: float
    speaker: str = ""
    text: str = ""
    stem_path: str = ""
    source_group: str = ""


@dataclass(frozen=True)
class VoiceInput:
    path: Path
    shot_id: str = ""
    source_group: str = ""
    source_stem: str = ""
    clip_id: str = ""
    clip_type: str = ""
    line_id: str = ""
    start_sec: float | None = None
    end_sec: float | None = None
    clip_start_sec: float | None = None
    clip_end_sec: float | None = None
    text: str = ""
    embedding: tuple[float, ...] | None = None
