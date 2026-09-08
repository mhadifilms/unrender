from __future__ import annotations

from pathlib import Path
from typing import Any

from unrender.lib.csv_io import write_rows


def _write_voice_clips_csv(path: Path, clips: list[dict[str, Any]]) -> None:
    fields = [
        "clip_id",
        "clip_type",
        "line_id",
        "shot_id",
        "source_group",
        "speaker",
        "source_stem",
        "path",
        "start_sec",
        "end_sec",
        "clip_start_sec",
        "clip_end_sec",
        "peak_dbfs",
        "rms_dbfs",
        "voiced_ratio",
        "kept",
        "reason",
        "text",
    ]
    write_rows(path, clips, fields)


def _write_clip_plan_csv(path: Path, plan: list[dict[str, Any]]) -> None:
    fields = [
        "clip_id",
        "clip_type",
        "line_id",
        "shot_id",
        "speaker",
        "on_screen_speakers",
        "on_screen_match",
        "stem_path",
        "source_stem",
        "source_group",
        "clip_start_sec",
        "clip_end_sec",
        "score",
        "status",
        "text",
    ]
    write_rows(path, plan, fields)


def _write_dialogue_plan_csv(path: Path, plan: list[dict[str, Any]]) -> None:
    fields = [
        "clip_id",
        "line_id",
        "speaker",
        "stem_path",
        "raw_stem_path",
        "dialogue_stem_path",
        "source_stem",
        "source_group",
        "score",
        "status",
        "text",
    ]
    write_rows(path, plan, fields)


def _write_shot_dialogue_csv(path: Path, mappings: list[dict[str, Any]]) -> None:
    fields = [
        "shot_id",
        "line_id",
        "speaker",
        "stem_path",
        "dialogue_stem_path",
        "raw_stem_path",
        "source_stem",
        "shot_stem_path",
        "source_group",
        "start_sec",
        "end_sec",
        "overlap_start_sec",
        "overlap_end_sec",
        "overlap_ratio",
        "status",
        "score",
        "text",
    ]
    write_rows(path, mappings, fields)
