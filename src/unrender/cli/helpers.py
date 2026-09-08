from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from unrender.project import RunPaths


def configure_logging(args: argparse.Namespace) -> None:
    if getattr(args, "quiet", False):
        level = logging.ERROR
    elif getattr(args, "debug", False):
        level = logging.DEBUG
    else:
        verbose = int(getattr(args, "verbose", 0) or 0)
        level = logging.DEBUG if verbose > 1 else logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def artifact_status_paths(run: RunPaths) -> list[tuple[str, Path]]:
    return [
        ("shots_manifest", run.shots_manifest_json),
        ("face_db", run.face_db),
        ("voice_db", run.voice_db),
        ("speaker_db", run.speaker_db),
        ("labels_csv", run.labels_csv),
        ("source_separation", run.source_separation_json),
        ("dialogue_lines", run.dialogue_lines_json),
        ("voice_clips", run.voice_clips_json),
        ("shot_matches", run.shot_matches_json),
        ("voice_matches", run.voice_matches_json),
        ("clip_stem_plan", run.clip_stem_plan_json),
        ("dialogue_stem_plan", run.dialogue_stem_plan_json),
        ("shot_dialogue_map", run.shot_dialogue_map_json),
        ("shot_stem_plan", run.shot_stem_plan_json),
        ("shot_dx_plan", run.shot_dx_plan_json),
        ("scenes_manifest", run.scenes_manifest_json),
        ("mne_fx_classification", run.mne_fx_classification_json),
        ("mne_music_cues", run.mne_music_cues_json),
        ("mne_room_tone", run.mne_room_tone_json),
        ("audio_analysis", run.audio_analysis_json),
        ("timeline", run.timeline_otio),
    ]


def threshold(value: Any, name: str) -> float:
    threshold_value = float(value)
    if threshold_value < 0.0 or threshold_value > 1.0:
        raise ValueError(f"--{name} must be between 0 and 1")
    return threshold_value


def positive_float(value: Any, name: str) -> float:
    number = float(value)
    if number <= 0:
        raise ValueError(f"--{name} must be positive")
    return number
