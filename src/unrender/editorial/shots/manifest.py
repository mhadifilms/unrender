"""Shot manifest planning and persistence.

The manifest written here is the same file the rest of the pipeline consumes
through ``load_shot_manifest`` (``paths.shots``): each entry carries
``shot_id``/``video_path``/``start_sec``/``end_sec`` plus frame-exact and
timecode representations. Media paths are planned deterministically, so a
detect-only run writes the same paths a later cut run fills in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unrender.editorial.shots.sources import CutRange
from unrender.lib.fingerprints import input_fingerprints
from unrender.lib.names import safe_name
from unrender.lib.timecode import frames_to_seconds, frames_to_smpte
from unrender.manifests import write_json

MEDIA_PLANNED = "planned"
MEDIA_CREATED = "created"


@dataclass(frozen=True)
class ShotMediaPlan:
    shot_id: str
    shot_name: str
    shot_dir: Path
    plate_path: Path  # cut video file, or EXR frame directory
    video_path: Path  # detection/matching-ready video (per-shot proxy for sequences)


def plan_shot_media(
    ranges: list[CutRange],
    *,
    shots_dir: Path,
    shot_prefix: str,
    master_kind: str,
    master_ext: str = ".mov",
) -> list[ShotMediaPlan]:
    prefix = safe_name(shot_prefix, fallback="shot")
    plans: list[ShotMediaPlan] = []
    for range_ in ranges:
        shot_name = f"{prefix}_{safe_name(range_.shot_id)}"
        shot_dir = shots_dir / shot_name
        if master_kind == "sequence":
            plate = shot_dir / f"{shot_name}_plate"
            video = shot_dir / f"{shot_name}_proxy.mov"
        else:
            plate = shot_dir / f"{shot_name}{master_ext}"
            video = plate
        plans.append(
            ShotMediaPlan(
                shot_id=range_.shot_id,
                shot_name=shot_name,
                shot_dir=shot_dir,
                plate_path=plate,
                video_path=video,
            )
        )
    return plans


def shot_manifest_entries(
    ranges: list[CutRange],
    plans: list[ShotMediaPlan],
    *,
    fps: float,
    origin_frame: int,
    drop_frame: bool = False,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for range_, plan in zip(ranges, plans, strict=True):
        entries.append(
            {
                "shot_id": range_.shot_id,
                "start_frame": range_.start_frame,
                "end_frame": range_.end_frame,
                "start_sec": round(frames_to_seconds(range_.start_frame, fps), 6),
                "end_sec": round(frames_to_seconds(range_.end_frame, fps), 6),
                "start_tc": frames_to_smpte(
                    origin_frame + range_.start_frame, fps, drop_frame=drop_frame
                ),
                "end_tc": frames_to_smpte(
                    origin_frame + range_.end_frame, fps, drop_frame=drop_frame
                ),
                "video_path": str(plan.video_path),
                "plate_path": str(plan.plate_path),
                "media_status": MEDIA_PLANNED,
                "label": range_.label,
            }
        )
    return entries


def write_shots_manifest(
    path: Path,
    *,
    entries: list[dict[str, Any]],
    fps: float,
    master: Path,
    proxy: Path | None,
    origin_frame: int,
    source: str,
    detector: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
    shot_list: Path | None = None,
) -> None:
    data: dict[str, Any] = {
        "version": "1.0",
        "fps": fps,
        "timecode_start": frames_to_smpte(origin_frame, fps),
        "master": str(master),
        "proxy": str(proxy) if proxy else "",
        "source": source,
        "inputs": input_fingerprints(inputs or {}),
        "shots": entries,
    }
    if detector:
        data["detector"] = detector
    if shot_list is not None:
        data["shot_list"] = str(shot_list)
    write_json(path, data)
    print(f"Shot manifest saved: {path} ({len(entries)} shot(s))", flush=True)


def spotting_csv_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "shot_id": entry["shot_id"],
            "start_tc": entry["start_tc"],
            "end_tc": entry["end_tc"],
            "start_sec": entry["start_sec"],
            "end_sec": entry["end_sec"],
        }
        for entry in entries
    ]
