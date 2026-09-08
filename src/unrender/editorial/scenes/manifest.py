"""Scene manifest persistence.

``scenes.json`` groups the shot manifest into scenes for timeline construction.
Shot identity, frames, and timecodes come straight from the shot manifest
entries so the two files always agree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from unrender.lib.fingerprints import input_fingerprints
from unrender.manifests import read_json, write_json


def scene_manifest_entries(
    shots: list[dict[str, Any]],
    assignment: dict[str, int],
    scores: list[float],
) -> list[dict[str, Any]]:
    """Fold shot-manifest entries into per-scene entries using the shot->scene
    assignment from :func:`unrender.editorial.scenes.group.assign_scenes`."""
    scenes: dict[int, dict[str, Any]] = {}
    for index, entry in enumerate(shots):
        shot_id = str(entry["shot_id"])
        scene_num = assignment[shot_id]
        scene = scenes.setdefault(
            scene_num,
            {
                "scene_id": f"{scene_num:03d}",
                "shot_ids": [],
                "start_frame": int(entry["start_frame"]),
                "end_frame": int(entry["end_frame"]),
                "start_sec": float(entry["start_sec"]),
                "end_sec": float(entry["end_sec"]),
                "start_tc": entry.get("start_tc", ""),
                "end_tc": entry.get("end_tc", ""),
                "boundary_score": 0.0,
            },
        )
        scene["shot_ids"].append(shot_id)
        scene["end_frame"] = int(entry["end_frame"])
        scene["end_sec"] = float(entry["end_sec"])
        scene["end_tc"] = entry.get("end_tc", "")
        scene["boundary_score"] = round(float(scores[index]), 4)
    return [scenes[num] for num in sorted(scenes)]


def write_scenes_manifest(
    path: Path,
    *,
    entries: list[dict[str, Any]],
    shots_manifest: Path,
    engine: str,
    variant: str,
    threshold: float,
    inputs: dict[str, Any] | None = None,
) -> None:
    write_json(
        path,
        {
            "version": "1.0",
            "shots_manifest": str(shots_manifest),
            "engine": engine,
            "variant": variant,
            "threshold": threshold,
            "inputs": input_fingerprints(inputs or {}),
            "scenes": entries,
        },
    )
    print(f"Scene manifest saved: {path} ({len(entries)} scene(s))", flush=True)


def load_scene_manifest(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    scenes = list(data.get("scenes") or [])
    if not scenes:
        raise ValueError(f"scene manifest has no scenes: {path}")
    return scenes


def scene_for_time(scenes: list[dict[str, Any]], moment_sec: float) -> str:
    """Scene id containing ``moment_sec``; clamps to first/last scene so
    slightly out-of-range dialogue timestamps still resolve."""
    for scene in scenes:
        if float(scene["start_sec"]) <= moment_sec < float(scene["end_sec"]):
            return str(scene["scene_id"])
    if moment_sec < float(scenes[0]["start_sec"]):
        return str(scenes[0]["scene_id"])
    return str(scenes[-1]["scene_id"])


def spotting_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "scene_id": entry["scene_id"],
            "start_tc": entry["start_tc"],
            "end_tc": entry["end_tc"],
            "start_sec": entry["start_sec"],
            "end_sec": entry["end_sec"],
            "shot_count": len(entry["shot_ids"]),
            "shot_ids": " ".join(entry["shot_ids"]),
        }
        for entry in entries
    ]
