"""Load run artifacts into adapter-independent timeline clips.

These loaders are pure: they read JSON/CSV artifacts and produce
:class:`TimelineClip` records with global (program-relative) timing in seconds.
The OpenTimelineIO assembly and any media probing happen in ``builder``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from unrender.manifests import load_shot_manifest, read_json
from unrender.speakers import canonical_speaker_name

# Source-stem keys (from source_separation.json) mapped to editorial track labels.
SOURCE_STEM_LABELS: tuple[tuple[str, str], ...] = (
    ("music_fx", "MX"),
    ("music", "MX"),
    ("effects", "FX"),
)
DIALOGUE_BED_KEY = "dialogue"


@dataclass
class TimelineClip:
    """One placement of media on the program timeline (seconds, global)."""

    name: str
    start_sec: float
    end_sec: float
    media: Path | None = None
    group: str = ""
    text: str = ""
    status: str = ""
    on_screen_speakers: tuple[str, ...] = ()
    full_length: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def load_video_clips(
    shots_path: Path,
    shot_matches: dict[str, list[str]],
) -> tuple[list[TimelineClip], dict[str, tuple[float, float]], int]:
    shots = load_shot_manifest(shots_path)
    clips: list[TimelineClip] = []
    windows: dict[str, tuple[float, float]] = {}
    skipped = 0
    for shot in shots:
        if shot.start_sec is None or shot.end_sec is None or shot.end_sec <= shot.start_sec:
            skipped += 1
            continue
        windows[shot.shot_id] = (shot.start_sec, shot.end_sec)
        on_screen = shot_matches.get(shot.shot_id) or _split_speakers(shot.existing_speaker)
        clips.append(
            TimelineClip(
                name=shot.shot_id,
                start_sec=shot.start_sec,
                end_sec=shot.end_sec,
                media=shot.video_path,
                on_screen_speakers=tuple(on_screen),
                metadata=_clean(
                    {
                        "shot_id": shot.shot_id,
                    }
                ),
            )
        )
    return clips, windows, skipped


def load_dialogue_clips(
    run: Any, windows: dict[str, tuple[float, float]], granularity: str
) -> tuple[list[TimelineClip], str]:
    if granularity in ("auto", "lines") and run.dialogue_stem_plan_json.exists():
        return _line_clips(run.dialogue_stem_plan_json), "lines"
    if granularity in ("auto", "shots") and run.shot_stem_plan_json.exists() and windows:
        return _shot_clips(run.shot_stem_plan_json, windows), "shots"
    if granularity in ("auto", "lines") and run.dialogue_lines_json.exists():
        return _line_clips(run.dialogue_lines_json, lines_only=True), "lines"
    return [], "none"


def load_scenes(scenes_path: Path | None) -> list[dict[str, Any]]:
    """Scene entries from a scenes manifest, or ``[]`` when absent/empty.

    Kept lazy (the scenes package pulls numpy) so timelines build without
    scenes present at no import cost.
    """
    if scenes_path is None or not scenes_path.exists():
        return []
    from unrender.editorial.scenes.manifest import load_scene_manifest

    try:
        return load_scene_manifest(scenes_path)
    except ValueError:
        return []


def load_shot_matches(run: Any) -> dict[str, list[str]]:
    if not run.shot_matches_json.exists():
        return {}
    data = read_json(run.shot_matches_json)
    matches: dict[str, list[str]] = {}
    for entry in (data.get("shots") if isinstance(data, dict) else None) or []:
        if not isinstance(entry, dict):
            continue
        shot_id = str(entry.get("shot_id") or "").strip()
        accepted = entry.get("accepted") or []
        if shot_id and isinstance(accepted, list) and accepted:
            matches[shot_id] = [str(name) for name in accepted]
    return matches


def load_shot_dx_clips(run: Any) -> list[TimelineClip]:
    """Per-shot DX clips (from ``audio shot-dx``) for a single timeline track."""
    if not run.shot_dx_plan_json.exists():
        return []
    data = read_json(run.shot_dx_plan_json)
    clips: list[TimelineClip] = []
    for entry in (data.get("shots") if isinstance(data, dict) else None) or []:
        if not isinstance(entry, dict):
            continue
        start = _first_float(entry, "start_sec")
        end = _first_float(entry, "end_sec")
        if start is None or end is None or end <= start:
            continue
        clips.append(
            TimelineClip(
                name=f"{entry.get('shot_id') or 'shot'} DX",
                start_sec=start,
                end_sec=end,
                media=_first_path(entry, "stem_path"),
                metadata=_clean({"shot_id": entry.get("shot_id"), "source_group": "dx_shot"}),
            )
        )
    return clips


def load_source_stems(run: Any) -> dict[str, Path]:
    stems: dict[str, Path] = {}
    if run.source_separation_json.exists():
        data = read_json(run.source_separation_json)
        raw = data.get("stems") if isinstance(data, dict) else None
        if isinstance(raw, dict):
            for key, value in raw.items():
                if value:
                    stems[str(key)] = Path(str(value)).expanduser()
    if not stems and run.source_stems_dir.exists():
        fallbacks = (("_MX_stem", "music_fx"), ("_FX_stem", "effects"), ("_DX_stem", "dialogue"))
        for suffix, key in fallbacks:
            matches = sorted(run.source_stems_dir.glob(f"*{suffix}.*"))
            if matches:
                stems[key] = matches[0]
    return stems


def source_stem_tracks(
    stems: dict[str, Path], *, include_music_effects: bool, include_dialogue_bed: bool
) -> list[tuple[str, str, Path]]:
    """Return ordered ``(label, key, path)`` for music/effects/other stems."""
    tracks: list[tuple[str, str, Path]] = []
    seen_labels: set[str] = set()
    if include_music_effects:
        for key, label in SOURCE_STEM_LABELS:
            path = stems.get(key)
            if path is not None and path.exists() and label not in seen_labels:
                seen_labels.add(label)
                tracks.append((label, key, path))
        for key, path in stems.items():
            if key in {k for k, _ in SOURCE_STEM_LABELS} or key == DIALOGUE_BED_KEY:
                continue
            if path.exists():
                tracks.append((key.upper(), key, path))
    if include_dialogue_bed:
        path = stems.get(DIALOGUE_BED_KEY)
        if path is not None and path.exists():
            tracks.append(("DX bed", DIALOGUE_BED_KEY, path))
    return tracks


# mne FX category keys mapped to editorial track labels.
MNE_FX_LABELS: tuple[tuple[str, str], ...] = (
    ("ambience", "AMB"),
    ("foley", "FOLEY"),
    ("hard_fx", "HFX"),
    ("design_fx", "DSGN"),
)


def load_mne_fx_tracks(run: Any) -> list[tuple[str, str, Path]]:
    """Full-length ``(label, key, path)`` tracks for the classified FX stems."""
    if not run.mne_fx_classification_json.exists():
        return []
    data = read_json(run.mne_fx_classification_json)
    categories = data.get("categories") if isinstance(data, dict) else None
    if not isinstance(categories, dict):
        return []
    tracks: list[tuple[str, str, Path]] = []
    for key, label in MNE_FX_LABELS:
        value = categories.get(key)
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if path.exists():
            tracks.append((label, f"mne_{key}", path))
    return tracks


def load_mne_room_tone_track(run: Any) -> tuple[str, str, Path] | None:
    """Full-length ``(label, key, path)`` track for the room-tone bed, if any."""
    if not run.mne_room_tone_json.exists():
        return None
    data = read_json(run.mne_room_tone_json)
    value = data.get("rt_stem") if isinstance(data, dict) else None
    if not value:
        return None
    path = Path(str(value)).expanduser()
    return ("RT", "mne_room_tone", path) if path.exists() else None


def load_mne_music_cue_clips(run: Any) -> list[TimelineClip]:
    """Per-cue clips (from ``mne music``) for a single MX-cues timeline track."""
    if not run.mne_music_cues_json.exists():
        return []
    data = read_json(run.mne_music_cues_json)
    clips: list[TimelineClip] = []
    for entry in (data.get("cues") if isinstance(data, dict) else None) or []:
        if not isinstance(entry, dict):
            continue
        start = _first_float(entry, "pad_start_sec", "start_sec")
        end = _first_float(entry, "pad_end_sec", "end_sec")
        if start is None or end is None or end <= start:
            continue
        clips.append(
            TimelineClip(
                name=str(entry.get("cue_id") or "cue"),
                start_sec=start,
                end_sec=end,
                media=_first_path(entry, "path"),
                metadata=_clean({"cue_id": entry.get("cue_id"), "source_group": "mx_cue"}),
            )
        )
    return clips


def _line_clips(path: Path, *, lines_only: bool = False) -> list[TimelineClip]:
    data = read_json(path)
    if not isinstance(data, dict):
        return []
    entries = data.get("clips") or data.get("lines") or []
    clips: list[TimelineClip] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        start = _first_float(entry, "clip_start_sec", "start_sec")
        end = _first_float(entry, "clip_end_sec", "end_sec")
        if start is None or end is None or end <= start:
            continue
        media = None if lines_only else _first_path(entry, "dialogue_stem_path", "stem_path")
        speaker = canonical_speaker_name(str(entry.get("speaker") or ""))
        clips.append(
            TimelineClip(
                name=str(entry.get("line_id") or entry.get("clip_id") or "line"),
                start_sec=start,
                end_sec=end,
                media=media,
                group=speaker,
                text=str(entry.get("text") or ""),
                status=str(entry.get("status") or ""),
                metadata=_clean(
                    {
                        "line_id": entry.get("line_id"),
                        "speaker": speaker,
                        "source_group": entry.get("source_group"),
                        "score": entry.get("score"),
                        "margin": entry.get("margin"),
                        "resolution_source": entry.get("resolution_source"),
                        "cluster_id": entry.get("cluster_id"),
                    }
                ),
            )
        )
    return clips


def _shot_clips(path: Path, windows: dict[str, tuple[float, float]]) -> list[TimelineClip]:
    data = read_json(path)
    if not isinstance(data, dict):
        return []
    clips: list[TimelineClip] = []
    for entry in data.get("shots") or []:
        if not isinstance(entry, dict):
            continue
        shot_id = str(entry.get("shot_id") or "")
        # Per-line shot cuts carry their own exact window; older plans fall
        # back to the shot window from the manifest.
        start = _first_float(entry, "start_sec")
        end = _first_float(entry, "end_sec")
        if start is None or end is None or end <= start:
            window = windows.get(shot_id)
            if window is None or window[1] <= window[0]:
                continue
            start, end = window
        speaker = canonical_speaker_name(str(entry.get("speaker") or ""))
        line_id = str(entry.get("line_id") or "")
        clips.append(
            TimelineClip(
                name=f"{shot_id} {line_id}".strip() if line_id else shot_id,
                start_sec=start,
                end_sec=end,
                media=_first_path(entry, "stem_path", "proposed_audio_path"),
                group=speaker,
                text=str(entry.get("text") or ""),
                status=str(entry.get("status") or ""),
                metadata=_clean(
                    {
                        "shot_id": shot_id,
                        "line_id": line_id,
                        "speaker": speaker,
                        "source_group": entry.get("source_group"),
                        "score": entry.get("score"),
                    }
                ),
            )
        )
    return clips


def _split_speakers(value: str) -> list[str]:
    raw = str(value or "").replace(";", ",").split(",")
    out: list[str] = []
    for item in raw:
        name = item.strip()
        if name and name not in out:
            out.append(name)
    return out


def _clean(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if value not in (None, "")}


def _first_float(entry: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = entry.get(key)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _first_path(entry: dict[str, Any], *keys: str) -> Path | None:
    for key in keys:
        value = entry.get(key)
        if value is not None and value != "":
            return Path(str(value)).expanduser()
    return None
