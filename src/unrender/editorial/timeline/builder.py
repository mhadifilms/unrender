"""Reconstruct an editor timeline (OpenTimelineIO) from run artifacts.

This is the final ``unrender`` step. It reads the per-shot video, the per-line
or per-shot dialogue stems, and the music/effects source stems produced earlier
and assembles them back into a single, editor-friendly timeline.

Highlights:

* Real media durations are probed (``ffprobe`` / ``wave``) so each clip carries
  an accurate ``available_range``; probing degrades gracefully when unavailable.
* Dialogue is laid out one track per speaker, ordered by speaking time, with
  overlapping lines fanned onto extra lanes.
* Clips carry editorial markers: the spoken line and speaker, plus color-coded
  flags for off-screen, unresolved, or missing-media lines.
* A program start timecode and full provenance metadata are recorded.

The timeline is written as ``.otio`` by default. With the
``OpenTimelineIO-Plugins`` package installed, other suffixes (``.edl``,
``.fcpxml``, ``.xml``, ``.aaf``) and the ``.otiod`` media bundle are written via
the matching adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from unrender import __version__
from unrender.editorial.timeline.media import MediaProber
from unrender.editorial.timeline.sources import (
    TimelineClip,
    load_dialogue_clips,
    load_mne_fx_tracks,
    load_mne_music_cue_clips,
    load_mne_room_tone_track,
    load_scenes,
    load_shot_dx_clips,
    load_shot_matches,
    load_source_stems,
    load_video_clips,
    source_stem_tracks,
)
from unrender.lib.timecode import smpte_to_seconds

DEFAULT_FPS = 24.0
DIALOGUE_GRANULARITIES = ("auto", "lines", "shots")
_TEXT_PREVIEW = 64


@dataclass
class TimelineSummary:
    """Counts and warnings describing the assembled timeline."""

    fps: float
    name: str
    dialogue_granularity: str = "none"
    start_timecode: str = ""
    duration_sec: float = 0.0
    tracks: int = 0
    video_clips: int = 0
    dialogue_clips: int = 0
    source_clips: int = 0
    scenes: int = 0
    markers: int = 0
    probed_media: int = 0
    missing_media: int = 0
    skipped_untimed: int = 0
    speakers: tuple[str, ...] = ()
    outputs: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> list[str]:
        start = f"  start: {self.start_timecode}" if self.start_timecode else ""
        lines = [
            f"Timeline: {self.name}",
            f"  fps: {self.fps:g}{start}",
            f"  duration: {self.duration_sec:.2f}s",
            f"  tracks: {self.tracks}",
            f"  video clips: {self.video_clips}",
            f"  dialogue clips ({self.dialogue_granularity}): {self.dialogue_clips}",
            f"  source-stem clips: {self.source_clips}",
            f"  scenes: {self.scenes}",
            f"  markers: {self.markers}",
            f"  media probed: {self.probed_media}",
        ]
        if self.speakers:
            lines.append(f"  speakers: {', '.join(self.speakers)}")
        if self.skipped_untimed:
            lines.append(f"  shots skipped (no start/end): {self.skipped_untimed}")
        if self.missing_media:
            lines.append(f"  clips with missing media (kept as placeholders): {self.missing_media}")
        for warning in self.warnings:
            lines.append(f"  ! {warning}")
        for output in self.outputs:
            lines.append(f"  written: {output}")
        return lines


@dataclass
class _Placed:
    clip: TimelineClip
    start_frame: int
    dur_frames: int
    media: Path | None
    media_start_frame: int
    available_frames: int | None

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.dur_frames


def _import_otio() -> Any:
    try:
        import opentimelineio as otio
    except ImportError as exc:  # pragma: no cover - exercised via CLI
        raise ImportError(
            "OpenTimelineIO is required for timeline assembly. "
            "Install it with: pip install 'unrender[timeline]'"
        ) from exc
    return otio


def build_timeline(
    run: Any,
    *,
    shots_path: Path | None = None,
    scenes_path: Path | None = None,
    scene_stacks: bool = False,
    fps: float = DEFAULT_FPS,
    dialogue: str = "auto",
    name: str = "unrender",
    start_timecode: str | None = None,
    include_dialogue_bed: bool = False,
    include_shot_dx: bool = False,
    include_music_effects: bool = True,
    include_mne: bool = False,
    markers: bool = True,
    prober: MediaProber | None = None,
) -> tuple[Any, TimelineSummary]:
    """Assemble an OpenTimelineIO ``Timeline`` from a run directory."""
    if dialogue not in DIALOGUE_GRANULARITIES:
        raise ValueError(
            f"dialogue granularity must be one of {DIALOGUE_GRANULARITIES}: {dialogue}"
        )
    if fps <= 0:
        raise ValueError("fps must be positive")

    otio = _import_otio()
    prober = prober or MediaProber()
    summary = TimelineSummary(fps=fps, name=name, start_timecode=start_timecode or "")

    shot_matches = load_shot_matches(run)
    video_clips: list[TimelineClip] = []
    windows: dict[str, tuple[float, float]] = {}
    if shots_path is not None:
        video_clips, windows, summary.skipped_untimed = load_video_clips(
            shots_path,
            shot_matches,
        )

    scenes = load_scenes(scenes_path)
    shot_scene = _assign_scene_ids(video_clips, scenes)
    summary.scenes = len(scenes)

    dialogue_clips, granularity = load_dialogue_clips(run, windows, dialogue)
    summary.dialogue_granularity = granularity
    shot_dx_clips = load_shot_dx_clips(run) if include_shot_dx else []
    mne_cue_clips = load_mne_music_cue_clips(run) if include_mne else []

    video_placed = [
        _place(clip, fps, prober, summary, clamp_to_window=True) for clip in video_clips
    ]
    dialogue_placed = [
        _place(clip, fps, prober, summary, clamp_to_window=False) for clip in dialogue_clips
    ]
    shot_dx_placed = [
        _place(clip, fps, prober, summary, clamp_to_window=True) for clip in shot_dx_clips
    ]
    mne_cue_placed = [
        _place(clip, fps, prober, summary, clamp_to_window=True) for clip in mne_cue_clips
    ]

    program_frames = max(
        (
            placed.end_frame
            for placed in (*video_placed, *dialogue_placed, *shot_dx_placed, *mne_cue_placed)
        ),
        default=0,
    )
    summary.duration_sec = program_frames / fps

    timeline = otio.schema.Timeline(name=name)
    timeline.global_start_time = _start_time(otio, start_timecode, fps)
    timeline.metadata["unrender"] = _provenance(run, fps, granularity, start_timecode)
    if scenes:
        timeline.metadata["unrender"]["scenes"] = [
            {
                "scene_id": str(scene["scene_id"]),
                "start_sec": float(scene["start_sec"]),
                "end_sec": float(scene["end_sec"]),
                "shot_ids": [str(s) for s in scene.get("shot_ids") or []],
            }
            for scene in scenes
        ]

    tracks: list[Any] = []
    if scene_stacks and scenes and video_placed:
        tracks.append(
            _build_scene_video_track(otio, video_placed, scenes, shot_scene, fps, summary, markers)
        )
    else:
        video_tracks = [
            _build_track(otio, f"V{index}", "Video", lane, fps, summary, markers, "video")
            for index, lane in enumerate(_assign_lanes(video_placed), 1)
        ]
        if scenes and video_tracks:
            for marker in _scene_markers(otio, scenes, fps):
                video_tracks[0].markers.append(marker)
                summary.markers += 1
        tracks.extend(video_tracks)

    speakers: list[str] = []
    for group, placed in _dialogue_groups(dialogue_placed):
        speakers.append(group or "DX")
        base = f"DX {group}" if group else "DX"
        lanes = _assign_lanes(placed)
        for index, lane in enumerate(lanes, 1):
            label = base if len(lanes) == 1 else f"{base} {index}"
            tracks.append(
                _build_track(otio, label, "Audio", lane, fps, summary, markers, "dialogue")
            )
    summary.speakers = tuple(speakers)

    if shot_dx_placed:
        # Shots are sequential, so this is normally exactly one track; any
        # overlapping shot windows fan out rather than corrupting the layer.
        dx_lanes = _assign_lanes(shot_dx_placed)
        for index, lane in enumerate(dx_lanes, 1):
            label = "DX shots" if len(dx_lanes) == 1 else f"DX shots {index}"
            tracks.append(_build_track(otio, label, "Audio", lane, fps, summary, False, "source"))
        if len(dx_lanes) > 1:
            summary.warnings.append(
                f"per-shot DX clips overlap; split across {len(dx_lanes)} tracks"
            )

    stems = load_source_stems(run)
    full_length_tracks: list[tuple[str, str, Path]] = list(
        source_stem_tracks(
            stems,
            include_music_effects=include_music_effects,
            include_dialogue_bed=include_dialogue_bed,
        )
    )
    if include_mne:
        full_length_tracks.extend(load_mne_fx_tracks(run))
        room_tone = load_mne_room_tone_track(run)
        if room_tone is not None:
            full_length_tracks.append(room_tone)
    if program_frames > 0:
        for label, key, path in full_length_tracks:
            clip = TimelineClip(
                name=label,
                start_sec=0.0,
                end_sec=program_frames / fps,
                media=path,
                full_length=True,
                metadata={"source_group": key},
            )
            source_placed = _place(
                clip, fps, prober, summary, clamp_to_window=False, program_frames=program_frames
            )
            tracks.append(
                _build_track(otio, label, "Audio", [source_placed], fps, summary, False, "source")
            )

    if mne_cue_placed:
        for index, lane in enumerate(_assign_lanes(mne_cue_placed), 1):
            label = "MX cues" if index == 1 else f"MX cues {index}"
            tracks.append(_build_track(otio, label, "Audio", lane, fps, summary, False, "source"))

    for track in tracks:
        timeline.tracks.append(track)
    summary.tracks = len(tracks)
    _finalize_warnings(summary, prober)
    return timeline, summary


def write_timeline(timeline: Any, path: Path) -> Path:
    """Write a timeline to ``path``; the suffix selects the OTIO adapter."""
    otio = _import_otio()
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lstrip(".").lower()
    available = set(_adapter_suffixes(otio))
    if suffix and suffix not in available:
        raise ValueError(
            f"no OpenTimelineIO adapter is registered for '.{suffix}'. "
            f"Available: {', '.join(sorted(available)) or 'otio'}. "
            "Install more with: pip install OpenTimelineIO-Plugins"
        )
    try:
        otio.adapters.write_to_file(timeline, str(path))
    except Exception as exc:  # adapters raise many types; surface a clean error
        raise RuntimeError(f"failed to write timeline to {path}: {exc}") from exc
    return path


def _place(
    clip: TimelineClip,
    fps: float,
    prober: MediaProber,
    summary: TimelineSummary,
    *,
    clamp_to_window: bool,
    program_frames: int | None = None,
) -> _Placed:
    media: Path | None = None
    media_start = 0.0
    media_dur: float | None = None
    if clip.media is not None and clip.media.exists():
        media = clip.media
        info = prober.probe(clip.media)
        if info is not None:
            media_start = info.start_sec
            media_dur = info.duration_sec
            summary.probed_media += 1

    if clip.full_length and program_frames is not None:
        timing = program_frames / fps
    else:
        timing = clip.end_sec - clip.start_sec
    if media_dur is None:
        duration = timing
    elif clamp_to_window:
        duration = min(timing, media_dur)
    else:
        duration = media_dur

    return _Placed(
        clip=clip,
        start_frame=max(0, round(clip.start_sec * fps)),
        dur_frames=max(1, round(duration * fps)),
        media=media,
        media_start_frame=max(0, round(media_start * fps)),
        available_frames=(round(media_dur * fps) if media_dur is not None else None),
    )


def _assign_scene_ids(
    video_clips: list[TimelineClip], scenes: list[dict[str, Any]]
) -> dict[str, str]:
    """Tag each video clip with its scene_id; returns the shot_id -> scene_id map."""
    if not scenes:
        return {}
    from unrender.editorial.scenes.manifest import scene_for_time

    shot_scene: dict[str, str] = {}
    for scene in scenes:
        for shot_id in scene.get("shot_ids") or []:
            shot_scene[str(shot_id)] = str(scene["scene_id"])
    for clip in video_clips:
        scene_id = shot_scene.get(clip.name) or scene_for_time(scenes, clip.start_sec)
        if scene_id:
            clip.metadata = {**clip.metadata, "scene_id": scene_id}
    return shot_scene


def _scene_markers(otio: Any, scenes: list[dict[str, Any]], fps: float) -> list[Any]:
    markers = []
    for scene in scenes:
        start = max(0, round(float(scene["start_sec"]) * fps))
        dur = max(1, round((float(scene["end_sec"]) - float(scene["start_sec"])) * fps))
        markers.append(
            otio.schema.Marker(
                name=f"SC{scene['scene_id']}",
                marked_range=otio.opentime.TimeRange(
                    otio.opentime.RationalTime(start, fps),
                    otio.opentime.RationalTime(dur, fps),
                ),
                color=otio.schema.MarkerColor.PURPLE,
                metadata={
                    "unrender": {
                        "scene_id": str(scene["scene_id"]),
                        "shot_ids": [str(s) for s in scene.get("shot_ids") or []],
                    }
                },
            )
        )
    return markers


def _build_scene_video_track(
    otio: Any,
    video_placed: list[_Placed],
    scenes: list[dict[str, Any]],
    shot_scene: dict[str, str],
    fps: float,
    summary: TimelineSummary,
    want_markers: bool,
) -> Any:
    """One video track whose children are per-scene Stacks (opt-in nesting).

    Each scene becomes an ``otio.schema.Stack`` (named ``SC{id}``) holding the
    scene's shot clips as lanes, rebased to the scene start. Inter-scene gaps
    keep program timing intact.
    """
    track = otio.schema.Track(name="V1", kind="Video")
    scene_order = {str(scene["scene_id"]): index for index, scene in enumerate(scenes)}
    by_scene: dict[str, list[_Placed]] = {}
    for placed in video_placed:
        scene_id = shot_scene.get(placed.clip.name, "")
        by_scene.setdefault(scene_id, []).append(placed)

    def order_key(scene_id: str) -> tuple[int, int]:
        rank = scene_order.get(scene_id, len(scene_order))
        first = min(p.start_frame for p in by_scene[scene_id])
        return (rank, first)

    cursor = 0
    for scene_id in sorted(by_scene, key=order_key):
        members = sorted(by_scene[scene_id], key=lambda p: p.start_frame)
        scene_start = members[0].start_frame
        scene_end = max(p.end_frame for p in members)
        gap = scene_start - cursor
        if gap > 0:
            track.append(
                otio.schema.Gap(
                    source_range=otio.opentime.TimeRange(
                        otio.opentime.RationalTime(0, fps),
                        otio.opentime.RationalTime(gap, fps),
                    )
                )
            )
        stack = otio.schema.Stack(name=f"SC{scene_id}" if scene_id else "SC?")
        stack.metadata["unrender"] = {"scene_id": scene_id}
        rebased = [_shift(placed, scene_start) for placed in members]
        for lane in _assign_lanes(rebased):
            stack.append(_build_track(otio, "", "Video", lane, fps, summary, want_markers, "video"))
        track.append(stack)
        cursor = scene_end
    return track


def _shift(placed: _Placed, offset: int) -> _Placed:
    """Copy a placement with its start rebased by ``offset`` frames."""
    return _Placed(
        clip=placed.clip,
        start_frame=max(0, placed.start_frame - offset),
        dur_frames=placed.dur_frames,
        media=placed.media,
        media_start_frame=placed.media_start_frame,
        available_frames=placed.available_frames,
    )


def _build_track(
    otio: Any,
    name: str,
    kind: str,
    lane: list[_Placed],
    fps: float,
    summary: TimelineSummary,
    want_markers: bool,
    counter: str,
) -> Any:
    track = otio.schema.Track(name=name, kind=kind)
    cursor = 0
    for placed in lane:
        gap = placed.start_frame - cursor
        if gap > 0:
            track.append(
                otio.schema.Gap(
                    source_range=otio.opentime.TimeRange(
                        otio.opentime.RationalTime(0, fps),
                        otio.opentime.RationalTime(gap, fps),
                    )
                )
            )
        track.append(_build_clip(otio, placed, fps, summary, want_markers, counter))
        cursor = placed.end_frame
        if counter == "video":
            summary.video_clips += 1
        elif counter == "dialogue":
            summary.dialogue_clips += 1
        else:
            summary.source_clips += 1
    return track


def _build_clip(
    otio: Any,
    placed: _Placed,
    fps: float,
    summary: TimelineSummary,
    want_markers: bool,
    counter: str,
) -> Any:
    start = otio.opentime.RationalTime(placed.media_start_frame, fps)
    source_range = otio.opentime.TimeRange(
        start, otio.opentime.RationalTime(placed.dur_frames, fps)
    )
    if placed.media is not None:
        available = source_range
        if placed.available_frames is not None:
            available = otio.opentime.TimeRange(
                start, otio.opentime.RationalTime(placed.available_frames, fps)
            )
        # Relative artifact paths are resolved against the cwd (where they were
        # checked for existence) so every reference is a portable file:// URI.
        media = placed.media if placed.media.is_absolute() else placed.media.resolve()
        reference: Any = otio.schema.ExternalReference(
            target_url=media.as_uri(),
            available_range=available,
        )
    else:
        reference = otio.schema.MissingReference()
        summary.missing_media += 1

    clip = otio.schema.Clip(
        name=placed.clip.name,
        media_reference=reference,
        source_range=source_range,
        metadata={"unrender": placed.clip.metadata} if placed.clip.metadata else {},
    )
    if want_markers:
        marker = _marker(otio, placed, source_range, counter)
        if marker is not None:
            clip.markers.append(marker)
            summary.markers += 1
    return clip


def _marker(otio: Any, placed: _Placed, source_range: Any, counter: str) -> Any | None:
    clip = placed.clip
    if counter == "video":
        if not clip.on_screen_speakers:
            return None
        return otio.schema.Marker(
            name="on-screen: " + ", ".join(clip.on_screen_speakers),
            marked_range=source_range,
            color=otio.schema.MarkerColor.CYAN,
        )

    if placed.media is None:
        color = otio.schema.MarkerColor.RED
        note = "missing stem"
    elif clip.status in ("offscreen_speaker", "unresolved", "no_matching_stem"):
        color = otio.schema.MarkerColor.ORANGE
        note = clip.status.replace("_", " ")
    else:
        color = otio.schema.MarkerColor.GREEN
        note = ""

    speaker = clip.group or "?"
    text = clip.text.strip()
    label = f"{speaker}: {text[:_TEXT_PREVIEW]}" if text else speaker
    if note:
        label = f"[{note}] {label}"
    metadata = {"speaker": speaker, "status": clip.status, "text": text}
    for key in ("score", "margin", "resolution_source", "cluster_id"):
        if key in clip.metadata:
            metadata[key] = clip.metadata[key]
    return otio.schema.Marker(
        name=label,
        marked_range=source_range,
        color=color,
        metadata={"unrender": metadata},
    )


def _assign_lanes(placed: list[_Placed]) -> list[list[_Placed]]:
    ordered = sorted(placed, key=lambda item: (item.start_frame, item.end_frame))
    lanes: list[list[_Placed]] = []
    lane_ends: list[int] = []
    for item in ordered:
        slot = None
        for index, end in enumerate(lane_ends):
            if item.start_frame >= end:
                slot = index
                break
        if slot is None:
            lanes.append([item])
            lane_ends.append(item.end_frame)
        else:
            lanes[slot].append(item)
            lane_ends[slot] = item.end_frame
    return lanes


def _dialogue_groups(placed: list[_Placed]) -> list[tuple[str, list[_Placed]]]:
    groups: dict[str, list[_Placed]] = {}
    for item in placed:
        groups.setdefault(item.clip.group, []).append(item)
    # Most prominent speakers (by total duration) first; unnamed group last.
    return sorted(
        groups.items(),
        key=lambda item: (item[0] == "", -sum(p.dur_frames for p in item[1]), item[0]),
    )


def _start_time(otio: Any, start_timecode: str | None, fps: float) -> Any:
    if not start_timecode:
        return otio.opentime.RationalTime(0, fps)
    seconds = smpte_to_seconds(start_timecode, fps)
    if seconds is None:
        raise ValueError(
            f"invalid --start-timecode: {start_timecode!r} (use HH:MM:SS:FF or seconds)"
        )
    return otio.opentime.RationalTime(round(seconds * fps), fps)


def _provenance(
    run: Any, fps: float, granularity: str, start_timecode: str | None
) -> dict[str, Any]:
    return {
        "version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_dir": str(run.root),
        "fps": fps,
        "dialogue_granularity": granularity,
        "start_timecode": start_timecode or "",
    }


def _finalize_warnings(summary: TimelineSummary, prober: MediaProber) -> None:
    if summary.missing_media:
        summary.warnings.append(
            f"{summary.missing_media} clip(s) reference missing media; run earlier steps or "
            "check paths."
        )
    if not prober.available and (summary.video_clips or summary.source_clips):
        summary.warnings.append(
            "ffprobe not found; clip durations fall back to artifact timing. Install ffmpeg "
            "for frame-accurate media ranges."
        )


def _adapter_suffixes(otio: Any) -> list[str]:
    suffixes: list[str] = []
    for name in otio.adapters.available_adapter_names():
        try:
            suffixes.extend(otio.adapters.from_name(name).suffixes or [])
        except Exception:  # pragma: no cover - defensive against odd plugins
            continue
    return suffixes
