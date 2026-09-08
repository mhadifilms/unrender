"""Cut a master (video file or image sequence) into per-shot media.

Metadata preservation is the ruling concern:

- Intra-frame masters (ProRes, DNx, ...) are cut with video **stream copy**
  (bit-identical frames) plus ``-map_metadata 0`` and a rewritten timecode
  track; audio is re-encoded to 24-bit PCM for sample-accurate boundaries.
- Inter-frame masters (H.264/HEVC) are re-encoded at mezzanine quality with
  output-seeking for frame accuracy, carrying the probed color tags forward.
- Image-sequence masters are cut by copying frame files verbatim — every EXR
  header attribute survives by definition. Renumbering is opt-in.
"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from unrender.editorial.shots.manifest import ShotMediaPlan
from unrender.editorial.shots.sources import CutRange
from unrender.lib.sequence import SequenceInfo
from unrender.lib.timecode import frames_to_seconds, frames_to_smpte
from unrender.lib.video import (
    VideoProbe,
    count_video_frames,
    probe_video,
    run_video_tool,
)

DEFAULT_CUT_TIMEOUT_SEC = 900
# Guards against header-only stubs from a killed encode; the frame-count
# probe in _video_output_complete does the real completeness validation.
_MIN_VALID_BYTES = 1_000

STATUS_CREATED = "created"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"
STATUS_PLANNED = "planned"


@dataclass(frozen=True)
class CutOutcome:
    shot_id: str
    status: str
    plate_path: Path | None = None
    video_path: Path | None = None
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status in (STATUS_CREATED, STATUS_SKIPPED, STATUS_PLANNED)


def cut_shots(
    *,
    master: Path,
    master_kind: str,
    ranges: list[CutRange],
    plans: list[ShotMediaPlan],
    fps: float,
    origin_frame: int,
    drop_frame: bool = False,
    proxy: Path | None = None,
    proxy_crf: int = 20,
    per_shot_proxy: bool = False,
    sequence: SequenceInfo | None = None,
    program_start: int | None = None,
    renumber: bool = False,
    renumber_start: int = 1001,
    link_mode: str = "copy",
    allow_gaps: bool = False,
    workers: int | None = None,
    force: bool = False,
    dry_run: bool = False,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    timeout: int = DEFAULT_CUT_TIMEOUT_SEC,
) -> list[CutOutcome]:
    if not ranges:
        raise ValueError("no shots to cut")
    if len(ranges) != len(plans):
        raise ValueError("ranges and media plans are out of step")

    if dry_run:
        for range_, plan in zip(ranges, plans, strict=True):
            print(
                f"  [dry-run] shot {range_.shot_id}: frames "
                f"{range_.start_frame}..{range_.end_frame} -> {plan.plate_path}",
                flush=True,
            )
        return [
            CutOutcome(
                shot_id=range_.shot_id,
                status=STATUS_PLANNED,
                plate_path=plan.plate_path,
                video_path=plan.video_path,
            )
            for range_, plan in zip(ranges, plans, strict=True)
        ]

    if master_kind == "sequence":
        if sequence is None:
            raise ValueError("sequence info is required to cut an image-sequence master")
        outcomes = _run_stage(
            "plates",
            [
                lambda r=range_, p=plan: _cut_sequence_shot(
                    sequence,
                    r,
                    p,
                    program_start=(
                        sequence.start_frame if program_start is None else program_start
                    ),
                    renumber=renumber,
                    renumber_start=renumber_start,
                    link_mode=link_mode,
                    allow_gaps=allow_gaps,
                    force=force,
                )
                for range_, plan in zip(ranges, plans, strict=True)
            ],
            workers=workers or min(8, len(ranges)),
        )
        proxy_needed = True
    else:
        probe = probe_video(master, ffprobe=ffprobe)
        branch = "stream copy" if probe.is_intra_frame else "re-encode"
        print(f"Master codec: {probe.codec} ({branch})", flush=True)
        outcomes = _run_stage(
            "plates",
            [
                lambda r=range_, p=plan: _cut_video_shot(
                    probe,
                    r,
                    p,
                    fps=fps,
                    origin_frame=origin_frame,
                    drop_frame=drop_frame,
                    force=force,
                    ffmpeg=ffmpeg,
                    ffprobe=ffprobe,
                    timeout=timeout,
                )
                for range_, plan in zip(ranges, plans, strict=True)
            ],
            workers=workers or min(4, len(ranges)),
        )
        proxy_needed = per_shot_proxy

    if proxy_needed:
        outcomes = _add_per_shot_proxies(
            outcomes,
            ranges=ranges,
            plans=plans,
            proxy=proxy,
            proxy_crf=proxy_crf,
            fps=fps,
            workers=workers,
            force=force,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            timeout=timeout,
        )
    return outcomes


# ---------------------------------------------------------------------------
# Video masters


def _cut_video_shot(
    probe: VideoProbe,
    range_: CutRange,
    plan: ShotMediaPlan,
    *,
    fps: float,
    origin_frame: int,
    drop_frame: bool,
    force: bool,
    ffmpeg: str,
    ffprobe: str,
    timeout: int,
) -> CutOutcome:
    target = plan.plate_path
    if not force and _video_output_complete(target, range_.num_frames, ffprobe=ffprobe):
        print(f"  shot {range_.shot_id}: plate exists, skipping", flush=True)
        return CutOutcome(range_.shot_id, STATUS_SKIPPED, target, plan.video_path)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    start_sec = frames_to_seconds(range_.start_frame, fps)
    shot_tc = frames_to_smpte(origin_frame + range_.start_frame, fps, drop_frame=drop_frame)
    rate = probe.r_frame_rate or f"{fps:.6f}"

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    if probe.is_intra_frame:
        # Input seeking is frame accurate when every frame is a keyframe, and
        # lets the video stream be copied without touching a single pixel.
        cmd += ["-ss", f"{start_sec:.6f}", "-i", str(probe.path)]
        cmd += ["-map", "0:v:0", "-map", "0:a?"]
        cmd += ["-c:v", "copy", "-c:a", "pcm_s24le"]
        cmd += ["-frames:v", str(range_.num_frames)]
        cmd += ["-avoid_negative_ts", "make_zero"]
    else:
        # Input seeking (before -i) stays frame accurate through the decode and
        # avoids re-decoding from the file start for every shot.
        cmd += ["-ss", f"{start_sec:.6f}", "-i", str(probe.path)]
        cmd += ["-map", "0:v:0", "-map", "0:a?"]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "12"]
        cmd += ["-fps_mode", "cfr", "-r", rate]
        cmd += ["-c:a", "copy"]
        cmd += ["-frames:v", str(range_.num_frames)]
        if probe.color_primaries:
            cmd += ["-color_primaries", probe.color_primaries]
        if probe.color_transfer:
            cmd += ["-color_trc", probe.color_transfer]
        if probe.color_space:
            cmd += ["-colorspace", probe.color_space]
    cmd += ["-map_metadata", "0"]
    if target.suffix.lower() in (".mov", ".mp4"):
        cmd += ["-movflags", "+write_colr+use_metadata_tags+faststart"]
        cmd += ["-timecode", shot_tc, "-write_tmcd", "1"]
    cmd.append(str(target))

    try:
        run_video_tool(cmd, timeout=timeout)
    except RuntimeError as exc:
        target.unlink(missing_ok=True)
        return CutOutcome(range_.shot_id, STATUS_FAILED, message=str(exc))
    if not _video_output_complete(target, range_.num_frames, ffprobe=ffprobe):
        target.unlink(missing_ok=True)
        return CutOutcome(
            range_.shot_id,
            STATUS_FAILED,
            message=f"output frame count did not match the expected {range_.num_frames}",
        )
    print(
        f"  shot {range_.shot_id}: {shot_tc} +{range_.num_frames}f -> {target.name}",
        flush=True,
    )
    return CutOutcome(range_.shot_id, STATUS_CREATED, target, plan.video_path)


def _video_output_complete(path: Path, expected_frames: int, *, ffprobe: str) -> bool:
    if not path.is_file() or path.stat().st_size < _MIN_VALID_BYTES:
        return False
    try:
        actual = count_video_frames(path, ffprobe=ffprobe)
    except (RuntimeError, ValueError):
        return False
    # The final shot can land one frame short when the container duration
    # rounds differently than frame math; treat that as complete.
    return abs(actual - expected_frames) <= 1


# ---------------------------------------------------------------------------
# Image-sequence masters


def _cut_sequence_shot(
    sequence: SequenceInfo,
    range_: CutRange,
    plan: ShotMediaPlan,
    *,
    program_start: int,
    renumber: bool,
    renumber_start: int,
    link_mode: str,
    allow_gaps: bool,
    force: bool,
) -> CutOutcome:
    target_dir = plan.plate_path
    pairs: list[tuple[Path, Path]] = []
    missing = 0
    padding = max(sequence.padding, 4)
    for index in range(range_.num_frames):
        source_frame = program_start + range_.start_frame + index
        source = sequence.frame_path(source_frame)
        if not source.exists():
            missing += 1
            continue
        if renumber:
            name = f"{plan.shot_name}.{renumber_start + index:0{padding}d}{sequence.ext}"
        else:
            name = source.name
        pairs.append((source, target_dir / name))

    if missing and not allow_gaps:
        return CutOutcome(
            range_.shot_id,
            STATUS_FAILED,
            message=(
                f"{missing} source frame(s) missing in "
                f"{sequence.directory} (rerun with --allow-gaps to skip them)"
            ),
        )
    if not pairs:
        return CutOutcome(range_.shot_id, STATUS_FAILED, message="no source frames exist")

    if not force and all(target.exists() for _, target in pairs):
        print(f"  shot {range_.shot_id}: {len(pairs)} frame(s) exist, skipping", flush=True)
        return CutOutcome(
            range_.shot_id,
            STATUS_SKIPPED,
            target_dir,
            plan.video_path,
            message=f"{missing} missing frame(s)" if missing else "",
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    fallbacks = 0
    for source, target in pairs:
        if target.exists() and not force:
            continue
        target.unlink(missing_ok=True)
        if link_mode == "hardlink":
            try:
                os.link(source, target)
            except OSError:
                fallbacks += 1
                shutil.copy2(source, target)
        elif link_mode == "symlink":
            target.symlink_to(source)
        else:
            shutil.copy2(source, target)
    if fallbacks:
        print(
            f"  WARNING: shot {range_.shot_id}: {fallbacks} hardlink(s) crossed "
            "devices and were copied instead.",
            flush=True,
        )
    note = f"{missing} missing frame(s)" if missing else ""
    print(
        f"  shot {range_.shot_id}: {len(pairs)} frame(s) -> {target_dir.name}"
        + (f" ({note})" if note else ""),
        flush=True,
    )
    return CutOutcome(range_.shot_id, STATUS_CREATED, target_dir, plan.video_path, message=note)


# ---------------------------------------------------------------------------
# Per-shot proxies (cut from the full-program proxy)


def _add_per_shot_proxies(
    outcomes: list[CutOutcome],
    *,
    ranges: list[CutRange],
    plans: list[ShotMediaPlan],
    proxy: Path | None,
    proxy_crf: int,
    fps: float,
    workers: int | None,
    force: bool,
    ffmpeg: str,
    ffprobe: str,
    timeout: int,
) -> list[CutOutcome]:
    if proxy is None or not proxy.exists():
        print(
            "  WARNING: no full-program proxy available; per-shot proxies were not "
            "created, so `shots match` will not work until they exist.",
            flush=True,
        )
        return [
            (
                CutOutcome(o.shot_id, o.status, o.plate_path, None, o.message)
                if o.ok and plans[i].video_path != plans[i].plate_path
                else o
            )
            for i, o in enumerate(outcomes)
        ]

    by_id = {outcome.shot_id: outcome for outcome in outcomes}
    jobs = []
    for range_, plan in zip(ranges, plans, strict=True):
        plate = by_id.get(range_.shot_id)
        if plate is None or not plate.ok:
            continue
        if plan.video_path != plan.plate_path:
            target = plan.video_path  # sequence master: proxy IS the manifest video
        else:
            target = plan.shot_dir / f"{plan.shot_name}_proxy.mov"
        jobs.append(
            lambda r=range_, p=plan, t=target: _cut_proxy_segment(
                proxy,
                r,
                p,
                t,
                fps=fps,
                crf=proxy_crf,
                force=force,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
                timeout=timeout,
            )
        )
    if not jobs:
        return outcomes
    proxy_outcomes = _run_stage("proxies", jobs, workers=workers or min(4, len(jobs)))
    failed = {o.shot_id: o for o in proxy_outcomes if not o.ok}
    merged: list[CutOutcome] = []
    for outcome in outcomes:
        failure = failed.get(outcome.shot_id)
        if failure is not None and outcome.ok:
            merged.append(
                CutOutcome(
                    outcome.shot_id,
                    STATUS_FAILED,
                    outcome.plate_path,
                    None,
                    f"per-shot proxy failed: {failure.message}",
                )
            )
        else:
            merged.append(outcome)
    return merged


def _cut_proxy_segment(
    proxy: Path,
    range_: CutRange,
    plan: ShotMediaPlan,
    target: Path,
    *,
    fps: float,
    crf: int,
    force: bool,
    ffmpeg: str,
    ffprobe: str,
    timeout: int,
) -> CutOutcome:
    if not force and _video_output_complete(target, range_.num_frames, ffprobe=ffprobe):
        return CutOutcome(range_.shot_id, STATUS_SKIPPED, plan.plate_path, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    start_sec = frames_to_seconds(range_.start_frame, fps)
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.6f}",
        "-i",
        str(proxy),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-fps_mode",
        "cfr",
        "-r",
        f"{fps:.6f}",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-frames:v",
        str(range_.num_frames),
        "-movflags",
        "+faststart",
        str(target),
    ]
    try:
        run_video_tool(cmd, timeout=timeout)
    except RuntimeError as exc:
        target.unlink(missing_ok=True)
        return CutOutcome(range_.shot_id, STATUS_FAILED, plan.plate_path, message=str(exc))
    print(f"  shot {range_.shot_id}: proxy -> {target.name}", flush=True)
    return CutOutcome(range_.shot_id, STATUS_CREATED, plan.plate_path, target)


# ---------------------------------------------------------------------------


def _run_stage(name: str, jobs: list, *, workers: int) -> list[CutOutcome]:
    print(f"Cutting {len(jobs)} shot {name} ({max(1, workers)} worker(s))...", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(lambda job: job(), jobs))
