"""CLI handlers for `shots proxy`, `shots detect`, and `shots cut`.

These are the ingest stage of the pipeline: they take a project whose config
only names a master (video file or EXR sequence) and produce the shot
manifest plus per-shot media everything downstream consumes via
``paths.shots``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal, overload

from unrender.cli.commands.context import (
    _path_arg,
    _project,
    _run_paths,
    _shots_arg,
    _string_arg,
)
from unrender.editorial.shots.cut import DEFAULT_CUT_TIMEOUT_SEC, CutOutcome, cut_shots
from unrender.editorial.shots.detect import (
    DetectOptions,
    build_shots,
    detect_cut_frames,
    resolve_detection_input,
)
from unrender.editorial.shots.manifest import (
    MEDIA_CREATED,
    plan_shot_media,
    shot_manifest_entries,
    spotting_csv_rows,
    write_shots_manifest,
)
from unrender.editorial.shots.proxy import (
    DEFAULT_PROXY_CRF,
    DEFAULT_PROXY_WIDTH,
    create_master_proxy,
    default_proxy_path,
    master_usable_as_proxy,
    resolve_program_range,
)
from unrender.editorial.shots.sources import CutRange, load_cut_ranges, resolve_tc_origin
from unrender.lib.csv_io import write_rows
from unrender.lib.fingerprints import warn_if_inputs_changed
from unrender.lib.names import safe_name
from unrender.lib.sequence import SequenceInfo, classify_master, detect_sequence
from unrender.lib.timecode import seconds_to_frames, smpte_to_frames
from unrender.lib.video import count_video_frames, probe_video
from unrender.manifests import read_json
from unrender.project import RunPaths

_SPOTTING_FIELDS = ["shot_id", "start_tc", "end_tc", "start_sec", "end_sec"]


def _shots_proxy(args: argparse.Namespace) -> int:
    run = _run_paths(args, explicit=args.run_dir)
    master = _master_arg(args, required=True)
    output = (
        args.output.expanduser()
        if args.output
        else (_proxy_arg(args) or default_proxy_path(run, master))
    )
    result = create_master_proxy(
        master=master,
        output=output,
        run=run,
        fps=_fps_arg(args),
        width=int(_shots_arg(args, "proxy_width", args.width, DEFAULT_PROXY_WIDTH)),
        crf=int(_shots_arg(args, "proxy_crf", args.crf, DEFAULT_PROXY_CRF)),
        include_audio=not args.no_audio and bool(_shots_arg(args, "proxy_audio", None, True)),
        exr_transfer=str(_shots_arg(args, "exr_transfer", args.exr_transfer, "bt709")),
        timecode_start=_timecode_start(args),
        frame_range=_frame_range_arg(args),
        force=args.force,
        ffmpeg=_ffmpeg_arg(args),
        ffprobe=_ffprobe_arg(args),
    )
    if not result.created:
        print("Use --force to recreate it.", flush=True)
    return 0


def _shots_detect(args: argparse.Namespace) -> int:
    run = _run_paths(args, explicit=args.run_dir)
    manifest_out = _manifest_out(args, run)
    if manifest_out.exists() and not args.force:
        resumed = _resume_from_manifest(args, run, manifest_out)
        if resumed is not None:
            return resumed
    master = _master_arg(args, required=not args.no_cut)
    if master is not None:
        master_kind, master = classify_master(master)
    else:
        master_kind = "video"

    fps, sequence = _resolve_fps(args, master=master, master_kind=master_kind)
    timecode_start = _timecode_start(args)

    detect_input, _created = resolve_detection_input(
        master=master,
        proxy=_proxy_arg(args),
        run=run,
        auto_create=not args.no_proxy_create,
        force_proxy=args.force_proxy,
        fps=fps,
        width=int(_shots_arg(args, "proxy_width", args.width, DEFAULT_PROXY_WIDTH)),
        crf=int(_shots_arg(args, "proxy_crf", args.crf, DEFAULT_PROXY_CRF)),
        exr_transfer=str(_shots_arg(args, "exr_transfer", args.exr_transfer, "bt709")),
        timecode_start=timecode_start,
        frame_range=_frame_range_arg(args) if master_kind == "sequence" else None,
        ffmpeg=_ffmpeg_arg(args),
        ffprobe=_ffprobe_arg(args),
    )
    if fps is None:
        fps = probe_video(detect_input, ffprobe=_ffprobe_arg(args)).fps

    options = _detect_options(args)
    print(f"Detecting cuts on {detect_input} ({options.detector} mode)...", flush=True)
    cut_frames = detect_cut_frames(detect_input, options=options, ffprobe=_ffprobe_arg(args))
    print(f"  {len(cut_frames)} raw cut(s) detected", flush=True)

    if master is not None:
        total_frames = _total_frames(
            args, master=master, master_kind=master_kind, sequence=sequence, fps=fps
        )
        proxy_frames = count_video_frames(detect_input, ffprobe=_ffprobe_arg(args))
        if abs(proxy_frames - total_frames) > 2:
            print(
                f"  WARNING: proxy has {proxy_frames} frames but the master has "
                f"{total_frames}; the proxy may be stale (rerun `shots proxy --force`).",
                flush=True,
            )
    else:
        total_frames = count_video_frames(detect_input, ffprobe=_ffprobe_arg(args))
    padding = int(_shots_arg(args, "shot_id_padding", args.padding, 3))
    ranges = build_shots(
        cut_frames,
        total_frames=total_frames,
        fps=fps,
        min_gap_sec=float(_shots_arg(args, "min_gap_sec", args.min_gap_sec, 0.33)),
        padding=padding,
    )
    print(f"  {len(ranges)} shot(s) after coalescing", flush=True)

    probed_tc = None
    if master is not None and master_kind == "video":
        probed_tc = probe_video(master, ffprobe=_ffprobe_arg(args)).start_timecode
    origin_frame, origin_source = resolve_tc_origin(
        args.tc_origin or timecode_start, fps=fps, probed_tc=probed_tc
    )
    print(f"Timecode origin: frame {origin_frame} ({origin_source})", flush=True)

    return _finish(
        args,
        run=run,
        master=master if master is not None else detect_input,
        master_kind=master_kind,
        sequence=sequence,
        ranges=ranges,
        fps=fps,
        origin_frame=origin_frame,
        source_kind="detect",
        detector_summary=options.summary(),
        cut_media=not args.no_cut,
        proxy=detect_input if detect_input != master else None,
        extra_inputs={"proxy": detect_input},
    )


def _shots_cut(args: argparse.Namespace) -> int:
    run = _run_paths(args, explicit=args.run_dir)
    master = _master_arg(args, required=True)
    master_kind, master = classify_master(master)
    fps, sequence = _resolve_fps(args, master=master, master_kind=master_kind)
    if fps is None:
        raise ValueError("could not determine fps; pass --fps or set `fps` in the config")

    shots_path = _path_arg(args, "shots", args.shots, required=True)
    total_frames = _total_frames(
        args, master=master, master_kind=master_kind, sequence=sequence, fps=fps
    )
    probed_tc = None
    if master_kind == "video":
        probed_tc = probe_video(master, ffprobe=_ffprobe_arg(args)).start_timecode
    padding = int(_shots_arg(args, "shot_id_padding", args.padding, 3))
    cut_list = load_cut_ranges(
        shots_path,
        fps=fps,
        tc_origin=args.tc_origin or _timecode_start(args),
        probed_tc=probed_tc,
        duration_frames=total_frames,
        padding=padding,
        edl_source_tc=args.edl_tc == "source",
    )
    print(f"Loaded {len(cut_list.ranges)} shot(s) from {shots_path}", flush=True)

    return _finish(
        args,
        run=run,
        master=master,
        master_kind=master_kind,
        sequence=sequence,
        ranges=list(cut_list.ranges),
        fps=fps,
        origin_frame=cut_list.origin_frame,
        source_kind=cut_list.source_kind,
        detector_summary=None,
        cut_media=not args.no_cut,
        extra_inputs={"shot_list": shots_path},
    )


# ---------------------------------------------------------------------------


def _resume_from_manifest(
    args: argparse.Namespace, run: RunPaths, manifest_out: Path
) -> int | None:
    """Skip re-detection when the manifest already exists (not ``--force``).

    Returns an exit code when the existing manifest was used, or ``None`` when
    it is unusable (empty/foreign) and detection should run normally. Cutting
    still resumes so interrupted runs pick up where they left off.
    """
    data = read_json(manifest_out)
    entries = list(data.get("shots") or [])
    if not entries or not all("start_frame" in entry and "end_frame" in entry for entry in entries):
        return None
    print(f"Shot manifest already exists, skipping detection: {manifest_out}", flush=True)
    warn_if_inputs_changed(data, _manifest_inputs(data), manifest_out)
    if args.no_cut:
        print("Use --force to re-detect.", flush=True)
        return 0

    master = _master_arg(args, required=True)
    master_kind, master = classify_master(master)
    fps = float(data.get("fps") or 0)
    if not fps:
        fps = _resolve_fps(args, master=master, master_kind=master_kind)[0] or 0.0
    if not fps:
        raise ValueError("could not determine fps; pass --fps or set `fps` in the config")
    sequence = detect_sequence(master) if master_kind == "sequence" else None
    origin_frame = smpte_to_frames(str(data.get("timecode_start") or ""), fps) or 0
    ranges = [
        CutRange(
            shot_id=str(entry["shot_id"]),
            start_frame=int(entry["start_frame"]),
            end_frame=int(entry["end_frame"]),
            label=str(entry.get("label") or ""),
        )
        for entry in entries
    ]
    recorded_proxy = Path(str(data.get("proxy") or "")).expanduser()
    return _finish(
        args,
        run=run,
        master=master,
        master_kind=master_kind,
        sequence=sequence,
        ranges=ranges,
        fps=fps,
        origin_frame=origin_frame,
        source_kind=str(data.get("source") or "detect"),
        detector_summary=data.get("detector"),
        cut_media=True,
        proxy=recorded_proxy if recorded_proxy.is_file() else None,
        extra_inputs={"proxy": data.get("proxy") or None},
    )


def _manifest_inputs(data: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the fingerprint spec recorded when the manifest was written."""
    inputs: dict[str, Any] = {}
    master = Path(str(data.get("master") or "")).expanduser()
    if master.is_file():
        inputs["master"] = master
    elif master.is_dir():
        try:
            sequence = detect_sequence(master)
            inputs["master"] = [
                sequence.frame_path(sequence.start_frame),
                sequence.frame_path(sequence.end_frame),
            ]
        except ValueError:
            pass
    if data.get("proxy"):
        inputs["proxy"] = data["proxy"]
    if data.get("shot_list"):
        inputs["shot_list"] = data["shot_list"]
    return inputs


def _finish(
    args: argparse.Namespace,
    *,
    run: RunPaths,
    master: Path,
    master_kind: str,
    sequence: SequenceInfo | None,
    ranges: list[CutRange],
    fps: float,
    origin_frame: int,
    source_kind: str,
    detector_summary: dict[str, Any] | None,
    cut_media: bool,
    extra_inputs: dict[str, Any],
    proxy: Path | None = None,
) -> int:
    """Plan media paths, optionally cut, and write the shot manifest."""
    if not ranges:
        raise ValueError("no shots to write")
    shots_dir = _shots_dir(args, run)
    prefix = _shot_prefix(args, master)
    master_ext = master.suffix.lower() if master_kind == "video" else ".mov"
    if master_ext == ".mxf":
        print("  MXF master: plates are remuxed into .mov containers.", flush=True)
        master_ext = ".mov"
    plans = plan_shot_media(
        ranges,
        shots_dir=shots_dir,
        shot_prefix=prefix,
        master_kind=master_kind,
        master_ext=master_ext,
    )
    drop_frame = ";" in (_timecode_start(args) or "") or ";" in (
        getattr(args, "tc_origin", "") or ""
    )
    entries = shot_manifest_entries(
        ranges, plans, fps=fps, origin_frame=origin_frame, drop_frame=drop_frame
    )

    program_start = (
        resolve_program_range(sequence, _frame_range_arg(args))[0] if sequence is not None else None
    )

    outcomes: list[CutOutcome] = []
    if cut_media:
        if proxy is None:
            proxy = _cutting_proxy(
                args,
                run,
                master=master,
                master_kind=master_kind,
                fps=fps,
                sequence=sequence,
            )
        outcomes = cut_shots(
            master=master,
            master_kind=master_kind,
            ranges=ranges,
            plans=plans,
            fps=fps,
            origin_frame=origin_frame,
            drop_frame=drop_frame,
            proxy=proxy,
            proxy_crf=int(_shots_arg(args, "proxy_crf", args.crf, DEFAULT_PROXY_CRF)),
            per_shot_proxy=args.per_shot_proxy,
            sequence=sequence,
            program_start=program_start,
            renumber=_renumber(args)[0],
            renumber_start=_renumber(args)[1],
            link_mode=args.link,
            allow_gaps=args.allow_gaps,
            workers=_workers(args),
            force=args.force,
            dry_run=args.dry_run,
            ffmpeg=_ffmpeg_arg(args),
            ffprobe=_ffprobe_arg(args),
            timeout=int(args.timeout or DEFAULT_CUT_TIMEOUT_SEC),
        )
        by_id = {outcome.shot_id: outcome for outcome in outcomes}
        for entry in entries:
            outcome = by_id.get(str(entry["shot_id"]))
            if outcome is not None and outcome.ok and outcome.video_path is not None:
                entry["media_status"] = MEDIA_CREATED

    if args.dry_run:
        print("Dry run: no manifest written.", flush=True)
        return 0

    manifest_out = _manifest_out(args, run)
    inputs: dict[str, Any] = dict(extra_inputs)
    if master_kind == "video":
        inputs["master"] = master
    elif sequence is not None:
        inputs["master"] = [
            sequence.frame_path(sequence.start_frame),
            sequence.frame_path(sequence.end_frame),
        ]
    shot_list = extra_inputs.get("shot_list")
    write_shots_manifest(
        manifest_out,
        entries=entries,
        fps=fps,
        master=master,
        proxy=proxy or _proxy_arg(args),
        origin_frame=origin_frame,
        source=source_kind,
        detector=detector_summary,
        inputs=inputs,
        shot_list=Path(str(shot_list)) if shot_list else None,
    )
    if args.csv is not None:
        csv_path = (
            Path(args.csv).expanduser()
            if isinstance(args.csv, str)
            else manifest_out.with_suffix(".csv")
        )
        write_rows(csv_path, spotting_csv_rows(entries), _SPOTTING_FIELDS)
        print(f"Spotting CSV saved: {csv_path}", flush=True)

    failures = [outcome for outcome in outcomes if not outcome.ok]
    if failures:
        details = "; ".join(f"{o.shot_id}: {o.message}" for o in failures[:5])
        raise RuntimeError(f"{len(failures)} shot(s) failed to cut ({details})")
    return 0


def _cutting_proxy(
    args: argparse.Namespace,
    run: RunPaths,
    *,
    master: Path,
    master_kind: str,
    fps: float | None,
    sequence: SequenceInfo | None = None,
) -> Path | None:
    """Resolve the full-program proxy used for per-shot proxy segments."""
    needs_proxy = (master_kind == "sequence" or args.per_shot_proxy) and not args.no_proxy
    if not needs_proxy or args.dry_run:
        return None
    configured = _proxy_arg(args)
    if configured is not None and configured.exists():
        return configured
    if configured is not None:
        raise ValueError(f"configured proxy does not exist: {configured}")
    if master_kind == "video" and master_usable_as_proxy(master, ffprobe=_ffprobe_arg(args)):
        return master
    default = default_proxy_path(run, master)
    if default.exists():
        return default
    print("Per-shot proxies need a full-program proxy; creating one first.", flush=True)
    result = create_master_proxy(
        master=master,
        output=default,
        run=run,
        fps=fps,
        width=int(_shots_arg(args, "proxy_width", args.width, DEFAULT_PROXY_WIDTH)),
        crf=int(_shots_arg(args, "proxy_crf", args.crf, DEFAULT_PROXY_CRF)),
        include_audio=False,
        exr_transfer=str(_shots_arg(args, "exr_transfer", args.exr_transfer, "bt709")),
        timecode_start=_timecode_start(args),
        frame_range=_frame_range_arg(args) if sequence is not None else None,
        ffmpeg=_ffmpeg_arg(args),
        ffprobe=_ffprobe_arg(args),
    )
    return result.path


# ---------------------------------------------------------------------------
# Argument resolution helpers


@overload
def _master_arg(args: argparse.Namespace, *, required: Literal[True]) -> Path: ...


@overload
def _master_arg(args: argparse.Namespace, *, required: bool) -> Path | None: ...


def _master_arg(args: argparse.Namespace, *, required: bool) -> Path | None:
    explicit = getattr(args, "master", None)
    value = _string_arg(
        args,
        "master",
        str(explicit) if explicit else None,
        required=required,
        aliases=("master_path",),
    )
    return Path(value).expanduser() if value else None


def _proxy_arg(args: argparse.Namespace) -> Path | None:
    explicit = getattr(args, "proxy", None)
    value = _string_arg(
        args,
        "proxy_path",
        str(explicit) if explicit else None,
        required=False,
        aliases=("proxy_master", "proxy"),
    )
    return Path(value).expanduser() if value else None


def _frame_range_arg(args: argparse.Namespace) -> tuple[int, int] | None:
    """The program span inside a sequence master, as ``START-END`` or ``START END``."""
    value = _shots_arg(args, "frame_range", getattr(args, "frame_range", None), None)
    if not value:
        return None
    if isinstance(value, str):
        parts = value.replace(":", "-").split("-")
    else:
        parts = list(value)
    if len(parts) != 2:
        raise ValueError(f"frame range must be START-END, got {value!r}")
    try:
        return int(parts[0]), int(parts[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"frame range must be two integers, got {value!r}") from exc


def _fps_arg(args: argparse.Namespace) -> float | None:
    if getattr(args, "fps", None) is not None:
        return float(args.fps)
    project = _project(args)
    if project is not None:
        value = project.data.get("fps")
        if value is not None and value != "":
            return float(value)
    return None


def _resolve_fps(
    args: argparse.Namespace,
    *,
    master: Path | None,
    master_kind: str,
) -> tuple[float | None, SequenceInfo | None]:
    configured = _fps_arg(args)
    if master is None:
        return configured, None
    if master_kind == "sequence":
        sequence = detect_sequence(master)
        if not configured:
            raise ValueError(
                "an image-sequence master needs an explicit frame rate: pass --fps or "
                "set top-level `fps` in the project config"
            )
        return configured, sequence
    probe = probe_video(master, ffprobe=_ffprobe_arg(args))
    if configured and abs(configured - probe.fps) > 0.01:
        print(
            f"  WARNING: config fps {configured} differs from probed master fps "
            f"{probe.fps:.3f}; using the probed rate.",
            flush=True,
        )
    return probe.fps, None


def _total_frames(
    args: argparse.Namespace,
    *,
    master: Path | None,
    master_kind: str,
    sequence: SequenceInfo | None,
    fps: float,
) -> int:
    if master_kind == "sequence" and sequence is not None:
        start, end = resolve_program_range(sequence, _frame_range_arg(args))
        return end - start + 1
    if master is not None and master.is_file():
        probe = probe_video(master, ffprobe=_ffprobe_arg(args))
        return seconds_to_frames(probe.duration_sec, fps)
    raise ValueError("cannot determine master duration")


def _timecode_start(args: argparse.Namespace) -> str | None:
    value = _shots_arg(args, "timecode_start", getattr(args, "timecode_start", None), None)
    return str(value) if value else None


def _detect_options(args: argparse.Namespace) -> DetectOptions:
    defaults = DetectOptions()
    return DetectOptions(
        detector=str(_shots_arg(args, "detector", args.detector, defaults.detector)),
        content_threshold=float(
            _shots_arg(args, "content_threshold", args.threshold, defaults.content_threshold)
        ),
        adaptive_threshold=float(
            _shots_arg(args, "adaptive_threshold", None, defaults.adaptive_threshold)
        ),
        min_content_val=float(_shots_arg(args, "min_content_val", None, defaults.min_content_val)),
        ensemble_content_threshold=float(
            _shots_arg(
                args, "ensemble_content_threshold", None, defaults.ensemble_content_threshold
            )
        ),
        edge_weight=float(_shots_arg(args, "edge_weight", None, defaults.edge_weight)),
        detect_fades=bool(_shots_arg(args, "detect_fades", None, defaults.detect_fades)),
        fade_threshold=int(_shots_arg(args, "fade_threshold", None, defaults.fade_threshold)),
        min_shot_frames=int(
            _shots_arg(args, "min_shot_frames", args.min_shot_frames, defaults.min_shot_frames)
        ),
        min_gap_sec=float(_shots_arg(args, "min_gap_sec", args.min_gap_sec, defaults.min_gap_sec)),
    )


def _shots_dir(args: argparse.Namespace, run: RunPaths) -> Path:
    explicit = getattr(args, "shots_dir", None)
    if explicit is not None:
        return Path(explicit).expanduser()
    configured = _path_arg(args, "shots_dir", None, required=False)
    return configured if configured is not None else run.shots_media_dir


def _manifest_out(args: argparse.Namespace, run: RunPaths) -> Path:
    explicit = getattr(args, "shots_out", None)
    if explicit is not None:
        return Path(explicit).expanduser()
    project = _project(args)
    if project is not None:
        configured = project.path_value("shots")
        if configured is not None and configured.suffix.lower() == ".json":
            return configured
    return run.shots_manifest_json


def _shot_prefix(args: argparse.Namespace, master: Path) -> str:
    configured = _shots_arg(args, "shot_name_prefix", getattr(args, "shot_prefix", None), None)
    if configured:
        return str(configured)
    project = _project(args)
    if project is not None:
        return project.name
    return safe_name(master.stem if master.is_file() else master.name, fallback="shot")


def _renumber(args: argparse.Namespace) -> tuple[bool, int]:
    explicit = getattr(args, "renumber", None)
    if explicit is not None:
        return True, int(explicit) if str(explicit) != "default" else 1001
    enabled = bool(_shots_arg(args, "renumber_frames", None, False))
    start = int(_shots_arg(args, "renumber_start", None, 1001))
    return enabled, start


def _workers(args: argparse.Namespace) -> int | None:
    value = _shots_arg(args, "workers", getattr(args, "workers", None), None)
    return int(value) if value not in (None, "") else None


def _ffmpeg_arg(args: argparse.Namespace) -> str:
    return str(getattr(args, "ffmpeg", None) or "ffmpeg")


def _ffprobe_arg(args: argparse.Namespace) -> str:
    return str(getattr(args, "ffprobe", None) or "ffprobe")
