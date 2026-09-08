"""CLI handler for `scenes group`.

Optional stage after `shots detect`/`shots cut`: groups the shot manifest
into narrative scenes and writes ``scenes.json`` for timeline assembly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from unrender.cli.commands.context import (
    _path_arg,
    _proxy_master_arg,
    _run_paths,
    _scenes_arg,
    _scenes_manifest_arg,
    _shots_manifest_arg,
)
from unrender.editorial.scenes import (
    assign_scenes,
    fuse_features,
    scene_manifest_entries,
    write_scenes_manifest,
)
from unrender.editorial.scenes.group import DEFAULT_THRESHOLD, BoundaryScores
from unrender.lib.csv_io import write_rows
from unrender.lib.fingerprints import warn_if_inputs_changed
from unrender.manifests import read_json

_SCENE_CSV_FIELDS = [
    "scene_id",
    "start_tc",
    "end_tc",
    "start_sec",
    "end_sec",
    "shot_count",
    "shot_ids",
]


def _scenes_group(args: argparse.Namespace) -> int:
    run = _run_paths(args, explicit=args.run_dir)
    run.ensure()
    manifest_out = _scenes_manifest_arg(args, run, explicit=args.scenes_out)
    shots_path = _shots_manifest_arg(args, run, explicit=args.shots, required=True)
    assert shots_path is not None
    shots_data = read_json(shots_path)
    shots = list(shots_data.get("shots") or [])
    if not shots:
        raise ValueError(f"shot manifest has no shots: {shots_path}")

    inputs = {"shots": shots_path}
    if manifest_out.exists() and not args.force:
        data = read_json(manifest_out)
        warn_if_inputs_changed(data, inputs, manifest_out)
        print(f"Scene manifest already exists, skipping: {manifest_out}", flush=True)
        print("Use --force to regroup.", flush=True)
        return 0

    video = _grouping_video(args, run, shots_path=shots_path)
    engine = str(_scenes_arg(args, "engine", args.engine, "fusion"))
    threshold = float(_scenes_arg(args, "threshold", args.threshold, DEFAULT_THRESHOLD))
    print(f"Grouping {len(shots)} shot(s) into scenes ({engine} engine) on {video}", flush=True)

    if engine == "fusion":
        result = _fusion_scores(args, run, video=video, shots=shots)
    else:
        raise ValueError(f"unknown scene engine {engine!r} (fusion)")

    shot_ids = [str(entry["shot_id"]) for entry in shots]
    assignment = assign_scenes(shot_ids, result.scores, threshold=threshold)
    entries = scene_manifest_entries(shots, assignment, [float(s) for s in result.scores])
    print(f"  {len(entries)} scene(s) at threshold {threshold}", flush=True)
    write_scenes_manifest(
        manifest_out,
        entries=entries,
        shots_manifest=shots_path,
        engine=result.engine,
        variant=result.variant,
        threshold=threshold,
        inputs=inputs,
    )
    if args.csv is not None:
        csv_path = (
            Path(args.csv).expanduser()
            if isinstance(args.csv, str)
            else manifest_out.with_suffix(".csv")
        )
        from unrender.editorial.scenes.manifest import spotting_rows

        write_rows(csv_path, spotting_rows(entries), _SCENE_CSV_FIELDS)
        print(f"Scene CSV saved: {csv_path}", flush=True)
    return 0


def _fusion_scores(args, run, *, video: Path, shots: list[dict]) -> BoundaryScores:
    from unrender.editorial.scenes.features import (
        KEYFRAMES_PER_SHOT,
        dialogue_spans,
        shot_audio_features,
        shot_embeddings,
    )
    from unrender.editorial.scenes.group import NOVELTY_WINDOW

    keyframes = int(_scenes_arg(args, "keyframes", None, KEYFRAMES_PER_SHOT))
    novelty_window = int(_scenes_arg(args, "novelty_window", None, NOVELTY_WINDOW))
    clip = shot_embeddings(video, shots, model="clip", keyframes=keyframes)
    dino = shot_embeddings(video, shots, model="dino", keyframes=keyframes)
    audio = None
    if not args.no_audio:
        audio_source = _path_arg(args, "master", None, required=False) or video
        audio = shot_audio_features(audio_source, shots, ffmpeg=args.ffmpeg or "ffmpeg")
        if audio is None:
            print("  no decodable audio; continuing with visual features", flush=True)
    spans = None
    lines_path = args.dialogue_lines or run.dialogue_lines_json
    if lines_path.exists():
        lines = list(read_json(lines_path).get("lines") or [])
        spans = dialogue_spans(lines, shots)
        if spans is not None:
            print(f"  dialogue continuity from {lines_path.name}", flush=True)
    result = fuse_features(
        clip_embeddings=clip,
        dino_embeddings=dino,
        audio_features=audio,
        dialogue_spans=spans,
        novelty_window=novelty_window,
    )
    print(f"  fused boundary scores ({result.variant} weights)", flush=True)
    return result


def _grouping_video(args, run, *, shots_path: Path) -> Path:
    """Keyframes come from the detection proxy (or a compressed master),
    mirroring the detector's never-touch-the-master rule."""
    explicit = getattr(args, "video", None)
    if explicit:
        return Path(explicit).expanduser()
    recorded = str(read_json(shots_path).get("proxy") or "")
    if recorded and Path(recorded).expanduser().is_file():
        return Path(recorded).expanduser()
    proxy = _proxy_master_arg(args, run, explicit=None, required=False)
    if proxy is not None and proxy.is_file():
        return proxy
    master = _path_arg(args, "master", None, required=False)
    if master is not None and master.is_file():
        return master
    raise ValueError(
        "no video to extract keyframes from: the shot manifest records no proxy and "
        "neither paths.proxy_path nor paths.master resolves to a file. Pass --video."
    )
