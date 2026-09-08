from __future__ import annotations

import argparse

from unrender.cli.commands.context import (
    _project,
    _run_paths,
    _scenes_manifest_arg,
    _shots_manifest_arg,
    _timeline_arg,
    _timeline_fps,
)
from unrender.editorial.timeline import DEFAULT_FPS, MediaProber, build_timeline, write_timeline


def _timeline_build(args: argparse.Namespace) -> int:
    run = _run_paths(args, explicit=args.run_dir)
    run.ensure()
    shots_path = _shots_manifest_arg(args, run, explicit=args.shots, required=False)
    scenes_path = None if args.no_scenes else _scenes_manifest_arg(args, run, explicit=args.scenes)
    fps = _timeline_fps(args, DEFAULT_FPS)
    project = _project(args)
    name = str(
        _timeline_arg(args, "name", args.name, None)
        or (project.name if project is not None else run.root.name)
    )

    include_mne = _timeline_arg(args, "include_mne", args.include_mne, None)
    if include_mne is None:
        include_mne = (
            run.mne_fx_classification_json.exists()
            or run.mne_music_cues_json.exists()
            or run.mne_room_tone_json.exists()
        )

    timeline, summary = build_timeline(
        run,
        shots_path=shots_path,
        scenes_path=scenes_path,
        scene_stacks=bool(_timeline_arg(args, "scene_stacks", args.scene_stacks, False)),
        fps=fps,
        dialogue=str(_timeline_arg(args, "dialogue", args.dialogue, "auto")),
        name=name,
        start_timecode=_timeline_arg(args, "start_timecode", args.start_timecode, None),
        include_dialogue_bed=bool(_timeline_arg(args, "dialogue_bed", args.dialogue_bed, False)),
        include_shot_dx=bool(_timeline_arg(args, "shot_dx", args.shot_dx, False)),
        include_music_effects=not bool(
            _timeline_arg(args, "no_source_stems", args.no_source_stems, False)
        ),
        include_mne=bool(include_mne),
        markers=not bool(_timeline_arg(args, "no_markers", args.no_markers, False)),
        prober=MediaProber(str(_timeline_arg(args, "ffprobe", args.ffprobe, "ffprobe"))),
    )
    if summary.tracks == 0:
        raise ValueError(
            "no timeline content found. Run the dialogue/shot steps first, "
            "or pass --shots for a video timeline."
        )

    if not args.dry_run:
        out = args.out.expanduser() if args.out else run.timeline_otio
        summary.outputs.append(write_timeline(timeline, out))
        if bool(_timeline_arg(args, "bundle", args.bundle, False)):
            summary.outputs.append(
                write_timeline(timeline, run.timeline_otio.with_suffix(".otiod"))
            )

    for line in summary.describe():
        print(line, flush=True)
    return 0
