from __future__ import annotations

import argparse

from unrender.analysis.identity.face import build_face_db, match_shots
from unrender.cli.commands.context import (
    _face_arg,
    _proxy_master_arg,
    _run_paths,
    _shots_arg,
    _shots_manifest_arg,
    _speaker_db_path,
)
from unrender.cli.helpers import threshold
from unrender.manifests import load_shot_manifest


def _face_build(args: argparse.Namespace) -> int:
    run = _run_paths(args, explicit=args.run_dir)
    run.ensure()
    video_path = _proxy_master_arg(args, run, explicit=args.video, required=True)
    assert video_path is not None
    build_face_db(
        video_path=video_path,
        face_db_path=run.face_db,
        review_dir=run.face_review_dir,
        interval_sec=float(_face_arg(args, "interval", args.interval, 1.0)),
        max_frames=args.max_frames or None,
        threshold=float(_face_arg(args, "threshold", args.threshold, 0.5)),
        min_cluster_size=int(_face_arg(args, "min_cluster_size", args.min_cluster_size, 2)),
        target_clusters=(
            None
            if _face_arg(args, "face_count", args.face_count, None) in (None, "")
            else int(_face_arg(args, "face_count", args.face_count, None))
        ),
        min_confidence=float(_face_arg(args, "min_confidence", args.min_confidence, 0.5)),
        min_face_px=int(_face_arg(args, "min_face_px", args.min_face_px, 36)),
        force=args.force,
    )
    return 0


def _shots_match(args: argparse.Namespace) -> int:
    run = _run_paths(args, explicit=args.run_dir)
    run.ensure()
    shots_path = _shots_manifest_arg(args, run, explicit=args.shots, required=True)
    assert shots_path is not None
    shots = load_shot_manifest(shots_path)
    match_shots(
        shots=shots,
        speaker_db_path=_speaker_db_path(args, args.speaker_db, run),
        output_json=run.shot_matches_json,
        output_csv=run.shot_matches_csv,
        samples_per_shot=int(_shots_arg(args, "samples_per_shot", args.samples_per_shot, 8)),
        sim_threshold=threshold(
            _shots_arg(args, "sim_threshold", args.sim_threshold, 0.45), "sim-threshold"
        ),
        min_confidence=threshold(
            _shots_arg(args, "min_confidence", args.min_confidence, 0.5), "min-confidence"
        ),
        min_votes=int(_shots_arg(args, "min_votes", args.min_votes, 2)),
        min_margin=float(_shots_arg(args, "min_margin", args.min_margin, 0.05)),
    )
    return 0
