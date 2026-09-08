from __future__ import annotations

import argparse

from unrender.analysis.identity.voice import ContinuityPolicy, build_voice_db, match_voice_db
from unrender.cli.commands.context import (
    _run_paths,
    _speaker_db_path,
    _voice_arg,
    _voice_clips_spec_arg,
)
from unrender.cli.helpers import threshold
from unrender.manifests import load_voice_inputs


def _voice_build(args: argparse.Namespace) -> int:
    run = _run_paths(args, explicit=args.run_dir)
    run.ensure()
    clips = load_voice_inputs(_voice_clips_spec_arg(args, run))
    build_voice_db(
        clips=clips,
        voice_db_path=run.voice_db,
        review_dir=run.voice_review_dir,
        backend=args.backend,
        threshold=float(_voice_arg(args, "cluster_threshold", args.cluster_threshold, 0.35)),
        min_cluster_size=int(_voice_arg(args, "min_cluster_size", args.min_cluster_size, 2)),
        target_clusters=(
            None
            if _voice_arg(args, "voice_count", args.voice_count, None) in (None, "")
            else int(_voice_arg(args, "voice_count", args.voice_count, None))
        ),
        samples_per_cluster=int(
            _voice_arg(args, "samples_per_cluster", args.samples_per_cluster, 4)
        ),
        continuity_policy=_continuity_policy(args),
        force=args.force,
    )
    return 0


def _continuity_policy(args: argparse.Namespace) -> ContinuityPolicy:
    return ContinuityPolicy(
        max_gap_sec=float(
            _voice_arg(args, "continuity_max_gap_sec", args.continuity_max_gap_sec, 45.0)
        ),
        min_similarity=float(
            _voice_arg(args, "continuity_min_similarity", args.continuity_min_similarity, 0.35)
        ),
        min_margin=float(
            _voice_arg(args, "continuity_min_margin", args.continuity_min_margin, 0.02)
        ),
        short_max_duration_sec=float(
            _voice_arg(
                args,
                "continuity_short_max_duration_sec",
                args.continuity_short_max_duration_sec,
                1.25,
            )
        ),
        short_max_gap_sec=float(
            _voice_arg(
                args,
                "continuity_short_max_gap_sec",
                args.continuity_short_max_gap_sec,
                8.0,
            )
        ),
        interjection_max_duration_sec=float(
            _voice_arg(
                args,
                "continuity_interjection_max_duration_sec",
                args.continuity_interjection_max_duration_sec,
                0.75,
            )
        ),
        interjection_max_gap_sec=float(
            _voice_arg(
                args,
                "continuity_interjection_max_gap_sec",
                args.continuity_interjection_max_gap_sec,
                20.0,
            )
        ),
    )


def _voice_match(args: argparse.Namespace) -> int:
    run = _run_paths(args, explicit=args.run_dir)
    run.ensure()
    clips = load_voice_inputs(_voice_clips_spec_arg(args, run))
    match_voice_db(
        clips=clips,
        speaker_db_path=_speaker_db_path(args, args.speaker_db, run),
        output_json=run.voice_matches_json,
        backend=args.backend,
        sim_threshold=threshold(args.sim_threshold, "sim-threshold"),
    )
    return 0
