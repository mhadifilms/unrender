from __future__ import annotations

import argparse

from unrender.cli.commands.context import _path_arg, _run_paths, _speaker_config_arg
from unrender.speakers.labeling import (
    apply_labels,
    label_interactive,
    write_label_template,
)


def _labels_template(args: argparse.Namespace) -> int:
    write_label_template(
        _run_paths(args, explicit=args.run_dir), args.out.expanduser() if args.out else None
    )
    return 0


def _labels_apply(args: argparse.Namespace) -> int:
    run = _run_paths(args, explicit=args.run_dir)
    apply_labels(
        run,
        _path_arg(args, "labels", args.labels, required=False) or run.labels_csv,
        config=_speaker_config_arg(args),
    )
    return 0


def _labels_interactive(args: argparse.Namespace) -> int:
    label_interactive(
        _run_paths(args, explicit=args.run_dir),
        config=_speaker_config_arg(args),
        cluster_type=args.type,
        relabel_all=args.relabel_all,
    )
    return 0
