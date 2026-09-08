from __future__ import annotations

import argparse

from unrender.cli.commands.context import _run_paths
from unrender.editorial.export import export_artifacts


def _export_artifacts(args: argparse.Namespace) -> int:
    export_artifacts(
        _run_paths(args, explicit=args.run_dir), args.out.expanduser() if args.out else None
    )
    return 0
