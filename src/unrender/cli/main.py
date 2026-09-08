from __future__ import annotations

import sys

from unrender.cli.commands import _projects_from_args
from unrender.cli.helpers import configure_logging
from unrender.cli.parser import build_parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    configure_logging(args)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 2
    try:
        projects = _projects_from_args(args)
        if not projects:
            args.project_config = None
            return int(args.handler(args) or 0)
        for project in projects:
            print(f"==> {project.name}", flush=True)
            args.project_config = project
            rc = int(args.handler(args) or 0)
            if rc:
                return rc
        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        if getattr(args, "debug", False):
            raise
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
