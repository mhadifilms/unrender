from __future__ import annotations

import argparse
import importlib.util
import shutil

from unrender.cli.commands.context import (
    _project,
    _project_audio_asset_issue,
    _project_path_issue,
    _run_paths,
)
from unrender.cli.helpers import artifact_status_paths


def _status(args: argparse.Namespace) -> int:
    run = _run_paths(args, explicit=args.run_dir)
    print(f"Run: {run.root}")
    for label, path in artifact_status_paths(run):
        print(f"{'ok' if path.exists() else 'missing':7} {label:24} {path}")
    return 0


def _doctor(args: argparse.Namespace) -> int:
    project = _project(args)
    issues: list[str] = []
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            issues.append(f"{binary} is not available on PATH")
    if project is not None:
        for key in (
            "master",
            "master_path",
            "proxy_master",
            "proxy",
            "proxy_path",
            "dx_stem",
            "full_audio",
            "transcript",
            "shots",
        ):
            issue = _project_path_issue(project, key)
            if issue:
                issues.append(issue)
        for key in ("full_mix", "dme.dx", "dme.mx", "dme.fx"):
            issue = _project_audio_asset_issue(project, key)
            if issue:
                issues.append(issue)
        for key in ("source_backend", "speaker_strategy"):
            value = project.audio_value(key)
            if value not in (None, "", "audioshake"):
                issues.append(f"audio.{key} must be audioshake")
        if project.string_value("master") and importlib.util.find_spec("scenedetect") is None:
            issues.append('Shot detection needs the detect extra: pip install "unrender[detect]"')
    for issue in issues:
        print(f"ERROR: {issue}")
    if not issues:
        print("Configured paths and local media prerequisites are ready.")
    return 1 if issues else 0
