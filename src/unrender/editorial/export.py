"""Export available editorial artifacts without host-specific state mutation."""

from __future__ import annotations

import shutil
from pathlib import Path

from unrender.project import RunPaths


def export_artifacts(run: RunPaths, output_dir: Path | None = None) -> Path:
    out = (output_dir or run.export_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    for source in (
        run.shots_manifest_json,
        run.scenes_manifest_json,
        run.face_db,
        run.voice_db,
        run.speaker_db,
        run.shot_matches_json,
        run.voice_matches_json,
        run.dialogue_lines_json,
        run.dialogue_stem_plan_json,
        run.shot_stem_plan_json,
        run.source_separation_json,
        run.audio_separation_json,
        run.timeline_otio,
    ):
        if source.is_file() and source.resolve() != (out / source.name).resolve():
            shutil.copy2(source, out / source.name)
    print(f"Artifacts exported: {out}", flush=True)
    return out
