from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unrender.manifests.loaders import (
    load_dialogue_lines,
    load_shot_manifest,
    load_voice_inputs,
    read_json,
    write_dialogue_lines_csv,
    write_json,
)
from unrender.manifests.models import DialogueLineRecord, ShotRecord, VoiceInput
from unrender.project.paths import RunPaths

MANIFEST_VERSION = "1.0"


@dataclass(frozen=True)
class ManifestStore:
    """Typed-ish persistence facade over the stable run artifact layout."""

    run: RunPaths

    def load_dialogue_lines(self, path: Path | None = None) -> list[DialogueLineRecord]:
        return load_dialogue_lines(path or self.run.dialogue_lines_json)

    def load_shots(self, path: Path) -> list[ShotRecord]:
        return load_shot_manifest(path)

    def load_voice_inputs(self, spec: str | None = None) -> list[VoiceInput]:
        return load_voice_inputs(spec or str(self.run.voice_clips_json))

    def read(self, path: Path) -> Any:
        return read_json(path)

    def write(self, path: Path, data: Any) -> None:
        write_json(path, with_manifest_version(data))

    def write_dialogue_lines(self, lines: list[dict[str, Any]]) -> None:
        self.write(self.run.dialogue_lines_json, {"lines": lines})
        write_dialogue_lines_csv(self.run.dialogue_lines_csv, lines)


def with_manifest_version(data: Any, *, version: str = MANIFEST_VERSION) -> Any:
    if isinstance(data, dict) and "version" not in data:
        return {"version": version, **data}
    return data
