from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SpeakerVariant = Literal["two_speaker", "n_speaker"]
SourceRole = Literal["dialogue", "music_fx", "effects"]


@dataclass(frozen=True)
class AudioSeparationResult:
    source: str
    output_dir: Path
    stems: list[Path]
    skipped: bool = False


@dataclass(frozen=True)
class SourceSeparationResult:
    source: str
    output_dir: Path
    dialogue_stem: Path | None
    stems: dict[str, Path]
    skipped: bool = False
