"""Shared helpers for locating and naming separated speaker stems.

The primary pipeline works on full-length per-speaker stems split by dialogue
line; these helpers load stem sources from globs, directories, or a
config-defined ``stem_map`` and infer speaker/source-group names from file
names.
"""

from __future__ import annotations

import glob
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from unrender.lib.audio import measure_peak_dbfs
from unrender.lib.csv_io import write_rows
from unrender.lib.ffmpeg import run_ffmpeg_slice as _shared_run_ffmpeg_slice
from unrender.lib.names import safe_name, source_group_from_text
from unrender.speakers import canonical_speaker_name, speaker_key

__all__ = [
    "StemSource",
    "load_stem_map",
    "load_stem_sources",
    "measure_peak_dbfs",
    "normalize_stem_sources",
    "run_ffmpeg_slice",
    "source_group_for_stem",
    "speaker_list",
    "write_plan_csv",
]

from dataclasses import dataclass


@dataclass(frozen=True)
class StemSource:
    path: Path
    speaker: str = ""
    source_group: str = ""


def load_stem_sources(spec: str | Path) -> list[StemSource]:
    candidate = Path(spec).expanduser()
    if candidate.exists() and candidate.is_dir():
        paths = sorted({*candidate.glob("*.wav"), *candidate.glob("*.WAV")})
    else:
        paths = [Path(path).expanduser() for path in sorted(glob.glob(str(spec)))]
    if not paths:
        raise FileNotFoundError(f"no separated speaker stems matched: {spec}")
    return [StemSource(path=path) for path in paths]


def load_stem_map(
    stem_map: Mapping[str, Any],
    *,
    speaker_names: list[str] | None = None,
) -> list[StemSource]:
    configured = {
        speaker_key(name): canonical_speaker_name(name)
        for name in speaker_names or []
        if speaker_key(name)
    }
    sources: list[StemSource] = []
    for speaker, raw_path in stem_map.items():
        if raw_path in (None, ""):
            continue
        configured_name = configured.get(speaker_key(str(speaker)))
        if configured and configured_name is None:
            allowed = ", ".join(sorted(configured.values()))
            raise ValueError(
                f"stem_map speaker is not configured: {speaker} (expected one of {allowed})"
            )
        path = Path(str(raw_path)).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"stem_map path for {speaker} not found: {path}")
        sources.append(
            StemSource(
                path=path,
                speaker=configured_name or canonical_speaker_name(str(speaker)),
            )
        )
    if not sources:
        raise ValueError("stem_map did not contain any usable speaker stems")
    return sources


def normalize_stem_sources(
    stems: Sequence[Path | StemSource],
    *,
    speaker_names: list[str] | None = None,
    infer_named_speakers: bool = False,
) -> list[StemSource]:
    sources: list[StemSource] = []
    for stem in stems:
        if isinstance(stem, StemSource):
            path = stem.path
            speaker = stem.speaker
            source_group = stem.source_group
        else:
            path = stem
            speaker = ""
            source_group = ""
        if infer_named_speakers and not speaker:
            speaker = _matched_config_speaker(path, speaker_names or [])
        sources.append(
            StemSource(
                path=path,
                speaker=canonical_speaker_name(speaker),
                source_group=source_group or _source_group(path, speaker=speaker),
            )
        )
    if not sources:
        raise ValueError("no separated speaker stems provided")
    return sources


def run_ffmpeg_slice(
    *,
    ffmpeg: str,
    source: Path,
    target: Path,
    start_sec: float,
    end_sec: float,
) -> None:
    _shared_run_ffmpeg_slice(
        ffmpeg=ffmpeg,
        source=source,
        target=target,
        start_sec=start_sec,
        end_sec=end_sec,
    )


def _source_group(path: Path, *, speaker: str = "") -> str:
    group = source_group_from_text(path.stem)
    if group:
        return group
    if speaker:
        return speaker_key(speaker)
    return safe_name(_strip_stem_suffix(path.stem))


def source_group_for_stem(path: Path, *, speaker: str = "") -> str:
    return _source_group(path, speaker=speaker)


def _matched_config_speaker(path: Path, speaker_names: list[str]) -> str:
    if not speaker_names:
        return ""
    stem_key = speaker_key(_strip_stem_suffix(path.stem))
    token_keys = {speaker_key(token) for token in re.split(r"[^A-Za-z0-9]+", path.stem)}
    candidates = sorted(
        (canonical_speaker_name(speaker) for speaker in speaker_names if speaker_key(speaker)),
        key=lambda speaker: len(speaker_key(speaker)),
        reverse=True,
    )
    for speaker in candidates:
        key = speaker_key(speaker)
        # Exact key/token equality only: substring matching would let a
        # speaker named "AL" claim a stem file named SALLY_stem.wav.
        if key in token_keys or key == stem_key:
            return speaker
    return ""


def _strip_stem_suffix(value: str) -> str:
    return re.sub(r"[_-]?stem$", "", value, flags=re.IGNORECASE)


def _speaker_list(value: Any) -> list[str]:
    if isinstance(value, list):
        raw = value
    else:
        raw = str(value or "").replace(";", ",").split(",")
    out: list[str] = []
    for item in raw:
        speaker = canonical_speaker_name(str(item))
        if speaker and speaker not in out:
            out.append(speaker)
    return out


def speaker_list(value: Any) -> list[str]:
    return _speaker_list(value)


def _write_plan_csv(path: Path, plan: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "shot_id",
        "line_id",
        "speaker",
        "stem_path",
        "source_group",
        "start_sec",
        "end_sec",
        "offset_in_shot_sec",
        "score",
        "status",
        "candidate_count",
    ]
    write_rows(path, plan, fields)


def write_plan_csv(path: Path, plan: list[dict[str, Any]]) -> None:
    _write_plan_csv(path, plan)
