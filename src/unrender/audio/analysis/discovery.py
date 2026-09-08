"""Role-aware discovery of audio analysis inputs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unrender.project import RunPaths

_AUDIO_SUFFIXES = {".wav", ".wave", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".aif", ".aiff"}

AudioInputs = (
    Mapping[str, str | Path | Sequence[str | Path]] | Sequence[str | Path] | str | Path | None
)


@dataclass(frozen=True)
class DiscoveredAudio:
    id: str
    role: str
    path: Path
    discovered_from: str
    metadata: dict[str, Any]


def discover_audio_inputs(
    run: RunPaths | Path | str,
    inputs: AudioInputs = None,
) -> list[DiscoveredAudio]:
    """Discover existing audio files, with manifests winning path collisions."""
    paths = run if isinstance(run, RunPaths) else RunPaths.from_path(run)
    candidates: list[tuple[str, Path, str, dict[str, Any]]] = []

    _source_manifest(candidates, paths, paths.source_separation_json)
    _speaker_manifest(candidates, paths, paths.audio_separation_json)
    _fx_manifest(candidates, paths, paths.mne_fx_classification_json)
    _music_manifest(candidates, paths, paths.mne_music_cues_json)
    _room_tone_manifest(candidates, paths, paths.mne_room_tone_json)
    _explicit_inputs(candidates, paths, inputs)

    _directory_fallback(candidates, paths.source_stems_dir, "source_stem")
    _directory_fallback(candidates, paths.unmapped_speakers_dir, "speaker")

    discovered: list[DiscoveredAudio] = []
    seen: set[Path] = set()
    for role, candidate, source, metadata in candidates:
        resolved = candidate.expanduser().resolve()
        if (
            resolved in seen
            or not resolved.is_file()
            or resolved.suffix.lower() not in _AUDIO_SUFFIXES
        ):
            continue
        seen.add(resolved)
        slug = re.sub(r"[^a-z0-9]+", "-", role.lower()).strip("-") or "audio"
        identity = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
        discovered.append(
            DiscoveredAudio(
                id=f"{slug}-{identity}",
                role=role,
                path=resolved,
                discovered_from=source,
                metadata=metadata,
            )
        )
    return discovered


def _source_manifest(
    output: list[tuple[str, Path, str, dict[str, Any]]],
    run: RunPaths,
    manifest: Path,
) -> None:
    data = _read_object(manifest)
    if data is None:
        return
    source = f"manifest:{manifest.name}"
    stems = data.get("stems")
    if isinstance(stems, Mapping):
        for raw_role, raw_path in stems.items():
            role = _source_role(str(raw_role))
            _append_path(output, role, raw_path, source, manifest, run, {"manifest_role": raw_role})
    _append_path(output, "dialogue", data.get("dialogue_stem"), source, manifest, run)
    _append_path(output, "source_mix", data.get("source"), source, manifest, run)


def _speaker_manifest(
    output: list[tuple[str, Path, str, dict[str, Any]]],
    run: RunPaths,
    manifest: Path,
) -> None:
    data = _read_object(manifest)
    if data is None:
        return
    source = f"manifest:{manifest.name}"
    recorded: set[str] = set()
    speakers = data.get("speakers")
    if isinstance(speakers, list):
        for index, speaker in enumerate(speakers, 1):
            if not isinstance(speaker, Mapping):
                continue
            raw_path = speaker.get("stem") or speaker.get("path")
            label = str(speaker.get("speaker") or speaker.get("speaker_id") or index)
            if raw_path:
                recorded.add(str(raw_path))
            _append_path(
                output,
                f"speaker:{label}",
                raw_path,
                source,
                manifest,
                run,
                {"speaker": label},
            )
    stems = data.get("stems")
    if isinstance(stems, list):
        for index, raw_path in enumerate(stems, 1):
            if str(raw_path) not in recorded:
                _append_path(
                    output,
                    f"speaker:{index:02d}",
                    raw_path,
                    source,
                    manifest,
                    run,
                )
    elif isinstance(stems, Mapping):
        for role, raw_path in stems.items():
            _append_path(output, f"speaker:{role}", raw_path, source, manifest, run)
    _append_path(output, "dialogue_residual", data.get("residual"), source, manifest, run)


def _fx_manifest(
    output: list[tuple[str, Path, str, dict[str, Any]]],
    run: RunPaths,
    manifest: Path,
) -> None:
    data = _read_object(manifest)
    if data is None:
        return
    source = f"manifest:{manifest.name}"
    categories = data.get("categories")
    if isinstance(categories, Mapping):
        for category, raw_path in categories.items():
            _append_path(
                output,
                f"effects:{category}",
                raw_path,
                source,
                manifest,
                run,
                {"category": str(category)},
            )


def _music_manifest(
    output: list[tuple[str, Path, str, dict[str, Any]]],
    run: RunPaths,
    manifest: Path,
) -> None:
    data = _read_object(manifest)
    if data is None:
        return
    source = f"manifest:{manifest.name}"
    _append_path(output, "music", data.get("cleaned_source"), source, manifest, run)
    cues = data.get("cues")
    if not isinstance(cues, list):
        return
    for index, cue in enumerate(cues, 1):
        if not isinstance(cue, Mapping):
            continue
        cue_id = str(cue.get("cue_id") or index)
        _append_path(
            output,
            f"music_cue:{cue_id}",
            cue.get("path"),
            source,
            manifest,
            run,
            {"cue_id": cue_id},
        )
        instruments = cue.get("instruments")
        if isinstance(instruments, Mapping):
            for instrument, raw_path in instruments.items():
                _append_path(
                    output,
                    f"music_instrument:{instrument}",
                    raw_path,
                    source,
                    manifest,
                    run,
                    {"cue_id": cue_id, "instrument": str(instrument)},
                )
        elif isinstance(instruments, list):
            instruments_dir = cue.get("instruments_dir")
            for instrument in instruments:
                raw_path = (
                    Path(str(instruments_dir)) / str(instrument) if instruments_dir else instrument
                )
                _append_path(
                    output,
                    f"music_instrument:{Path(str(instrument)).stem}",
                    raw_path,
                    source,
                    manifest,
                    run,
                    {
                        "cue_id": cue_id,
                        "instrument": Path(str(instrument)).stem,
                    },
                )


def _room_tone_manifest(
    output: list[tuple[str, Path, str, dict[str, Any]]],
    run: RunPaths,
    manifest: Path,
) -> None:
    data = _read_object(manifest)
    if data is None:
        return
    source = f"manifest:{manifest.name}"
    _append_path(output, "room_tone", data.get("rt_stem"), source, manifest, run)
    groups = data.get("groups")
    if isinstance(groups, list):
        for group in groups:
            if isinstance(group, Mapping):
                group_id = str(group.get("group_id") or "")
                _append_path(
                    output,
                    f"room_tone_loop:{group_id}",
                    group.get("loop_path"),
                    source,
                    manifest,
                    run,
                    {"group_id": group_id},
                )


def _explicit_inputs(
    output: list[tuple[str, Path, str, dict[str, Any]]],
    run: RunPaths,
    inputs: AudioInputs,
) -> None:
    if inputs is None:
        return
    if isinstance(inputs, Mapping):
        for role, raw_values in inputs.items():
            values: Sequence[str | Path]
            if isinstance(raw_values, (str, Path)):
                values = [raw_values]
            else:
                values = raw_values
            for value in values:
                _append_explicit(output, run, str(role), value)
        return
    values = [inputs] if isinstance(inputs, (str, Path)) else inputs
    for value in values:
        text = str(value)
        if "=" in text:
            role, raw_path = text.split("=", 1)
            _append_explicit(output, run, role.strip() or "explicit", raw_path)
        else:
            _append_explicit(output, run, "explicit", value)


def _append_explicit(
    output: list[tuple[str, Path, str, dict[str, Any]]],
    run: RunPaths,
    role: str,
    value: str | Path,
) -> None:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = run.root / path
    output.append((role, path, "explicit", {}))


def _directory_fallback(
    output: list[tuple[str, Path, str, dict[str, Any]]],
    directory: Path,
    role_prefix: str,
) -> None:
    if not directory.is_dir():
        return
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if path.suffix.lower() in _AUDIO_SUFFIXES:
            role = f"{role_prefix}:{path.stem}"
            output.append((role, path, f"directory:{directory.name}", {}))


def _append_path(
    output: list[tuple[str, Path, str, dict[str, Any]]],
    role: str,
    raw_path: Any,
    source: str,
    manifest: Path,
    run: RunPaths,
    metadata: dict[str, Any] | None = None,
) -> None:
    if not isinstance(raw_path, (str, Path)) or not str(raw_path).strip():
        return
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        candidates = (manifest.parent / path, run.root / path, path)
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    output.append((role, path, source, metadata or {}))


def _read_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _source_role(role: str) -> str:
    normalized = role.lower().replace("-", "_")
    if normalized in {"dialogue", "dialog", "dx", "speech", "vocals"}:
        return "dialogue"
    if normalized in {"music", "mx", "instrumental"}:
        return "music"
    if normalized in {"effects", "effect", "fx", "sfx"}:
        return "effects"
    if normalized in {"music_fx", "mx_fx", "instrumental_fx", "background"}:
        return "music_effects"
    return f"source:{role}"
