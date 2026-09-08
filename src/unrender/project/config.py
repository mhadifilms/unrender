from __future__ import annotations

import copy
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any

from unrender.editorial.dialogue.transcript import extract_speakers_from_transcript
from unrender.manifests import read_json
from unrender.speakers import configured_speakers

DEFAULT_CONFIG_DIR = Path("config")


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    path: Path
    data: dict[str, Any]

    def path_value(self, key: str) -> Path | None:
        value = self.paths.get(key)
        if value in (None, ""):
            return None
        return Path(str(value)).expanduser()

    def string_value(self, key: str) -> str | None:
        value = self.paths.get(key)
        if value in (None, ""):
            return None
        return str(value)

    def audio_value(self, key: str) -> Any:
        value: Any = self.data.get("audio") or {}
        for part in key.split("."):
            if not isinstance(value, dict):
                return None
            value = value.get(part)
        return value

    def audio_asset_value(self, key: str) -> Any:
        value: Any = self.audio_assets.get("original") or {}
        for part in key.split("."):
            if not isinstance(value, dict):
                return None
            value = value.get(part)
        return value

    def audio_asset_string(self, key: str) -> str | None:
        value = self.audio_asset_value(key)
        if value in (None, ""):
            return None
        if isinstance(value, (dict, list)):
            raise ValueError(
                f"project config audio_assets.original.{key} must be a string: {self.path}"
            )
        return str(value)

    def audio_asset_path(self, key: str) -> Path | None:
        value = self.audio_asset_string(key)
        return Path(value).expanduser() if value is not None else None

    @property
    def paths(self) -> dict[str, Any]:
        paths = self.data.get("paths") or {}
        if not isinstance(paths, dict):
            raise ValueError(f"project config paths must be an object: {self.path}")
        return paths

    @property
    def audio_assets(self) -> dict[str, Any]:
        assets = self.data.get("audio_assets") or {}
        if not isinstance(assets, dict):
            raise ValueError(f"project config audio_assets must be an object: {self.path}")
        original = assets.get("original")
        if original is not None and not isinstance(original, dict):
            raise ValueError(f"project config audio_assets.original must be an object: {self.path}")
        return assets


def load_speaker_config(path: Path) -> dict[str, Any]:
    data = _load_config_data(path)
    if not configured_speakers(data):
        raise ValueError("speaker config must define a non-empty 'speakers' set")
    return data


def _load_config_data(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"speaker config not found: {path}")
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError("speaker config must be a JSON object")
    return _seed_speakers_from_transcript(data, config_path=path)


def load_project_config(path: Path) -> ProjectConfig:
    data = _load_config_data(path)
    return ProjectConfig(name=_project_name(path), path=path.expanduser().resolve(), data=data)


def resolve_project_configs(
    project_specs: list[str],
    *,
    config_dir: Path = DEFAULT_CONFIG_DIR,
) -> list[ProjectConfig]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for spec in project_specs:
        matches = _project_matches(spec, config_dir=config_dir)
        if not matches:
            raise FileNotFoundError(f"project config not found: {spec}")
        for path in matches:
            resolved = path.expanduser().resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)
    return [load_project_config(path) for path in paths]


def _project_matches(spec: str, *, config_dir: Path) -> list[Path]:
    candidate = Path(spec).expanduser()
    if candidate.exists():
        return [candidate]

    has_glob = any(char in spec for char in "*?[]")
    search_specs: list[str] = []
    if has_glob:
        if candidate.parent != Path("."):
            search_specs.append(str(candidate))
        search_specs.append(str(config_dir / spec))
        if not spec.endswith(".json"):
            search_specs.append(str(config_dir / f"{spec}.json"))
    else:
        if candidate.suffix == ".json" or candidate.parent != Path("."):
            search_specs.append(str(candidate))
        search_specs.append(str(config_dir / (spec if spec.endswith(".json") else f"{spec}.json")))

    matches = sorted(Path(path) for pattern in search_specs for path in glob(pattern))
    return [
        path
        for path in matches
        if path.is_file() and path.name not in {"sample-config.json", "sample.json"}
    ]


def _project_name(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(DEFAULT_CONFIG_DIR.resolve())
        return rel.with_suffix("").as_posix()
    except ValueError:
        return path.stem


def _seed_speakers_from_transcript(data: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    if configured_speakers(data):
        return data
    paths = data.get("paths") or {}
    if not isinstance(paths, dict):
        return data
    script_value = paths.get("transcript")
    if script_value in (None, ""):
        return data
    script_path = Path(str(script_value)).expanduser()
    if not script_path.is_absolute():
        script_path = config_path.parent / script_path
    fps = _transcript_fps(data)
    speakers = extract_speakers_from_transcript(script_path, fps=fps)
    if speakers:
        seeded = copy.deepcopy(data)
        seeded["speakers"] = {speaker: {"aliases": []} for speaker in speakers}
        return seeded
    return data


def _transcript_fps(data: dict[str, Any]) -> float:
    audio = data.get("audio") or {}
    if isinstance(audio, dict):
        value = audio.get("transcript_fps") or audio.get("fps")
        if value is not None and value != "":
            return float(value)
    value = data.get("fps")
    if value is not None and value != "":
        return float(value)
    return 24.0
