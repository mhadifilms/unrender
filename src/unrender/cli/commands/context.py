from __future__ import annotations

import argparse
import glob as glob_module
from pathlib import Path
from typing import Any, Literal, overload

from unrender.editorial.shots.stems import (
    StemSource,
    load_stem_map,
    load_stem_sources,
)
from unrender.lib.names import safe_name
from unrender.manifests import read_json
from unrender.project import RunPaths
from unrender.project.config import (
    ProjectConfig,
    load_speaker_config,
    resolve_project_configs,
)
from unrender.speakers import configured_speakers


def _project_path_issue(
    project: ProjectConfig,
    key: str,
    *,
    allow_glob: bool = False,
) -> str | None:
    value = project.string_value(key)
    if not value:
        return None
    if value.startswith(("http://", "https://")):
        return None
    if allow_glob and any(char in value for char in "*?[]"):
        if glob_module.glob(value):
            return None
        return f"paths.{key} matched no files: {value}"
    path = project.path_value(key)
    if path is not None and not path.exists():
        return f"paths.{key} does not exist: {path}"
    return None


def _project_audio_asset_issue(
    project: ProjectConfig,
    key: str,
    *,
    allow_glob: bool = False,
) -> str | None:
    value = project.audio_asset_string(key)
    if not value:
        return None
    if value.startswith(("http://", "https://")):
        return None
    label = f"audio_assets.original.{key}"
    if allow_glob and any(char in value for char in "*?[]"):
        if glob_module.glob(value):
            return None
        return f"{label} matched no files: {value}"
    path = project.audio_asset_path(key)
    if path is not None and not path.exists():
        return f"{label} does not exist: {path}"
    return None


def _projects_from_args(args: argparse.Namespace) -> list[ProjectConfig]:
    specs = getattr(args, "projects", None) or []
    return resolve_project_configs([str(spec) for spec in specs]) if specs else []


def _project(args: argparse.Namespace) -> ProjectConfig | None:
    return getattr(args, "project_config", None)


def _run_paths(args: argparse.Namespace, *, explicit: Path | None) -> RunPaths:
    if explicit is not None:
        return RunPaths.from_path(explicit)
    project = _project(args)
    if project is not None:
        value = project.path_value("run_dir")
        if value is not None:
            return RunPaths.from_path(value)
    raise ValueError("missing --run-dir (or paths.run_dir in project config)")


@overload
def _path_arg(
    args: argparse.Namespace,
    key: str,
    explicit: Path | None,
    *,
    required: Literal[True],
) -> Path: ...


@overload
def _path_arg(
    args: argparse.Namespace,
    key: str,
    explicit: Path | None,
    *,
    required: Literal[False],
) -> Path | None: ...


def _path_arg(
    args: argparse.Namespace,
    key: str,
    explicit: Path | None,
    *,
    required: bool,
) -> Path | None:
    if explicit is not None:
        return explicit.expanduser()
    project = _project(args)
    if project is not None:
        value = project.path_value(key)
        if value is not None:
            return value
    if required:
        raise ValueError(f"missing --{key.replace('_', '-')} (or paths.{key} in project config)")
    return None


def _shots_manifest_arg(
    args: argparse.Namespace,
    run: RunPaths,
    *,
    explicit: Path | None,
    required: bool,
) -> Path | None:
    """Resolve the shot manifest, falling back to the one `shots detect` wrote."""
    configured = _path_arg(args, "shots", explicit, required=False)
    if configured is not None:
        return configured
    if run.shots_manifest_json.exists():
        return run.shots_manifest_json
    if required:
        raise ValueError(
            "missing --shots (or paths.shots in project config). Run `unrender shots detect` "
            "or `unrender shots cut` first to create the shot manifest."
        )
    return None


def _scenes_manifest_arg(
    args: argparse.Namespace,
    run: RunPaths,
    *,
    explicit: Path | None,
) -> Path:
    """Resolve the scene manifest path (explicit -> paths.scenes -> run default).

    Scenes are optional, so this always returns a path (defaulting to the run
    directory's ``scenes.json``); callers check existence and skip scene-scoped
    behavior when it is absent.
    """
    configured = _path_arg(args, "scenes", explicit, required=False)
    return configured if configured is not None else run.scenes_manifest_json


def _proxy_master_arg(
    args: argparse.Namespace,
    run: RunPaths,
    *,
    explicit: Path | None,
    required: bool,
) -> Path | None:
    """Resolve the program proxy, falling back to the one `shots proxy` created."""
    configured = _string_arg(
        args,
        "proxy_path",
        str(explicit) if explicit else None,
        required=False,
        aliases=("proxy_master", "proxy"),
    )
    if configured is not None:
        return Path(configured).expanduser()
    if run.proxy_manifest_json.exists():
        data = read_json(run.proxy_manifest_json)
        candidate = Path(str(data.get("output") or "")).expanduser()
        if str(candidate) != "." and candidate.exists():
            return candidate
    if required:
        raise ValueError(
            "missing --video (or paths.proxy_path in project config). Run "
            "`unrender shots proxy` or `unrender shots detect` first to create one."
        )
    return None


def _dx_stem_arg(
    args: argparse.Namespace,
    run: RunPaths,
    *,
    explicit: str | None,
    required: bool,
) -> str:
    resolved = _resolve_dx_stem(args, run, explicit=explicit, required=required)
    _refuse_wrong_language(args, resolved)
    return resolved


def _resolve_dx_stem(
    args: argparse.Namespace,
    run: RunPaths,
    *,
    explicit: str | None,
    required: bool,
) -> str:
    if explicit:
        return explicit
    project = _project(args)
    if project is not None:
        configured_asset = project.audio_asset_string("dme.dx")
        if configured_asset:
            return configured_asset
    derived = _derived_dialogue_stem(args, run)
    if derived is not None:
        return str(derived)
    if required:
        nested_key = "audio_assets.original.dme.dx"
        raise ValueError(
            f"missing --dx-stem (or {nested_key} in project config). "
            "Run `unrender audio separate -p <project>` first if the dialogue stem "
            "must be generated from the configured full audio."
        )
    return ""


def _refuse_wrong_language(
    args: argparse.Namespace,
    dx_stem: str,
) -> None:
    """Check a configured dialogue language before consuming its audio."""
    expected = _audio_language_arg(args)
    if not expected or not dx_stem or getattr(args, "skip_language_check", False):
        return
    path = Path(dx_stem).expanduser()
    if not path.is_file():
        return

    from unrender.audio.language import probe_audio_language

    try:
        probe = probe_audio_language(path, expected=expected)
    except (ImportError, RuntimeError) as exc:
        # Not checkable is not the same as wrong, but it must not be silent:
        # the whole point is that this mistake leaves no other trace.
        print(f"WARNING: could not verify the language of {path.name}: {exc}", flush=True)
        return
    if probe.issue is not None:
        raise ValueError(f"{probe.issue} Pass --skip-language-check to run it anyway.")


def _derived_dialogue_stem(args: argparse.Namespace, run: RunPaths) -> Path | None:
    if run.source_separation_json.exists():
        data = read_json(run.source_separation_json)
        raw = str(data.get("dialogue_stem") or "").strip()
        if raw:
            path = Path(raw).expanduser()
            if path.exists():
                return path
    prefix = _audio_output_prefix(args, run)
    candidates = sorted(run.source_stems_dir.glob(f"{prefix}_DX_stem.*"))
    return candidates[0] if candidates else None


def _shot_speakers_arg(args: argparse.Namespace, run: RunPaths) -> dict[str, list[str]]:
    explicit = getattr(args, "shot_matches", None)
    path = explicit.expanduser() if explicit else run.shot_matches_json
    out = _shot_speaker_matches(path)
    return out


def _shot_speaker_matches(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    data = read_json(path)
    out: dict[str, list[str]] = {}
    for row in (data.get("shots") if isinstance(data, dict) else None) or []:
        if not isinstance(row, dict):
            continue
        shot_id = str(row.get("shot_id") or "").strip()
        if not shot_id:
            continue
        speakers = _speaker_values(row.get("accepted") or row.get("proposed_value") or "")
        if speakers:
            out[shot_id] = speakers
    return out


def _speaker_values(value: Any) -> list[str]:
    if isinstance(value, list):
        raw = value
    else:
        raw = str(value or "").replace(";", ",").split(",")
    speakers: list[str] = []
    for item in raw:
        speaker = str(item or "").strip().upper()
        if speaker and speaker not in speakers:
            speakers.append(speaker)
    return speakers


def _string_arg(
    args: argparse.Namespace,
    key: str,
    explicit: str | None,
    *,
    required: bool,
    aliases: tuple[str, ...] = (),
) -> str | None:
    if explicit:
        return explicit
    project = _project(args)
    if project is not None:
        for candidate in (key, *aliases):
            value = project.string_value(candidate)
            if value:
                return value
    if required:
        raise ValueError(f"missing --{key.replace('_', '-')} (or paths.{key} in project config)")
    return None


def _speaker_db_path(args: argparse.Namespace, explicit: Path | None, run: RunPaths) -> Path:
    return _path_arg(args, "speaker_db", explicit, required=False) or run.speaker_db


def _speaker_config_arg(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "config", None) is not None:
        return load_speaker_config(args.config.expanduser())
    project = _project(args)
    if project is not None:
        return project.data
    raise ValueError("missing --config (or -p/--project with speakers)")


def _config_speaker_names(args: argparse.Namespace) -> list[str]:
    project = _project(args)
    if project is None:
        return []
    return [meta["name"] for meta in configured_speakers(project.data).values()]


def _stem_sources_arg(
    args: argparse.Namespace,
    run: RunPaths,
) -> list[StemSource]:
    if args.stems:
        return load_stem_sources(args.stems)
    project = _project(args)
    if project is not None:
        stem_map = project.data.get("stem_map")
        if stem_map:
            if not isinstance(stem_map, dict):
                raise ValueError(f"project config stem_map must be an object: {project.path}")
            return load_stem_map(stem_map, speaker_names=_config_speaker_names(args))
        separated = project.audio_asset_string("separated_stems")
        if separated:
            return load_stem_sources(separated)
    return load_stem_sources(run.unmapped_speaker_glob)


def _voice_clips_spec_arg(args: argparse.Namespace, run: RunPaths) -> str:
    return (
        args.clips
        or _string_arg(args, "voice_clips", None, required=False, aliases=("clips",))
        or _default_voice_clip_spec(run)
    )


def _default_voice_clip_spec(run: RunPaths) -> str:
    return str(run.voice_clips_json) if run.voice_clips_json.exists() else run.voice_clips_glob


def _section_arg(
    args: argparse.Namespace,
    section: str,
    key: str,
    explicit: Any,
    default: Any,
) -> Any:
    if explicit is not None:
        return explicit
    project = _project(args)
    if project is not None:
        section_data = project.data.get(section) or {}
        if not isinstance(section_data, dict):
            raise ValueError(f"project config {section} must be an object: {project.path}")
        value = section_data.get(key)
        if value is not None:
            return value
    return default


def _audio_arg(args: argparse.Namespace, key: str, explicit: Any, default: Any) -> Any:
    return _section_arg(args, "audio", key, explicit, default)


def _full_audio_arg(
    args: argparse.Namespace,
    *,
    explicit: str | None,
) -> str | None:
    if explicit:
        return explicit
    project = _project(args)
    if project is not None:
        configured_asset = project.audio_asset_string("full_mix")
        if configured_asset:
            return configured_asset
    return None


def _audio_language_arg(
    args: argparse.Namespace,
) -> str | None:
    """The language a project declares its audio is spoken in, if any."""
    project = _project(args)
    return project.audio_asset_string("language") if project is not None else None


def _audio_output_prefix(
    args: argparse.Namespace,
    run: RunPaths,
) -> str:
    configured = _audio_arg(args, "audioshake_prefix", getattr(args, "prefix", None), None)
    if configured:
        base = safe_name(configured, fallback="audio")
    else:
        project = _project(args)
        if project is not None:
            base = safe_name(project.name, fallback="audio")
        else:
            base = safe_name(run.root.name, fallback="audio")
    return base


def _face_arg(args: argparse.Namespace, key: str, explicit: Any, default: Any) -> Any:
    return _section_arg(args, "face", key, explicit, default)


def _voice_arg(args: argparse.Namespace, key: str, explicit: Any, default: Any) -> Any:
    return _section_arg(args, "voice", key, explicit, default)


def _shots_arg(args: argparse.Namespace, key: str, explicit: Any, default: Any) -> Any:
    return _section_arg(args, "shots", key, explicit, default)


def _scenes_arg(args: argparse.Namespace, key: str, explicit: Any, default: Any) -> Any:
    return _section_arg(args, "scenes", key, explicit, default)


def _timeline_arg(args: argparse.Namespace, key: str, explicit: Any, default: Any) -> Any:
    return _section_arg(args, "timeline", key, explicit, default)


def _mne_arg(args: argparse.Namespace, key: str, explicit: Any, default: Any) -> Any:
    return _section_arg(args, "mne", key, explicit, default)


def _timeline_fps(args: argparse.Namespace, default: float) -> float:
    explicit = getattr(args, "fps", None)
    if explicit is not None:
        return float(explicit)
    project = _project(args)
    if project is not None:
        audio = project.data.get("audio")
        if isinstance(audio, dict):
            value = audio.get("fps") or audio.get("transcript_fps")
            if value is not None and value != "":
                return float(value)
        value = project.data.get("fps")
        if value is not None and value != "":
            return float(value)
    return default


def _audioshake_client(api_key: str):
    from unrender.audio.separation.audioshake import AudioShakeClient

    return AudioShakeClient(api_key=api_key)
