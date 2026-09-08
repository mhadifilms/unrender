from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from unrender.audio.audio_io import (
    ensure_2d,
    fit_length,
    match_channels,
    read_audio,
    resample_to,
    write_audio,
)

from .contracts import RecipeContract, RecipeSettings, RoleInput

ALGORITHM_VERSION = "unrender.audio.transforms/1"
EPSILON = np.finfo(np.float32).eps


@dataclass(frozen=True)
class LoadedRole:
    role: str
    path: Path
    audio: np.ndarray
    sample_rate: int
    sha256: str


@dataclass
class RecipeProducts:
    outputs: dict[str, tuple[np.ndarray, int]] = field(default_factory=dict)
    automation: dict[str, np.ndarray] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    latency_samples: int = 0
    tail_samples: int = 0


def db_to_gain(value: float | np.ndarray) -> float | np.ndarray:
    return np.power(10.0, np.asarray(value) / 20.0)


def gain_to_db(value: float | np.ndarray, *, floor: float = -120.0) -> float | np.ndarray:
    result = 20.0 * np.log10(np.maximum(np.asarray(value), 10.0 ** (floor / 20.0)))
    return np.maximum(result, floor)


def peak_protect(audio: np.ndarray, ceiling_dbfs: float) -> tuple[np.ndarray, float]:
    arr = ensure_2d(audio)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    ceiling = float(db_to_gain(ceiling_dbfs))
    gain = min(1.0, ceiling / max(peak, EPSILON))
    return np.asarray(arr * gain, dtype=np.float32), gain


def dry_wet_mix(dry: np.ndarray, wet: np.ndarray, amount: float) -> np.ndarray:
    dry_arr = ensure_2d(dry)
    wet_arr = fit_length(match_channels(wet, dry_arr.shape[1]), len(dry_arr))
    if amount <= 0.0:
        return dry_arr.astype(np.float32, copy=True)
    if amount >= 1.0:
        return wet_arr.astype(np.float32, copy=True)
    return np.asarray(dry_arr * (1.0 - amount) + wet_arr * amount, dtype=np.float32)


def audio_metrics(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    arr = ensure_2d(audio)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(arr), dtype=np.float64))) if arr.size else 0.0
    dc_offset = float(np.max(np.abs(np.mean(arr, axis=0)))) if arr.size else 0.0
    return {
        "sample_rate": int(sample_rate),
        "sample_count": len(arr),
        "channels": int(arr.shape[1]),
        "duration_seconds": float(len(arr) / sample_rate),
        "peak": peak,
        "peak_dbfs": float(gain_to_db(peak)),
        "rms": rms,
        "rms_dbfs": float(gain_to_db(rms)),
        "estimated_loudness_lufs": float(gain_to_db(rms) - 0.691),
        "dc_offset": dc_offset,
    }


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_analysis_bundle(bundle: Mapping[str, Any] | Path | str | None) -> dict[str, Any]:
    if bundle is None:
        return {}
    if isinstance(bundle, Mapping):
        return _project_analysis_payload(dict(bundle), Path.cwd())
    path = Path(bundle).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"analysis bundle does not exist: {path}")
    if path.suffix.lower() == ".npy":
        return {"data": np.load(path, allow_pickle=False)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"data": payload}
    return _project_analysis_payload(payload, path.parent)


def _project_analysis_payload(payload: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    descriptors = payload.get("features")
    if not isinstance(descriptors, list):
        return payload
    projected = dict(payload)
    projected["feature_descriptors"] = descriptors
    events: list[dict[str, Any]] = []
    named: dict[str, Any] = {}
    for descriptor in descriptors:
        if not isinstance(descriptor, Mapping):
            continue
        name = str(descriptor.get("name") or "")
        if not name:
            continue
        value: Any = descriptor.get("value")
        raw_path = descriptor.get("path")
        if raw_path:
            path = Path(str(raw_path))
            if not path.is_absolute():
                path = base_dir / path
            try:
                value = np.load(path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError):
                value = None
        if descriptor.get("kind") == "events" and isinstance(value, list):
            events.extend(item for item in value if isinstance(item, dict))
        if value is not None and (name not in named or not descriptor.get("artifact_id")):
            named[name] = value
    projected.update(named)
    projected["events"] = events
    if "spectral_flux" in named:
        projected.setdefault("novelty", named["spectral_flux"])
    if "dialogue_line_outliers" in named:
        projected.setdefault("review_regions", named["dialogue_line_outliers"])
    return projected


def stable_digest(value: Any) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def coerce_settings(
    settings: RecipeSettings | Mapping[str, Any] | None,
    params: Mapping[str, Any] | None,
) -> tuple[RecipeSettings, dict[str, Any]]:
    recipe_params = dict(params or {})
    if settings is None:
        return RecipeSettings(), recipe_params
    if isinstance(settings, RecipeSettings):
        return settings, recipe_params
    if not isinstance(settings, Mapping):
        raise TypeError("settings must be RecipeSettings or a mapping")
    setting_names = {"dry_wet", "seed", "peak_ceiling_dbfs", "sample_subtype"}
    setting_values = {key: value for key, value in settings.items() if key in setting_names}
    embedded_params = {key: value for key, value in settings.items() if key not in setting_names}
    embedded_params.update(recipe_params)
    return RecipeSettings(**setting_values), embedded_params


def normalize_role_paths(
    role_paths: Mapping[str, str | Path] | Sequence[RoleInput],
) -> dict[str, Path]:
    if isinstance(role_paths, Mapping):
        items: Iterable[tuple[str, str | Path]] = role_paths.items()
    else:
        items = ((item.role, item.path) for item in role_paths)
    normalized: dict[str, Path] = {}
    aliases = {
        "dialogue": "dx",
        "dialog": "dx",
        "music": "mx",
        "effects": "fx",
        "sfx": "fx",
        "room-tone": "room_tone",
        "roomtone": "room_tone",
    }
    for role, path in items:
        key = aliases.get(str(role).strip().lower(), str(role).strip().lower())
        if key in normalized:
            raise ValueError(f"duplicate role input: {key}")
        normalized[key] = Path(path).expanduser().resolve()
    if not normalized:
        raise ValueError("at least one role path is required")
    return normalized


def load_roles(
    role_paths: Mapping[str, Path],
    contract: RecipeContract,
) -> dict[str, LoadedRole]:
    declared = {alias for spec in contract.role_inputs for alias in (spec.role, *spec.aliases)}
    unknown = sorted(set(role_paths) - declared)
    if unknown:
        raise ValueError(f"undeclared roles for {contract.name}: {', '.join(unknown)}")
    for spec in contract.role_inputs:
        if spec.required and not any(key in role_paths for key in (spec.role, *spec.aliases)):
            raise ValueError(f"{contract.name} requires role '{spec.role}'")
    loaded: dict[str, LoadedRole] = {}
    for role, path in role_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} source does not exist: {path}")
        audio, sample_rate = read_audio(path)
        loaded[role] = LoadedRole(
            role=role,
            path=path,
            audio=np.nan_to_num(ensure_2d(audio), copy=False).astype(np.float32),
            sample_rate=sample_rate,
            sha256=hash_file(path),
        )
    return loaded


def find_role(roles: Mapping[str, LoadedRole], *names: str) -> LoadedRole | None:
    return next((roles[name] for name in names if name in roles), None)


def align_to(
    source: LoadedRole,
    sample_rate: int,
    sample_count: int,
    channels: int,
) -> np.ndarray:
    audio = resample_to(source.audio, source.sample_rate, sample_rate)
    return fit_length(match_channels(audio, channels), sample_count)


def combine_audio(
    tracks: Sequence[tuple[np.ndarray, int]],
    *,
    sample_rate: int | None = None,
    channels: int | None = None,
    sample_count: int | None = None,
) -> tuple[np.ndarray, int]:
    if not tracks:
        raise ValueError("cannot combine an empty track list")
    target_rate = sample_rate or tracks[0][1]
    target_channels = channels or max(ensure_2d(track).shape[1] for track, _ in tracks)
    if sample_count is None:
        sample_count = max(
            round(len(ensure_2d(track)) * target_rate / rate) for track, rate in tracks
        )
    mixed = np.zeros((sample_count, target_channels), dtype=np.float32)
    for track, rate in tracks:
        aligned = resample_to(track, rate, target_rate)
        aligned = fit_length(match_channels(aligned, target_channels), sample_count)
        mixed += aligned
    return mixed, target_rate


def extract_events(bundle: Mapping[str, Any]) -> list[tuple[float, float]]:
    raw: Any = bundle.get("events", bundle.get("review_regions", bundle.get("beats", [])))
    if isinstance(raw, Mapping):
        raw = raw.get("events", raw.get("regions", raw.get("items", [])))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    events: list[tuple[float, float]] = []
    for item in raw:
        if isinstance(item, (int, float)):
            start = float(item)
            end = start
        elif isinstance(item, Mapping):
            start_value = item.get(
                "start_seconds",
                item.get(
                    "start_sec",
                    item.get(
                        "start",
                        item.get("time_seconds", item.get("time_sec", item.get("time"))),
                    ),
                ),
            )
            if start_value is None:
                continue
            start = float(start_value)
            end_value = item.get("end_seconds", item.get("end_sec", item.get("end")))
            if end_value is None:
                end = start + float(
                    item.get(
                        "duration_seconds",
                        item.get("duration_sec", item.get("duration", 0.0)),
                    )
                )
            else:
                end = float(end_value)
        else:
            continue
        if math.isfinite(start) and math.isfinite(end) and end >= start:
            events.append((max(0.0, start), max(0.0, end)))
    return sorted(events)


def numeric_analysis_series(bundle: Mapping[str, Any], *keys: str) -> np.ndarray | None:
    value: Any = None
    for key in keys:
        if key in bundle:
            value = bundle[key]
            break
    if isinstance(value, Mapping):
        value = value.get("values", value.get("scores", value.get("data")))
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    array = array[np.isfinite(array)]
    return array if array.size else None


def materialize_products(
    *,
    products: RecipeProducts,
    contract: RecipeContract,
    settings: RecipeSettings,
    parameters: Mapping[str, Any],
    roles: Mapping[str, LoadedRole],
    output_dir: Path,
    analysis_bundle: Mapping[str, Any],
) -> tuple[
    dict[str, Path],
    dict[str, Path],
    Path,
    dict[str, Any],
    dict[str, Any],
]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = {role.path for role in roles.values()}
    output_paths: dict[str, Path] = {}
    after_metrics: dict[str, Any] = {}
    output_records: dict[str, Any] = {}
    for name, (audio, sample_rate) in products.outputs.items():
        safe_name = _safe_name(name)
        path = output_dir / f"{contract.name}__{safe_name}.wav"
        if path.resolve() in source_paths:
            raise ValueError(f"refusing to overwrite source audio: {path}")
        clean = np.nan_to_num(ensure_2d(audio), copy=False).astype(np.float32)
        _write_audio_atomic(path, clean, sample_rate, subtype=settings.sample_subtype)
        output_paths[name] = path
        after_metrics[name] = audio_metrics(clean, sample_rate)
        output_records[name] = {
            "path": path.name,
            "sha256": hash_file(path),
            **after_metrics[name],
        }
    automation_paths: dict[str, Path] = {}
    automation_records: dict[str, Any] = {}
    for name, array in products.automation.items():
        path = output_dir / f"{contract.name}__{_safe_name(name)}.npy"
        _save_npy_atomic(path, np.asarray(array))
        automation_paths[name] = path
        automation_records[name] = {
            "path": path.name,
            "sha256": hash_file(path),
            "shape": list(np.asarray(array).shape),
            "dtype": str(np.asarray(array).dtype),
        }
    for role in roles.values():
        if hash_file(role.path) != role.sha256:
            raise RuntimeError(f"source changed while rendering: {role.path}")
    before_metrics = {
        role_name: audio_metrics(role.audio, role.sample_rate) for role_name, role in roles.items()
    }
    metrics = {
        "before": before_metrics,
        "after": after_metrics,
        **_jsonable(products.metrics),
    }
    provenance_basis = {
        "algorithm": ALGORITHM_VERSION,
        "recipe": contract.name,
        "settings": settings.as_dict(),
        "parameters": dict(parameters),
        "inputs": {name: role.sha256 for name, role in sorted(roles.items())},
        "analysis_sha256": stable_digest(analysis_bundle),
    }
    provenance = {
        **provenance_basis,
        "deterministic": True,
        "fingerprint": stable_digest(provenance_basis),
    }
    manifest_data = {
        "schema_version": 1,
        "recipe": contract.as_dict(),
        "settings": settings.as_dict(),
        "parameters": dict(parameters),
        "inputs": {
            name: {
                "path": str(role.path),
                "sha256": role.sha256,
                **before_metrics[name],
            }
            for name, role in sorted(roles.items())
        },
        "outputs": output_records,
        "automation": automation_records,
        "metrics": metrics,
        "latency_samples": int(products.latency_samples),
        "tail_samples": int(products.tail_samples),
        "report": _jsonable(products.report),
        "provenance": provenance,
    }
    manifest = output_dir / f"{contract.name}__recipe_manifest.json"
    if manifest.resolve() in source_paths:
        raise ValueError(f"refusing to overwrite source: {manifest}")
    _write_json_atomic(manifest, manifest_data)
    return output_paths, automation_paths, manifest, metrics, provenance


def _safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return cleaned.strip("_") or "output"


def _write_audio_atomic(path: Path, audio: np.ndarray, sample_rate: int, *, subtype: str) -> None:
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}-",
        suffix=".wav",
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        write_audio(temporary_path, audio, sample_rate, subtype=subtype)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}-",
        suffix=".npy",
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        np.save(temporary_path, array, allow_pickle=False)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}-",
        suffix=".json",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(data), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value
