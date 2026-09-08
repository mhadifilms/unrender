"""Tolerant access to a generic canonical audio-analysis bundle."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_MISSING = object()


def _get_path(data: Mapping[str, Any], dotted: str) -> Any:
    current: Any = data
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


@dataclass(frozen=True)
class AnalysisBundle:
    """Bundle data plus the directory used to resolve relative array paths."""

    data: Mapping[str, Any]
    base_dir: Path
    source_path: Path | None = None

    def get(self, *paths: str, default: Any = None) -> Any:
        for path in paths:
            value = _get_path(self.data, path)
            if value is not _MISSING:
                return value
        return default

    def array(self, *paths: str, default: Any = None) -> np.ndarray | None:
        value = self.get(*paths, default=default)
        if value is None:
            return None
        if isinstance(value, Mapping):
            value = next(
                (
                    value[key]
                    for key in ("values", "data", "array", "path", "npy", "uri")
                    if key in value
                ),
                None,
            )
        if value is None:
            return None
        if isinstance(value, (str, Path)):
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = self.base_dir / path
            if path.suffix.lower() == ".npy":
                return np.asarray(np.load(path, allow_pickle=False))
            if path.suffix.lower() == ".npz":
                with np.load(path, allow_pickle=False) as archive:
                    if not archive.files:
                        return None
                    return np.asarray(archive[archive.files[0]])
            raise ValueError(f"analysis array reference must be .npy or .npz: {path}")
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return None
        if array.dtype.kind not in "biufc":
            return None
        return array

    @property
    def duration_sec(self) -> float:
        raw = self.get(
            "duration_sec",
            "duration",
            "metadata.duration_sec",
            "program.duration_sec",
            default=None,
        )
        try:
            duration = float(raw)
        except (TypeError, ValueError):
            duration = 0.0
        if duration > 0:
            return duration
        times = self.array("time_sec", "times_sec", "timeline.time_sec", "features.time_sec")
        if times is not None and times.size:
            return max(0.001, float(np.nanmax(times)))
        frame_count = self.frame_count
        raw_hop = self.get("hop_sec", "frame_hop_sec", "metadata.hop_sec", default=1.0)
        try:
            hop = max(0.001, float(raw_hop))
        except (TypeError, ValueError):
            hop = 1.0
        return max(0.001, max(1, frame_count - 1) * hop)

    @property
    def frame_count(self) -> int:
        candidates = (
            "time_sec",
            "times_sec",
            "features.loudness",
            "loudness",
            "features.masking",
            "masking",
            "features.source_dominance",
            "mix.source_dominance",
            "structure.self_similarity",
            "self_similarity",
        )
        for name in candidates:
            try:
                value = self.array(name)
            except (OSError, ValueError):
                continue
            if value is not None and value.ndim and value.shape[0]:
                return int(value.shape[0])
        return 1

    def events(self) -> list[dict[str, Any]]:
        raw = self.get("events", "timeline.events", "annotations.events", default=[])
        events: list[dict[str, Any]] = []
        if isinstance(raw, Mapping):
            for role, values in raw.items():
                if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                    for item in values:
                        event = _event_dict(item, default_role=str(role))
                        if event is not None:
                            events.append(event)
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            for item in raw:
                event = _event_dict(item)
                if event is not None:
                    events.append(event)
        return events

    def source_names(self, count: int) -> list[str]:
        raw = self.get("source_names", "sources", "mix.sources", default=[])
        names: list[str] = []
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            for index, item in enumerate(raw):
                if isinstance(item, Mapping):
                    names.append(str(item.get("name") or item.get("role") or f"Source {index + 1}"))
                else:
                    names.append(str(item))
        return [names[i] if i < len(names) else f"Source {i + 1}" for i in range(count)]


def _event_dict(value: Any, *, default_role: str = "event") -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    event = dict(value)
    event["role"] = str(
        value.get("role") or value.get("type") or value.get("category") or default_role
    ).lower()
    start = value.get("start_sec", value.get("start", value.get("time_sec", 0.0)))
    end = value.get("end_sec", value.get("end", start))
    try:
        event["start_sec"] = float(start)
        event["end_sec"] = max(float(end), event["start_sec"])
    except (TypeError, ValueError):
        return None
    event["label"] = str(value.get("label") or value.get("name") or event["role"])
    return event


def load_analysis_bundle(value: AnalysisBundle | Mapping[str, Any] | str | Path) -> AnalysisBundle:
    """Load a bundle mapping or JSON file without assuming an application run layout."""

    if isinstance(value, AnalysisBundle):
        return value
    if isinstance(value, Mapping):
        raw_base = value.get("_base_dir", value.get("base_dir", Path.cwd()))
        try:
            base_dir = Path(raw_base).expanduser().resolve()
        except TypeError:
            base_dir = Path.cwd()
        return AnalysisBundle(data=_canonical_projection(value, base_dir), base_dir=base_dir)
    path = Path(value).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("audio analysis bundle must contain a JSON object")
    return AnalysisBundle(
        data=_canonical_projection(data, path.parent),
        base_dir=path.parent,
        source_path=path,
    )


def _canonical_projection(data: Mapping[str, Any], base_dir: Path) -> Mapping[str, Any]:
    """Project canonical descriptor lists into renderer-friendly named features."""
    raw_features = data.get("features")
    if not isinstance(raw_features, list):
        return data
    descriptors = [item for item in raw_features if isinstance(item, Mapping)]
    artifacts = [item for item in data.get("artifacts", []) if isinstance(item, Mapping)]
    preferred = ("mix", "dialogue", "dx", "music", "mx", "effects", "fx")
    reference_id = ""
    for role in preferred:
        match = next(
            (item for item in artifacts if str(item.get("role", "")).lower() == role),
            None,
        )
        if match is not None:
            reference_id = str(match.get("id") or "")
            break
    if not reference_id and artifacts:
        reference_id = str(artifacts[0].get("id") or "")

    def descriptor(name: str, *, global_first: bool = False) -> Mapping[str, Any] | None:
        matches = [item for item in descriptors if item.get("name") == name]
        if global_first:
            shared = next((item for item in matches if not item.get("artifact_id")), None)
            if shared is not None:
                return shared
        return next(
            (item for item in matches if str(item.get("artifact_id") or "") == reference_id),
            matches[0] if matches else None,
        )

    def resolved(item: Mapping[str, Any] | None) -> Any:
        if item is None:
            return None
        if item.get("path"):
            path = Path(str(item["path"]))
            if not path.is_absolute():
                path = base_dir / path
            try:
                return np.load(path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError):
                return None
        return item.get("value")

    def columns(names: tuple[str, ...]) -> np.ndarray | None:
        values = [resolved(descriptor(name)) for name in names]
        arrays = [np.asarray(item).reshape(-1) for item in values if item is not None]
        if not arrays:
            return None
        count = min(len(item) for item in arrays)
        return np.column_stack([item[:count] for item in arrays])

    source_dominance = resolved(descriptor("source_dominance", global_first=True))
    source_dominance_by_band = resolved(descriptor("source_dominance_by_band", global_first=True))
    masking = resolved(descriptor("dialogue_masking_risk", global_first=True))
    masking_by_band = resolved(descriptor("dialogue_masking_by_band", global_first=True))
    loudness = resolved(descriptor("momentary_loudness_lufs"))
    dynamics = resolved(descriptor("crest_factor"))
    timbre = columns(("spectral_centroid_hz", "spectral_flatness", "harmonicity"))
    stereo = columns(("stereo_balance_db", "mid_side_width"))
    band_power = resolved(descriptor("bark_band_power"))
    similarity_descriptor = next(
        (
            item
            for item in descriptors
            if str(item.get("name") or "").startswith("self_similarity_")
            and str(item.get("artifact_id") or "") == reference_id
        ),
        next(
            (
                item
                for item in descriptors
                if str(item.get("name") or "").startswith("self_similarity_")
            ),
            None,
        ),
    )
    events: list[dict[str, Any]] = []
    motif_spans: list[dict[str, Any]] = []
    for item in descriptors:
        if item.get("kind") != "events":
            continue
        value = item.get("value")
        if isinstance(value, list):
            for raw_event in value:
                if not isinstance(raw_event, dict):
                    continue
                event = dict(raw_event)
                event.setdefault("type", str(item.get("name") or "event"))
                event.setdefault("role", str(item.get("name") or "event"))
                events.append(event)
                if item.get("name") == "repeated_motifs":
                    duration_sec = float(event.get("duration_sec") or 0.0)
                    for key in ("first_start_sec", "second_start_sec"):
                        if event.get(key) is None:
                            continue
                        start_sec = float(event[key])
                        motif_spans.append(
                            {
                                "start_sec": start_sec,
                                "end_sec": start_sec + duration_sec,
                                "label": "motif",
                            }
                        )
    boundaries = [
        float(event.get("time_sec", event.get("start_sec", 0.0)))
        for event in events
        if str(event.get("type") or event.get("role") or "").lower()
        in {"boundary", "event_boundary", "event_boundaries"}
    ]
    motifs = motif_spans
    duration = max(
        (
            float(item.get("audio", {}).get("duration_sec", 0.0))
            for item in artifacts
            if isinstance(item.get("audio"), Mapping)
        ),
        default=0.0,
    )
    projected = dict(data)
    projected["feature_descriptors"] = raw_features
    projected["duration_sec"] = duration
    projected["sources"] = [
        {
            "name": str(item.get("role") or item.get("id") or "source"),
            "role": str(item.get("role") or ""),
            "path": str(item.get("path") or ""),
        }
        for item in artifacts
    ]
    projected["features"] = {
        "role_activity": source_dominance,
        "source_dominance": source_dominance,
        "source_dominance_by_band": source_dominance_by_band,
        "masking": masking,
        "masking_by_band": masking_by_band,
        "loudness": loudness,
        "dynamics": dynamics,
        "timbral_color": timbre,
        "stereo_space": stereo,
        "band_power": band_power,
    }
    projected["structure"] = {
        "self_similarity": resolved(similarity_descriptor),
        "boundaries": boundaries,
        "motifs": motifs,
    }
    projected["events"] = events
    return projected


def first_array(bundle: AnalysisBundle, aliases: Iterable[str]) -> np.ndarray | None:
    """Return the first available numeric array among ``aliases``."""

    for name in aliases:
        try:
            array = bundle.array(name)
        except (FileNotFoundError, ValueError):
            continue
        if array is not None:
            return array
    return None


def json_safe_bundle(bundle: AnalysisBundle, *, max_values: int = 4096) -> dict[str, Any]:
    """Resolve array references and convert a bundle into bounded JSON-safe data."""

    def convert(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            array = value
            if array.size > max_values and array.ndim:
                step = max(1, int(np.ceil(array.shape[0] / max_values)))
                array = array[::step]
            return array.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Mapping):
            if any(key in value for key in ("path", "npy", "uri")):
                try:
                    resolved = AnalysisBundle({"value": value}, bundle.base_dir).array("value")
                except (OSError, ValueError):
                    resolved = None
                if resolved is not None:
                    return convert(resolved)
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    converted = convert(bundle.data)
    return dict(converted) if isinstance(converted, Mapping) else {}
