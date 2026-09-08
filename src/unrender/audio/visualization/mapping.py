"""Versioned, data-only rules for audio-analysis visualizations."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAPPING_VERSION = 1

_DEFAULT_COLORS = {
    "background": "#0b1020",
    "panel": "#141b2d",
    "grid": "#334155",
    "text": "#e5e7eb",
    "muted_text": "#94a3b8",
    "dialogue": "#38bdf8",
    "music": "#a78bfa",
    "fx": "#fb923c",
    "masking": "#ef4444",
    "loudness": "#34d399",
    "dynamics": "#facc15",
    "boundary": "#f8fafc",
    "motif": "#f472b6",
    "playhead": "#ffffff",
}

_DEFAULT_GEOMETRY: dict[str, int | float] = {
    "timeline_width": 1200,
    "timeline_height": 680,
    "structure_size": 640,
    "territory_width": 960,
    "territory_height": 480,
    "poster_width": 1280,
    "poster_height": 960,
    "overlay_width": 960,
    "overlay_height": 180,
    "overlay_fps": 12.0,
    "overlay_max_fps": 30.0,
    "overlay_max_duration": 30.0,
    "margin": 24,
}

_DEFAULT_LAYERS = {
    "role_activity": True,
    "timbral_color": True,
    "stereo_space": True,
    "masking": True,
    "loudness_dynamics": True,
    "events": True,
    "boundaries": True,
    "motifs": True,
}

_DEFAULT_RULES: dict[str, dict[str, Any]] = {
    "role_activity": {
        "feature": "features.role_activity",
        "layer": "role_activity",
        "geometry": "stacked_area",
        "palette": ["#38bdf8", "#a78bfa", "#fb923c"],
        "opacity": 0.9,
    },
    "timbral_color": {
        "feature": "features.timbral_color",
        "layer": "timbral_color",
        "geometry": "color_strip",
        "palette": ["#2563eb", "#a855f7", "#f59e0b"],
        "opacity": 0.95,
    },
    "stereo_space": {
        "feature": "features.stereo_space",
        "layer": "stereo_space",
        "geometry": "center_width",
        "palette": ["#22d3ee"],
        "opacity": 0.8,
    },
    "masking": {
        "feature": "features.masking",
        "layer": "masking",
        "geometry": "heat_strip",
        "palette": ["#172554", "#ef4444", "#fef08a"],
        "opacity": 0.9,
    },
    "loudness_dynamics": {
        "feature": "features.loudness",
        "layer": "loudness_dynamics",
        "geometry": "line_pair",
        "palette": ["#34d399", "#facc15"],
        "opacity": 1.0,
    },
    "events": {
        "feature": "events",
        "layer": "events",
        "geometry": "intervals",
        "palette": ["#38bdf8", "#a78bfa", "#fb923c"],
        "opacity": 0.95,
    },
}


def _valid_color(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if value.startswith("#") and len(value) in {4, 7, 9}:
        try:
            int(value[1:], 16)
        except ValueError:
            return False
        return True
    return value.lower() in {"black", "white", "transparent"}


@dataclass(frozen=True)
class VisualRule:
    """Maps one analysis feature to a named visual layer and geometry."""

    feature: str
    layer: str
    geometry: str
    palette: tuple[str, ...] = ()
    opacity: float = 1.0

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], fallback: VisualRule | None = None) -> VisualRule:
        fallback = fallback or cls(feature="", layer="", geometry="line")
        palette = value.get("palette", fallback.palette)
        if not isinstance(palette, (list, tuple)):
            palette = fallback.palette
        safe_palette = tuple(color for color in palette if _valid_color(color))
        if not safe_palette:
            safe_palette = fallback.palette
        try:
            opacity = min(1.0, max(0.0, float(value.get("opacity", fallback.opacity))))
        except (TypeError, ValueError):
            opacity = fallback.opacity
        return cls(
            feature=str(value.get("feature", fallback.feature)),
            layer=str(value.get("layer", fallback.layer)),
            geometry=str(value.get("geometry", fallback.geometry)),
            palette=safe_palette,
            opacity=opacity,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "layer": self.layer,
            "geometry": self.geometry,
            "palette": list(self.palette),
            "opacity": self.opacity,
        }


def _default_rules() -> dict[str, VisualRule]:
    return {name: VisualRule.from_dict(value) for name, value in _DEFAULT_RULES.items()}


@dataclass(frozen=True)
class VisualMapping:
    """Serializable visualization policy.

    Missing or malformed individual values fall back to conservative defaults.
    A newer mapping version is rejected because its rule semantics may differ.
    """

    version: int = MAPPING_VERSION
    colors: dict[str, str] = field(default_factory=lambda: copy.deepcopy(_DEFAULT_COLORS))
    geometry: dict[str, int | float] = field(
        default_factory=lambda: copy.deepcopy(_DEFAULT_GEOMETRY)
    )
    layers: dict[str, bool] = field(default_factory=lambda: copy.deepcopy(_DEFAULT_LAYERS))
    rules: dict[str, VisualRule] = field(default_factory=_default_rules)

    @classmethod
    def default(cls) -> VisualMapping:
        return cls()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> VisualMapping:
        if value is None:
            return cls.default()
        if not isinstance(value, Mapping):
            raise TypeError("visual mapping must be a JSON object")
        try:
            version = int(value.get("version", MAPPING_VERSION))
        except (TypeError, ValueError) as exc:
            raise ValueError("visual mapping version must be an integer") from exc
        if version > MAPPING_VERSION:
            raise ValueError(
                f"visual mapping version {version} is newer than supported version "
                f"{MAPPING_VERSION}"
            )
        if version < 1:
            raise ValueError(f"unsupported visual mapping version {version}")

        colors = copy.deepcopy(_DEFAULT_COLORS)
        raw_colors = value.get("colors")
        if isinstance(raw_colors, Mapping):
            colors.update(
                {str(name): str(color) for name, color in raw_colors.items() if _valid_color(color)}
            )

        geometry = copy.deepcopy(_DEFAULT_GEOMETRY)
        raw_geometry = value.get("geometry")
        if isinstance(raw_geometry, Mapping):
            for name, raw in raw_geometry.items():
                try:
                    number = float(raw)
                except (TypeError, ValueError):
                    continue
                if number <= 0 or not number < float("inf"):
                    continue
                geometry[str(name)] = int(number) if number.is_integer() else number

        layers = copy.deepcopy(_DEFAULT_LAYERS)
        raw_layers = value.get("layers")
        if isinstance(raw_layers, Mapping):
            layers.update(
                {
                    str(name): enabled
                    for name, enabled in raw_layers.items()
                    if isinstance(enabled, bool)
                }
            )

        rules = _default_rules()
        raw_rules = value.get("rules")
        if isinstance(raw_rules, Mapping):
            for name, raw_rule in raw_rules.items():
                if isinstance(raw_rule, Mapping):
                    rules[str(name)] = VisualRule.from_dict(raw_rule, rules.get(str(name)))

        return cls(
            version=MAPPING_VERSION,
            colors=colors,
            geometry=geometry,
            layers=layers,
            rules=rules,
        )

    @classmethod
    def from_json(cls, value: str | bytes) -> VisualMapping:
        return cls.from_dict(json.loads(value))

    @classmethod
    def load(cls, path: str | Path) -> VisualMapping:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "colors": dict(self.colors),
            "geometry": dict(self.geometry),
            "layers": dict(self.layers),
            "rules": {name: rule.to_dict() for name, rule in self.rules.items()},
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_json() + "\n", encoding="utf-8")
        return output


def load_visual_mapping(
    value: VisualMapping | Mapping[str, Any] | str | Path | None,
) -> VisualMapping:
    """Normalize an instance, mapping object, or JSON path to ``VisualMapping``."""

    if value is None:
        return VisualMapping.default()
    if isinstance(value, VisualMapping):
        return value
    if isinstance(value, Mapping):
        return VisualMapping.from_dict(value)
    return VisualMapping.load(value)
