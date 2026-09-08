from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class RoleInput:
    """A concrete, immutable role-to-source assignment."""

    role: str
    path: Path


@dataclass(frozen=True)
class RoleInputSpec:
    """A declared input role in a recipe contract."""

    role: str
    required: bool = True
    description: str = ""
    aliases: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParameterSpec:
    """A scalar recipe parameter with an explicit safe operating range."""

    default: float | int | bool | str
    minimum: float | int | None = None
    maximum: float | int | None = None
    description: str = ""

    def validate(self, name: str, value: Any) -> float | int | bool | str:
        if isinstance(self.default, bool):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool")
            return value
        if isinstance(self.default, int) and not isinstance(self.default, bool):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            parsed: float | int | bool | str = int(value)
        elif isinstance(self.default, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            parsed = float(value)
        elif isinstance(self.default, str):
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            parsed = value
        else:
            raise TypeError(f"unsupported default type for {name}")
        if isinstance(parsed, (int, float)) and not isinstance(parsed, bool):
            if self.minimum is not None and parsed < self.minimum:
                raise ValueError(f"{name} must be >= {self.minimum}, got {parsed}")
            if self.maximum is not None and parsed > self.maximum:
                raise ValueError(f"{name} must be <= {self.maximum}, got {parsed}")
        elif self.minimum is not None or self.maximum is not None:
            raise TypeError(f"{name} has numeric bounds but is not numeric")
        return parsed

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecipeContract:
    """Static declaration of a transform's inputs, dependencies, and controls."""

    name: str
    category: str
    description: str
    role_inputs: tuple[RoleInputSpec, ...]
    parameters: Mapping[str, ParameterSpec] = field(default_factory=dict)
    dependencies: tuple[str, ...] = ("numpy", "scipy")
    analysis_inputs: tuple[str, ...] = ()
    preserves_duration: bool = True
    preserves_sample_rate: bool = True
    preserves_channel_layout: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    def resolve_parameters(self, supplied: Mapping[str, Any] | None = None) -> dict[str, Any]:
        supplied_values = dict(supplied or {})
        unknown = sorted(set(supplied_values) - set(self.parameters))
        if unknown:
            raise ValueError(f"unknown parameters for {self.name}: {', '.join(unknown)}")
        return {
            name: spec.validate(name, supplied_values.get(name, spec.default))
            for name, spec in self.parameters.items()
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "role_inputs": [role.as_dict() for role in self.role_inputs],
            "parameters": {name: spec.as_dict() for name, spec in self.parameters.items()},
            "dependencies": list(self.dependencies),
            "analysis_inputs": list(self.analysis_inputs),
            "preserves_duration": self.preserves_duration,
            "preserves_sample_rate": self.preserves_sample_rate,
            "preserves_channel_layout": self.preserves_channel_layout,
        }


@dataclass(frozen=True)
class RecipeSettings:
    """Cross-recipe controls that affect rendering and reproducibility."""

    dry_wet: float = 1.0
    seed: int = 0
    peak_ceiling_dbfs: float = -1.0
    sample_subtype: str = "FLOAT"

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.dry_wet) <= 1.0:
            raise ValueError("dry_wet must be between 0 and 1")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        if not -24.0 <= float(self.peak_ceiling_dbfs) <= 0.0:
            raise ValueError("peak_ceiling_dbfs must be between -24 and 0")
        normalized_subtype = self.sample_subtype.upper()
        if normalized_subtype not in {"FLOAT", "PCM_16", "PCM16"}:
            raise ValueError("sample_subtype must be FLOAT or PCM_16")
        object.__setattr__(self, "dry_wet", float(self.dry_wet))
        object.__setattr__(self, "peak_ceiling_dbfs", float(self.peak_ceiling_dbfs))
        object.__setattr__(
            self,
            "sample_subtype",
            "PCM_16" if normalized_subtype == "PCM16" else normalized_subtype,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


TransformSettings = RecipeSettings


@dataclass(frozen=True)
class RecipeResult:
    """Materialized outputs and audit data from one recipe invocation."""

    name: str
    contract: RecipeContract
    output_dir: Path
    outputs: Mapping[str, Path]
    automation: Mapping[str, Path]
    manifest: Path
    metrics: Mapping[str, Any]
    provenance: Mapping[str, Any]
    report: Mapping[str, Any]
    latency_samples: int = 0
    tail_samples: int = 0

    @property
    def output_wavs(self) -> Mapping[str, Path]:
        return self.outputs

    @property
    def automation_arrays(self) -> Mapping[str, Path]:
        return self.automation

    @property
    def manifest_path(self) -> Path:
        return self.manifest

    @property
    def before_metrics(self) -> Mapping[str, Any]:
        value = self.metrics.get("before", {})
        return value if isinstance(value, Mapping) else {}

    @property
    def after_metrics(self) -> Mapping[str, Any]:
        value = self.metrics.get("after", {})
        return value if isinstance(value, Mapping) else {}

    @property
    def mixes(self) -> Mapping[str, Path]:
        return {name: path for name, path in self.outputs.items() if "mix" in name}
