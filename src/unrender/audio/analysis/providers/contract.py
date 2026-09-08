"""Shared contracts for optional audio-analysis providers.

This module deliberately depends only on the Python standard library.  Provider
modules may therefore be imported, inspected, and preflighted without importing
their model runtimes or triggering model downloads.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Protocol


class CapabilityState(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    BLOCKED_BY_POLICY = "blocked_by_policy"
    FAILED = "failed"
    COMPLETE = "complete"


class Profile(str, Enum):
    CORE = "core"
    AUTO = "auto"
    FULL = "full"


@dataclass(frozen=True)
class PackageProvenance:
    name: str
    version: str | None

    @property
    def exact(self) -> bool:
        return bool(self.version)


@dataclass(frozen=True)
class ModelProvenance:
    model_id: str
    revision: str | None
    source: str = "huggingface"

    @property
    def exact(self) -> bool:
        return bool(self.model_id and self.revision and self.revision != "main")


@dataclass(frozen=True)
class LicenseMetadata:
    license_id: str
    source: str
    permissive: bool | None
    gated: bool = False
    terms_url: str | None = None


@dataclass(frozen=True)
class ProviderProvenance:
    packages: tuple[PackageProvenance, ...] = ()
    models: tuple[ModelProvenance, ...] = ()

    @property
    def exact(self) -> bool:
        return (
            bool(self.packages or self.models)
            and all(package.exact for package in self.packages)
            and all(model.exact for model in self.models)
        )


@dataclass(frozen=True)
class ProviderPolicy:
    """Policy applied before an optional provider can run.

    Unknown licenses are rejected by ``permissive_only``.  Gated providers must
    both be enabled globally and named in ``explicit_opt_ins``.
    """

    permissive_only: bool = False
    allow_gated: bool = False
    explicit_opt_ins: frozenset[str] = frozenset()

    def block_reason(
        self,
        provider_id: str,
        license_metadata: LicenseMetadata,
        *,
        requires_opt_in: bool = False,
    ) -> str | None:
        if self.permissive_only and license_metadata.permissive is not True:
            return (
                f"{provider_id} is not verified as permissively licensed "
                f"({license_metadata.license_id})"
            )
        if license_metadata.gated and not self.allow_gated:
            return f"{provider_id} is gated and gated providers are disabled"
        if (license_metadata.gated or requires_opt_in) and provider_id not in self.explicit_opt_ins:
            return f"{provider_id} requires explicit opt-in"
        return None


@dataclass(frozen=True)
class CapabilityStatus:
    provider_id: str
    capability: str
    state: CapabilityState
    reason: str = ""
    provenance: ProviderProvenance = field(default_factory=ProviderProvenance)
    license: LicenseMetadata | None = None
    backend: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["provenance"]["exact"] = self.provenance.exact
        return payload


@dataclass
class ProviderResult:
    provider_id: str
    capability: str
    state: CapabilityState
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: ProviderProvenance = field(default_factory=ProviderProvenance)
    error: str | None = None

    @property
    def complete(self) -> bool:
        return self.state is CapabilityState.COMPLETE

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["provenance"]["exact"] = self.provenance.exact
        return payload


class AudioAnalysisProvider(Protocol):
    provider_id: str
    capabilities: tuple[str, ...]

    def preflight(self, policy: ProviderPolicy | None = None) -> list[CapabilityStatus]: ...


def package_provenance(distribution: str) -> PackageProvenance:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = None
    return PackageProvenance(name=distribution, version=version)


def module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def statuses_for_capabilities(
    *,
    provider_id: str,
    capabilities: tuple[str, ...],
    state: CapabilityState,
    reason: str,
    provenance: ProviderProvenance,
    license_metadata: LicenseMetadata,
    backend: str | None = None,
) -> list[CapabilityStatus]:
    return [
        CapabilityStatus(
            provider_id=provider_id,
            capability=capability,
            state=state,
            reason=reason,
            provenance=provenance,
            license=license_metadata,
            backend=backend,
        )
        for capability in capabilities
    ]
