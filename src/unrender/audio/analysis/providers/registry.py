"""Registry and profile preflight for optional audio-analysis providers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from .contract import (
    AudioAnalysisProvider,
    CapabilityState,
    CapabilityStatus,
    Profile,
    ProviderPolicy,
)


class ProfilePreflightError(RuntimeError):
    def __init__(self, report: ProfilePreflightReport) -> None:
        self.report = report
        failures = [
            f"{status.provider_id}/{status.capability}: {status.state.value}"
            for status in report.statuses
            if status.state is not CapabilityState.AVAILABLE
        ]
        detail = ", ".join(failures) or "no providers were configured"
        super().__init__(f"{report.profile.value} profile preflight failed: {detail}")


@dataclass(frozen=True)
class ProfilePreflightReport:
    profile: Profile
    statuses: tuple[CapabilityStatus, ...]
    configured_providers: tuple[str, ...]
    strict: bool

    @property
    def ok(self) -> bool:
        if not self.strict:
            return True
        return bool(self.configured_providers) and all(
            status.state is CapabilityState.AVAILABLE for status in self.statuses
        )

    def by_provider(self) -> dict[str, list[CapabilityStatus]]:
        grouped: dict[str, list[CapabilityStatus]] = {}
        for status in self.statuses:
            grouped.setdefault(status.provider_id, []).append(status)
        return grouped


class ProviderRegistry:
    def __init__(self, providers: Iterable[AudioAnalysisProvider] = ()) -> None:
        self._providers: dict[str, AudioAnalysisProvider] = {}
        for provider in providers:
            self.register(provider)

    def register(self, provider: AudioAnalysisProvider) -> None:
        if provider.provider_id in self._providers:
            raise ValueError(f"audio-analysis provider already registered: {provider.provider_id}")
        self._providers[provider.provider_id] = provider

    def get(self, provider_id: str) -> AudioAnalysisProvider:
        try:
            return self._providers[provider_id]
        except KeyError as exc:
            raise KeyError(f"unknown audio-analysis provider: {provider_id}") from exc

    def provider_ids(self) -> tuple[str, ...]:
        return tuple(self._providers)

    def preflight(
        self,
        profile: Profile | str,
        *,
        configured_providers: Iterable[str] | None = None,
        policy: ProviderPolicy | None = None,
        raise_on_failure: bool | None = None,
    ) -> ProfilePreflightReport:
        active_profile = Profile(profile)
        configured = self._select_providers(active_profile, configured_providers)
        strict = active_profile is Profile.FULL
        statuses: list[CapabilityStatus] = []
        for provider_id in configured:
            provider = self._providers.get(provider_id)
            if provider is None:
                statuses.append(
                    CapabilityStatus(
                        provider_id=provider_id,
                        capability="provider",
                        state=CapabilityState.UNAVAILABLE,
                        reason="provider is not registered",
                    )
                )
                continue
            try:
                statuses.extend(provider.preflight(policy))
            except Exception as exc:
                statuses.extend(
                    CapabilityStatus(
                        provider_id=provider_id,
                        capability=capability,
                        state=CapabilityState.FAILED,
                        reason=f"preflight failed: {exc}",
                    )
                    for capability in provider.capabilities
                )
        report = ProfilePreflightReport(
            profile=active_profile,
            statuses=tuple(statuses),
            configured_providers=configured,
            strict=strict,
        )
        should_raise = strict if raise_on_failure is None else raise_on_failure
        if should_raise and not report.ok:
            raise ProfilePreflightError(report)
        return report

    def _select_providers(
        self,
        profile: Profile,
        configured_providers: Iterable[str] | None,
    ) -> tuple[str, ...]:
        if configured_providers is not None:
            return tuple(dict.fromkeys(configured_providers))
        if profile is Profile.CORE:
            return ()
        return self.provider_ids()


def default_registry(*, enabled_gated: Iterable[str] = ()) -> ProviderRegistry:
    """Build the default registry without importing any model runtime."""

    from .beat_this import BeatThisProvider
    from .brouhaha import BrouhahaProvider
    from .clap import ClapProvider
    from .mfa import MfaProvider
    from .whisperx import WhisperXProvider

    enabled = set(enabled_gated)
    providers = cast(
        tuple[AudioAnalysisProvider, ...],
        (
            WhisperXProvider(),
            MfaProvider(),
            BrouhahaProvider(enabled="brouhaha" in enabled),
            ClapProvider(),
            BeatThisProvider(),
        ),
    )
    return ProviderRegistry(providers)
