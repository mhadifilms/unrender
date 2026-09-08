"""Deferred WhisperX word and character timing adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contract import (
    CapabilityState,
    LicenseMetadata,
    ModelProvenance,
    ProviderPolicy,
    ProviderProvenance,
    ProviderResult,
    module_available,
    package_provenance,
    statuses_for_capabilities,
)


class WhisperXProvider:
    provider_id = "whisperx"
    capabilities = ("word_alignment", "character_alignment")
    license_metadata = LicenseMetadata(
        license_id="model-dependent",
        source="https://github.com/m-bain/whisperX",
        permissive=None,
    )

    def __init__(
        self,
        *,
        whisper_model: str = "large-v3",
        model_revision: str | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.whisper_model = whisper_model
        self.model_revision = model_revision
        self.device = device
        self.compute_type = compute_type

    @property
    def provenance(self) -> ProviderProvenance:
        return ProviderProvenance(
            packages=(package_provenance("whisperx"),),
            models=(
                ModelProvenance(
                    model_id=self.whisper_model,
                    revision=self.model_revision,
                    source="whisperx",
                ),
            ),
        )

    def preflight(self, policy: ProviderPolicy | None = None):
        active_policy = policy or ProviderPolicy()
        blocked = active_policy.block_reason(self.provider_id, self.license_metadata)
        if blocked:
            state, reason = CapabilityState.BLOCKED_BY_POLICY, blocked
        elif not module_available("whisperx"):
            state = CapabilityState.UNAVAILABLE
            reason = "whisperx is not installed; install the transcription extra"
        else:
            state = CapabilityState.AVAILABLE
            reason = "WhisperX runtime is installed; models load only when alignment runs"
        return statuses_for_capabilities(
            provider_id=self.provider_id,
            capabilities=self.capabilities,
            state=state,
            reason=reason,
            provenance=self.provenance,
            license_metadata=self.license_metadata,
            backend="whisperx",
        )

    def align(
        self,
        audio_path: str | Path,
        *,
        language: str | None = None,
        include_characters: bool = True,
    ) -> ProviderResult:
        """Align via the repository's existing transcription helper.

        That helper exposes genuine WhisperX word alignment but not WhisperX's
        raw character alignment.  Character intervals are consequently
        interpolated inside aligned word bounds and are marked as degraded
        rather than being represented as model-produced timings.
        """

        if not module_available("whisperx"):
            return ProviderResult(
                provider_id=self.provider_id,
                capability="word_alignment",
                state=CapabilityState.UNAVAILABLE,
                provenance=self.provenance,
                error="whisperx is not installed",
            )

        try:
            from unrender.audio import asr

            words, helper_meta = asr.transcribe_word_aligned(
                Path(audio_path),
                model_name=self.whisper_model,
                device=self.device,
                compute_type=self.compute_type,
                language=language,
            )
        except Exception as exc:
            return ProviderResult(
                provider_id=self.provider_id,
                capability="word_alignment",
                state=CapabilityState.FAILED,
                provenance=self.provenance,
                error=f"WhisperX alignment failed: {exc}",
            )

        aligned = bool(helper_meta.get("aligned"))
        characters = _interpolate_characters(words) if include_characters else []
        degradations: list[str] = []
        if not aligned:
            degradations.append("word timings fell back to segment bounds")
        if include_characters:
            degradations.append("character timings interpolated from word bounds")
        timing_quality = {
            "word": "model_aligned" if aligned else "segment_fallback",
            "character": "interpolated_from_words" if include_characters else "not_requested",
            "degraded": bool(degradations),
            "degradations": degradations,
        }
        return ProviderResult(
            provider_id=self.provider_id,
            capability="word_alignment",
            state=CapabilityState.COMPLETE,
            data={"words": words, "characters": characters},
            metadata={
                "language": helper_meta.get("language") or language,
                "timing_quality": timing_quality,
            },
            provenance=self.provenance,
        )


def _interpolate_characters(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    characters: list[dict[str, Any]] = []
    for word_index, word in enumerate(words):
        text = str(word.get("word") or word.get("text") or "").strip()
        if not text or word.get("start") is None or word.get("end") is None:
            continue
        start = float(word["start"])
        end = float(word["end"])
        step = max(0.0, end - start) / len(text)
        for character_index, character in enumerate(text):
            characters.append(
                {
                    "character": character,
                    "start": start + character_index * step,
                    "end": start + (character_index + 1) * step,
                    "word_index": word_index,
                    "timing_source": "interpolated_from_word",
                }
            )
    return characters
