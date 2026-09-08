"""Deferred CLAP semantic embeddings and cosine retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

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

CLAP_SAMPLE_RATE = 48_000


class ClapProvider:
    provider_id = "clap"
    capabilities = ("semantic_embedding", "cosine_search", "prompt_scoring")
    license_metadata = LicenseMetadata(
        license_id="Apache-2.0",
        source="https://huggingface.co/laion/clap-htsat-unfused",
        permissive=True,
    )

    def __init__(
        self,
        *,
        model_id: str = "laion/clap-htsat-unfused",
        model_revision: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.model_revision = model_revision
        self.device = device
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._resolved_revision = model_revision

    @property
    def provenance(self) -> ProviderProvenance:
        return ProviderProvenance(
            packages=(
                package_provenance("transformers"),
                package_provenance("torch"),
            ),
            models=(
                ModelProvenance(
                    model_id=self.model_id,
                    revision=self._resolved_revision,
                ),
            ),
        )

    def preflight(self, policy: ProviderPolicy | None = None):
        active_policy = policy or ProviderPolicy()
        blocked = active_policy.block_reason(self.provider_id, self.license_metadata)
        missing = [module for module in ("transformers", "torch") if not module_available(module)]
        if blocked:
            state, reason = CapabilityState.BLOCKED_BY_POLICY, blocked
        elif missing:
            state = CapabilityState.UNAVAILABLE
            reason = "CLAP dependencies are not installed: " + ", ".join(missing)
        else:
            state = CapabilityState.AVAILABLE
            reason = "CLAP runtime is installed; weights load only when embedding runs"
        return statuses_for_capabilities(
            provider_id=self.provider_id,
            capabilities=self.capabilities,
            state=state,
            reason=reason,
            provenance=self.provenance,
            license_metadata=self.license_metadata,
            backend="transformers_clap",
        )

    def embed(
        self,
        clips: Sequence[np.ndarray],
        *,
        sample_rate: int,
        times: Sequence[tuple[float, float]] | None = None,
    ) -> ProviderResult:
        if not clips:
            return ProviderResult(
                provider_id=self.provider_id,
                capability="semantic_embedding",
                state=CapabilityState.COMPLETE,
                data={"embeddings": []},
                metadata={"dtype": "float16", "normalized": True},
                provenance=self.provenance,
            )
        if times is not None and len(times) != len(clips):
            return self._failure("times must contain one (start, end) pair per clip")
        try:
            model, processor, torch = self._load_backend()
            prepared = [
                _resample_audio(np.asarray(clip, dtype=np.float32), sample_rate) for clip in clips
            ]
            inputs = processor(
                audios=prepared,
                sampling_rate=CLAP_SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            inputs = _move_inputs(inputs, self.device)
            with torch.no_grad():
                values = model.get_audio_features(**inputs)
            vectors = _normalize_float16(_to_numpy(values))
        except Exception as exc:
            return self._failure(f"CLAP embedding failed: {exc}")

        active_times = times or tuple(
            (float(index), float(index + 1)) for index in range(len(clips))
        )
        records = [
            {
                "start_sec": float(start),
                "end_sec": float(end),
                "embedding": vector,
            }
            for (start, end), vector in zip(active_times, vectors, strict=True)
        ]
        return ProviderResult(
            provider_id=self.provider_id,
            capability="semantic_embedding",
            state=CapabilityState.COMPLETE,
            data={"embeddings": records},
            metadata={
                "dtype": "float16",
                "normalized": True,
                "input_sample_rate": sample_rate,
                "model_sample_rate": CLAP_SAMPLE_RATE,
                "backend": "transformers_clap",
            },
            provenance=self.provenance,
        )

    def embed_text(self, prompts: Sequence[str]) -> np.ndarray:
        if not prompts:
            return np.empty((0, 0), dtype=np.float16)
        model, processor, torch = self._load_backend()
        inputs = processor(text=list(prompts), return_tensors="pt", padding=True)
        inputs = _move_inputs(inputs, self.device)
        with torch.no_grad():
            values = model.get_text_features(**inputs)
        return _normalize_float16(_to_numpy(values))

    def search(
        self,
        query_embedding: np.ndarray,
        embedding_result: ProviderResult | Sequence[dict[str, Any]],
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        records = (
            embedding_result.data.get("embeddings", [])
            if isinstance(embedding_result, ProviderResult)
            else list(embedding_result)
        )
        query = _normalize_float32(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))[0]
        scored: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            candidate = _normalize_float32(
                np.asarray(record["embedding"], dtype=np.float32).reshape(1, -1)
            )[0]
            scored.append(
                {
                    "index": index,
                    "start_sec": float(record.get("start_sec", index)),
                    "end_sec": float(record.get("end_sec", index + 1)),
                    "cosine_score": float(np.dot(query, candidate)),
                }
            )
        scored.sort(key=lambda item: item["cosine_score"], reverse=True)
        return scored if limit is None else scored[: max(0, limit)]

    def score_prompts(
        self,
        audio_embedding: np.ndarray,
        prompts: Sequence[str],
    ) -> dict[str, Any]:
        text_embeddings = self.embed_text(prompts)
        query = _normalize_float32(np.asarray(audio_embedding, dtype=np.float32).reshape(1, -1))[0]
        text = _normalize_float32(text_embeddings.astype(np.float32))
        cosine_scores = text @ query
        probabilities = _softmax(cosine_scores)
        return {
            "scores": [
                {
                    "prompt": prompt,
                    "cosine_score": float(cosine),
                    "prompt_set_probability": float(probability),
                }
                for prompt, cosine, probability in zip(
                    prompts, cosine_scores, probabilities, strict=True
                )
            ],
            "calibrated": False,
            "probability_scope": "prompt_set_softmax",
        }

    def _load_backend(self) -> tuple[Any, Any, Any]:
        if self._model is not None:
            return self._model, self._processor, self._torch
        import torch
        from transformers import ClapModel, ClapProcessor

        kwargs: dict[str, Any] = {}
        if self.model_revision:
            kwargs["revision"] = self.model_revision
        model = ClapModel.from_pretrained(self.model_id, **kwargs).to(self.device)
        model.eval()
        processor = ClapProcessor.from_pretrained(self.model_id, **kwargs)
        self._model, self._processor, self._torch = model, processor, torch
        self._resolved_revision = _resolved_model_revision(model) or self.model_revision
        return model, processor, torch

    def _failure(self, error: str) -> ProviderResult:
        return ProviderResult(
            provider_id=self.provider_id,
            capability="semantic_embedding",
            state=CapabilityState.FAILED,
            provenance=self.provenance,
            error=error,
        )


CLAPProvider = ClapProvider


def _move_inputs(inputs: Any, device: str) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
    return inputs


def _resample_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    flattened = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate == CLAP_SAMPLE_RATE or flattened.size < 2:
        return flattened
    output_size = max(1, round(flattened.size * CLAP_SAMPLE_RATE / sample_rate))
    source_points = np.linspace(0.0, 1.0, flattened.size, endpoint=False)
    target_points = np.linspace(0.0, 1.0, output_size, endpoint=False)
    return np.interp(target_points, source_points, flattened).astype(np.float32)


def _to_numpy(value: Any) -> np.ndarray:
    for method in ("detach", "cpu"):
        function = getattr(value, method, None)
        if function is not None:
            value = function()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


def _normalize_float32(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("CLAP embeddings must be a two-dimensional matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float32).eps):
        raise ValueError("CLAP returned a zero-length embedding")
    return values / norms


def _normalize_float16(values: np.ndarray) -> np.ndarray:
    return _normalize_float32(values).astype(np.float16)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


def _resolved_model_revision(model: Any) -> str | None:
    config = getattr(model, "config", None)
    revision = getattr(config, "_commit_hash", None)
    return str(revision) if revision else None
