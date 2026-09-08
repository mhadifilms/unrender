"""Deferred, gated Brouhaha VAD and acoustic-condition adapter."""

from __future__ import annotations

import os
from pathlib import Path
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


class BrouhahaProvider:
    provider_id = "brouhaha"
    capabilities = ("vad", "snr", "c50_estimate_db")
    license_metadata = LicenseMetadata(
        license_id="OpenRAIL",
        source="https://huggingface.co/pyannote/brouhaha",
        permissive=False,
        gated=True,
        terms_url="https://huggingface.co/pyannote/brouhaha",
    )

    def __init__(
        self,
        *,
        model_id: str = "pyannote/brouhaha",
        model_revision: str | None = None,
        hf_token: str | None = None,
        enabled: bool = False,
    ) -> None:
        self.model_id = model_id
        self.model_revision = model_revision
        self.hf_token = hf_token
        self.enabled = enabled
        self._inference: Any = None
        self._resolved_revision = model_revision

    @property
    def provenance(self) -> ProviderProvenance:
        return ProviderProvenance(
            packages=(
                package_provenance("pyannote.audio"),
                package_provenance("brouhaha"),
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
        blocked = active_policy.block_reason(
            self.provider_id,
            self.license_metadata,
            requires_opt_in=True,
        )
        if not self.enabled:
            blocked = blocked or "brouhaha is disabled until explicitly enabled"
        token = self._token()
        if blocked:
            state, reason = CapabilityState.BLOCKED_BY_POLICY, blocked
        elif not token:
            state = CapabilityState.UNAVAILABLE
            reason = "Brouhaha requires HF_TOKEN or HUGGINGFACE_TOKEN"
        elif not module_available("pyannote.audio") or not module_available("brouhaha"):
            state = CapabilityState.UNAVAILABLE
            reason = "pyannote.audio and brouhaha-vad are required"
        else:
            state = CapabilityState.AVAILABLE
            reason = "gated Brouhaha runtime is enabled; the model loads only when analysis runs"
        return statuses_for_capabilities(
            provider_id=self.provider_id,
            capabilities=self.capabilities,
            state=state,
            reason=reason,
            provenance=self.provenance,
            license_metadata=self.license_metadata,
            backend="pyannote/brouhaha",
        )

    def analyze(
        self,
        audio_path: str | Path,
        *,
        policy: ProviderPolicy | None = None,
    ) -> ProviderResult:
        statuses = self.preflight(policy)
        first = statuses[0]
        if first.state is not CapabilityState.AVAILABLE:
            return ProviderResult(
                provider_id=self.provider_id,
                capability="vad_snr_c50",
                state=first.state,
                provenance=self.provenance,
                error=first.reason,
            )
        try:
            inference = self._load_inference()
            output = inference(str(audio_path))
            series = _extract_joint_series(output)
        except Exception as exc:
            return ProviderResult(
                provider_id=self.provider_id,
                capability="vad_snr_c50",
                state=CapabilityState.FAILED,
                provenance=self.provenance,
                error=f"Brouhaha analysis failed: {exc}",
            )
        return ProviderResult(
            provider_id=self.provider_id,
            capability="vad_snr_c50",
            state=CapabilityState.COMPLETE,
            data={"series": series},
            metadata={
                "backend": "pyannote/brouhaha",
                "gated": True,
                "explicit_opt_in": True,
            },
            provenance=self.provenance,
        )

    def _token(self) -> str | None:
        return self.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    def _load_inference(self) -> Any:
        if self._inference is not None:
            return self._inference
        from pyannote.audio import Inference, Model

        kwargs: dict[str, Any] = {"use_auth_token": self._token()}
        if self.model_revision:
            kwargs["revision"] = self.model_revision
        model = Model.from_pretrained(self.model_id, **kwargs)
        if model is None:
            raise RuntimeError(f"unable to load Brouhaha model {self.model_id!r}")
        self._inference = Inference(model)
        self._resolved_revision = _resolved_model_revision(model) or self.model_revision
        return self._inference


def _extract_joint_series(output: Any) -> dict[str, list[dict[str, float]]]:
    if isinstance(output, dict):
        return {
            "vad": _extract_single_series(_pick_output(output, "vad"), "probability"),
            "snr": _extract_single_series(_pick_output(output, "snr"), "snr_db"),
            "c50_estimate_db": _extract_single_series(
                _pick_output(output, "c50_estimate_db", "c50"), "c50_estimate_db"
            ),
        }

    data = getattr(output, "data", None)
    window = getattr(output, "sliding_window", None)
    if data is not None:
        values = np.asarray(data, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] < 3:
            raise ValueError("Brouhaha output must contain VAD, SNR, and C50 columns")
        start = float(getattr(window, "start", 0.0))
        step = float(getattr(window, "step", 1.0))
        rows = ((start + index * step, row) for index, row in enumerate(values))
    else:
        rows = (
            (float(getattr(frame, "middle", index)), np.asarray(values).reshape(-1))
            for index, (frame, values) in enumerate(output)
        )

    series: dict[str, list[dict[str, float]]] = {
        "vad": [],
        "snr": [],
        "c50_estimate_db": [],
    }
    for time_sec, values in rows:
        if len(values) < 3:
            raise ValueError("Brouhaha output row must contain VAD, SNR, and C50")
        series["vad"].append({"time_sec": time_sec, "probability": float(values[0])})
        series["snr"].append({"time_sec": time_sec, "snr_db": float(values[1])})
        series["c50_estimate_db"].append(
            {"time_sec": time_sec, "c50_estimate_db": float(values[2])}
        )
    return series


def _pick_output(output: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in output:
            return output[name]
    raise ValueError(f"Brouhaha output is missing {names[0]}")


def _extract_single_series(feature: Any, value_key: str) -> list[dict[str, float]]:
    if isinstance(feature, list):
        records: list[dict[str, float]] = []
        for index, item in enumerate(feature):
            if isinstance(item, dict):
                raw_time = item.get("time_sec", item.get("time", index))
                raw_value = item.get(value_key, item.get("value", 0.0))
                time_sec = float(index if raw_time is None else raw_time)
                value = float(0.0 if raw_value is None else raw_value)
            else:
                time_sec, value = float(index), float(item)
            records.append({"time_sec": time_sec, value_key: value})
        return records

    values = np.asarray(getattr(feature, "data", feature), dtype=np.float32).squeeze()
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1:
        raise ValueError(f"expected a one-dimensional {value_key} feature")
    window = getattr(feature, "sliding_window", None)
    start = float(getattr(window, "start", 0.0))
    step = float(getattr(window, "step", 1.0))
    return [
        {"time_sec": start + index * step, value_key: float(value)}
        for index, value in enumerate(values)
    ]


def _resolved_model_revision(model: Any) -> str | None:
    for owner in (model, getattr(model, "model", None), getattr(model, "_model", None)):
        config = getattr(owner, "config", None)
        revision = getattr(config, "_commit_hash", None)
        if revision:
            return str(revision)
    return None
