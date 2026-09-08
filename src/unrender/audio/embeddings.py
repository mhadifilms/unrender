"""Shared, import-light voice embedding backends and compatibility shims."""

from __future__ import annotations

import gc
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class VoiceEmbedder(Protocol):
    def embed(self, path: Path) -> tuple[float, ...]: ...


class WaveformVoiceEmbedder(Protocol):
    def embed_waveform(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> tuple[float, ...]: ...


def build_voice_embedder(backend: str = "pyannote") -> VoiceEmbedder:
    if backend != "pyannote":
        raise ValueError(f"unsupported voice embedding backend: {backend}")
    return PyannoteVoiceEmbedder()


class PyannoteVoiceEmbedder:
    def __init__(self, model_name: str = "pyannote/embedding") -> None:
        self.model_name = model_name
        self._inference: Any | None = None

    def embed(self, path: Path) -> tuple[float, ...]:
        return self._embed_preloaded(preload_audio(path))

    def embed_waveform(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> tuple[float, ...]:
        """Embed an in-memory fragment without temporary audio files."""

        try:
            import torch
        except ImportError as exc:
            raise ImportError("voice embeddings require torch from the 'voice' extra") from exc
        from unrender.lib.audio import resample_audio

        values = np.asarray(audio, dtype=np.float32)
        mono = values.mean(axis=1) if values.ndim == 2 else values.reshape(-1)
        if sample_rate != 16_000:
            mono = resample_audio(mono, sample_rate, 16_000)
        waveform = torch.from_numpy(np.ascontiguousarray(mono[None, :], dtype=np.float32))
        return self._embed_preloaded({"waveform": waveform, "sample_rate": 16_000})

    def _embed_preloaded(self, audio: dict[str, Any]) -> tuple[float, ...]:
        inference = self._load_inference()
        raw = inference(audio)
        embedding = getattr(raw, "data", raw)
        arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
        return tuple(float(value) for value in _normalize(np.asarray([arr], dtype=np.float32))[0])

    def _load_inference(self) -> Any:
        if self._inference is not None:
            return self._inference
        token = huggingface_token()
        try:
            import torch

            patch_numpy_legacy_aliases()
            patch_hf_hub_use_auth_token()
            from pyannote.audio import Inference, Model

            patch_hf_hub_use_auth_token()
        except ImportError as exc:
            raise ImportError(
                "voice embeddings require the 'voice' extra: pip install .[voice]"
            ) from exc

        with torch_load_legacy_checkpoints(torch):
            model = Model.from_pretrained(
                self.model_name,
                **pretrained_auth_kwargs(Model.from_pretrained, token),
            )
        if model is None:
            raise RuntimeError(f"failed to load pyannote model: {self.model_name}")
        device = torch_device(torch)
        if hasattr(model, "to"):
            model.to(device)
        self._inference = Inference(model, window="whole")
        return self._inference

    def release_gpu(self) -> None:
        """Release cached embedding weights before another large GPU model loads."""

        self._inference = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class SpeechBrainEcapaEmbedder:
    """Independent ECAPA-TDNN speaker encoder for fragment consensus."""

    def __init__(self) -> None:
        self._classifier: Any | None = None

    def embed_waveform(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> tuple[float, ...]:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("ECAPA embeddings require torch") from exc
        from unrender.lib.audio import resample_audio

        values = np.asarray(audio, dtype=np.float32)
        mono = values.mean(axis=1) if values.ndim == 2 else values.reshape(-1)
        if sample_rate != 16_000:
            mono = resample_audio(mono, sample_rate, 16_000)
        waveform = torch.from_numpy(np.ascontiguousarray(mono[None, :]))
        classifier = self._load()
        device = next(classifier.mods.embedding_model.parameters()).device
        with torch.no_grad():
            embedding = classifier.encode_batch(waveform.to(device))
        vector = np.asarray(embedding.squeeze().cpu(), dtype=np.float32)
        normalized = _normalize(vector[None, :])[0]
        return tuple(float(value) for value in normalized)

    def _load(self) -> Any:
        if self._classifier is not None:
            return self._classifier
        try:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:
            raise ImportError("ECAPA embeddings require speechbrain: pip install .[voice]") from exc
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(Path.home() / ".cache" / "unrender" / "ecapa"),
            run_opts={"device": device},
        )
        return self._classifier


def huggingface_token() -> str | None:
    import os

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    return token.strip() if token else None


def patch_hf_hub_use_auth_token() -> None:
    """Bridge older pyannote calls to newer huggingface_hub signatures."""

    try:
        import inspect
        from functools import wraps

        import huggingface_hub
        from huggingface_hub import file_download
    except ImportError:
        return

    targets = [file_download, huggingface_hub]
    for module_name in ("pyannote.audio.core.model", "pyannote.audio.core.pipeline"):
        pyannote_module = sys.modules.get(module_name)
        if pyannote_module is not None:
            targets.append(pyannote_module)

    for module in targets:
        original = getattr(module, "hf_hub_download", None)
        if original is None or getattr(original, "_unrender_token_patch", False):
            continue
        try:
            parameters = inspect.signature(original).parameters
        except (TypeError, ValueError):
            continue
        if "use_auth_token" in parameters:
            continue

        @wraps(original)
        def hf_hub_download(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
            use_auth_token = kwargs.pop("use_auth_token", None)
            if use_auth_token is not None and "token" not in kwargs:
                kwargs["token"] = use_auth_token
            return _original(*args, **kwargs)

        hf_hub_download._unrender_token_patch = True  # type: ignore[attr-defined]
        module.hf_hub_download = hf_hub_download  # type: ignore[attr-defined]


def patch_numpy_legacy_aliases() -> None:
    """Bridge aliases removed by NumPy 2 for the pinned pyannote 3.1 fork."""

    if not hasattr(np, "NaN"):
        np.NaN = np.nan  # type: ignore[attr-defined]
    if not hasattr(np, "NAN"):
        np.NAN = np.nan  # type: ignore[attr-defined]


def pretrained_auth_kwargs(loader: Any, token: str | None) -> dict[str, Any]:
    """Support both pyannote 3.x ``use_auth_token`` and newer ``token`` APIs."""

    if token is None:
        return {}
    try:
        import inspect

        parameters = inspect.signature(loader).parameters
    except (TypeError, ValueError):
        return {"token": token}
    if "token" in parameters:
        return {"token": token}
    if "use_auth_token" in parameters:
        return {"use_auth_token": token}
    return {"token": token}


@contextmanager
def torch_load_legacy_checkpoints(torch: Any):
    original = torch.load
    try:
        import inspect

        if "weights_only" not in inspect.signature(original).parameters:
            yield
            return
    except (TypeError, ValueError):
        yield
        return

    def torch_load(*args: Any, **kwargs: Any) -> Any:
        kwargs["weights_only"] = False
        return original(*args, **kwargs)

    torch.load = torch_load
    try:
        yield
    finally:
        torch.load = original


def torch_device(torch: Any) -> Any:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preload_audio(path: Path) -> dict[str, Any]:
    """Decode audio as a mono 16 kHz waveform for pyannote inference."""

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "voice embeddings require torch from the 'voice' extra: pip install .[voice]"
        ) from exc

    from unrender.lib.audio import read_audio_any

    data, _sr = read_audio_any(path, target_sr=16000)
    mono = data.mean(axis=1) if data.ndim == 2 else data
    waveform = torch.from_numpy(np.ascontiguousarray(mono[None, :], dtype=np.float32))
    return {"waveform": waveform, "sample_rate": 16000}


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


__all__ = [
    "PyannoteVoiceEmbedder",
    "SpeechBrainEcapaEmbedder",
    "VoiceEmbedder",
    "WaveformVoiceEmbedder",
    "build_voice_embedder",
    "huggingface_token",
    "patch_hf_hub_use_auth_token",
    "patch_numpy_legacy_aliases",
    "preload_audio",
    "pretrained_auth_kwargs",
    "torch_device",
    "torch_load_legacy_checkpoints",
]
