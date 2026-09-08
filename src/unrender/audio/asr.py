from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from unrender.lib.languages import to_whisper_code

_WHISPERX_MODELS: dict[tuple[str, str, str, str | None], Any] = {}
_WHISPERX_ALIGN_MODELS: dict[tuple[str, str], tuple[Any, Any]] = {}
_TRANSCRIPTION_PYTHON_ENV = "UNRENDER_TRANSCRIPTION_PYTHON"
_TRANSCRIPTION_WORKER_ENV = "_UNRENDER_TRANSCRIPTION_WORKER"
_TRANSCRIPTION_WORKER_MODULE = "unrender.audio.transcription_worker"


def whisper_language_code(language: str | None) -> str | None:
    """ISO code for Whisper from a name/code/alias (thin wrapper over the shared
    language registry; unknown values pass through lowercased)."""
    return to_whisper_code(language)


def load_whisperx_model(
    *,
    model_name: str,
    device: str,
    compute_type: str,
    language: str | None,
) -> Any | None:
    try:
        import torch
        import whisperx
    except ImportError:
        return None
    language_code = whisper_language_code(language)
    key = (model_name, device, compute_type, language_code)
    if key not in _WHISPERX_MODELS:
        from unrender.audio.embeddings import torch_load_legacy_checkpoints

        with torch_load_legacy_checkpoints(torch):
            _WHISPERX_MODELS[key] = whisperx.load_model(
                model_name,
                device,
                compute_type=compute_type,
                language=language_code,
            )
    return _WHISPERX_MODELS[key]


def transcribe_segments(
    audio: Any,
    *,
    model_name: str = "small",
    device: str = "cpu",
    compute_type: str = "float32",
    language: str | None = None,
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    """Return WhisperX segments, or an empty list when WhisperX is unavailable."""
    external_python = _external_transcription_python()
    if external_python is not None and isinstance(audio, (str, os.PathLike)):
        result = _run_transcription_worker(
            external_python,
            operation="transcribe_path_segments",
            payload=_transcription_payload(
                Path(audio),
                model_name=model_name,
                device=device,
                compute_type=compute_type,
                language=language,
                batch_size=batch_size,
            ),
        )
        segments = result.get("segments")
        if not isinstance(segments, list) or not all(
            isinstance(segment, dict) for segment in segments
        ):
            raise RuntimeError(f"{_TRANSCRIPTION_PYTHON_ENV} worker returned malformed segments")
        return [dict(segment) for segment in segments]

    return _transcribe_segments_in_process(
        audio,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        language=language,
        batch_size=batch_size,
    )


def _transcribe_segments_in_process(
    audio: Any,
    *,
    model_name: str,
    device: str,
    compute_type: str,
    language: str | None,
    batch_size: int,
    required: bool = False,
) -> list[dict[str, Any]]:
    result = _transcribe_result(
        audio,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        language=language,
        batch_size=batch_size,
    )
    if not isinstance(result, dict):
        if required:
            raise ImportError(
                "path transcription requires whisperx and torch in the interpreter selected by "
                "UNRENDER_TRANSCRIPTION_PYTHON: pip install .[transcription]"
            )
        return []
    return [dict(segment) for segment in result.get("segments") or []]


def transcribe_path(
    path: Path,
    *,
    model_name: str = "small",
    device: str = "cpu",
    compute_type: str = "float32",
    language: str | None = None,
    batch_size: int = 16,
) -> str:
    segments = transcribe_segments(
        str(path),
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        language=language,
        batch_size=batch_size,
    )
    return _segments_text(segments)


def transcribe_audio(
    audio: Any,
    *,
    model_name: str = "small",
    device: str = "cpu",
    compute_type: str = "float32",
    language: str | None = None,
    batch_size: int = 16,
) -> str:
    segments = transcribe_segments(
        audio,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        language=language,
        batch_size=batch_size,
    )
    return _segments_text(segments)


def transcribe_word_aligned(
    path: Path,
    *,
    model_name: str = "small",
    device: str = "cpu",
    compute_type: str = "float32",
    language: str | None = None,
    batch_size: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return aligned WhisperX words and detection/alignment metadata.

    If alignment fails, segment timestamps are returned and
    ``metadata["aligned"]`` is false.
    Unlike the text-only helpers, word ASR is required and therefore gives an
    actionable error when the optional WhisperX stack is unavailable.
    """
    external_python = _external_transcription_python()
    if external_python is not None:
        result = _run_transcription_worker(
            external_python,
            operation="transcribe_word_aligned",
            payload=_transcription_payload(
                path,
                model_name=model_name,
                device=device,
                compute_type=compute_type,
                language=language,
                batch_size=batch_size,
            ),
        )
        words = result.get("words")
        metadata = result.get("metadata")
        if (
            not isinstance(words, list)
            or not all(isinstance(word, dict) for word in words)
            or not isinstance(metadata, dict)
        ):
            raise RuntimeError(
                f"{_TRANSCRIPTION_PYTHON_ENV} worker returned malformed word-alignment data"
            )
        return [dict(word) for word in words], dict(metadata)

    return _transcribe_word_aligned_in_process(
        path,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        language=language,
        batch_size=batch_size,
    )


def _transcribe_word_aligned_in_process(
    path: Path,
    *,
    model_name: str,
    device: str,
    compute_type: str,
    language: str | None,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import whisperx
    except ImportError as exc:
        raise ImportError(
            "word-level transcription requires whisperx in the interpreter selected by "
            "UNRENDER_TRANSCRIPTION_PYTHON: pip install .[transcription]"
        ) from exc

    language_code = whisper_language_code(language)
    # WhisperX bundles an older trusted Pyannote VAD checkpoint. PyTorch 2.6+
    # otherwise rejects that package asset through its weights_only default.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    model = load_whisperx_model(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        language=language_code,
    )
    if model is None:
        raise ImportError(
            "word-level transcription requires whisperx and torch in the interpreter selected "
            "by UNRENDER_TRANSCRIPTION_PYTHON: pip install .[transcription]"
        )

    audio = whisperx.load_audio(str(path))
    result = model.transcribe(audio, batch_size=batch_size, language=language_code)
    # Alignment needs a concrete language, but callers that are *asking* what
    # was spoken must be able to tell a real detection from this fallback.
    detected = str(result.get("language")) if result.get("language") else None
    detected_language = detected or language_code or "en"
    aligned = False
    try:
        align_model, align_metadata = _load_whisperx_align_model(
            whisperx,
            language=detected_language,
            device=device,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        aligned = any(segment.get("words") for segment in result.get("segments") or [])
    except Exception as exc:
        print(f"  WARNING: WhisperX alignment failed, using segment timestamps: {exc}", flush=True)

    words: list[dict[str, Any]] = []
    for segment in result.get("segments") or []:
        segment_words = segment.get("words") or []
        if segment_words:
            for word in segment_words:
                if word.get("start") is None or word.get("end") is None:
                    continue
                words.append(
                    {
                        "word": str(word.get("word") or "").strip(),
                        "start": float(word["start"]),
                        "end": float(word["end"]),
                        "speaker": "",
                    }
                )
        elif segment.get("start") is not None and segment.get("end") is not None:
            words.append(
                {
                    "word": str(segment.get("text") or "").strip(),
                    "start": float(segment["start"]),
                    "end": float(segment["end"]),
                    "speaker": "",
                }
            )
    return words, {
        "language": detected_language,
        "detected_language": detected,
        "aligned": aligned,
    }


def _transcription_payload(
    path: Path,
    *,
    model_name: str,
    device: str,
    compute_type: str,
    language: str | None,
    batch_size: int,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "model_name": model_name,
        "device": device,
        "compute_type": compute_type,
        "language": language,
        "batch_size": batch_size,
    }


def _external_transcription_python() -> str | None:
    configured = os.environ.get(_TRANSCRIPTION_PYTHON_ENV, "").strip()
    if not configured or os.environ.get(_TRANSCRIPTION_WORKER_ENV) == "1":
        return None
    interpreter = _resolve_transcription_python(configured)
    if _same_interpreter(interpreter, sys.executable):
        return None
    return interpreter


def _resolve_transcription_python(configured: str) -> str:
    candidate = shutil.which(configured) if not Path(configured).parent.name else configured
    if not candidate:
        raise RuntimeError(
            f"{_TRANSCRIPTION_PYTHON_ENV} points to a missing interpreter: {configured}"
        )
    path = Path(candidate).expanduser()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(
            f"{_TRANSCRIPTION_PYTHON_ENV} must name an executable Python interpreter: "
            f"{configured}"
        )
    # Keep the launcher path intact. Distinct virtualenv launchers commonly
    # symlink to the same base binary but still select different environments.
    return str(path.absolute())


def _same_interpreter(left: str, right: str) -> bool:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))


def _run_transcription_worker(
    interpreter: str,
    *,
    operation: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="unrender-transcription-") as temporary:
        work_dir = Path(temporary)
        request_path = work_dir / "request.json"
        response_path = work_dir / "response.json"
        request_path.write_text(
            json.dumps({"version": 1, "operation": operation, "payload": payload}),
            encoding="utf-8",
        )
        env = os.environ.copy()
        env[_TRANSCRIPTION_WORKER_ENV] = "1"
        command = [
            interpreter,
            "-m",
            _TRANSCRIPTION_WORKER_MODULE,
            str(request_path),
            str(response_path),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                env=env,
            )
        except OSError as exc:
            raise RuntimeError(
                f"failed to launch {_TRANSCRIPTION_PYTHON_ENV} interpreter "
                f"{interpreter!r}: {exc}"
            ) from exc

        try:
            response = _read_transcription_response(response_path)
        except RuntimeError as exc:
            raise RuntimeError(f"{exc}; worker exited {completed.returncode}") from exc
        if completed.returncode != 0 or response.get("ok") is not True:
            error = response.get("error")
            if isinstance(error, dict):
                detail = str(error.get("message") or error.get("type") or "unknown error")
            else:
                detail = "unknown worker error"
            raise RuntimeError(
                f"{_TRANSCRIPTION_PYTHON_ENV} worker failed using {interpreter!r}: {detail}"
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(
                f"{_TRANSCRIPTION_PYTHON_ENV} worker returned a malformed response: "
                "result must be an object"
            )
        return dict(result)


def _read_transcription_response(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{_TRANSCRIPTION_PYTHON_ENV} worker did not create its JSON response")
    try:
        response = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{_TRANSCRIPTION_PYTHON_ENV} worker returned malformed JSON") from exc
    if not isinstance(response, dict) or not isinstance(response.get("ok"), bool):
        raise RuntimeError(
            f"{_TRANSCRIPTION_PYTHON_ENV} worker returned a malformed response envelope"
        )
    return response


def _transcribe_result(
    audio: Any,
    *,
    model_name: str,
    device: str,
    compute_type: str,
    language: str | None,
    batch_size: int,
) -> Any:
    model = load_whisperx_model(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        language=language,
    )
    if model is None:
        return None
    return model.transcribe(audio, batch_size=batch_size)


def _load_whisperx_align_model(
    whisperx: Any,
    *,
    language: str,
    device: str,
) -> tuple[Any, Any]:
    key = (language, device)
    if key not in _WHISPERX_ALIGN_MODELS:
        _WHISPERX_ALIGN_MODELS[key] = whisperx.load_align_model(
            language_code=language,
            device=device,
        )
    return _WHISPERX_ALIGN_MODELS[key]


def _segments_text(segments: Any) -> str:
    if isinstance(segments, dict):
        segments = segments.get("segments") or []
    if not isinstance(segments, list):
        return ""
    return " ".join(str(segment.get("text") or "").strip() for segment in segments)
