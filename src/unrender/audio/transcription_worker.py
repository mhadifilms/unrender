from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from unrender.audio import asr

_OPERATIONS = {"transcribe_path_segments", "transcribe_word_aligned"}
_PAYLOAD_FIELDS = {
    "path",
    "model_name",
    "device",
    "compute_type",
    "language",
    "batch_size",
}


def execute_request(request: Any) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        raise ValueError("transcription worker request must be an object")
    if request.get("version") != 1:
        raise ValueError("transcription worker request version must be 1")
    operation = request.get("operation")
    if operation not in _OPERATIONS:
        raise ValueError(f"unsupported transcription operation: {operation!r}")
    payload = _validate_payload(request.get("payload"))

    if operation == "transcribe_word_aligned":
        words, metadata = asr._transcribe_word_aligned_in_process(**payload)
        return {"words": words, "metadata": metadata}
    segments = asr._transcribe_segments_in_process(
        str(payload.pop("path")),
        **payload,
        required=True,
    )
    return {"segments": segments}


def _validate_payload(raw_payload: Any) -> dict[str, Any]:
    if not isinstance(raw_payload, Mapping):
        raise ValueError("transcription worker payload must be an object")
    unknown = sorted(set(raw_payload).difference(_PAYLOAD_FIELDS))
    missing = sorted(_PAYLOAD_FIELDS.difference(raw_payload))
    if unknown:
        raise ValueError(f"transcription worker payload has unknown fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"transcription worker payload is missing fields: {', '.join(missing)}")

    path = raw_payload["path"]
    model_name = raw_payload["model_name"]
    device = raw_payload["device"]
    compute_type = raw_payload["compute_type"]
    language = raw_payload["language"]
    batch_size = raw_payload["batch_size"]
    for name, value in (
        ("path", path),
        ("model_name", model_name),
        ("device", device),
        ("compute_type", compute_type),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"transcription worker payload {name} must be a non-empty string")
    if language is not None and not isinstance(language, str):
        raise ValueError("transcription worker payload language must be a string or null")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("transcription worker payload batch_size must be a positive integer")
    return {
        "path": Path(path),
        "model_name": model_name,
        "device": device,
        "compute_type": compute_type,
        "language": language,
        "batch_size": batch_size,
    }


def _write_response(path: Path, response: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(response), encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2:
        print("usage: transcription_worker REQUEST_JSON RESPONSE_JSON", file=sys.stderr)
        return 2
    request_path, response_path = map(Path, arguments)
    os.environ["_UNRENDER_TRANSCRIPTION_WORKER"] = "1"
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        result = execute_request(request)
        response: dict[str, Any] = {"ok": True, "result": result}
        exit_code = 0
    except Exception as exc:
        response = {
            "ok": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        exit_code = 1
    try:
        _write_response(response_path, response)
    except OSError as exc:
        print(f"failed to write transcription response: {exc}", file=sys.stderr)
        return 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
