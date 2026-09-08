"""Input fingerprints for restartable pipeline steps.

Steps skip their work when the output manifest already exists, which silently
serves stale artifacts if an input file changed in the meantime. Recording a
cheap content fingerprint of each input in the manifest lets a skipped step
detect the change and warn the user to rerun with ``--force``.

Fingerprints hash the first and last 4 MiB plus the file size, so multi-GB
media does not require a full read. A same-size edit confined to the middle
of a file is the one case this cannot detect.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_HASH_WINDOW_BYTES = 4 * 1024 * 1024

InputSpec = dict[str, Any]


def sha256_file(path: Path) -> str:
    """Full-content SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprints(inputs: InputSpec) -> dict[str, dict[str, Any]]:
    """Fingerprint named file, URL, scalar-setting, or aggregate inputs."""
    out: dict[str, dict[str, Any]] = {}
    for name, value in inputs.items():
        entry = _entry(value)
        if entry is not None:
            out[name] = entry
    return out


def changed_inputs(manifest: dict[str, Any], inputs: InputSpec) -> list[str]:
    """Names of inputs whose current fingerprint differs from the manifest."""
    recorded = manifest.get("inputs")
    if not isinstance(recorded, dict):
        return []
    if not recorded:
        return []
    changed: list[str] = []
    for name, value in inputs.items():
        previous = recorded.get(name)
        current = _entry(value)
        if not isinstance(previous, dict):
            if current is not None:
                changed.append(name)
            continue
        if current != previous:
            changed.append(name)
    return changed


def warn_if_inputs_changed(manifest: dict[str, Any], inputs: InputSpec, output: Path) -> list[str]:
    changed = changed_inputs(manifest, inputs)
    if changed:
        print(
            f"  WARNING: input(s) changed since {output.name} was written: "
            f"{', '.join(sorted(changed))}. Rerun with --force to regenerate.",
            flush=True,
        )
    return changed


def _entry(value: Any) -> dict[str, Any] | None:
    if value in (None, ""):
        return None
    if isinstance(value, list | tuple):
        return _aggregate_fingerprint(value)
    if isinstance(value, str | Path):
        path = Path(str(value)).expanduser()
        if path.is_file():
            return _file_fingerprint(path)
    return _value_fingerprint(value)


def _aggregate_fingerprint(values: Any) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    entries = [(str(value), _entry(value)) for value in values if value not in (None, "")]
    for label, fingerprint in sorted(entries, key=lambda item: item[0]):
        if fingerprint is None:
            continue
        digest.update(label.encode())
        digest.update(json.dumps(fingerprint, sort_keys=True).encode())
        count += 1
    return {"count": count, "sha256": digest.hexdigest()}


def _value_fingerprint(value: Any) -> dict[str, Any]:
    encoded = json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()
    return {
        "value_type": type(value).__name__,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _file_fingerprint(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(_HASH_WINDOW_BYTES))
        if size > 2 * _HASH_WINDOW_BYTES:
            handle.seek(size - _HASH_WINDOW_BYTES)
            digest.update(handle.read(_HASH_WINDOW_BYTES))
    return {"size": size, "sha256": digest.hexdigest()}
