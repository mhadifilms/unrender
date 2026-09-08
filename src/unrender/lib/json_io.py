from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON through a writer-unique sibling temp file and atomic replace.

    The temp name includes the writer's PID and thread id: status heartbeats
    can be written by concurrent writers (pipeline runner, remote pollers,
    scoring threads), and a shared ``<name>.tmp`` let one writer replace away
    the file another writer was about to rename, raising FileNotFoundError
    mid-run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}-{threading.get_ident()}.tmp")
    try:
        temporary.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_status_heartbeat(
    path: Path,
    *,
    reset: bool = False,
    timestamps: Literal["iso", "epoch"] = "iso",
    **values: Any,
) -> None:
    """Merge values into an externally observable status file, atomically.

    Preserves the file's original creation stamp and refreshes the heartbeat
    stamp on every write. ``timestamps="epoch"`` keeps the numeric field names
    used by remote-worker pollers.
    """
    previous: dict[str, Any] = {}
    if path.is_file() and not reset:
        try:
            loaded = read_json(path)
            if isinstance(loaded, dict):
                previous = loaded
        except (OSError, ValueError):
            previous = {}
    now: Any
    if timestamps == "epoch":
        now = time.time()
        created_key, heartbeat_key = "created_at_epoch", "last_heartbeat_epoch"
    else:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        created_key, heartbeat_key = "created_at", "last_heartbeat_at"
    write_json_atomic(
        path,
        {
            **previous,
            **values,
            created_key: previous.get(created_key, now),
            heartbeat_key: now,
        },
    )
