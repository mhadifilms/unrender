from __future__ import annotations

import threading
from pathlib import Path

from unrender.lib.json_io import read_json, write_json_atomic, write_status_heartbeat


def test_write_json_atomic_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "status.json"

    write_json_atomic(target, {"value": 1})

    assert read_json(target) == {"value": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["status.json"]


def test_concurrent_heartbeats_never_race_on_a_shared_temp_file(tmp_path: Path) -> None:
    """Regression: a shared <name>.tmp let one writer replace away the temp
    file another writer was about to rename, raising FileNotFoundError."""
    target = tmp_path / "heartbeat.json"
    errors: list[Exception] = []
    barrier = threading.Barrier(4)

    def hammer(worker: int) -> None:
        barrier.wait()
        for iteration in range(50):
            try:
                write_status_heartbeat(target, worker=worker, iteration=iteration)
            except Exception as exc:  # pragma: no cover - failure channel
                errors.append(exc)

    threads = [threading.Thread(target=hammer, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    payload = read_json(target)
    assert payload["last_heartbeat_at"]
    assert [p.name for p in tmp_path.iterdir()] == ["heartbeat.json"]
