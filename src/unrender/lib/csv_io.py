from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_rows_atomic(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write rows to a temp sibling then atomically replace the target."""
    tmp = path.with_name(f".{path.name}.tmp")
    write_rows(tmp, rows, fields)
    tmp.replace(path)
