"""Lenient numeric coercion for manifest and CSV fields.

Manifest rows and CSV cells carry numbers as strings (often empty). These
helpers define the shared fail-soft semantics; callers with stricter needs
(finiteness checks, raise-on-invalid) keep their own local variants.
"""

from __future__ import annotations

from typing import Any


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
