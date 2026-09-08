"""Shared workflow provenance carried by interoperable run artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

WorkflowProvenance = Literal["manual"]

MANUAL_WORKFLOW: WorkflowProvenance = "manual"
WORKFLOW_PROVENANCE_VALUES = frozenset({MANUAL_WORKFLOW})


def workflow_provenance(value: str) -> WorkflowProvenance:
    """Validate a serialized workflow provenance value."""

    if value not in WORKFLOW_PROVENANCE_VALUES:
        supported = ", ".join(sorted(WORKFLOW_PROVENANCE_VALUES))
        raise ValueError(f"unsupported workflow provenance {value!r}; choose from {supported}")
    return value


def artifact_workflow(
    payload: Mapping[str, Any],
    *,
    default: WorkflowProvenance | None = None,
) -> WorkflowProvenance | None:
    """Read optional provenance without rejecting legacy artifacts."""

    value = payload.get("workflow_provenance")
    if value in (None, ""):
        return default
    return workflow_provenance(str(value))


__all__ = [
    "MANUAL_WORKFLOW",
    "WORKFLOW_PROVENANCE_VALUES",
    "WorkflowProvenance",
    "artifact_workflow",
    "workflow_provenance",
]
