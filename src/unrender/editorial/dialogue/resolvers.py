"""Registry and future-facing automatic selection for existing clip resolvers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from unrender.editorial.dialogue.clusters import resolve_voice_clips_clustered
from unrender.editorial.dialogue.lines import (
    resolve_voice_clips,
    resolve_voice_clips_from_voice_db,
)
from unrender.workflows import (
    MANUAL_WORKFLOW,
    WorkflowProvenance,
)
from unrender.workflows import (
    workflow_provenance as validate_workflow_provenance,
)

ClipResolver = Callable[..., list[dict[str, Any]]]


@dataclass(frozen=True)
class ClipResolverSpec:
    name: str
    resolver: ClipResolver
    prerequisites: tuple[str, ...]
    description: str

    def missing_prerequisites(self, available: Iterable[str]) -> tuple[str, ...]:
        present = frozenset(str(value) for value in available)
        return tuple(name for name in self.prerequisites if name not in present)


CLIP_RESOLVERS: Mapping[str, ClipResolverSpec] = {
    "voice-db": ClipResolverSpec(
        name="voice-db",
        resolver=resolve_voice_clips_from_voice_db,
        prerequisites=("voice_db",),
        description="Preserve reviewed voice-cluster membership and labels.",
    ),
    "cluster": ClipResolverSpec(
        name="cluster",
        resolver=resolve_voice_clips_clustered,
        prerequisites=("clips", "speaker_db"),
        description="Cluster all clips, then assign clusters to labeled voice centroids.",
    ),
    "clip": ClipResolverSpec(
        name="clip",
        resolver=resolve_voice_clips,
        prerequisites=("clips", "speaker_db"),
        description="Match each clip independently against labeled voice centroids.",
    ),
}

_PREFERRED_RESOLVER = {
    MANUAL_WORKFLOW: "voice-db",
}


def get_clip_resolver(name: str) -> ClipResolverSpec:
    try:
        return CLIP_RESOLVERS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(CLIP_RESOLVERS))
        raise ValueError(f"unsupported clip resolver {name!r}; choose from {supported}") from exc


def select_clip_resolver(
    workflow_provenance: WorkflowProvenance,
    *,
    available_artifacts: Iterable[str] | None = None,
) -> ClipResolverSpec:
    """Select the existing resolver suited to an artifact-producing workflow.

    Supplying ``available_artifacts`` turns selection into a prerequisite check;
    omitting it only returns the workflow preference.
    """

    workflow = validate_workflow_provenance(workflow_provenance)
    spec = get_clip_resolver(_PREFERRED_RESOLVER[workflow])
    if available_artifacts is not None:
        missing = spec.missing_prerequisites(available_artifacts)
        if missing:
            raise ValueError(
                f"{workflow} workflow resolver {spec.name!r} requires: {', '.join(missing)}"
            )
    return spec


__all__ = [
    "CLIP_RESOLVERS",
    "ClipResolver",
    "ClipResolverSpec",
    "get_clip_resolver",
    "select_clip_resolver",
]
