"""Renderer contract and registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from unrender.audio.visualization.bundle import AnalysisBundle
from unrender.audio.visualization.mapping import VisualMapping


@dataclass(frozen=True)
class RenderContext:
    """Inputs shared by all standalone audio-analysis renderers."""

    bundle: AnalysisBundle
    output_dir: Path
    mapping: VisualMapping
    source_paths: tuple[Path, ...] = ()
    alternate_paths: tuple[Path, ...] = ()
    ffmpeg: str | None = None


@runtime_checkable
class AudioAnalysisRenderer(Protocol):
    """Callable renderer that writes one primary file or directory."""

    def __call__(self, context: RenderContext) -> Path:
        """Render an artifact and return its primary path."""


Renderer = Callable[[RenderContext], Path]


class RendererRegistry:
    """Small explicit registry with duplicate protection."""

    def __init__(self) -> None:
        self._renderers: dict[str, Renderer] = {}

    def register(
        self,
        name: str,
        renderer: Renderer | None = None,
        *,
        replace: bool = False,
    ) -> Renderer | Callable[[Renderer], Renderer]:
        normalized = self._normalize(name)

        def add(candidate: Renderer) -> Renderer:
            if normalized in self._renderers and not replace:
                raise ValueError(f"renderer already registered: {normalized}")
            if not callable(candidate):
                raise TypeError("renderer must be callable")
            self._renderers[normalized] = candidate
            return candidate

        if renderer is None:
            return add
        return add(renderer)

    def unregister(self, name: str) -> None:
        del self._renderers[self._normalize(name)]

    def get(self, name: str) -> Renderer:
        normalized = self._normalize(name)
        try:
            return self._renderers[normalized]
        except KeyError as exc:
            available = ", ".join(self.names())
            raise KeyError(f"unknown audio renderer {name!r}; available: {available}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._renderers))

    def items(self) -> tuple[tuple[str, Renderer], ...]:
        return tuple(sorted(self._renderers.items()))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self._normalize(name) in self._renderers

    def __len__(self) -> int:
        return len(self._renderers)

    @staticmethod
    def _normalize(name: str) -> str:
        normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
        if not normalized:
            raise ValueError("renderer name cannot be empty")
        return normalized


renderer_registry = RendererRegistry()
RENDERERS: Mapping[str, Renderer] = renderer_registry._renderers


def register_renderer(
    name: str, renderer: Renderer | None = None, *, replace: bool = False
) -> Renderer | Callable[[Renderer], Renderer]:
    return renderer_registry.register(name, renderer, replace=replace)


def get_renderer(name: str) -> Renderer:
    return renderer_registry.get(name)


def available_renderers() -> tuple[str, ...]:
    return renderer_registry.names()
