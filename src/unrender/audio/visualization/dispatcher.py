"""Built-in renderer registration and high-level dispatch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from unrender.audio.visualization.bundle import AnalysisBundle, load_analysis_bundle
from unrender.audio.visualization.html import render_standalone_html
from unrender.audio.visualization.images import (
    render_mix_territory,
    render_program_poster,
    render_structure_map,
    render_timeline_atlas,
)
from unrender.audio.visualization.mapping import VisualMapping, load_visual_mapping
from unrender.audio.visualization.overlay import encode_overlay_mov, render_overlay_sequence
from unrender.audio.visualization.registry import (
    RenderContext,
    available_renderers,
    register_renderer,
    renderer_registry,
)

DEFAULT_RENDERERS = ("timeline_atlas", "structure_map", "mix_territory", "poster", "html")

_ALIASES = {
    "timeline": "timeline_atlas",
    "atlas": "timeline_atlas",
    "structure": "structure_map",
    "similarity": "structure_map",
    "territory": "mix_territory",
    "mix": "mix_territory",
    "program_poster": "poster",
    "standalone_html": "html",
    "overlay_timeline": "overlay",
}


def _timeline(context: RenderContext) -> Path:
    return render_timeline_atlas(
        context.bundle,
        context.output_dir / "timeline_atlas.png",
        context.mapping,
    )


def _structure(context: RenderContext) -> Path:
    return render_structure_map(
        context.bundle,
        context.output_dir / "structure_map.png",
        context.mapping,
    )


def _territory(context: RenderContext) -> Path:
    return render_mix_territory(
        context.bundle,
        context.output_dir / "mix_territory.png",
        context.mapping,
    )


def _poster(context: RenderContext) -> Path:
    return render_program_poster(
        context.bundle,
        context.output_dir / "audio_analysis_poster.png",
        context.mapping,
    )


def _html(context: RenderContext) -> Path:
    return render_standalone_html(
        context.bundle,
        context.output_dir / "audio_analysis.html",
        context.mapping,
        source_paths=context.source_paths,
        alternate_paths=context.alternate_paths,
    )


def _overlay(context: RenderContext) -> Path:
    frame_dir = context.output_dir / "overlay_frames"
    render_overlay_sequence(context.bundle, frame_dir, context.mapping)
    if context.ffmpeg:
        encode_overlay_mov(
            frame_dir,
            context.output_dir / "overlay.mov",
            fps=float(context.mapping.geometry["overlay_fps"]),
            ffmpeg=context.ffmpeg,
        )
    return frame_dir


def _spectral(context: RenderContext) -> Path:
    if not context.source_paths:
        raise ValueError("the spectral renderer requires at least one source audio path")
    from unrender.audio.spectral import (
        encode_audio_to_image,
        load_audio_file,
        write_preview_png,
    )

    audio, sample_rate = load_audio_file(
        context.source_paths[0],
        ffmpeg=context.ffmpeg or "ffmpeg",
    )
    encoded = encode_audio_to_image(audio, sample_rate)
    return write_preview_png(context.output_dir / "spectral.png", encoded)


def _ensure_builtins() -> None:
    builtins = {
        "timeline_atlas": _timeline,
        "structure_map": _structure,
        "mix_territory": _territory,
        "poster": _poster,
        "html": _html,
        "overlay": _overlay,
        "spectral": _spectral,
    }
    for name, renderer in builtins.items():
        if name not in renderer_registry:
            register_renderer(name, renderer)


def _path_tuple(
    value: str | Path | Sequence[str | Path] | None,
) -> tuple[Path, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, Path)):
        return (Path(value),)
    return tuple(Path(path) for path in value)


def render_audio_analysis(
    bundle: AnalysisBundle | Mapping[str, Any] | str | Path,
    renderer_names: str | Sequence[str] | None = None,
    output_dir: str | Path | None = None,
    mapping: VisualMapping | Mapping[str, Any] | str | Path | None = None,
    source_paths: str | Path | Sequence[str | Path] | None = None,
    alternate_paths: str | Path | Sequence[str | Path] | None = None,
    ffmpeg: str | None = None,
    *,
    renderers: str | Sequence[str] | None = None,
    source_path: str | Path | None = None,
    alternate_path: str | Path | None = None,
) -> dict[str, Path]:
    """Render selected analysis artifacts and return their primary paths."""

    _ensure_builtins()
    if output_dir is None:
        raise ValueError("output_dir is required")
    if renderer_names is not None and renderers is not None:
        raise ValueError("pass either renderer_names or renderers, not both")
    loaded = load_analysis_bundle(bundle)
    visual = load_visual_mapping(mapping)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    requested_renderers = renderer_names if renderer_names is not None else renderers
    names = DEFAULT_RENDERERS if requested_renderers is None else requested_renderers
    if isinstance(names, str):
        names = (names,)
    selected_sources = source_paths if source_paths is not None else source_path
    selected_alternates = alternate_paths if alternate_paths is not None else alternate_path
    context = RenderContext(
        bundle=loaded,
        output_dir=output,
        mapping=visual,
        source_paths=_path_tuple(selected_sources),
        alternate_paths=_path_tuple(selected_alternates),
        ffmpeg=ffmpeg,
    )
    results: dict[str, Path] = {}
    for requested_name in names:
        key = str(requested_name).strip().lower().replace("-", "_").replace(" ", "_")
        canonical = _ALIASES.get(key, key)
        results[canonical] = renderer_registry.get(canonical)(context)
    return results


_ensure_builtins()

__all__ = [
    "DEFAULT_RENDERERS",
    "available_renderers",
    "render_audio_analysis",
]
