"""Standalone renderers for canonical audio-analysis bundles.

Renderers accept plain mappings or JSON files with list/array values and
relative ``.npy``/``.npz`` references.
"""

from unrender.audio.visualization.bundle import AnalysisBundle, load_analysis_bundle
from unrender.audio.visualization.dispatcher import (
    DEFAULT_RENDERERS,
    render_audio_analysis,
)
from unrender.audio.visualization.html import render_standalone_html
from unrender.audio.visualization.images import (
    render_mix_territory,
    render_program_poster,
    render_structure_map,
    render_timeline_atlas,
)
from unrender.audio.visualization.mapping import (
    MAPPING_VERSION,
    VisualMapping,
    VisualRule,
    load_visual_mapping,
)
from unrender.audio.visualization.overlay import (
    encode_overlay_mov,
    render_overlay_sequence,
)
from unrender.audio.visualization.registry import (
    AudioAnalysisRenderer,
    RenderContext,
    RendererRegistry,
    available_renderers,
    get_renderer,
    register_renderer,
    renderer_registry,
)

render_self_similarity_map = render_structure_map
render_whole_program_poster = render_program_poster
render_timeline_overlay_sequence = render_overlay_sequence

__all__ = [
    "DEFAULT_RENDERERS",
    "MAPPING_VERSION",
    "AnalysisBundle",
    "AudioAnalysisRenderer",
    "RenderContext",
    "RendererRegistry",
    "VisualMapping",
    "VisualRule",
    "available_renderers",
    "encode_overlay_mov",
    "get_renderer",
    "load_analysis_bundle",
    "load_visual_mapping",
    "register_renderer",
    "render_audio_analysis",
    "render_mix_territory",
    "render_overlay_sequence",
    "render_program_poster",
    "render_self_similarity_map",
    "render_standalone_html",
    "render_structure_map",
    "render_timeline_atlas",
    "render_timeline_overlay_sequence",
    "render_whole_program_poster",
    "renderer_registry",
]
