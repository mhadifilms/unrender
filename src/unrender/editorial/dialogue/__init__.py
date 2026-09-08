from __future__ import annotations

from unrender.editorial.dialogue.clusters import resolve_voice_clips_clustered
from unrender.editorial.dialogue.lines import (
    build_dialogue_lines,
    build_voice_clips,
    import_transcript_lines,
    load_stem_sources_for_project,
    map_dialogue_to_shots,
    materialize_dialogue_line_stems,
    refine_cut_window,
    resolve_voice_clips,
    resolve_voice_clips_from_voice_db,
    transcribe_dialogue_lines,
)
from unrender.editorial.dialogue.resolvers import (
    CLIP_RESOLVERS,
    ClipResolverSpec,
    get_clip_resolver,
    select_clip_resolver,
)

__all__ = [
    "CLIP_RESOLVERS",
    "ClipResolverSpec",
    "build_dialogue_lines",
    "build_voice_clips",
    "get_clip_resolver",
    "import_transcript_lines",
    "load_stem_sources_for_project",
    "map_dialogue_to_shots",
    "materialize_dialogue_line_stems",
    "refine_cut_window",
    "resolve_voice_clips",
    "resolve_voice_clips_clustered",
    "resolve_voice_clips_from_voice_db",
    "select_clip_resolver",
    "transcribe_dialogue_lines",
]
