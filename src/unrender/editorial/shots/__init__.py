from __future__ import annotations

from typing import Any

from unrender.editorial.shots.stems import (
    StemSource,
    load_stem_map,
    load_stem_sources,
    source_group_for_stem,
)

__all__ = [
    "DEFAULT_PROXY_WIDTH",
    "DetectOptions",
    "StemSource",
    "build_shots",
    "detect_cut_frames",
    "load_stem_map",
    "load_stem_sources",
    "master_usable_as_proxy",
    "resolve_detection_input",
    "source_group_for_stem",
]


def __getattr__(name: str) -> Any:
    if name in {"DetectOptions", "build_shots", "detect_cut_frames", "resolve_detection_input"}:
        from unrender.editorial.shots.detect import (
            DetectOptions,
            build_shots,
            detect_cut_frames,
            resolve_detection_input,
        )

        return {
            "DetectOptions": DetectOptions,
            "build_shots": build_shots,
            "detect_cut_frames": detect_cut_frames,
            "resolve_detection_input": resolve_detection_input,
        }[name]
    if name in {"DEFAULT_PROXY_WIDTH", "master_usable_as_proxy"}:
        from unrender.editorial.shots.proxy import DEFAULT_PROXY_WIDTH, master_usable_as_proxy

        return {
            "DEFAULT_PROXY_WIDTH": DEFAULT_PROXY_WIDTH,
            "master_usable_as_proxy": master_usable_as_proxy,
        }[name]
    raise AttributeError(name)
