"""Shared primitives used across unrender pipeline stages.

This package holds the low-level building blocks (JSON/CSV IO, timecode math,
subprocess runner, ffmpeg/ffprobe wrappers, audio/video probing, image-sequence
detection, name slugging, and input fingerprints) that the pipeline stages
build on. Nothing here imports a pipeline stage, so it stays free of cycles.

The cheap, dependency-light helpers are re-exported here for convenience;
numpy-backed modules (``audio``, ``ffmpeg``, ``video``) must be imported from
their submodules to keep import time low.
"""

from __future__ import annotations

from unrender.lib.csv_io import write_rows, write_rows_atomic
from unrender.lib.fingerprints import (
    changed_inputs,
    input_fingerprints,
    warn_if_inputs_changed,
)
from unrender.lib.json_io import read_json, write_json
from unrender.lib.names import safe_name
from unrender.lib.timecode import (
    frames_to_seconds,
    frames_to_smpte,
    parse_time_value,
    seconds_to_frames,
    smpte_to_frames,
    smpte_to_seconds,
)

__all__ = [
    "changed_inputs",
    "frames_to_seconds",
    "frames_to_smpte",
    "input_fingerprints",
    "parse_time_value",
    "read_json",
    "safe_name",
    "seconds_to_frames",
    "smpte_to_frames",
    "smpte_to_seconds",
    "warn_if_inputs_changed",
    "write_json",
    "write_rows",
    "write_rows_atomic",
]
