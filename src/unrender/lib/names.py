from __future__ import annotations

import re

# An anonymous audio group, as written by every producer of speaker stems and read
# back by every consumer of them. Zero padding is part of the contract: speaker_7
# and speaker_07 name the same group, and a mapping keyed on the unpadded form
# silently misses stems written with the padded one.
SOURCE_GROUP_RE = re.compile(r"speaker[_-]?(\d+)", flags=re.IGNORECASE)


def safe_name(value: object, *, fallback: str = "item") -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return safe.strip("._") or fallback


def source_group_label(index: int) -> str:
    """The canonical label for the ``index``-th anonymous group, counting from one."""
    return f"speaker_{int(index):02d}"


def source_group_from_text(value: object) -> str:
    """The group a filename or label refers to, or ``""`` when it names no group.

    Producers write ``PROG_speaker_07_stem.wav``, ``speaker-7``, and bare
    ``speaker_07``; the same search-and-repad was hand-rolled at five call sites,
    which is one contract with five chances to drift.
    """

    match = SOURCE_GROUP_RE.search(str(value))
    return source_group_label(int(match.group(1))) if match else ""
