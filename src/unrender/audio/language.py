"""Check spoken language against a project expectation using sampled ASR windows."""

from __future__ import annotations

import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from unrender.audio.separation.input_audio import probe_audio_streams
from unrender.lib.audio import measure_activity
from unrender.lib.ffmpeg import extract_audio_window, probe_media_duration_sec
from unrender.lib.languages import to_whisper_code

PROBE_SAMPLE_RATE = 16000
DEFAULT_WINDOW_SEC = 30.0
DEFAULT_WINDOWS = 3
# Below this share of agreeing windows a detection is treated as undecided
# rather than acted on, so one confused window cannot fail a run.
MIN_LANGUAGE_CONFIDENCE = 0.6

# A window with less audible content than this is skipped as silence rather
# than spent on detection. Separated speaker stems are mostly silent, so
# sampling retries elsewhere instead of landing in the gaps between lines.
MIN_VOICED_RATIO = 0.05
SILENCE_RETRY_ROUNDS = 4

# Detections already made in this process, keyed by file content and sampling.
_DETECTIONS: dict[
    tuple[object, ...], tuple[str | None, float, tuple[tuple[float, str | None], ...]]
] = {}


@dataclass(frozen=True)
class LanguageProbe:
    """What language a file is spoken in, and whether that is what was expected.

    ``detected`` and ``expected`` are ISO codes. ``detected`` is ``None`` when
    no sampled window contained recognizable speech. ``confidence`` is the
    share of the windows asked for that agreed on ``detected``, so windows lost
    to silence or a failed detection count against it.
    """

    path: Path
    detected: str | None
    expected: str | None
    confidence: float
    windows: tuple[tuple[float, str | None], ...]

    @property
    def matches(self) -> bool | None:
        """True/False when the comparison is decidable, ``None`` when it is not."""
        if self.expected is None or self.detected is None:
            return None
        return self.detected == self.expected

    @property
    def issue(self) -> str | None:
        """A human-readable problem statement, or ``None`` when nothing is wrong.

        A low-confidence disagreement is not reported: a single confused window
        should not fail a run.
        """
        if self.matches is not False or self.confidence < MIN_LANGUAGE_CONFIDENCE:
            return None
        return (
            f"{self.path} is spoken in {self.detected!r} but {self.expected!r} was expected "
            f"({self.confidence:.0%} of sampled windows agree). Check the selected source audio "
            f"and the expected language in the project configuration."
        )


def probe_audio_language(
    path: Path,
    *,
    expected: str | None = None,
    window_sec: float = DEFAULT_WINDOW_SEC,
    windows: int = DEFAULT_WINDOWS,
    model_name: str = "small",
    device: str = "cpu",
    compute_type: str = "float32",
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> LanguageProbe:
    """Detect the language spoken in ``path`` from a few windows of it.

    Windows are cut by seeking rather than by decoding the whole file, because
    a feature program is 90 minutes long and this only needs 90 seconds of it.
    Requires the optional transcription extra, since detection runs through the
    same WhisperX path the rest of the codebase transcribes with.

    Suited to dense audio: a full mix, a program master, a dialogue stem. A
    single separated speaker stem can be 98% silence, and sampling will
    usually miss the speech and report ``detected=None`` rather than guess.
    """
    if window_sec <= 0 or windows < 1:
        raise ValueError("window_sec must be positive and windows must be at least 1")

    expected_code = to_whisper_code(expected)
    # What a file is spoken in does not depend on who is asking, and several
    # stages ask about the same 90-minute file in one run. ``expected`` is not
    # part of the key because it changes the verdict, not the detection.
    key = _cache_key(path, window_sec=window_sec, windows=windows, model_name=model_name)
    cached = _DETECTIONS.get(key) if key is not None else None
    if cached is not None:
        detected, confidence, sampled = cached
        return LanguageProbe(
            path=path,
            detected=detected,
            expected=expected_code,
            confidence=confidence,
            windows=sampled,
        )

    # A multi-track master keeps dialogue on its own discrete track, and
    # ffmpeg's default pick can be a channel carrying only music, so every
    # stream is merged.
    try:
        stream_count = len(probe_audio_streams(path, ffprobe=ffprobe))
    except ValueError as exc:
        # Callers treat this as "cannot check", so it must not surface as the
        # ValueError that means "refuse this run".
        raise RuntimeError(f"no audio to detect a language from: {path}") from exc
    duration_sec = probe_media_duration_sec(path, ffprobe=ffprobe)

    results: list[tuple[float, str | None]] = []
    with tempfile.TemporaryDirectory(prefix="unrender-language-") as directory:
        excerpt = Path(directory) / "excerpt.wav"
        for start_sec in _candidate_starts(duration_sec, window_sec=window_sec, count=windows):
            try:
                extract_audio_window(
                    path,
                    excerpt,
                    ffmpeg=ffmpeg,
                    start_sec=start_sec,
                    duration_sec=window_sec,
                    sample_rate=PROBE_SAMPLE_RATE,
                    audio_streams=stream_count,
                )
            except RuntimeError:
                # One unseekable offset should cost one offset, not the run.
                continue
            if measure_activity(excerpt).voiced_ratio < MIN_VOICED_RATIO:
                continue
            results.append(
                (
                    start_sec,
                    _detect_window(
                        excerpt,
                        model_name=model_name,
                        device=device,
                        compute_type=compute_type,
                    ),
                )
            )
            if len(results) >= windows:
                break

    decided = Counter(code for _, code in results if code)
    detected, agreeing = decided.most_common(1)[0] if decided else (None, 0)
    # Out of the windows asked for, not the ones that happened to decide, so a
    # lone window agreeing with itself cannot reach the threshold.
    if key is not None:
        _DETECTIONS[key] = (detected, agreeing / windows, tuple(results))
    return LanguageProbe(
        path=path,
        detected=detected,
        expected=expected_code,
        confidence=agreeing / windows,
        windows=tuple(results),
    )


def _detect_window(
    excerpt: Path,
    *,
    model_name: str,
    device: str,
    compute_type: str,
) -> str | None:
    from unrender.audio import asr

    _, metadata = asr.transcribe_word_aligned(
        excerpt,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        language=None,
    )
    # Deliberately the raw detection, not ``language``: that one falls back to
    # English when nothing was recognized, which is indistinguishable from
    # confidently detecting English speech.
    return to_whisper_code(metadata.get("detected_language"))


def _cache_key(
    path: Path, *, window_sec: float, windows: int, model_name: str
) -> tuple[object, ...] | None:
    """Identify a file by content, or ``None`` when it cannot be stat'd."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return (path.resolve(), stat.st_mtime_ns, stat.st_size, window_sec, windows, model_name)


def _candidate_starts(
    duration_sec: float,
    *,
    window_sec: float,
    count: int,
    rounds: int = SILENCE_RETRY_ROUNDS,
) -> list[float]:
    """Offsets to try, spread across the program and refined on each round.

    A separated speaker stem is mostly silence, so more offsets are offered
    than are needed and the caller skips the quiet ones. Every round spans the
    whole program rather than continuing where the last one stopped, so the
    first samples taken are always well separated.
    """
    span = duration_sec - window_sec
    if span <= 0:
        return [0.0]
    starts: list[float] = []
    for current_round in range(rounds):
        for index in range(count):
            start = span * (index + (current_round + 0.5) / rounds) / count
            if start not in starts:
                starts.append(start)
    return starts
