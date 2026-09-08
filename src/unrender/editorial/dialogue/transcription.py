from __future__ import annotations

import re
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np

from unrender.editorial.shots.stems import StemSource, normalize_stem_sources
from unrender.lib.audio import WavReader

# Matches SpeakerIdentitySettings.change_confidence_floor for manifests that
# predate publishing the floor in their settings block.
DEFAULT_CHANGE_CONFIDENCE_FLOOR = 0.5

DX_RESIDUAL_REVIEW_REASON = "dx_residual_without_identity_evidence"

# "Fragments cover most of the line": identity fragments must overlap at least
# this share of a line's word time before a DX-only residual word inside the
# line downgrades from a blocking conflict to an advisory.
RESIDUAL_ADVISORY_COVERAGE_RATIO = 0.5


@dataclass(frozen=True)
class SpeechIsland:
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class StemActivity:
    source_group: str
    path: Path
    levels_db: np.ndarray
    hop_sec: float


def detect_speech_islands(
    path: Path,
    *,
    threshold_db: float = -45.0,
    hop_sec: float = 0.02,
    min_silence_sec: float = 0.25,
    min_island_sec: float = 0.08,
) -> list[SpeechIsland]:
    """Detect contiguous speech-energy regions in a (separated) DX stem.

    Cut windows snap to these islands so line boundaries do not clip breaths
    or word onsets that word-level timestamps miss. Separated dialogue stems
    are near-silent between lines, so a simple RMS gate is reliable here.
    Returns an empty list for unreadable or non-WAV audio, in which case
    callers fall back to plain handle padding.
    """
    levels, hop_dur = _wav_rms_levels(path, hop_sec=hop_sec)
    if levels.size == 0:
        return []
    return _levels_to_islands(
        levels,
        hop_dur=hop_dur,
        threshold_db=threshold_db,
        min_silence_sec=min_silence_sec,
        min_island_sec=min_island_sec,
    )


def annotate_words_with_stem_activity(
    words: list[dict[str, Any]],
    stems: Sequence[Path | StemSource],
    *,
    min_activity_db: float = -50.0,
    min_margin_db: float = 1.0,
    hop_sec: float = 0.02,
) -> list[dict[str, Any]]:
    """Tag ASR words with the locally dominant separated-stem channel.

    AudioShake speaker stems are not stable identities over a whole project, but
    they are useful local separation channels. The identity clustering step
    still happens later; this only prevents mixed-DX ASR gaps from merging
    adjacent turns into long multi-speaker blobs.
    """
    sources = normalize_stem_sources(stems)
    tracks: list[StemActivity] = []
    for source in sources:
        levels, actual_hop = _wav_rms_levels(source.path, hop_sec=hop_sec)
        if levels.size:
            tracks.append(
                StemActivity(
                    source_group=source.source_group,
                    path=source.path,
                    levels_db=levels,
                    hop_sec=actual_hop,
                )
            )
    if not tracks:
        return [dict(word) for word in words]

    annotated: list[dict[str, Any]] = []
    for word in words:
        updated = dict(word)
        if word.get("start") is None or word.get("end") is None:
            annotated.append(updated)
            continue
        start = float(word["start"])
        end = float(word["end"])
        scores = [
            (track.source_group, _activity_score(track, start_sec=start, end_sec=end))
            for track in tracks
        ]
        scores.sort(key=lambda item: item[1], reverse=True)
        best_group, best_score = scores[0]
        second_score = scores[1][1] if len(scores) > 1 else float("-inf")
        if best_score >= min_activity_db and best_score - second_score >= min_margin_db:
            updated["speaker"] = best_group
            updated["source_group"] = best_group
            updated["source_activity_db"] = round(best_score, 3)
        annotated.append(updated)
    return annotated


def _wav_rms_levels(path: Path, *, hop_sec: float) -> tuple[np.ndarray, float]:
    try:
        with WavReader(path) as reader:
            rate = reader.sample_rate
            channels = reader.channels
            if rate <= 0:
                return np.empty(0, dtype=np.float64), hop_sec
            hop_frames = max(1, int(rate * hop_sec))
            hop_dur = hop_frames / float(rate)
            samples_per_hop = hop_frames * max(1, channels)
            levels_parts: list[np.ndarray] = []
            carry = np.empty(0, dtype=np.float32)
            # Read ~512 hops at a time and reduce per-hop RMS vectorized;
            # per-hop reads cost minutes on feature-length stems.
            block_frames = hop_frames * 512
            while True:
                values = reader.read(block_frames)
                if not values.size:
                    break
                if carry.size:
                    values = np.concatenate([carry, values])
                full = (values.size // samples_per_hop) * samples_per_hop
                carry = values[full:]
                if full:
                    hops = values[:full].reshape(-1, samples_per_hop)
                    levels_parts.append(np.mean(np.square(hops), axis=1))
            if carry.size:
                levels_parts.append(
                    np.asarray([float(np.mean(np.square(carry)))], dtype=np.float64)
                )
    except (wave.Error, OSError, EOFError, ValueError):
        return np.empty(0, dtype=np.float64), hop_sec

    if not levels_parts:
        return np.empty(0, dtype=np.float64), hop_sec
    mean_squares = np.concatenate(levels_parts)
    levels = np.full(mean_squares.shape, -np.inf)
    positive = mean_squares > 0
    levels[positive] = 10.0 * np.log10(mean_squares[positive])
    return levels, hop_dur


def _levels_to_islands(
    levels: np.ndarray,
    *,
    hop_dur: float,
    threshold_db: float,
    min_silence_sec: float,
    min_island_sec: float,
) -> list[SpeechIsland]:
    islands: list[SpeechIsland] = []
    start: float | None = None
    voiced_end = 0.0

    for index, level in enumerate(levels):
        hop_start = index * hop_dur
        if level > threshold_db:
            if start is None:
                start = hop_start
            voiced_end = hop_start + hop_dur
        elif start is not None and hop_start - voiced_end >= min_silence_sec:
            if voiced_end - start >= min_island_sec:
                islands.append(SpeechIsland(start_sec=start, end_sec=voiced_end))
            start = None
    if start is not None and voiced_end - start >= min_island_sec:
        islands.append(SpeechIsland(start_sec=start, end_sec=voiced_end))
    return islands


def _activity_score(track: StemActivity, *, start_sec: float, end_sec: float) -> float:
    start = max(0, int(np.floor(start_sec / track.hop_sec)))
    end = max(start + 1, int(np.ceil(end_sec / track.hop_sec)))
    window = track.levels_db[start:end]
    if window.size == 0:
        return float("-inf")
    finite = window[np.isfinite(window)]
    if finite.size == 0:
        return float("-inf")
    # A high percentile is stable against alignment padding while still
    # reflecting the locally dominant separation channel for the word.
    return float(np.percentile(finite, 90))


def _group_words(
    words: list[dict[str, Any]],
    *,
    max_gap_sec: float,
    parallel_sources: bool = False,
) -> list[dict[str, Any]]:
    if parallel_sources:
        buckets: dict[str, list[dict[str, Any]]] = {}
        for word in words:
            source = str(word.get("source_group") or word.get("speaker") or "")
            buckets.setdefault(source, []).append(word)
        parallel_groups = [
            group
            for bucket in buckets.values()
            for group in _group_words(bucket, max_gap_sec=max_gap_sec)
        ]
        return sorted(
            parallel_groups,
            key=lambda group: (
                float(group["start_sec"]),
                float(group["end_sec"]),
                str(group.get("source_group") or ""),
            ),
        )

    valid = [
        word
        for word in words
        if word.get("start") is not None
        and word.get("end") is not None
        and str(word.get("word") or word.get("text") or "").strip()
    ]
    valid.sort(key=lambda word: float(word["start"]))
    groups: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for word in valid:
        start = float(word["start"])
        end = float(word["end"])
        text = str(word.get("word") or word.get("text") or "").strip()
        speaker = str(
            word.get("source_group") or word.get("speaker") or word.get("diarized_speaker") or ""
        ).strip()
        transcription_source = str(word.get("transcription_source") or "").strip()
        turn_speaker = str(word.get("diarized_turn_speaker") or "").strip()
        if current is None:
            current = _new_word_group(
                word,
                start=start,
                end=end,
                text=text,
                speaker=speaker,
                transcription_source=transcription_source,
            )
            continue
        gap = start - float(current["end_sec"])
        speaker_changed = bool(
            speaker and current.get("diarized_speaker") and speaker != current["diarized_speaker"]
        )
        turn_speaker_changed = bool(
            turn_speaker
            and current.get("diarized_turn_speaker")
            and turn_speaker != current["diarized_turn_speaker"]
        )
        if gap > max_gap_sec or speaker_changed or turn_speaker_changed:
            groups.append(current)
            current = _new_word_group(
                word,
                start=start,
                end=end,
                text=text,
                speaker=speaker,
                transcription_source=transcription_source,
            )
        else:
            current["end_sec"] = end
            current["text"] = f"{current['text']} {text}".strip()
            if not current.get("diarized_speaker"):
                current["diarized_speaker"] = speaker
            if not current.get("source_group"):
                current["source_group"] = speaker
            if not current.get("diarized_turn_speaker"):
                current["diarized_turn_speaker"] = turn_speaker
            if (
                transcription_source
                and current.get("transcription_source")
                and transcription_source != current["transcription_source"]
            ):
                current["transcription_source"] = "mixed"
            elif not current.get("transcription_source"):
                current["transcription_source"] = transcription_source
            current["review_required"] = bool(
                current.get("review_required") or word.get("review_required")
            )
    if current is not None:
        groups.append(current)
    return groups


def _new_word_group(
    word: dict[str, Any],
    *,
    start: float,
    end: float,
    text: str,
    speaker: str,
    transcription_source: str,
) -> dict[str, Any]:
    group: dict[str, Any] = {
        "start_sec": start,
        "end_sec": end,
        "text": text,
        "diarized_speaker": speaker,
        "source_group": speaker,
        "transcription_source": transcription_source,
        "diarized_turn_speaker": str(word.get("diarized_turn_speaker") or ""),
        "review_required": bool(word.get("review_required")),
    }
    return group


def transcribe_stem_words(
    stems: Sequence[Path | StemSource],
    *,
    dx_words: Sequence[dict[str, Any]] = (),
    whisper_model: str = "small",
    device: str = "cpu",
    compute_type: str = "float32",
    language: str | None = None,
    meta: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """ASR each stem independently and merge it with unmatched DX words."""
    from unrender.audio import asr

    sources = normalize_stem_sources(stems)
    stem_words: list[dict[str, Any]] = []
    stem_runs: list[dict[str, Any]] = []
    for source in sources:
        words, asr_meta = asr.transcribe_word_aligned(
            source.path,
            model_name=whisper_model,
            device=device,
            compute_type=compute_type,
            language=language,
        )
        for word in words:
            stamped = dict(word)
            stamped["speaker"] = source.source_group
            stamped["source_group"] = source.source_group
            stamped["transcription_source"] = "stem"
            stamped["review_required"] = True
            stem_words.append(stamped)
        stem_runs.append(
            {
                "path": str(source.path),
                "source_group": source.source_group,
                "language": asr_meta.get("language"),
                "aligned": bool(asr_meta.get("aligned")),
                "word_count": len(words),
            }
        )

    raw_stem_word_count = len(stem_words)
    merged = reconcile_stem_words(stem_words, dx_words)
    if meta is not None:
        meta.update(
            {
                "stem_count": len(sources),
                "stem_word_count": raw_stem_word_count,
                "cross_source_duplicate_word_count": raw_stem_word_count - len(stem_words),
                "dx_word_count": len(dx_words),
                "dx_residual_word_count": sum(
                    word.get("transcription_source") == "dx_residual" for word in merged
                ),
                "dx_residual_review_count": sum(
                    bool(word.get("review_required"))
                    for word in merged
                    if word.get("transcription_source") == "dx_residual"
                ),
                "stems": stem_runs,
            }
        )
    return merged


def reconcile_stem_words(
    stem_words: Sequence[dict[str, Any]],
    dx_words: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep all stem words and add only DX words not represented temporally/textually."""
    authoritative = [dict(word) for word in stem_words]
    stem_match_counts: dict[int, int] = {}
    residuals: list[dict[str, Any]] = []
    for dx_word in sorted(dx_words, key=_word_sort_key):
        candidates = [
            (score, index)
            for index, stem_word in enumerate(authoritative)
            if stem_match_counts.get(index, 0) < _word_match_capacity(stem_word)
            if (score := _word_match_score(stem_word, dx_word)) is not None
        ]
        if candidates:
            _, best_index = max(candidates)
            stem_match_counts[best_index] = stem_match_counts.get(best_index, 0) + 1
            continue
        residual = dict(dx_word)
        source_group, ambiguous = _nearest_residual_source(dx_word, authoritative)
        residual["speaker"] = source_group
        residual["source_group"] = source_group
        residual["transcription_source"] = "dx_residual"
        residual["residual_assignment"] = "nearest_stem" if source_group else "unresolved"
        # Temporal assignment is a useful suggestion, not an identity proof.
        # Every DX-only residual must be reviewed before synthesis can ship it.
        residual["review_required"] = True
        residual["assignment_ambiguous"] = ambiguous
        residuals.append(residual)
    return sorted([*authoritative, *residuals], key=_word_sort_key)


def _word_match_score(stem_word: dict[str, Any], dx_word: dict[str, Any]) -> float | None:
    if any(word.get(key) is None for word in (stem_word, dx_word) for key in ("start", "end")):
        return None
    stem_text = _normalized_word_text(stem_word)
    dx_text = _normalized_word_text(dx_word)
    if not stem_text or not dx_text:
        return None

    stem_start, stem_end = float(stem_word["start"]), float(stem_word["end"])
    dx_start, dx_end = float(dx_word["start"]), float(dx_word["end"])
    overlap = max(0.0, min(stem_end, dx_end) - max(stem_start, dx_start))
    stem_duration = max(0.001, stem_end - stem_start)
    dx_duration = max(0.001, dx_end - dx_start)
    center_distance = abs((stem_start + stem_end - dx_start - dx_end) / 2.0)
    if overlap <= 0.0 and center_distance > 0.25:
        return None

    if stem_text == dx_text:
        text_score = 1.0
    else:
        stem_tokens = set(stem_text.split())
        dx_tokens = set(dx_text.split())
        token_overlap = len(stem_tokens & dx_tokens) / max(1, min(len(stem_tokens), len(dx_tokens)))
        text_score = max(SequenceMatcher(None, stem_text, dx_text).ratio(), token_overlap)
    if text_score < 0.72:
        return None

    temporal_score = overlap / min(stem_duration, dx_duration)
    if overlap <= 0.0:
        temporal_score = max(0.0, 1.0 - center_distance / 0.25)
    return text_score * 2.0 + min(1.0, temporal_score)


def _nearest_residual_source(
    dx_word: dict[str, Any],
    stem_words: Sequence[dict[str, Any]],
    *,
    max_distance_sec: float = 0.75,
    ambiguity_sec: float = 0.05,
) -> tuple[str, bool]:
    if dx_word.get("start") is None or dx_word.get("end") is None:
        return "", True
    start = float(dx_word["start"])
    end = float(dx_word["end"])
    by_source: dict[str, float] = {}
    for stem_word in stem_words:
        source = str(stem_word.get("source_group") or "").strip()
        if not source or stem_word.get("start") is None or stem_word.get("end") is None:
            continue
        stem_start = float(stem_word["start"])
        stem_end = float(stem_word["end"])
        distance = max(stem_start - end, start - stem_end, 0.0)
        by_source[source] = min(distance, by_source.get(source, float("inf")))
    ranked = sorted((distance, source) for source, distance in by_source.items())
    if not ranked or ranked[0][0] > max_distance_sec:
        return "", True
    if len(ranked) > 1 and ranked[1][0] - ranked[0][0] <= ambiguity_sec:
        return "", True
    return ranked[0][1], False


def _normalized_word_text(word: dict[str, Any]) -> str:
    text = str(word.get("word") or word.get("text") or "").casefold().replace("_", " ")
    return " ".join(re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).split())


def _word_match_capacity(word: dict[str, Any]) -> int:
    return max(1, len(_normalized_word_text(word).split()))


def _word_sort_key(word: dict[str, Any]) -> tuple[float, float, str]:
    start = float(word["start"]) if word.get("start") is not None else float("inf")
    end = float(word["end"]) if word.get("end") is not None else start
    return start, end, str(word.get("source_group") or "")
