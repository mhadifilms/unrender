from __future__ import annotations

import shutil
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from unrender.analysis.identity.voice import match_voice_db
from unrender.editorial.dialogue.plan_writers import (
    _write_clip_plan_csv,
    _write_dialogue_plan_csv,
    _write_shot_dialogue_csv,
    _write_voice_clips_csv,
)
from unrender.editorial.dialogue.transcript import parse_transcript
from unrender.editorial.dialogue.transcription import (
    SpeechIsland,
    _group_words,
    annotate_words_with_stem_activity,
    detect_speech_islands,
    transcribe_stem_words,
)
from unrender.editorial.shots.stems import (
    StemSource,
    load_stem_map,
    load_stem_sources,
    normalize_stem_sources,
    source_group_for_stem,
    write_plan_csv,
)
from unrender.lib.audio import measure_activity
from unrender.lib.ffmpeg import run_ffmpeg_timeline_mix
from unrender.lib.fingerprints import input_fingerprints, warn_if_inputs_changed
from unrender.lib.names import safe_name
from unrender.manifests import (
    DialogueLineRecord,
    ShotRecord,
    VoiceInput,
    read_json,
    write_dialogue_lines_csv,
    write_json,
)
from unrender.speakers import speaker_key


def transcribe_dialogue_lines(
    *,
    audio_path: Path,
    output_json: Path,
    output_csv: Path,
    shots: list[ShotRecord] | None = None,
    whisper_model: str = "small",
    device: str = "cpu",
    compute_type: str = "float32",
    language: str | None = None,
    max_gap_sec: float = 0.5,
    handle_sec: float = 0.18,
    source_stems: Sequence[Path | StemSource] | None = None,
    per_stem: bool = True,
    source_activity_threshold_db: float = -50.0,
    source_margin_db: float = 1.0,
    force: bool = False,
    words: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    inputs: dict[str, Any] = {"audio": audio_path}
    stem_sources = normalize_stem_sources(source_stems) if source_stems else []
    if stem_sources:
        inputs["source_stems"] = [source.path for source in stem_sources]
    if output_json.exists() and not force:
        data = read_json(output_json)
        changed = warn_if_inputs_changed(data, inputs, output_json)
        if not changed:
            print(f"Dialogue lines already exist, skipping: {output_json}", flush=True)
            return list(data.get("lines") or [])
        print("Dialogue inputs changed; regenerating line transcription", flush=True)
    if not audio_path.exists() and words is None:
        raise FileNotFoundError(f"DX stem not found: {audio_path}")

    if words is None:
        from unrender.audio import asr

    transcription_metadata: dict[str, Any] = {}
    if words is not None:
        raw_words = words
        transcription_mode = "injected"
        transcription_metadata["word_count"] = len(words)
    elif stem_sources and per_stem:
        dx_words, dx_meta = asr.transcribe_word_aligned(
            audio_path,
            model_name=whisper_model,
            device=device,
            compute_type=compute_type,
            language=language,
        )
        raw_words = transcribe_stem_words(
            stem_sources,
            dx_words=dx_words,
            whisper_model=whisper_model,
            device=device,
            compute_type=compute_type,
            language=language,
            meta=transcription_metadata,
        )
        transcription_mode = "per_stem"
        transcription_metadata["dx"] = {
            **dx_meta,
            "path": str(audio_path),
            "word_count": len(dx_words),
        }
    else:
        raw_words, dx_meta = asr.transcribe_word_aligned(
            audio_path,
            model_name=whisper_model,
            device=device,
            compute_type=compute_type,
            language=language,
        )
        transcription_mode = "dx"
        transcription_metadata["dx"] = {
            **dx_meta,
            "path": str(audio_path),
            "word_count": len(raw_words),
        }

    uses_stem_activity = bool(stem_sources and transcription_mode != "per_stem")
    if uses_stem_activity:
        raw_words = annotate_words_with_stem_activity(
            raw_words,
            stem_sources,
            min_activity_db=source_activity_threshold_db,
            min_margin_db=source_margin_db,
        )
    islands_by_source = {
        source.source_group: detect_speech_islands(source.path) for source in stem_sources
    }
    islands = detect_speech_islands(audio_path) if audio_path.exists() and not stem_sources else []
    lines = build_dialogue_lines(
        raw_words,
        max_gap_sec=max_gap_sec,
        handle_sec=handle_sec,
        shots=shots or [],
        islands=islands,
        islands_by_source=islands_by_source,
        parallel_sources=transcription_mode == "per_stem",
    )
    write_json(
        output_json,
        {
            "version": "1.0",
            "source_audio": str(audio_path),
            "whisper_model": whisper_model,
            "language": language or "",
            "transcription_mode": transcription_mode,
            "transcription_metadata": transcription_metadata,
            "max_gap_sec": max_gap_sec,
            "handle_sec": handle_sec,
            "source_activity_threshold_db": (
                source_activity_threshold_db if uses_stem_activity else None
            ),
            "source_margin_db": source_margin_db if uses_stem_activity else None,
            "inputs": input_fingerprints(inputs),
            "lines": lines,
        },
    )
    write_dialogue_lines_csv(output_csv, lines)
    print(f"Dialogue lines saved: {output_json} ({len(lines)} line(s))", flush=True)
    return lines


def import_transcript_lines(
    *,
    transcript_path: Path,
    output_json: Path,
    output_csv: Path,
    fps: float = 24.0,
    default_duration_sec: float = 3.0,
    min_duration_sec: float = 0.3,
    handle_sec: float = 0.18,
    force: bool = False,
) -> list[dict[str, Any]]:
    inputs = {"transcript": transcript_path}
    if output_json.exists() and not force:
        print(f"Dialogue lines already exist, skipping: {output_json}", flush=True)
        data = read_json(output_json)
        warn_if_inputs_changed(data, inputs, output_json)
        return list(data.get("lines") or [])
    lines = parse_transcript(
        transcript_path,
        fps=fps,
        default_duration_sec=default_duration_sec,
        min_duration_sec=min_duration_sec,
        handle_sec=handle_sec,
    )
    write_json(
        output_json,
        {
            "version": "1.0",
            "source": "transcript",
            "transcript": str(transcript_path),
            "fps": fps,
            "default_duration_sec": default_duration_sec,
            "min_duration_sec": min_duration_sec,
            "handle_sec": handle_sec,
            "inputs": input_fingerprints(inputs),
            "lines": lines,
        },
    )
    write_dialogue_lines_csv(output_csv, lines)
    print(f"Dialogue lines imported: {output_json} ({len(lines)} line(s))", flush=True)
    return lines


def build_dialogue_lines(
    words: list[dict[str, Any]],
    *,
    max_gap_sec: float = 0.5,
    handle_sec: float = 0.18,
    shots: list[ShotRecord] | None = None,
    islands: list[SpeechIsland] | None = None,
    islands_by_source: dict[str, list[SpeechIsland]] | None = None,
    parallel_sources: bool = False,
    identity_provenance: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    grouped = _group_words(
        words,
        max_gap_sec=max_gap_sec,
        parallel_sources=parallel_sources,
    )
    lines: list[dict[str, Any]] = []
    for index, group in enumerate(grouped, 1):
        start_sec = float(group["start_sec"])
        end_sec = float(group["end_sec"])
        source_group = str(group.get("source_group") or group.get("diarized_speaker") or "")
        line_islands = (islands_by_source or {}).get(source_group, islands or [])
        cut_start, cut_end = refine_cut_window(
            start_sec,
            end_sec,
            handle_sec=handle_sec,
            islands=line_islands,
        )
        line_id = f"DL_{index:06d}"
        line = {
            "line_id": line_id,
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "cut_start_sec": round(cut_start, 3),
            "cut_end_sec": round(cut_end, 3),
            "diarized_speaker": group.get("diarized_speaker", ""),
            "diarized_turn_speaker": group.get("diarized_turn_speaker", ""),
            "source_group": source_group,
            "speaker": "",
            "text": group["text"],
            "shot_overlaps": _shot_overlaps(start_sec, end_sec, shots or []),
        }
        if group.get("transcription_source"):
            line["transcription_source"] = group["transcription_source"]
        if group.get("review_required"):
            line["review_required"] = True
            line["status"] = "needs_speaker_review"
        lines.append(line)
    return lines


def refine_cut_window(
    start_sec: float,
    end_sec: float,
    *,
    handle_sec: float,
    islands: list[SpeechIsland],
    max_expand_sec: float = 0.6,
) -> tuple[float, float]:
    """Snap the padded cut window outward to overlapping speech-island edges.

    Expansion is capped at ``max_expand_sec`` beyond the word window so that an
    island shared with an adjacent line (e.g. a fast speaker change) cannot
    pull a whole neighboring line into this clip.
    """
    cut_start = max(0.0, start_sec - handle_sec)
    cut_end = max(cut_start, end_sec + handle_sec)
    overlaps = [
        island for island in islands if island.end_sec > start_sec and island.start_sec < end_sec
    ]
    if overlaps:
        island_start = min(island.start_sec for island in overlaps)
        island_end = max(island.end_sec for island in overlaps)
        cut_start = max(0.0, min(cut_start, max(island_start, start_sec - max_expand_sec)))
        cut_end = max(cut_end, min(island_end, end_sec + max_expand_sec))
    return cut_start, cut_end


def build_voice_clips(
    *,
    lines: list[DialogueLineRecord],
    stems: Sequence[Path | StemSource],
    output_dir: Path,
    output_json: Path,
    output_csv: Path,
    clip_type: str = "dialogue_line",
    silence_threshold_db: float = -60.0,
    min_rms_dbfs: float = -55.0,
    activity_threshold_db: float = -50.0,
    min_voiced_ratio: float = 0.15,
    force: bool = False,
    ffmpeg: str = "ffmpeg",
) -> list[dict[str, Any]]:
    inputs = {"stems": [stem.path if isinstance(stem, StemSource) else stem for stem in stems]}
    if output_json.exists() and not force:
        print(f"Voice clips already exist, skipping: {output_json}", flush=True)
        data = read_json(output_json)
        warn_if_inputs_changed(data, inputs, output_json)
        return list(data.get("clips") or [])
    stem_sources = normalize_stem_sources(stems)
    output_dir.mkdir(parents=True, exist_ok=True)
    clips: list[dict[str, Any]] = []
    for line in lines:
        unit_id = line.line_id
        line_id = line.line_id if clip_type == "dialogue_line" else ""
        shot_id = line.line_id if clip_type == "shot" else ""
        line_source_group = str(
            getattr(line, "source_group", "") or line.diarized_speaker or ""
        ).strip()
        selected_sources = _sources_for_line(
            stem_sources,
            line_id=line.line_id,
            source_group=line_source_group,
        )
        start_sec = line.cut_start_sec if line.cut_start_sec is not None else line.start_sec
        end_sec = line.cut_end_sec if line.cut_end_sec is not None else line.end_sec
        if end_sec <= start_sec:
            raise ValueError(f"dialogue line {line.line_id} has non-positive cut duration")
        line_dir = (
            output_dir / ("dialogue_lines" if clip_type == "dialogue_line" else "shots") / unit_id
        )
        for source in selected_sources:
            source_group = source.source_group or source_group_for_stem(
                source.path, speaker=source.speaker
            )
            target = line_dir / f"{unit_id}_{source_group}_stem.wav"
            print(
                f"  clip {unit_id} {source_group}: {start_sec:.3f}-{end_sec:.3f}",
                flush=True,
            )
            if force or not target.exists():
                from unrender.editorial.shots.stems import run_ffmpeg_slice

                run_ffmpeg_slice(
                    ffmpeg=ffmpeg,
                    source=source.path,
                    target=target,
                    start_sec=start_sec,
                    end_sec=end_sec,
                )
            activity = measure_activity(target, threshold_db=activity_threshold_db)
            peak_dbfs = activity.peak_dbfs
            reasons = []
            if peak_dbfs <= silence_threshold_db:
                reasons.append(f"peak_dbfs <= {silence_threshold_db:g}")
            if activity.rms_dbfs < min_rms_dbfs:
                reasons.append(f"rms_dbfs < {min_rms_dbfs:g}")
            if activity.voiced_ratio < min_voiced_ratio:
                reasons.append(f"voiced_ratio < {min_voiced_ratio:g}")
            keep = not reasons
            reason = "; ".join(reasons)
            if not keep and target.exists():
                target.unlink()
            clips.append(
                {
                    "clip_id": f"{unit_id}_{source_group}",
                    "clip_type": clip_type,
                    "line_id": line_id,
                    "shot_id": shot_id,
                    "source_group": source_group,
                    "speaker": source.speaker,
                    "source_stem": str(source.path),
                    "path": str(target),
                    "clip_path": str(target),
                    "start_sec": line.start_sec,
                    "end_sec": line.end_sec,
                    "clip_start_sec": start_sec,
                    "clip_end_sec": end_sec,
                    "text": line.text,
                    "peak_dbfs": peak_dbfs,
                    "rms_dbfs": activity.rms_dbfs,
                    "voiced_ratio": activity.voiced_ratio,
                    "kept": keep,
                    "reason": reason,
                }
            )
    write_json(
        output_json,
        {
            "version": "1.0",
            "clip_type": clip_type,
            "output_dir": str(output_dir),
            "inputs": input_fingerprints(inputs),
            "clips": clips,
        },
    )
    _write_voice_clips_csv(output_csv, clips)
    kept = sum(1 for clip in clips if clip["kept"])
    print(f"Voice clips saved: {output_json} ({kept}/{len(clips)} kept)", flush=True)
    return clips


def _sources_for_line(
    sources: list[StemSource],
    *,
    line_id: str,
    source_group: str,
) -> list[StemSource]:
    if not source_group:
        return sources
    selected = [source for source in sources if source.source_group == source_group]
    if not selected:
        available = ", ".join(source.source_group for source in sources)
        raise ValueError(
            f"dialogue line {line_id} references source_group {source_group!r}, "
            f"but stems provide: {available}"
        )
    return selected


def resolve_voice_clips(
    *,
    clips: list[VoiceInput],
    speaker_db_path: Path,
    voice_matches_json: Path,
    output_json: Path,
    output_csv: Path,
    backend: str = "pyannote",
    sim_threshold: float = 0.65,
    min_margin: float = 0.0,
) -> list[dict[str, Any]]:
    voice_matches = match_voice_db(
        clips=clips,
        speaker_db_path=speaker_db_path,
        output_json=voice_matches_json,
        backend=backend,
        sim_threshold=sim_threshold,
        min_margin=min_margin,
    )
    plan: list[dict[str, Any]] = []
    for match in voice_matches:
        best_matches = list(match.get("matches") or [])
        best = best_matches[0] if best_matches else {}
        accepted = list(match.get("accepted") or [])
        clip_path = str(match.get("clip") or "")
        plan.append(
            {
                "clip_id": str(match.get("clip_id") or ""),
                "clip_type": str(match.get("clip_type") or ""),
                "line_id": str(match.get("line_id") or ""),
                "shot_id": str(match.get("shot_id") or ""),
                "speaker": accepted[0] if accepted else str(best.get("speaker") or ""),
                "stem_path": clip_path if accepted else "",
                "source_stem": str(match.get("source_stem") or ""),
                "source_group": str(match.get("source_group") or ""),
                "start_sec": match.get("start_sec"),
                "end_sec": match.get("end_sec"),
                "clip_start_sec": match.get("clip_start_sec"),
                "clip_end_sec": match.get("clip_end_sec"),
                "score": float(best.get("score") or 0.0),
                "margin": float(match.get("margin") or 0.0),
                "status": "matched" if accepted else "no_matching_stem",
                "resolution_source": "clip",
                "text": str(match.get("text") or ""),
            }
        )
    write_json(
        output_json,
        {
            "version": "1.0",
            "resolver": "clip",
            "speaker_db": str(speaker_db_path),
            "voice_matches": str(voice_matches_json),
            "sim_threshold": sim_threshold,
            "min_margin": min_margin,
            "clips": plan,
        },
    )
    _write_clip_plan_csv(output_csv, plan)
    matched = sum(1 for entry in plan if entry["status"] == "matched")
    print(f"Clip stem plan saved: {output_json} ({matched}/{len(plan)} matched)", flush=True)
    return plan


def resolve_voice_clips_from_voice_db(
    *,
    voice_db_path: Path,
    output_json: Path,
    output_csv: Path,
    clips: list[VoiceInput] | None = None,
) -> list[dict[str, Any]]:
    """Resolve clips from the already-labeled voice DB membership.

    ``voice build`` is the global identity clustering step. Once the user
    labels those clusters, resolution should preserve that exact membership
    instead of reclustering the clips and reassigning identities.
    """
    if not voice_db_path.exists():
        raise FileNotFoundError(f"voice DB not found: {voice_db_path}")
    voice_db = read_json(voice_db_path)
    wanted = {clip.clip_id for clip in clips or [] if clip.clip_id}
    seen: set[str] = set()
    plan: list[dict[str, Any]] = []

    for cluster in voice_db.get("clusters") or []:
        cluster_id = int(cluster.get("cluster_id", 0))
        speaker = str(cluster.get("speaker") or cluster.get("name") or "").strip()
        skipped = bool(cluster.get("skipped"))
        if skipped:
            status = "skipped_cluster"
        elif not speaker_key(speaker):
            status = "unlabeled_cluster"
        else:
            status = "matched"
        for clip in cluster.get("clips") or []:
            entry = _voice_db_clip_plan_entry(
                clip,
                cluster_id=cluster_id,
                speaker=speaker if status == "matched" else "",
                status=status,
            )
            clip_id = str(entry.get("clip_id") or "")
            if wanted and clip_id not in wanted:
                continue
            seen.add(clip_id)
            plan.append(entry)

    for clip in voice_db.get("ungrouped") or []:
        entry = _voice_db_clip_plan_entry(
            clip,
            cluster_id=None,
            speaker="",
            status="ungrouped",
        )
        clip_id = str(entry.get("clip_id") or "")
        if wanted and clip_id not in wanted:
            continue
        seen.add(clip_id)
        plan.append(entry)

    for clip in clips or []:
        if not clip.clip_id or clip.clip_id in seen:
            continue
        plan.append(
            _voice_db_clip_plan_entry(
                {
                    "path": str(clip.path),
                    "clip_id": clip.clip_id,
                    "clip_type": clip.clip_type,
                    "line_id": clip.line_id,
                    "shot_id": clip.shot_id,
                    "source_group": clip.source_group,
                    "source_stem": clip.source_stem,
                    "start_sec": clip.start_sec,
                    "end_sec": clip.end_sec,
                    "clip_start_sec": clip.clip_start_sec,
                    "clip_end_sec": clip.clip_end_sec,
                    "text": clip.text,
                },
                cluster_id=None,
                speaker="",
                status="missing_voice_db_cluster",
            )
        )

    plan.sort(key=lambda entry: (entry["line_id"], entry["shot_id"], entry["clip_id"]))
    write_json(
        output_json,
        {
            "version": "1.0",
            "resolver": "voice-db",
            "voice_db": str(voice_db_path),
            "clips": plan,
        },
    )
    _write_clip_plan_csv(output_csv, plan)
    matched = sum(1 for entry in plan if entry["status"] == "matched")
    print(f"Clip stem plan saved: {output_json} ({matched}/{len(plan)} matched)", flush=True)
    return plan


def _voice_db_clip_plan_entry(
    clip: dict[str, Any],
    *,
    cluster_id: int | None,
    speaker: str,
    status: str,
) -> dict[str, Any]:
    clip_path = str(clip.get("path") or clip.get("clip_path") or "")
    matched = status == "matched"
    return {
        "clip_id": str(clip.get("clip_id") or ""),
        "clip_type": str(clip.get("clip_type") or ""),
        "line_id": str(clip.get("line_id") or ""),
        "shot_id": str(clip.get("shot_id") or ""),
        "speaker": speaker if matched else "",
        "stem_path": clip_path if matched else "",
        "source_stem": str(clip.get("source_stem") or ""),
        "source_group": str(clip.get("source_group") or ""),
        "start_sec": clip.get("start_sec"),
        "end_sec": clip.get("end_sec"),
        "clip_start_sec": clip.get("clip_start_sec"),
        "clip_end_sec": clip.get("clip_end_sec"),
        "score": 1.0 if matched else 0.0,
        "margin": 1.0 if matched else 0.0,
        "cluster_id": cluster_id,
        "status": status,
        "resolution_source": "voice-db",
        "text": str(clip.get("text") or ""),
    }


def materialize_dialogue_line_stems(
    *,
    clip_plan: list[dict[str, Any]],
    output_dir: Path,
    output_json: Path,
    output_csv: Path,
    force: bool = False,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan: list[dict[str, Any]] = []
    for entry in clip_plan:
        materialized = dict(entry)
        if entry.get("status") != "matched":
            plan.append(materialized)
            continue
        if str(entry.get("clip_type") or "") != "dialogue_line":
            plan.append(materialized)
            continue
        line_id = str(entry.get("line_id") or "").strip()
        speaker = str(entry.get("speaker") or "").strip()
        source_path = Path(str(entry.get("stem_path") or "")).expanduser()
        if not line_id:
            raise ValueError(f"matched clip is missing line_id: {entry}")
        if not speaker_key(speaker):
            raise ValueError(f"matched clip {line_id} is missing speaker")
        if not source_path.exists():
            raise FileNotFoundError(f"resolved dialogue clip not found: {source_path}")
        target = output_dir / line_id / f"{line_id}_{speaker_key(speaker)}_stem.wav"
        print(f"  name dialogue {line_id} {speaker}: {target}", flush=True)
        if force or not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)
        materialized["raw_stem_path"] = str(source_path)
        materialized["dialogue_stem_path"] = str(target)
        materialized["stem_path"] = str(target)
        plan.append(materialized)

    write_json(
        output_json,
        {
            "version": "1.0",
            "source": "clip_stem_plan",
            "output_dir": str(output_dir),
            "clips": plan,
        },
    )
    _write_dialogue_plan_csv(output_csv, plan)
    matched = sum(1 for entry in plan if entry.get("dialogue_stem_path"))
    print(f"Dialogue stem plan saved: {output_json} ({matched} named clip(s))", flush=True)
    return plan


def map_dialogue_to_shots(
    *,
    lines: list[DialogueLineRecord],
    shots: list[ShotRecord],
    clip_plan: list[dict[str, Any]],
    output_json: Path,
    output_csv: Path,
    shot_plan_json: Path,
    shot_plan_csv: Path,
    mapped_dir: Path | None = None,
    shot_speakers_by_id: dict[str, list[str]] | None = None,
    min_overlap_ratio: float = 0.01,
    force: bool = False,
    ffmpeg: str = "ffmpeg",
) -> list[dict[str, Any]]:
    """Map resolved dialogue lines onto shots.

    The mapping itself is metadata; the labeled per-line dialogue stems remain
    the primary deliverable. When ``mapped_dir`` is provided (optional final
    step), each line overlapping a shot is additionally cut to the exact
    intersection of the line window and the shot timecode: a shot with two
    lines gets two stem files, a shot with one line gets one.
    """
    best_by_line = _best_clip_by_line(clip_plan)
    shot_speakers_by_id = shot_speakers_by_id or {}
    mappings: list[dict[str, Any]] = []
    for shot in shots:
        if shot.start_sec is None or shot.end_sec is None:
            raise ValueError(f"shot {shot.shot_id} is missing start_sec/end_sec")
        for line in lines:
            overlap = _overlap(shot.start_sec, shot.end_sec, line.start_sec, line.end_sec)
            if overlap <= 0.0:
                continue
            line_duration = max(0.001, line.end_sec - line.start_sec)
            overlap_ratio = overlap / line_duration
            if overlap_ratio < min_overlap_ratio:
                continue
            selected = best_by_line.get(line.line_id, {})
            speaker = str(selected.get("speaker") or line.speaker)
            on_screen_speakers = shot_speakers_by_id.get(shot.shot_id, [])
            on_screen_match = _matches_on_screen_speaker(speaker, on_screen_speakers)
            status = str(selected.get("status", "unresolved"))
            if status == "matched" and not on_screen_match:
                status = "offscreen_speaker"
            mappings.append(
                {
                    "shot_id": shot.shot_id,
                    "line_id": line.line_id,
                    "start_sec": line.start_sec,
                    "end_sec": line.end_sec,
                    "overlap_start_sec": max(shot.start_sec, line.start_sec),
                    "overlap_end_sec": min(shot.end_sec, line.end_sec),
                    "overlap_ratio": overlap_ratio,
                    "speaker": speaker,
                    "on_screen_speakers": on_screen_speakers,
                    "on_screen_match": on_screen_match,
                    "stem_path": selected.get("stem_path", ""),
                    "dialogue_stem_path": selected.get("dialogue_stem_path", ""),
                    "raw_stem_path": selected.get("raw_stem_path", ""),
                    "source_stem": selected.get("source_stem", ""),
                    "source_group": selected.get("source_group", ""),
                    "text": line.text,
                    "status": status,
                    "score": selected.get("score", 0.0),
                }
            )
    if mapped_dir is not None:
        shot_plan = _cut_shot_line_stems(
            mappings=mappings,
            shots_by_id={shot.shot_id: shot for shot in shots},
            mapped_dir=mapped_dir,
            force=force,
            ffmpeg=ffmpeg,
        )
    else:
        shot_plan = _shot_plan_from_dialogue_map(mappings)
    write_json(
        output_json,
        {
            "version": "1.0",
            "min_overlap_ratio": min_overlap_ratio,
            "shots": mappings,
        },
    )
    _write_shot_dialogue_csv(output_csv, mappings)
    write_json(
        shot_plan_json, {"version": "1.0", "source": "shot_dialogue_map", "shots": shot_plan}
    )
    write_plan_csv(shot_plan_csv, shot_plan)
    print(f"Shot dialogue map saved: {output_json} ({len(mappings)} mapping(s))", flush=True)
    return mappings


def load_stem_sources_for_project(
    *,
    stem_spec: str | None,
    stem_map: dict[str, Any] | None,
    fallback_spec: str,
    speaker_names: list[str],
) -> list[StemSource]:
    if stem_spec:
        return load_stem_sources(stem_spec)
    if stem_map:
        return load_stem_map(stem_map, speaker_names=speaker_names)
    return load_stem_sources(fallback_spec)


def _shot_overlaps(
    start_sec: float, end_sec: float, shots: list[ShotRecord]
) -> list[dict[str, Any]]:
    overlaps: list[dict[str, Any]] = []
    for shot in shots:
        if shot.start_sec is None or shot.end_sec is None:
            continue
        overlap = _overlap(shot.start_sec, shot.end_sec, start_sec, end_sec)
        if overlap <= 0:
            continue
        overlaps.append(
            {
                "shot_id": shot.shot_id,
                "overlap_start_sec": max(shot.start_sec, start_sec),
                "overlap_end_sec": min(shot.end_sec, end_sec),
                "overlap_ratio": overlap / max(0.001, end_sec - start_sec),
            }
        )
    return overlaps


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _matches_on_screen_speaker(speaker: str, on_screen_speakers: list[str]) -> bool:
    if not on_screen_speakers:
        return True
    speaker_id = speaker_key(speaker)
    return bool(speaker_id and speaker_id in {speaker_key(item) for item in on_screen_speakers})


def _best_clip_by_line(clip_plan: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for entry in clip_plan:
        if entry.get("status") != "matched":
            continue
        line_id = str(entry.get("line_id") or "")
        if not line_id:
            continue
        current = out.get(line_id)
        if current is None or float(entry.get("score") or 0.0) > float(current.get("score") or 0.0):
            out[line_id] = entry
    return out


def _shot_plan_from_dialogue_map(mappings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate_counts = Counter(str(mapping.get("shot_id") or "") for mapping in mappings)
    best_by_shot: dict[str, dict[str, Any]] = {}
    for mapping in mappings:
        if mapping.get("status") != "matched":
            continue
        shot_id = str(mapping.get("shot_id") or "")
        current = best_by_shot.get(shot_id)
        if current is None or float(mapping.get("score") or 0.0) > float(
            current.get("score") or 0.0
        ):
            best_by_shot[shot_id] = mapping
    return [
        {
            "shot_id": shot_id,
            "speaker": str(mapping.get("speaker") or ""),
            "stem_path": str(mapping.get("stem_path") or ""),
            "source_group": str(mapping.get("source_group") or ""),
            "score": float(mapping.get("score") or 0.0),
            "status": "matched",
            "candidate_count": candidate_counts[shot_id],
            "line_id": str(mapping.get("line_id") or ""),
            "proposed_audio_path": str(mapping.get("stem_path") or ""),
        }
        for shot_id, mapping in sorted(best_by_shot.items())
    ]


def _cut_shot_line_stems(
    *,
    mappings: list[dict[str, Any]],
    shots_by_id: dict[str, ShotRecord],
    mapped_dir: Path,
    force: bool,
    ffmpeg: str,
) -> list[dict[str, Any]]:
    """Build one exact-shot-length stem per active speaker.

    Each assigned dialogue interval is taken from its resolved local source
    stem and placed at the original offset within a silent shot-length
    timeline. This preserves timing without admitting bleed from unrelated
    speakers elsewhere on an AudioShake source channel.
    """
    matched = [
        mapping
        for mapping in mappings
        if mapping.get("status") == "matched"
        and str(mapping.get("shot_id") or "")
        and speaker_key(str(mapping.get("speaker") or ""))
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for mapping in matched:
        key = (
            str(mapping.get("shot_id") or ""),
            speaker_key(str(mapping.get("speaker") or "")),
        )
        grouped.setdefault(key, []).append(mapping)
    plan: list[dict[str, Any]] = []
    for (shot_id, _speaker_key), speaker_mappings in sorted(grouped.items()):
        shot = shots_by_id.get(shot_id)
        if shot is None or shot.start_sec is None or shot.end_sec is None:
            raise ValueError(f"shot {shot_id} is missing start_sec/end_sec")
        shot_start = float(shot.start_sec)
        shot_end = float(shot.end_sec)
        shot_duration = shot_end - shot_start
        if shot_duration <= 0:
            raise ValueError(f"shot {shot_id} has a non-positive duration")

        ordered = sorted(speaker_mappings, key=lambda mapping: str(mapping.get("line_id") or ""))
        speaker = str(ordered[0].get("speaker") or "")
        segments: list[tuple[Path, float, float, float]] = []
        line_ids: list[str] = []
        for mapping in ordered:
            line_id = str(mapping.get("line_id") or "")
            source_stem = str(mapping.get("source_stem") or "")
            if not source_stem:
                raise ValueError(f"shot {shot_id} line {line_id} is missing source_stem")
            source = Path(source_stem).expanduser()
            if not source.exists():
                raise FileNotFoundError(f"source stem for shot {shot_id} not found: {source}")
            cut_start = float(mapping["overlap_start_sec"])
            cut_end = float(mapping["overlap_end_sec"])
            if cut_end <= cut_start:
                continue
            segments.append((source, cut_start, cut_end, round(cut_start - shot_start, 6)))
            line_ids.append(line_id)
        if not segments:
            continue

        name = f"{safe_name(shot_id)}_{speaker_key(speaker)}_stem.wav"
        target = mapped_dir / safe_name(shot_id) / name
        print(
            f"  build shot stem {shot_id} {speaker}: "
            f"{len(segments)} dialogue interval(s), {shot_duration:.3f}s",
            flush=True,
        )
        if force or not target.exists():
            run_ffmpeg_timeline_mix(
                ffmpeg=ffmpeg,
                segments=segments,
                target=target,
                duration_sec=shot_duration,
            )
        plan.append(
            {
                "shot_id": shot_id,
                "line_id": ",".join(line_ids),
                "speaker": speaker,
                "stem_path": str(target),
                "source_stem": ",".join(sorted({str(segment[0]) for segment in segments})),
                "source_group": ",".join(
                    sorted(
                        {
                            str(mapping.get("source_group") or "")
                            for mapping in ordered
                            if mapping.get("source_group")
                        }
                    )
                ),
                "start_sec": shot_start,
                "end_sec": shot_end,
                "offset_in_shot_sec": 0.0,
                "score": max(float(mapping.get("score") or 0.0) for mapping in ordered),
                "status": "materialized",
                "candidate_count": len(ordered),
                "text": " ".join(
                    str(mapping.get("text") or "").strip()
                    for mapping in ordered
                    if mapping.get("text")
                ),
                "proposed_audio_path": str(target),
            }
        )
        for mapping in ordered:
            mapping["shot_stem_path"] = str(target)
    return plan
