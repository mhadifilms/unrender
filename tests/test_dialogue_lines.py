from __future__ import annotations

import json
import os
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from conftest import write_wav
from unrender.editorial.dialogue.lines import (
    SpeechIsland,
    build_dialogue_lines,
    build_voice_clips,
    map_dialogue_to_shots,
    materialize_dialogue_line_stems,
    refine_cut_window,
    resolve_voice_clips,
    resolve_voice_clips_from_voice_db,
)
from unrender.manifests import DialogueLineRecord, ShotRecord, VoiceInput, read_json


def test_build_dialogue_lines_splits_on_gap_and_speaker_change() -> None:
    lines = build_dialogue_lines(
        [
            {"word": "hello", "start": 1.0, "end": 1.2, "speaker": "SPEAKER_00"},
            {"word": "there", "start": 1.3, "end": 1.5, "speaker": "SPEAKER_00"},
            {"word": "yes", "start": 1.6, "end": 1.8, "speaker": "SPEAKER_01"},
            {"word": "again", "start": 3.0, "end": 3.2, "speaker": "SPEAKER_01"},
        ],
        max_gap_sec=0.5,
        handle_sec=0.18,
    )

    assert [line["line_id"] for line in lines] == ["DL_000001", "DL_000002", "DL_000003"]
    assert [line["text"] for line in lines] == ["hello there", "yes", "again"]
    assert lines[0]["diarized_speaker"] == "SPEAKER_00"
    assert lines[0]["cut_start_sec"] == 0.82
    assert lines[0]["cut_end_sec"] == 1.68


def test_refine_cut_window_uses_speech_island_edges() -> None:
    assert refine_cut_window(
        10.0,
        11.0,
        handle_sec=0.18,
        islands=[SpeechIsland(9.7, 11.4)],
    ) == (9.7, 11.4)


def test_refine_cut_window_caps_expansion_into_shared_islands() -> None:
    # An island spanning a fast speaker change must not pull the whole
    # neighboring line into this clip.
    assert refine_cut_window(
        10.0,
        11.0,
        handle_sec=0.18,
        islands=[SpeechIsland(8.0, 13.0)],
        max_expand_sec=0.6,
    ) == (9.4, 11.6)


def test_detect_speech_islands_finds_voiced_regions(tmp_path: Path) -> None:
    from unrender.editorial.dialogue.transcription import detect_speech_islands

    rate = 16_000
    samples = np.concatenate(
        [
            np.zeros(int(rate * 0.5), dtype=np.int16),
            np.full(int(rate * 1.0), 8000, dtype=np.int16),
            np.zeros(int(rate * 0.5), dtype=np.int16),
            np.full(int(rate * 0.6), 8000, dtype=np.int16),
            np.zeros(int(rate * 0.3), dtype=np.int16),
        ]
    )
    audio = tmp_path / "dx.wav"
    write_wav(audio, samples)

    islands = detect_speech_islands(audio)

    assert len(islands) == 2
    assert islands[0].start_sec == pytest.approx(0.5, abs=0.05)
    assert islands[0].end_sec == pytest.approx(1.5, abs=0.05)
    assert islands[1].start_sec == pytest.approx(2.0, abs=0.05)
    assert islands[1].end_sec == pytest.approx(2.6, abs=0.05)
    assert detect_speech_islands(tmp_path / "missing.wav") == []


def test_detect_speech_islands_matches_naive_reference(tmp_path: Path) -> None:
    """The vectorized block reader must reproduce the per-hop loop exactly."""
    import math

    from unrender.editorial.dialogue.transcription import detect_speech_islands
    from unrender.lib.audio import full_scale, pcm_values

    rng = np.random.default_rng(42)
    rate = 16_000
    pieces = []
    for _ in range(6):
        pieces.append(np.zeros(int(rate * rng.uniform(0.1, 0.6)), dtype=np.int16))
        burst = (rng.normal(0.0, 4000.0, int(rate * rng.uniform(0.1, 0.8)))).astype(np.int16)
        pieces.append(burst)
    samples = np.concatenate(pieces)
    audio = tmp_path / "dx.wav"
    write_wav(audio, samples)

    def naive_levels() -> list[float]:
        levels = []
        with wave.open(str(audio), "rb") as handle:
            width = handle.getsampwidth()
            hop_frames = max(1, int(handle.getframerate() * 0.02))
            scale = full_scale(width)
            while True:
                raw = handle.readframes(hop_frames)
                if not raw:
                    break
                values = pcm_values(raw, width) / scale
                rms = float(np.sqrt(np.mean(np.square(values))))
                levels.append(20.0 * math.log10(rms) if rms > 0 else float("-inf"))
        return levels

    hop_dur = max(1, int(rate * 0.02)) / rate
    reference = []
    start = None
    voiced_end = 0.0
    for index, level in enumerate(naive_levels()):
        hop_start = index * hop_dur
        if level > -45.0:
            if start is None:
                start = hop_start
            voiced_end = hop_start + hop_dur
        elif start is not None and hop_start - voiced_end >= 0.25:
            if voiced_end - start >= 0.08:
                reference.append((start, voiced_end))
            start = None
    if start is not None and voiced_end - start >= 0.08:
        reference.append((start, voiced_end))

    islands = detect_speech_islands(audio)
    assert [(i.start_sec, i.end_sec) for i in islands] == [
        (pytest.approx(s), pytest.approx(e)) for s, e in reference
    ]


def test_transcribe_lines_snaps_cut_windows_to_speech(tmp_path: Path) -> None:
    from unrender.editorial.dialogue.lines import transcribe_dialogue_lines

    rate = 16_000
    samples = np.concatenate(
        [
            np.zeros(int(rate * 0.7), dtype=np.int16),
            np.full(int(rate * 1.7), 8000, dtype=np.int16),
            np.zeros(int(rate * 0.4), dtype=np.int16),
        ]
    )
    audio = tmp_path / "dx.wav"
    write_wav(audio, samples)

    lines = transcribe_dialogue_lines(
        audio_path=audio,
        output_json=tmp_path / "lines.json",
        output_csv=tmp_path / "lines.csv",
        words=[{"word": "hello", "start": 1.0, "end": 2.0}],
    )

    assert len(lines) == 1
    # The word window (1.0-2.0) expands past the 0.18s handles to the
    # detected speech island (0.7-2.4).
    assert lines[0]["cut_start_sec"] == pytest.approx(0.7, abs=0.05)
    assert lines[0]["cut_end_sec"] == pytest.approx(2.4, abs=0.05)


def test_transcribe_lines_splits_fast_turns_by_local_stem_activity(tmp_path: Path) -> None:
    from unrender.editorial.dialogue.lines import transcribe_dialogue_lines

    rate = 16_000
    duration = int(rate * 2.5)
    speaker_01 = np.zeros(duration, dtype=np.int16)
    speaker_02 = np.zeros(duration, dtype=np.int16)
    speaker_01[int(rate * 0.05) : int(rate * 0.45)] = 8000
    speaker_02[int(rate * 0.65) : int(rate * 1.05)] = 8000
    speaker_01[int(rate * 1.55) : int(rate * 1.95)] = 8000
    source_a = tmp_path / "clip_speaker_01_stem.wav"
    source_b = tmp_path / "clip_speaker_02_stem.wav"
    dx = tmp_path / "dx.wav"
    write_wav(source_a, speaker_01)
    write_wav(source_b, speaker_02)
    write_wav(dx, speaker_01 + speaker_02)

    lines = transcribe_dialogue_lines(
        audio_path=dx,
        output_json=tmp_path / "lines.json",
        output_csv=tmp_path / "lines.csv",
        source_stems=[source_a, source_b],
        max_gap_sec=0.5,
        words=[
            {"word": "hello", "start": 0.10, "end": 0.30},
            # Gap from the previous word is under max_gap_sec; source change
            # must still split the turn before voice clustering.
            {"word": "jesus", "start": 0.70, "end": 0.90},
            {"word": "again", "start": 1.60, "end": 1.80},
        ],
        force=True,
    )

    assert [line["text"] for line in lines] == ["hello", "jesus", "again"]
    assert [line["source_group"] for line in lines] == [
        "speaker_01",
        "speaker_02",
        "speaker_01",
    ]
    assert lines[0]["cut_end_sec"] == pytest.approx(0.45, abs=0.05)


def test_whisperx_loader_sets_torch_safe_load_env(monkeypatch, tmp_path: Path) -> None:
    from unrender.audio import asr

    class FakeModel:
        def transcribe(self, audio, *, batch_size: int, language: str | None):
            return {"language": "en", "segments": []}

    fake_whisperx = types.SimpleNamespace(
        load_audio=lambda path: np.zeros(160, dtype=np.float32),
        load_align_model=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("skip")),
    )
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setattr(asr, "load_whisperx_model", lambda **_kwargs: FakeModel())
    monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)

    assert (
        asr.transcribe_word_aligned(
            tmp_path / "audio.wav",
            model_name="tiny",
            device="cpu",
            compute_type="float32",
            language="en",
        )[0]
        == []
    )
    assert os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"


def test_build_voice_clips_writes_manifest_and_prunes_silence(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_a = tmp_path / "speaker_01_stem.wav"
    source_b = tmp_path / "speaker_02_stem.wav"
    source_a.write_bytes(b"source")
    source_b.write_bytes(b"source")

    def fake_slice(*, source: Path, target: Path, **kwargs) -> None:
        if "speaker_01" in source.name:
            write_wav(target, np.full(1600, 1000, dtype=np.int16))
        else:
            write_wav(target, np.zeros(1600, dtype=np.int16))

    monkeypatch.setattr("unrender.editorial.shots.stems.run_ffmpeg_slice", fake_slice)

    clips = build_voice_clips(
        lines=[
            DialogueLineRecord(
                line_id="DL_000001",
                start_sec=1.0,
                end_sec=2.0,
                cut_start_sec=0.8,
                cut_end_sec=2.2,
                text="hello there",
            )
        ],
        stems=[source_a, source_b],
        output_dir=tmp_path / "audio" / "voice_clips",
        output_json=tmp_path / "audio" / "voice_clips.json",
        output_csv=tmp_path / "audio" / "voice_clips.csv",
        force=True,
    )

    kept = [clip for clip in clips if clip["kept"]]
    dropped = [clip for clip in clips if not clip["kept"]]
    assert kept[0]["clip_id"] == "DL_000001_speaker_01"
    assert kept[0]["line_id"] == "DL_000001"
    assert kept[0]["text"] == "hello there"
    assert Path(kept[0]["path"]).exists()
    assert not Path(dropped[0]["path"]).exists()
    assert (
        read_json(tmp_path / "audio" / "voice_clips.json")["clips"][0]["clip_type"]
        == "dialogue_line"
    )


def test_build_voice_clips_cuts_only_matching_source_group(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_a = tmp_path / "speaker_01_stem.wav"
    source_b = tmp_path / "speaker_02_stem.wav"
    source_a.write_bytes(b"source")
    source_b.write_bytes(b"source")
    sliced_sources: list[Path] = []

    def fake_slice(*, source: Path, target: Path, **kwargs) -> None:
        sliced_sources.append(source)
        write_wav(target, np.full(1600, 1000, dtype=np.int16))

    monkeypatch.setattr("unrender.editorial.shots.stems.run_ffmpeg_slice", fake_slice)

    clips = build_voice_clips(
        lines=[
            DialogueLineRecord(
                line_id="DL_000001",
                start_sec=1.0,
                end_sec=2.0,
                source_group="speaker_02",
                cut_start_sec=1.0,
                cut_end_sec=2.0,
                text="hello",
            )
        ],
        stems=[source_a, source_b],
        output_dir=tmp_path / "audio" / "voice_clips",
        output_json=tmp_path / "audio" / "voice_clips.json",
        output_csv=tmp_path / "audio" / "voice_clips.csv",
        force=True,
    )

    assert sliced_sources == [source_b]
    assert [clip["clip_id"] for clip in clips] == ["DL_000001_speaker_02"]
    assert clips[0]["kept"] is True


def test_build_voice_clips_rejects_peak_only_bleed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "speaker_01_stem.wav"
    source.write_bytes(b"source")

    def fake_slice(*, target: Path, **kwargs) -> None:
        samples = np.zeros(16000, dtype=np.int16)
        samples[:400] = 16000
        write_wav(target, samples)

    monkeypatch.setattr("unrender.editorial.shots.stems.run_ffmpeg_slice", fake_slice)

    clips = build_voice_clips(
        lines=[
            DialogueLineRecord(
                line_id="DL_000001",
                start_sec=1.0,
                end_sec=2.0,
                cut_start_sec=1.0,
                cut_end_sec=2.0,
                text="bleed",
            )
        ],
        stems=[source],
        output_dir=tmp_path / "audio" / "voice_clips",
        output_json=tmp_path / "audio" / "voice_clips.json",
        output_csv=tmp_path / "audio" / "voice_clips.csv",
        force=True,
    )

    assert clips[0]["peak_dbfs"] > -10.0
    assert clips[0]["voiced_ratio"] < 0.15
    assert clips[0]["kept"] is False
    assert "voiced_ratio" in clips[0]["reason"]


def test_resolve_voice_clips_writes_line_level_plan(tmp_path: Path) -> None:
    speaker_db = tmp_path / "speaker_db.json"
    speaker_db.write_text(
        json.dumps(
            {
                "version": "1.0",
                "speakers": {},
                "face_clusters": [],
                "voice_clusters": [
                    {
                        "cluster_id": 1,
                        "speaker": "ALEX",
                        "name": "ALEX",
                        "centroid": [1.0, 0.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    clip_path = tmp_path / "DL_000001_speaker_01_stem.wav"
    clip_path.write_bytes(b"audio")

    plan = resolve_voice_clips(
        clips=[
            VoiceInput(
                path=clip_path,
                clip_id="DL_000001_speaker_01",
                clip_type="dialogue_line",
                line_id="DL_000001",
                source_group="speaker_01",
                text="hello there",
                embedding=(0.99, 0.01),
            )
        ],
        speaker_db_path=speaker_db,
        voice_matches_json=tmp_path / "voice_matches.json",
        output_json=tmp_path / "clip_stem_plan.json",
        output_csv=tmp_path / "clip_stem_plan.csv",
    )

    assert plan[0]["line_id"] == "DL_000001"
    assert plan[0]["speaker"] == "ALEX"
    assert plan[0]["stem_path"] == str(clip_path)
    assert (
        read_json(tmp_path / "voice_matches.json")["shots"][0]["clip_id"] == "DL_000001_speaker_01"
    )


def test_resolve_voice_clips_from_voice_db_preserves_labeled_membership_and_skips(
    tmp_path: Path,
) -> None:
    matched_clip = tmp_path / "DL_000001_speaker_01_stem.wav"
    skipped_clip = tmp_path / "DL_000002_speaker_02_stem.wav"
    matched_clip.write_bytes(b"audio")
    skipped_clip.write_bytes(b"audio")
    voice_db = tmp_path / "voice_db.json"
    voice_db.write_text(
        json.dumps(
            {
                "version": "1.0",
                "clusters": [
                    {
                        "cluster_id": 0,
                        "speaker": "ALEX",
                        "clips": [
                            {
                                "path": str(matched_clip),
                                "clip_id": "DL_000001_speaker_01",
                                "clip_type": "dialogue_line",
                                "line_id": "DL_000001",
                                "source_group": "speaker_01",
                                "source_stem": "/tmp/full_speaker_01.wav",
                                "clip_start_sec": 1.0,
                                "clip_end_sec": 2.0,
                                "text": "hello",
                            }
                        ],
                    },
                    {
                        "cluster_id": 1,
                        "skipped": True,
                        "clips": [
                            {
                                "path": str(skipped_clip),
                                "clip_id": "DL_000002_speaker_02",
                                "clip_type": "dialogue_line",
                                "line_id": "DL_000002",
                                "source_group": "speaker_02",
                                "source_stem": "/tmp/full_speaker_02.wav",
                                "clip_start_sec": 3.0,
                                "clip_end_sec": 4.0,
                                "text": "music",
                            }
                        ],
                    },
                ],
                "ungrouped": [
                    {
                        "path": str(tmp_path / "DL_000003_speaker_03_stem.wav"),
                        "clip_id": "DL_000003_speaker_03",
                        "clip_type": "dialogue_line",
                        "line_id": "DL_000003",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    plan = resolve_voice_clips_from_voice_db(
        voice_db_path=voice_db,
        output_json=tmp_path / "clip_stem_plan.json",
        output_csv=tmp_path / "clip_stem_plan.csv",
    )

    by_line = {entry["line_id"]: entry for entry in plan}
    assert by_line["DL_000001"]["status"] == "matched"
    assert by_line["DL_000001"]["speaker"] == "ALEX"
    assert by_line["DL_000001"]["stem_path"] == str(matched_clip)
    assert by_line["DL_000002"]["status"] == "skipped_cluster"
    assert by_line["DL_000002"]["speaker"] == ""
    assert by_line["DL_000002"]["stem_path"] == ""
    assert by_line["DL_000003"]["status"] == "ungrouped"
    assert read_json(tmp_path / "clip_stem_plan.json")["resolver"] == "voice-db"


def test_materialize_dialogue_line_stems_renames_resolved_clips(tmp_path: Path) -> None:
    raw_clip = tmp_path / "voice_clips" / "DL_000001_speaker_01_stem.wav"
    write_wav(raw_clip, np.full(1600, 1000, dtype=np.int16))

    plan = materialize_dialogue_line_stems(
        clip_plan=[
            {
                "clip_id": "DL_000001_speaker_01",
                "clip_type": "dialogue_line",
                "line_id": "DL_000001",
                "speaker": "ALEX",
                "stem_path": str(raw_clip),
                "source_stem": "/tmp/full_speaker_01.wav",
                "source_group": "speaker_01",
                "score": 0.9,
                "status": "matched",
                "text": "hello",
            }
        ],
        output_dir=tmp_path / "audio" / "dialogue_mapped",
        output_json=tmp_path / "dialogue_stem_plan.json",
        output_csv=tmp_path / "dialogue_stem_plan.csv",
        force=True,
    )

    named = tmp_path / "audio" / "dialogue_mapped" / "DL_000001" / "DL_000001_ALEX_stem.wav"
    assert named.exists()
    assert plan[0]["raw_stem_path"] == str(raw_clip)
    assert plan[0]["stem_path"] == str(named)
    assert plan[0]["dialogue_stem_path"] == str(named)
    assert read_json(tmp_path / "dialogue_stem_plan.json")["clips"][0]["stem_path"] == str(named)


def test_map_dialogue_to_shots_keeps_full_line_across_multiple_shots(tmp_path: Path) -> None:
    mappings = map_dialogue_to_shots(
        lines=[
            DialogueLineRecord(
                line_id="DL_000001",
                start_sec=10.0,
                end_sec=14.0,
                text="one full line",
            )
        ],
        shots=[
            ShotRecord("001", Path("/tmp/001.mov"), start_sec=9.5, end_sec=11.0),
            ShotRecord("002", Path("/tmp/002.mov"), start_sec=11.0, end_sec=13.0),
            ShotRecord("003", Path("/tmp/003.mov"), start_sec=13.0, end_sec=15.0),
        ],
        clip_plan=[
            {
                "line_id": "DL_000001",
                "speaker": "ALEX",
                "stem_path": "/tmp/DL_000001_speaker_01_stem.wav",
                "source_group": "speaker_01",
                "score": 0.9,
                "status": "matched",
            }
        ],
        output_json=tmp_path / "shot_dialogue_map.json",
        output_csv=tmp_path / "shot_dialogue_map.csv",
        shot_plan_json=tmp_path / "shot_stem_plan.json",
        shot_plan_csv=tmp_path / "shot_stem_plan.csv",
    )

    assert [mapping["shot_id"] for mapping in mappings] == ["001", "002", "003"]
    assert {mapping["line_id"] for mapping in mappings} == {"DL_000001"}
    assert {mapping["text"] for mapping in mappings} == {"one full line"}
    assert read_json(tmp_path / "shot_stem_plan.json")["source"] == "shot_dialogue_map"


def test_map_dialogue_builds_exact_shot_length_speaker_stem(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Dialogue intervals share one silent, shot-length timeline per speaker."""
    first_source = tmp_path / "full_speaker_01.wav"
    second_source = tmp_path / "full_speaker_02.wav"
    first_source.write_bytes(b"source")
    second_source.write_bytes(b"source")
    for line_id in ("DL_000001", "DL_000002"):
        named = tmp_path / "dialogue_mapped" / line_id / f"{line_id}_ALEX_stem.wav"
        write_wav(named, np.full(1600, 1000, dtype=np.int16))

    timeline_mixes: list[tuple[list[tuple[Path, float, float, float]], float]] = []

    def fake_timeline_mix(*, target: Path, segments, duration_sec: float, **kwargs) -> None:
        timeline_mixes.append((list(segments), duration_sec))
        write_wav(target, np.full(1600, 1000, dtype=np.int16))

    monkeypatch.setattr(
        "unrender.editorial.dialogue.lines.run_ffmpeg_timeline_mix", fake_timeline_mix
    )

    def plan_entry(line_id: str, source: Path) -> dict:
        named = tmp_path / "dialogue_mapped" / line_id / f"{line_id}_ALEX_stem.wav"
        return {
            "line_id": line_id,
            "speaker": "ALEX",
            "stem_path": str(named),
            "dialogue_stem_path": str(named),
            "source_stem": str(source),
            "source_group": "speaker_01",
            "score": 0.9,
            "status": "matched",
        }

    mappings = map_dialogue_to_shots(
        lines=[
            DialogueLineRecord(line_id="DL_000001", start_sec=10.5, end_sec=11.0, text="one"),
            # Crosses the shot-out boundary: must be clamped to 12.0.
            DialogueLineRecord(line_id="DL_000002", start_sec=11.4, end_sec=12.6, text="two"),
        ],
        shots=[ShotRecord("001", Path("/tmp/001.mov"), start_sec=10.0, end_sec=12.0)],
        clip_plan=[
            plan_entry("DL_000001", first_source),
            plan_entry("DL_000002", second_source),
        ],
        output_json=tmp_path / "shot_dialogue_map.json",
        output_csv=tmp_path / "shot_dialogue_map.csv",
        shot_plan_json=tmp_path / "shot_stem_plan.json",
        shot_plan_csv=tmp_path / "shot_stem_plan.csv",
        mapped_dir=tmp_path / "audio" / "mapped",
        force=True,
    )

    output = tmp_path / "audio" / "mapped" / "001" / "001_ALEX_stem.wav"
    assert timeline_mixes == [
        (
            [
                (first_source, 10.5, 11.0, 0.5),
                (second_source, 11.4, 12.0, 1.4),
            ],
            2.0,
        )
    ]
    assert output.exists()
    assert {mapping["shot_stem_path"] for mapping in mappings} == {str(output)}
    shot_plan = read_json(tmp_path / "shot_stem_plan.json")["shots"]
    assert len(shot_plan) == 1
    assert shot_plan[0]["line_id"] == "DL_000001,DL_000002"
    assert shot_plan[0]["status"] == "materialized"
    assert shot_plan[0]["candidate_count"] == 2
    assert shot_plan[0]["offset_in_shot_sec"] == 0.0
    assert shot_plan[0]["start_sec"] == 10.0
    assert shot_plan[0]["end_sec"] == 12.0


def test_map_dialogue_without_mapped_dir_writes_plan_only(tmp_path: Path) -> None:
    named = tmp_path / "dialogue_mapped" / "DL_000001" / "DL_000001_ALEX_stem.wav"
    write_wav(named, np.full(1600, 1000, dtype=np.int16))

    map_dialogue_to_shots(
        lines=[DialogueLineRecord(line_id="DL_000001", start_sec=10.5, end_sec=11.0, text="x")],
        shots=[ShotRecord("001", Path("/tmp/001.mov"), start_sec=10.0, end_sec=12.0)],
        clip_plan=[
            {
                "line_id": "DL_000001",
                "speaker": "ALEX",
                "stem_path": str(named),
                "source_group": "speaker_01",
                "score": 0.9,
                "status": "matched",
            }
        ],
        output_json=tmp_path / "shot_dialogue_map.json",
        output_csv=tmp_path / "shot_dialogue_map.csv",
        shot_plan_json=tmp_path / "shot_stem_plan.json",
        shot_plan_csv=tmp_path / "shot_stem_plan.csv",
    )

    plan = read_json(tmp_path / "shot_stem_plan.json")["shots"]
    assert plan[0]["status"] == "matched"
    assert not (tmp_path / "audio" / "mapped").exists()


def test_map_dialogue_keeps_offscreen_script_lines_out_of_final_stems(
    monkeypatch,
    tmp_path: Path,
) -> None:
    full_source = tmp_path / "full_speaker_01.wav"
    full_source.write_bytes(b"source")
    named_line = tmp_path / "dialogue_mapped" / "DL_000001" / "DL_000001_ALEX_stem.wav"
    write_wav(named_line, np.full(1600, 1000, dtype=np.int16))

    calls: list[Path] = []

    def fake_timeline_mix(*, target: Path, **kwargs) -> None:
        calls.append(target)
        write_wav(target, np.full(1600, 1000, dtype=np.int16))

    monkeypatch.setattr(
        "unrender.editorial.dialogue.lines.run_ffmpeg_timeline_mix", fake_timeline_mix
    )

    mappings = map_dialogue_to_shots(
        lines=[
            DialogueLineRecord(
                line_id="DL_000001",
                start_sec=10.0,
                end_sec=11.0,
                speaker="ALEX",
                text="offscreen line",
            )
        ],
        shots=[ShotRecord("001", Path("/tmp/001.mov"), start_sec=9.5, end_sec=11.5)],
        clip_plan=[
            {
                "line_id": "DL_000001",
                "speaker": "ALEX",
                "stem_path": str(named_line),
                "dialogue_stem_path": str(named_line),
                "source_stem": str(full_source),
                "source_group": "speaker_01",
                "score": 0.9,
                "status": "matched",
            }
        ],
        output_json=tmp_path / "shot_dialogue_map.json",
        output_csv=tmp_path / "shot_dialogue_map.csv",
        shot_plan_json=tmp_path / "shot_stem_plan.json",
        shot_plan_csv=tmp_path / "shot_stem_plan.csv",
        mapped_dir=tmp_path / "audio" / "mapped",
        shot_speakers_by_id={"001": ["JORDAN"]},
        force=True,
    )

    assert calls == []
    assert mappings[0]["status"] == "offscreen_speaker"
    assert mappings[0]["on_screen_speakers"] == ["JORDAN"]
    assert read_json(tmp_path / "shot_stem_plan.json")["shots"] == []
