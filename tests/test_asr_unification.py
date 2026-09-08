from __future__ import annotations

import json
from pathlib import Path

from unrender.audio import asr
from unrender.editorial.dialogue import transcription
from unrender.editorial.dialogue.lines import transcribe_dialogue_lines
from unrender.editorial.shots.stems import StemSource
from unrender.manifests import load_dialogue_lines


def test_editorial_word_alignment_uses_shared_asr(monkeypatch, tmp_path: Path) -> None:
    calls: list[Path] = []

    def fake_word_asr(path: Path, **_kwargs):
        calls.append(path)
        return (
            [{"word": "hello", "start": 0.0, "end": 0.2, "speaker": ""}],
            {
                "language": "en",
                "aligned": True,
            },
        )

    monkeypatch.setattr(asr, "transcribe_word_aligned", fake_word_asr)
    audio = tmp_path / "audio.wav"
    audio.touch()
    output_json = tmp_path / "lines.json"

    lines = transcribe_dialogue_lines(
        audio_path=audio,
        output_json=output_json,
        output_csv=tmp_path / "lines.csv",
        whisper_model="tiny",
        device="cpu",
        compute_type="float32",
        language="English",
        force=True,
    )

    assert calls == [audio]
    assert lines[0]["text"] == "hello"
    assert json.loads(output_json.read_text())["transcription_metadata"]["dx"] == {
        "language": "en",
        "aligned": True,
        "path": str(audio),
        "word_count": 1,
    }


def test_stem_asr_preserves_overlap_and_adds_dx_residual(monkeypatch) -> None:
    source_a = StemSource(Path("a.wav"), source_group="speaker_01")
    source_b = StemSource(Path("b.wav"), source_group="speaker_02")
    calls: list[Path] = []

    def fake_word_asr(path: Path, **_kwargs):
        calls.append(path)
        words = {
            source_a.path: [{"word": "hello", "start": 0.0, "end": 0.4}],
            source_b.path: [{"word": "world", "start": 0.1, "end": 0.5}],
        }[path]
        return words, {"language": "en", "aligned": True}

    monkeypatch.setattr(asr, "transcribe_word_aligned", fake_word_asr)
    meta: dict[str, object] = {}

    merged = transcription.transcribe_stem_words(
        [source_a, source_b],
        dx_words=[
            {"word": "hello", "start": 0.02, "end": 0.38},
            {"word": "world", "start": 0.12, "end": 0.48},
            {"word": "again", "start": 0.7, "end": 1.0},
        ],
        meta=meta,
    )

    assert calls == [source_a.path, source_b.path]
    assert [(word["word"], word["source_group"]) for word in merged] == [
        ("hello", "speaker_01"),
        ("world", "speaker_02"),
        ("again", "speaker_02"),
    ]
    assert merged[-1]["transcription_source"] == "dx_residual"
    assert all(word["review_required"] for word in merged)
    assert merged[0]["review_required"] is True
    assert meta["dx_residual_word_count"] == 1
    assert meta["stem_count"] == 2


def test_parallel_source_grouping_keeps_overlapping_speaker_lines_intact() -> None:
    groups = transcription._group_words(
        [
            {
                "word": "hello",
                "start": 0.0,
                "end": 0.2,
                "source_group": "speaker_01",
            },
            {
                "word": "world",
                "start": 0.1,
                "end": 0.3,
                "source_group": "speaker_02",
            },
            {
                "word": "again",
                "start": 0.35,
                "end": 0.55,
                "source_group": "speaker_01",
            },
        ],
        max_gap_sec=0.5,
        parallel_sources=True,
    )

    assert [(group["text"], group["source_group"]) for group in groups] == [
        ("hello again", "speaker_01"),
        ("world", "speaker_02"),
    ]


def test_ambiguous_dx_residual_requires_speaker_review() -> None:
    merged = transcription.reconcile_stem_words(
        [
            {
                "word": "alpha",
                "start": 0.0,
                "end": 0.4,
                "source_group": "speaker_01",
            },
            {
                "word": "beta",
                "start": 0.0,
                "end": 0.4,
                "source_group": "speaker_02",
            },
        ],
        [{"word": "missing", "start": 0.1, "end": 0.3}],
    )

    residual = next(word for word in merged if word["word"] == "missing")
    assert residual["source_group"] == ""
    assert residual["review_required"] is True
    assert residual["residual_assignment"] == "unresolved"


def test_transcribe_lines_per_stem_mode_and_manifest(monkeypatch, tmp_path: Path) -> None:
    dx = tmp_path / "dx.wav"
    source_a = tmp_path / "speaker_01_stem.wav"
    source_b = tmp_path / "speaker_02_stem.wav"
    for path in (dx, source_a, source_b):
        path.touch()
    calls: list[Path] = []

    def fake_word_asr(path: Path, **_kwargs):
        calls.append(path)
        words = {
            dx: [
                {"word": "hello", "start": 0.0, "end": 0.4, "speaker": ""},
                {"word": "world", "start": 0.1, "end": 0.5, "speaker": ""},
                {"word": "again", "start": 1.0, "end": 1.2, "speaker": ""},
            ],
            source_a: [{"word": "hello", "start": 0.0, "end": 0.4, "speaker": ""}],
            source_b: [{"word": "world", "start": 0.1, "end": 0.5, "speaker": ""}],
        }[path]
        return words, {"language": "en", "aligned": True}

    monkeypatch.setattr(asr, "transcribe_word_aligned", fake_word_asr)
    output_json = tmp_path / "lines.json"

    lines = transcribe_dialogue_lines(
        audio_path=dx,
        source_stems=[source_a, source_b],
        output_json=output_json,
        output_csv=tmp_path / "lines.csv",
        force=True,
    )

    assert calls == [dx, source_a, source_b]
    assert [(line["text"], line["source_group"]) for line in lines] == [
        ("hello", "speaker_01"),
        ("world again", "speaker_02"),
    ]
    assert lines[1]["review_required"] is True
    loaded = load_dialogue_lines(output_json)
    assert loaded[0].review_required is True
    assert loaded[1].review_required is True
    assert loaded[1].transcription_source == "mixed"
    manifest = json.loads(output_json.read_text())
    assert manifest["transcription_mode"] == "per_stem"
    assert manifest["transcription_metadata"]["stem_count"] == 2
    assert manifest["transcription_metadata"]["dx_residual_word_count"] == 1
    assert manifest["transcription_metadata"]["dx"]["aligned"] is True


def test_transcribe_lines_legacy_mode_uses_dx_only(monkeypatch, tmp_path: Path) -> None:
    dx = tmp_path / "dx.wav"
    stem = tmp_path / "speaker_01_stem.wav"
    dx.touch()
    stem.touch()
    calls: list[Path] = []

    def fake_word_asr(path: Path, **_kwargs):
        calls.append(path)
        return (
            [{"word": "legacy", "start": 0.0, "end": 0.2, "speaker": ""}],
            {
                "language": "en",
                "aligned": True,
            },
        )

    monkeypatch.setattr(asr, "transcribe_word_aligned", fake_word_asr)
    output_json = tmp_path / "legacy.json"

    lines = transcribe_dialogue_lines(
        audio_path=dx,
        source_stems=[stem],
        per_stem=False,
        output_json=output_json,
        output_csv=tmp_path / "legacy.csv",
        force=True,
    )

    assert calls == [dx]
    assert lines[0]["text"] == "legacy"
    manifest = json.loads(output_json.read_text())
    assert manifest["transcription_mode"] == "dx"
    assert manifest["source_activity_threshold_db"] == -50.0


def test_injected_words_never_invoke_asr(monkeypatch, tmp_path: Path) -> None:
    def fail_asr(*_args, **_kwargs):
        raise AssertionError("ASR must not run for injected words")

    monkeypatch.setattr(asr, "transcribe_word_aligned", fail_asr)
    output_json = tmp_path / "injected.json"

    lines = transcribe_dialogue_lines(
        audio_path=tmp_path / "missing.wav",
        words=[{"word": "injected", "start": 0.0, "end": 0.2}],
        output_json=output_json,
        output_csv=tmp_path / "injected.csv",
        force=True,
    )

    assert lines[0]["text"] == "injected"
    assert json.loads(output_json.read_text())["transcription_mode"] == "injected"
