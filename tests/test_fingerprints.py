"""Input fingerprints and stale-artifact warnings for restartable steps."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from conftest import write_wav
from unrender.editorial.dialogue.lines import transcribe_dialogue_lines
from unrender.lib.fingerprints import changed_inputs, input_fingerprints


def test_input_fingerprints_files_and_lists(tmp_path: Path) -> None:
    one = tmp_path / "one.bin"
    two = tmp_path / "two.bin"
    one.write_bytes(b"aaa")
    two.write_bytes(b"bbb")

    fingerprints = input_fingerprints({"single": one, "many": [one, two], "missing": None})

    assert fingerprints["single"]["size"] == 3
    assert fingerprints["many"]["count"] == 2
    assert "missing" not in fingerprints


def test_changed_inputs_detects_content_change(tmp_path: Path) -> None:
    source = tmp_path / "input.bin"
    source.write_bytes(b"original")
    manifest = {"inputs": input_fingerprints({"source": source})}

    assert changed_inputs(manifest, {"source": source}) == []

    source.write_bytes(b"modified!")
    assert changed_inputs(manifest, {"source": source}) == ["source"]


def test_fingerprints_detect_url_and_scalar_setting_changes() -> None:
    original = {
        "source": "https://example.com/dx-v1.wav",
        "settings": {"model": "one", "speakers": 2},
    }
    manifest = {"inputs": input_fingerprints(original)}

    assert changed_inputs(manifest, original) == []
    assert changed_inputs(
        manifest,
        {
            "source": "https://example.com/dx-v2.wav",
            "settings": {"model": "two", "speakers": 2},
        },
    ) == ["source", "settings"]
    assert changed_inputs(manifest, {**original, "format": "wav"}) == ["format"]


def test_changed_inputs_ignores_unrecorded_inputs(tmp_path: Path) -> None:
    source = tmp_path / "input.bin"
    source.write_bytes(b"data")
    assert changed_inputs({}, {"source": source}) == []
    assert changed_inputs({"inputs": {}}, {"source": source}) == []


def test_skipped_step_warns_when_input_changed(tmp_path: Path, capsys) -> None:
    audio = tmp_path / "dx.wav"
    write_wav(audio, np.zeros(16_000, dtype=np.int16))
    words = [{"word": "hello", "start": 0.2, "end": 0.6}]

    transcribe_dialogue_lines(
        audio_path=audio,
        output_json=tmp_path / "lines.json",
        output_csv=tmp_path / "lines.csv",
        words=words,
    )
    capsys.readouterr()

    write_wav(audio, np.full(16_000, 500, dtype=np.int16))
    transcribe_dialogue_lines(
        audio_path=audio,
        output_json=tmp_path / "lines.json",
        output_csv=tmp_path / "lines.csv",
        words=words,
    )

    output = capsys.readouterr().out
    assert "WARNING: input(s) changed" in output
    assert "audio" in output


def test_skipped_step_stays_quiet_when_input_unchanged(tmp_path: Path, capsys) -> None:
    audio = tmp_path / "dx.wav"
    write_wav(audio, np.zeros(16_000, dtype=np.int16))
    words = [{"word": "hello", "start": 0.2, "end": 0.6}]

    for _ in range(2):
        transcribe_dialogue_lines(
            audio_path=audio,
            output_json=tmp_path / "lines.json",
            output_csv=tmp_path / "lines.csv",
            words=words,
        )

    assert "WARNING" not in capsys.readouterr().out
