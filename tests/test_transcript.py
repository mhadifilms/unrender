from __future__ import annotations

import json
from pathlib import Path

from unrender.editorial.dialogue.transcript import (
    extract_speakers_from_transcript,
    parse_transcript,
)
from unrender.project.config import load_project_config


def test_parse_transcript_csv_infers_windows_and_speakers(tmp_path: Path) -> None:
    script = tmp_path / "TRANSCRIPT.csv"
    script.write_text(
        "Timecode,Character,English Dialogue\n"
        '00:00:10:00,Alex,"Hello there."\n'
        '00:00:12:00,Jordan,"General Kenobi."\n'
        '00:00:14:00,ON-SCREEN TEXT,"Title card."\n',
        encoding="utf-8",
    )

    lines = parse_transcript(script, fps=24.0, handle_sec=0.2)

    assert [line["line_id"] for line in lines] == ["DL_000001", "DL_000002"]
    assert [line["speaker"] for line in lines] == ["ALEX", "JORDAN"]
    assert lines[0]["start_sec"] == 10.0
    assert lines[0]["end_sec"] == 12.0
    assert lines[0]["cut_start_sec"] == 9.8
    assert lines[0]["cut_end_sec"] == 12.2
    assert lines[0]["text"] == "Hello there"


def test_extract_speakers_from_transcript_is_unique(tmp_path: Path) -> None:
    script = tmp_path / "script.csv"
    script.write_text(
        "start_sec,end_sec,speaker,text\n" "1,2,Alex,one\n" "2,3,ALEX,two\n" "3,4,Jordan,three\n",
        encoding="utf-8",
    )

    assert extract_speakers_from_transcript(script) == ["ALEX", "JORDAN"]


def test_project_config_can_seed_speakers_from_transcript(tmp_path: Path) -> None:
    script = tmp_path / "script.csv"
    script.write_text(
        "start_sec,end_sec,speaker,text\n1,2,Alex,hello\n2,3,Jordan,bye\n",
        encoding="utf-8",
    )
    config = tmp_path / "show.json"
    config.write_text(
        json.dumps({"paths": {"run_dir": str(tmp_path / "run"), "transcript": str(script)}}),
        encoding="utf-8",
    )

    project = load_project_config(config)

    assert sorted(project.data["speakers"]) == ["ALEX", "JORDAN"]
