from __future__ import annotations

import json
from pathlib import Path

import pytest

from unrender.editorial.shots.sources import load_cut_ranges, resolve_tc_origin

FPS = 24.0


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# tc origin


def test_resolve_tc_origin_precedence() -> None:
    assert resolve_tc_origin("zero", fps=FPS, probed_tc="01:00:00:00") == (0, "explicit zero")
    frame, source = resolve_tc_origin("01:00:00:00", fps=FPS)
    assert frame == 86_400 and source.startswith("explicit")
    frame, source = resolve_tc_origin(None, fps=FPS, probed_tc="01:00:00:00")
    assert frame == 86_400 and "master start timecode" in source
    frame, source = resolve_tc_origin(None, fps=FPS, auto_candidate_frames=100)
    assert frame == 100 and "auto" in source
    assert resolve_tc_origin(None, fps=FPS) == (0, "default 00:00:00:00")
    with pytest.raises(ValueError, match="invalid --tc-origin"):
        resolve_tc_origin("garbage:tc", fps=FPS)


# ---------------------------------------------------------------------------
# CSV


def test_csv_with_timecodes_and_origin(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "shots.csv",
        "shot_id,start_tc,end_tc\n001,01:00:00:00,01:00:02:00\n002,01:00:02:00,01:00:04:12\n",
    )
    cut_list = load_cut_ranges(path, fps=FPS, tc_origin="01:00:00:00", padding=3)
    assert [r.shot_id for r in cut_list.ranges] == ["001", "002"]
    assert cut_list.ranges[0].start_frame == 0
    assert cut_list.ranges[0].end_frame == 48
    assert cut_list.ranges[1].end_frame == 108
    assert cut_list.source_kind == "csv"


def test_csv_column_aliases_and_seconds(tmp_path: Path) -> None:
    path = _write(tmp_path / "shots.csv", "id,start,end\nA,0.0,2.0\nB,2.0,4.0\n")
    cut_list = load_cut_ranges(path, fps=FPS, padding=3)
    assert [r.shot_id for r in cut_list.ranges] == ["A", "B"]
    assert cut_list.ranges[1].start_frame == 48


def test_csv_auto_ids_are_padded(tmp_path: Path) -> None:
    path = _write(tmp_path / "shots.csv", "start_tc,end_tc\n00:00:00:00,00:00:01:00\n")
    cut_list = load_cut_ranges(path, fps=FPS, padding=4)
    assert cut_list.ranges[0].shot_id == "0001"


def test_csv_missing_columns(tmp_path: Path) -> None:
    path = _write(tmp_path / "shots.csv", "shot_id,foo\n1,2\n")
    with pytest.raises(ValueError, match="needs start/end columns"):
        load_cut_ranges(path, fps=FPS)


def test_csv_drop_frame_timecodes(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "shots.csv",
        "shot_id,start_tc,end_tc\n1,00:00:00;00,00:01:00;02\n",
    )
    cut_list = load_cut_ranges(path, fps=29.97, padding=3)
    assert cut_list.ranges[0].end_frame == 1_800


# ---------------------------------------------------------------------------
# TXT


def test_txt_ranges_two_and_three_tokens(tmp_path: Path) -> None:
    two = _write(tmp_path / "two.txt", "# comment\n00:00:00:00 00:00:02:00\n2.0, 4.0\n")
    cut_list = load_cut_ranges(two, fps=FPS, padding=3)
    assert [(r.start_frame, r.end_frame) for r in cut_list.ranges] == [(0, 48), (48, 96)]

    three = _write(tmp_path / "three.txt", "010 0 2.0\n020 2.0 4.0\n")
    cut_list = load_cut_ranges(three, fps=FPS, padding=3)
    assert [r.shot_id for r in cut_list.ranges] == ["010", "020"]


def test_txt_cut_points_build_segments(tmp_path: Path) -> None:
    path = _write(tmp_path / "cuts.txt", "00:00:02:00\n00:00:04:00\n")
    cut_list = load_cut_ranges(path, fps=FPS, duration_frames=144, padding=3)
    assert [(r.start_frame, r.end_frame) for r in cut_list.ranges] == [
        (0, 48),
        (48, 96),
        (96, 144),
    ]
    assert [r.shot_id for r in cut_list.ranges] == ["001", "002", "003"]


def test_txt_cut_points_need_duration(tmp_path: Path) -> None:
    path = _write(tmp_path / "cuts.txt", "00:00:02:00\n")
    with pytest.raises(ValueError, match="needs the master duration"):
        load_cut_ranges(path, fps=FPS, padding=3)


def test_txt_mixed_token_counts_error(tmp_path: Path) -> None:
    path = _write(tmp_path / "bad.txt", "0 2.0\n1 2.0 4.0\n")
    with pytest.raises(ValueError, match="one format throughout"):
        load_cut_ranges(path, fps=FPS, padding=3)


# ---------------------------------------------------------------------------
# EDL


EDL_TEXT = """TITLE: TEST
FCM: NON-DROP FRAME

001  REEL0001 V     C        00:00:00:00 00:00:02:00 01:00:00:00 01:00:02:00
* FROM CLIP NAME: OPENING

002  REEL0001 V     D    030 00:00:02:00 00:00:04:00 01:00:02:00 01:00:04:00
* FROM CLIP NAME: SECOND

003  REEL0001 AA    C        00:00:00:00 00:00:01:00 01:00:04:00 01:00:05:00
"""


def test_edl_record_tc_auto_origin(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(tmp_path / "shots.edl", EDL_TEXT)
    cut_list = load_cut_ranges(path, fps=FPS, padding=3)
    # Audio-only event skipped; origin auto-resolves to the earliest rec_in.
    assert len(cut_list.ranges) == 2
    assert cut_list.origin_frame == 86_400
    assert cut_list.ranges[0].start_frame == 0
    assert cut_list.ranges[0].label == "OPENING"
    assert cut_list.ranges[1].shot_id == "002"
    captured = capsys.readouterr()
    assert "transition 'D'" in captured.out


def test_edl_source_tc(tmp_path: Path) -> None:
    path = _write(tmp_path / "shots.edl", EDL_TEXT)
    cut_list = load_cut_ranges(path, fps=FPS, padding=3, edl_source_tc=True, tc_origin="zero")
    assert cut_list.ranges[0].start_frame == 0
    assert cut_list.ranges[1].start_frame == 48


# ---------------------------------------------------------------------------
# JSON


def test_json_manifest_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "shots.json"
    path.write_text(
        json.dumps(
            {
                "shots": [
                    {"shot_id": "001", "start_frame": 0, "end_frame": 48},
                    {"shot_id": "002", "start_sec": 2.0, "end_sec": 4.0},
                    {"shot_id": "003", "start_tc": "00:00:04:00", "end_tc": "00:00:05:00"},
                ]
            }
        ),
        encoding="utf-8",
    )
    cut_list = load_cut_ranges(path, fps=FPS, tc_origin="zero", padding=3)
    assert [(r.start_frame, r.end_frame) for r in cut_list.ranges] == [
        (0, 48),
        (48, 96),
        (96, 120),
    ]


# ---------------------------------------------------------------------------
# Validation


def test_duplicate_shot_ids_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "shots.csv", "shot_id,start,end\n1,0,1\n001,1,2\n")
    with pytest.raises(ValueError, match="duplicate shot_id"):
        load_cut_ranges(path, fps=FPS, padding=3)


def test_non_positive_duration_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "shots.csv", "shot_id,start,end\n1,2.0,2.0\n")
    with pytest.raises(ValueError, match="non-positive duration"):
        load_cut_ranges(path, fps=FPS, padding=3)


def test_negative_start_suggests_tc_origin(tmp_path: Path) -> None:
    path = _write(tmp_path / "shots.csv", "shot_id,start_tc,end_tc\n1,00:00:01:00,00:00:02:00\n")
    with pytest.raises(ValueError, match="tc-origin"):
        load_cut_ranges(path, fps=FPS, tc_origin="01:00:00:00", padding=3)


def test_end_clamped_within_one_frame(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(tmp_path / "shots.csv", "shot_id,start,end\n1,0,2.0\n")
    cut_list = load_cut_ranges(path, fps=FPS, duration_frames=47, padding=3)
    assert cut_list.ranges[0].end_frame == 47
    assert "clamped" in capsys.readouterr().out


def test_end_far_past_master_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "shots.csv", "shot_id,start,end\n1,0,4.0\n")
    with pytest.raises(ValueError, match="past the master"):
        load_cut_ranges(path, fps=FPS, duration_frames=48, padding=3)


def test_overlap_and_gap_warnings(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(
        tmp_path / "shots.csv",
        "shot_id,start,end\n1,0,2.0\n2,1.5,3.0\n3,3.5,4.0\n",
    )
    cut_list = load_cut_ranges(path, fps=FPS, padding=3)
    assert len(cut_list.ranges) == 3
    captured = capsys.readouterr().out
    assert "overlaps" in captured
    assert "uncovered" in captured


def test_empty_list_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "shots.csv", "shot_id,start,end\n")
    with pytest.raises(ValueError, match="no shots found"):
        load_cut_ranges(path, fps=FPS)


def test_unsupported_extension(tmp_path: Path) -> None:
    path = _write(tmp_path / "shots.xml", "<x/>")
    with pytest.raises(ValueError, match="unsupported shot list format"):
        load_cut_ranges(path, fps=FPS)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_cut_ranges(tmp_path / "nope.csv", fps=FPS)
