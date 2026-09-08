"""Unit tests for the OTIO-independent timeline loaders and media prober."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from unrender.editorial.timeline import MediaProber
from unrender.editorial.timeline.sources import (
    load_dialogue_clips,
    load_shot_matches,
    load_source_stems,
    source_stem_tracks,
)
from unrender.project import RunPaths


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_load_shot_matches_ignores_malformed_entries(tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()
    _write_json(
        run.shot_matches_json,
        {
            "shots": [
                {"shot_id": "001", "accepted": ["ALEX", "JORDAN"]},
                {"shot_id": "002", "accepted": []},
                {"shot_id": "", "accepted": ["SAM"]},
                {"shot_id": "003", "accepted": "not-a-list"},
                "not-a-dict",
            ]
        },
    )

    assert load_shot_matches(run) == {"001": ["ALEX", "JORDAN"]}


def test_load_shot_matches_missing_file_returns_empty(tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()
    assert load_shot_matches(run) == {}


def test_load_source_stems_prefers_manifest(tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()
    _write_json(
        run.source_separation_json,
        {"stems": {"music_fx": "/tmp/mx.wav", "effects": "", "dialogue": "/tmp/dx.wav"}},
    )

    stems = load_source_stems(run)

    assert stems == {"music_fx": Path("/tmp/mx.wav"), "dialogue": Path("/tmp/dx.wav")}


def test_load_source_stems_falls_back_to_directory_glob(tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()
    mx = run.source_stems_dir / "show_MX_stem.wav"
    dx = run.source_stems_dir / "show_DX_stem.wav"
    mx.write_bytes(b"\x00")
    dx.write_bytes(b"\x00")

    stems = load_source_stems(run)

    assert stems == {"music_fx": mx, "dialogue": dx}


def test_source_stem_tracks_orders_and_dedupes(tmp_path: Path) -> None:
    paths = {}
    for key in ("music_fx", "music", "effects", "dialogue", "ambience"):
        path = tmp_path / f"{key}.wav"
        path.write_bytes(b"\x00")
        paths[key] = path

    tracks = source_stem_tracks(paths, include_music_effects=True, include_dialogue_bed=True)

    labels = [label for label, _, _ in tracks]
    # music_fx wins the MX label over music; unknown keys get their own track;
    # the dialogue bed is always last.
    assert labels == ["MX", "FX", "AMBIENCE", "DX bed"]
    assert tracks[0][2] == paths["music_fx"]

    no_bed = source_stem_tracks(paths, include_music_effects=True, include_dialogue_bed=False)
    assert [label for label, _, _ in no_bed] == ["MX", "FX", "AMBIENCE"]

    bed_only = source_stem_tracks(paths, include_music_effects=False, include_dialogue_bed=True)
    assert [label for label, _, _ in bed_only] == ["DX bed"]


def test_source_stem_tracks_skips_missing_files(tmp_path: Path) -> None:
    tracks = source_stem_tracks(
        {"music_fx": tmp_path / "missing.wav"},
        include_music_effects=True,
        include_dialogue_bed=True,
    )
    assert tracks == []


def test_line_clips_skip_entries_with_bad_timing(tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()
    _write_json(
        run.dialogue_stem_plan_json,
        {
            "clips": [
                {"line_id": "L1", "speaker": "alex", "start_sec": 0.0, "end_sec": 1.0},
                {"line_id": "L2", "start_sec": "not-a-number", "end_sec": 2.0},
                {"line_id": "L3", "start_sec": 3.0},
                {"line_id": "L4", "start_sec": 5.0, "end_sec": 4.0},
            ]
        },
    )

    clips, granularity = load_dialogue_clips(run, {}, "lines")

    assert granularity == "lines"
    assert [clip.name for clip in clips] == ["L1"]
    assert clips[0].group == "ALEX"


def test_shot_granularity_uses_shot_windows(tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()
    stem = run.shot_mapped_dir / "001_stem.wav"
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.write_bytes(b"\x00")
    _write_json(
        run.shot_stem_plan_json,
        {
            "shots": [
                {"shot_id": "001", "speaker": "ALEX", "stem_path": str(stem), "status": "matched"},
                {"shot_id": "999", "speaker": "SAM", "stem_path": str(stem)},
            ]
        },
    )
    windows = {"001": (10.0, 12.0)}

    clips, granularity = load_dialogue_clips(run, windows, "shots")

    assert granularity == "shots"
    assert len(clips) == 1
    clip = clips[0]
    assert (clip.name, clip.start_sec, clip.end_sec) == ("001", 10.0, 12.0)
    assert clip.media == stem
    assert clip.group == "ALEX"


def test_load_dialogue_clips_reports_none_when_no_artifacts(tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()
    assert load_dialogue_clips(run, {}, "auto") == ([], "none")


def _ffprobe_payload() -> str:
    return json.dumps(
        {
            "format": {"duration": "4.25", "start_time": "0.5"},
            "streams": [
                {"codec_type": "video", "duration": "4.25"},
                {"codec_type": "audio", "duration": "4.20"},
            ],
        }
    )


def test_media_prober_uses_ffprobe_for_video(tmp_path: Path, monkeypatch) -> None:
    media = tmp_path / "clip.mov"
    media.write_bytes(b"\x00")
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=_ffprobe_payload(), stderr="")

    monkeypatch.setattr(
        "unrender.editorial.timeline.media.shutil.which", lambda name: "/usr/bin/ffprobe"
    )
    monkeypatch.setattr("unrender.lib.proc.subprocess.run", fake_run)

    prober = MediaProber()
    info = prober.probe(media)

    assert info is not None
    assert info.duration_sec == pytest.approx(4.25)
    assert info.start_sec == pytest.approx(0.5)
    assert info.has_video is True and info.has_audio is True

    # Second probe of the same path is served from the cache.
    assert prober.probe(media) is info
    assert len(calls) == 1


def test_media_prober_handles_ffprobe_failure(tmp_path: Path, monkeypatch) -> None:
    media = tmp_path / "clip.mov"
    media.write_bytes(b"\x00")

    def fail_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(
        "unrender.editorial.timeline.media.shutil.which", lambda name: "/usr/bin/ffprobe"
    )
    monkeypatch.setattr("unrender.lib.proc.subprocess.run", fail_run)

    assert MediaProber().probe(media) is None


def test_media_prober_without_ffprobe_returns_none(tmp_path: Path, monkeypatch) -> None:
    media = tmp_path / "clip.mov"
    media.write_bytes(b"\x00")
    monkeypatch.setattr("unrender.editorial.timeline.media.shutil.which", lambda name: None)

    prober = MediaProber()
    assert prober.available is False
    assert prober.probe(media) is None
