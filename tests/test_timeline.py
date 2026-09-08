from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from unrender.cli import main
from unrender.editorial.timeline import MediaProber, build_timeline, write_timeline
from unrender.project import RunPaths

otio = pytest.importorskip("opentimelineio")


def _write_plan(run: RunPaths, clips: list[dict]) -> None:
    run.dialogue_stem_plan_json.write_text(
        json.dumps({"version": "1.0", "clips": clips}), encoding="utf-8"
    )


def _wav(path: Path, seconds: float, sample_rate: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(np.zeros(int(sample_rate * seconds), dtype=np.int16).tobytes())


def _seed_run(tmp_path: Path) -> tuple[RunPaths, Path]:
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()

    shot_video = tmp_path / "shot_001.mov"
    shot_video.write_bytes(b"\x00")
    shots_csv = tmp_path / "shots.csv"
    shots_csv.write_text(
        f"shot_id,video_path,start_sec,end_sec,existing_speaker\n001,{shot_video},10.0,12.0,ALEX\n",
        encoding="utf-8",
    )

    dx = run.dialogue_mapped_dir / "DL_000001" / "DL_000001_ALEX_stem.wav"
    _wav(dx, 1.2)
    run.dialogue_stem_plan_json.write_text(
        json.dumps(
            {
                "version": "1.0",
                "source": "clip_stem_plan",
                "clips": [
                    {
                        "clip_id": "DL_000001_speaker_01",
                        "clip_type": "dialogue_line",
                        "line_id": "DL_000001",
                        "speaker": "ALEX",
                        "stem_path": str(dx),
                        "dialogue_stem_path": str(dx),
                        "source_group": "speaker_01",
                        "start_sec": 10.4,
                        "end_sec": 11.4,
                        "clip_start_sec": 10.3,
                        "clip_end_sec": 11.5,
                        "status": "matched",
                        "text": "hello",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    mx = run.source_stems_dir / "show_MX_stem.wav"
    fx = run.source_stems_dir / "show_FX_stem.wav"
    _wav(mx, 3.0)
    _wav(fx, 3.0)
    run.source_separation_json.write_text(
        json.dumps(
            {
                "version": "1.0",
                "backend": "audioshake",
                "stems": {"music_fx": str(mx), "effects": str(fx)},
            }
        ),
        encoding="utf-8",
    )
    return run, shots_csv


def test_build_timeline_places_video_dialogue_and_source_tracks(tmp_path: Path) -> None:
    run, shots_csv = _seed_run(tmp_path)

    timeline, summary = build_timeline(run, shots_path=shots_csv, fps=24.0, name="show")

    assert summary.video_clips == 1
    assert summary.dialogue_clips == 1
    assert summary.source_clips == 2
    assert summary.dialogue_granularity == "lines"
    assert summary.missing_media == 0

    track_names = {track.name: track for track in timeline.tracks}
    assert "V1" in track_names
    assert "DX ALEX" in track_names
    assert "MX" in track_names and "FX" in track_names

    video = track_names["V1"]
    clips = list(video.find_clips())
    assert len(clips) == 1
    placed = clips[0].range_in_parent()
    assert placed.start_time.to_seconds() == pytest.approx(10.0, abs=0.05)
    assert placed.duration.to_seconds() == pytest.approx(2.0, abs=0.05)


def test_overlapping_dialogue_splits_into_lanes(tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()
    a = run.dialogue_mapped_dir / "a.wav"
    b = run.dialogue_mapped_dir / "b.wav"
    _wav(a, 2.0)
    _wav(b, 2.0)
    run.dialogue_stem_plan_json.write_text(
        json.dumps(
            {
                "version": "1.0",
                "clips": [
                    {
                        "line_id": "L1",
                        "speaker": "ALEX",
                        "dialogue_stem_path": str(a),
                        "start_sec": 0.0,
                        "end_sec": 2.0,
                        "status": "matched",
                    },
                    {
                        "line_id": "L2",
                        "speaker": "ALEX",
                        "dialogue_stem_path": str(b),
                        "start_sec": 1.0,
                        "end_sec": 3.0,
                        "status": "matched",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    timeline, summary = build_timeline(run, fps=24.0, name="overlap")

    dx_tracks = [t for t in timeline.tracks if t.name.startswith("DX ALEX")]
    assert len(dx_tracks) == 2
    assert summary.dialogue_clips == 2


def test_missing_media_becomes_placeholder(tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()
    run.dialogue_lines_json.write_text(
        json.dumps(
            {
                "version": "1.0",
                "lines": [{"line_id": "L1", "speaker": "ALEX", "start_sec": 0.0, "end_sec": 1.0}],
            }
        ),
        encoding="utf-8",
    )

    timeline, summary = build_timeline(run, fps=24.0, dialogue="lines", name="m")

    assert summary.dialogue_clips == 1
    assert summary.missing_media == 1
    clip = next(iter(timeline.find_clips()))
    assert isinstance(clip.media_reference, otio.schema.MissingReference)


def test_write_timeline_rejects_unknown_suffix(tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()
    run.dialogue_lines_json.write_text(
        json.dumps({"lines": [{"line_id": "L1", "start_sec": 0.0, "end_sec": 1.0}]}),
        encoding="utf-8",
    )
    timeline, _ = build_timeline(run, fps=24.0, dialogue="lines", name="m")
    with pytest.raises(ValueError, match="adapter"):
        write_timeline(timeline, tmp_path / "out.unknownext")


def test_cli_timeline_build_writes_otio(tmp_path: Path) -> None:
    run, shots_csv = _seed_run(tmp_path)

    assert main(["timeline", "build", "--run-dir", str(run.root), "--shots", str(shots_csv)]) == 0

    out = run.timeline_otio
    assert out.exists()
    timeline = otio.adapters.read_from_file(str(out))
    assert timeline.name == run.root.name
    assert len(list(timeline.tracks)) >= 3


def test_markers_are_color_coded_by_status(tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()
    matched = run.dialogue_mapped_dir / "m.wav"
    offscreen = run.dialogue_mapped_dir / "o.wav"
    _wav(matched, 1.0)
    _wav(offscreen, 1.0)
    _write_plan(
        run,
        [
            {
                "line_id": "L1",
                "speaker": "ALEX",
                "dialogue_stem_path": str(matched),
                "start_sec": 0.0,
                "end_sec": 1.0,
                "status": "matched",
                "text": "hi",
            },
            {
                "line_id": "L2",
                "speaker": "JORDAN",
                "dialogue_stem_path": str(offscreen),
                "start_sec": 2.0,
                "end_sec": 3.0,
                "status": "offscreen_speaker",
                "text": "yo",
            },
            {
                "line_id": "L3",
                "speaker": "JORDAN",
                "stem_path": "",
                "start_sec": 4.0,
                "end_sec": 5.0,
                "status": "no_matching_stem",
                "text": "gone",
            },
        ],
    )

    timeline, summary = build_timeline(run, fps=24.0, name="m")

    colors = {}
    for clip in timeline.find_clips():
        if clip.markers:
            colors[clip.name] = clip.markers[0].color
    assert colors["L1"] == otio.schema.MarkerColor.GREEN
    assert colors["L2"] == otio.schema.MarkerColor.ORANGE
    assert colors["L3"] == otio.schema.MarkerColor.RED
    assert summary.markers == 3


def test_no_markers_flag_disables_markers(tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()
    stem = run.dialogue_mapped_dir / "m.wav"
    _wav(stem, 1.0)
    _write_plan(
        run,
        [
            {
                "line_id": "L1",
                "speaker": "ALEX",
                "dialogue_stem_path": str(stem),
                "start_sec": 0.0,
                "end_sec": 1.0,
                "status": "matched",
                "text": "hi",
            }
        ],
    )

    timeline, summary = build_timeline(run, fps=24.0, name="m", markers=False)

    assert summary.markers == 0
    assert all(not clip.markers for clip in timeline.find_clips())


def test_start_timecode_sets_global_start(tmp_path: Path) -> None:
    run, shots_csv = _seed_run(tmp_path)

    timeline, summary = build_timeline(
        run, shots_path=shots_csv, fps=24.0, name="tc", start_timecode="01:00:00:00"
    )

    assert otio.opentime.to_timecode(timeline.global_start_time, 24.0) == "01:00:00:00"
    assert summary.start_timecode == "01:00:00:00"


def test_invalid_start_timecode_raises(tmp_path: Path) -> None:
    run, shots_csv = _seed_run(tmp_path)
    with pytest.raises(ValueError, match="start-timecode"):
        build_timeline(run, shots_path=shots_csv, fps=24.0, start_timecode="not-a-tc:::")


def test_provenance_metadata_recorded(tmp_path: Path) -> None:
    run, shots_csv = _seed_run(tmp_path)

    timeline, _ = build_timeline(run, shots_path=shots_csv, fps=24.0, name="p")

    provenance = timeline.metadata["unrender"]
    assert provenance["fps"] == 24.0
    assert provenance["run_dir"] == str(run.root)
    assert provenance["dialogue_granularity"] == "lines"
    assert "version" in provenance and "generated_at" in provenance


def test_dialogue_tracks_ordered_by_speaking_time(tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()
    quiet = run.dialogue_mapped_dir / "q.wav"
    loud = run.dialogue_mapped_dir / "l.wav"
    _wav(quiet, 1.0)
    _wav(loud, 5.0)
    _write_plan(
        run,
        [
            {
                "line_id": "L1",
                "speaker": "QUIET",
                "dialogue_stem_path": str(quiet),
                "start_sec": 0.0,
                "end_sec": 1.0,
                "status": "matched",
            },
            {
                "line_id": "L2",
                "speaker": "LOUD",
                "dialogue_stem_path": str(loud),
                "start_sec": 2.0,
                "end_sec": 7.0,
                "status": "matched",
            },
        ],
    )

    timeline, summary = build_timeline(run, fps=24.0, name="order")

    dx_order = [t.name for t in timeline.tracks if t.name.startswith("DX")]
    assert dx_order == ["DX LOUD", "DX QUIET"]
    assert summary.speakers == ("LOUD", "QUIET")


def test_available_range_uses_probed_duration(tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()
    stem = run.dialogue_mapped_dir / "m.wav"
    _wav(stem, 1.5)
    _write_plan(
        run,
        [
            {
                "line_id": "L1",
                "speaker": "ALEX",
                "dialogue_stem_path": str(stem),
                "start_sec": 0.0,
                "end_sec": 9.9,
                "status": "matched",
            }
        ],
    )

    timeline, summary = build_timeline(run, fps=24.0, name="probe")

    clip = next(iter(timeline.find_clips()))
    # The 1.5s media wins over the 9.9s artifact window.
    assert clip.source_range.duration.to_seconds() == pytest.approx(1.5, abs=0.05)
    assert clip.available_range().duration.to_seconds() == pytest.approx(1.5, abs=0.05)
    assert summary.probed_media >= 1


def test_cli_dry_run_writes_nothing(tmp_path: Path) -> None:
    run, shots_csv = _seed_run(tmp_path)

    assert (
        main(
            [
                "timeline",
                "build",
                "--run-dir",
                str(run.root),
                "--shots",
                str(shots_csv),
                "--dry-run",
            ]
        )
        == 0
    )
    assert not run.timeline_otio.exists()


def test_cli_bundle_writes_otiod(tmp_path: Path) -> None:
    run, shots_csv = _seed_run(tmp_path)

    assert (
        main(
            ["timeline", "build", "--run-dir", str(run.root), "--shots", str(shots_csv), "--bundle"]
        )
        == 0
    )
    assert run.timeline_otio.exists()
    assert run.timeline_otio.with_suffix(".otiod").exists()


def test_cli_no_source_stems_and_no_markers(tmp_path: Path) -> None:
    run, shots_csv = _seed_run(tmp_path)

    assert (
        main(
            [
                "timeline",
                "build",
                "--run-dir",
                str(run.root),
                "--shots",
                str(shots_csv),
                "--no-source-stems",
                "--no-markers",
            ]
        )
        == 0
    )

    timeline = otio.adapters.read_from_file(str(run.timeline_otio))
    track_names = {track.name for track in timeline.tracks}
    assert "MX" not in track_names and "FX" not in track_names
    assert all(not clip.markers for clip in timeline.find_clips())


def test_dialogue_bed_flag_adds_dx_bed_track(tmp_path: Path) -> None:
    run, shots_csv = _seed_run(tmp_path)
    bed = run.source_stems_dir / "show_DX_stem.wav"
    _wav(bed, 3.0)
    data = json.loads(run.source_separation_json.read_text(encoding="utf-8"))
    data["stems"]["dialogue"] = str(bed)
    run.source_separation_json.write_text(json.dumps(data), encoding="utf-8")

    timeline, summary = build_timeline(
        run, shots_path=shots_csv, fps=24.0, name="bed", include_dialogue_bed=True
    )

    track_names = [track.name for track in timeline.tracks]
    assert "DX bed" in track_names
    assert summary.source_clips == 3


def test_include_mne_adds_fx_room_tone_and_cue_tracks(tmp_path: Path) -> None:
    run, shots_csv = _seed_run(tmp_path)
    amb = run.mne_fx_dir / "show_AMB_stem.wav"
    hfx = run.mne_fx_dir / "show_HFX_stem.wav"
    rt = run.mne_room_tone_dir / "show_RT_stem.wav"
    cue = run.mne_music_dir / "cues" / "show_cue_01.wav"
    for path in (amb, hfx, rt, cue):
        _wav(path, 2.0)
    run.mne_fx_classification_json.write_text(
        json.dumps(
            {
                "version": "1.0",
                "categories": {"ambience": str(amb), "hard_fx": str(hfx)},
                "events": [],
            }
        ),
        encoding="utf-8",
    )
    run.mne_room_tone_json.write_text(
        json.dumps({"version": "1.0", "rt_stem": str(rt), "groups": []}), encoding="utf-8"
    )
    run.mne_music_cues_json.write_text(
        json.dumps(
            {
                "version": "1.0",
                "cues": [
                    {
                        "cue_id": "cue_01",
                        "path": str(cue),
                        "start_sec": 10.0,
                        "end_sec": 12.0,
                        "pad_start_sec": 10.0,
                        "pad_end_sec": 12.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    timeline, _ = build_timeline(run, shots_path=shots_csv, fps=24.0, name="mne", include_mne=True)
    track_names = [track.name for track in timeline.tracks]
    assert "AMB" in track_names
    assert "HFX" in track_names
    assert "RT" in track_names
    assert "MX cues" in track_names

    # Off unless requested.
    without, _ = build_timeline(run, shots_path=shots_csv, fps=24.0, name="mne2")
    without_names = [track.name for track in without.tracks]
    assert "AMB" not in without_names
    assert "MX cues" not in without_names


def test_cli_include_mne_defaults_on_when_artifacts_present(tmp_path: Path, monkeypatch) -> None:
    run, shots_csv = _seed_run(tmp_path)
    cue = run.mne_music_dir / "cues" / "show_cue_01.wav"
    _wav(cue, 2.0)
    run.mne_music_cues_json.write_text(
        json.dumps(
            {
                "version": "1.0",
                "cues": [
                    {
                        "cue_id": "cue_01",
                        "path": str(cue),
                        "pad_start_sec": 10.0,
                        "pad_end_sec": 12.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out = run.root / "timeline.otio"
    assert (
        main(
            [
                "timeline",
                "build",
                "--run-dir",
                str(run.root),
                "--shots",
                str(shots_csv),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    timeline = otio.adapters.read_from_file(str(out))
    assert "MX cues" in [track.name for track in timeline.tracks]


def test_on_screen_speakers_get_cyan_video_marker(tmp_path: Path) -> None:
    run, shots_csv = _seed_run(tmp_path)

    timeline, _ = build_timeline(run, shots_path=shots_csv, fps=24.0, name="v")

    video = next(track for track in timeline.tracks if track.name == "V1")
    clip = next(iter(video.find_clips()))
    assert len(clip.markers) == 1
    marker = clip.markers[0]
    assert marker.color == otio.schema.MarkerColor.CYAN
    assert marker.name == "on-screen: ALEX"


def test_relative_media_paths_become_file_uris(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()
    _wav(tmp_path / "stem.wav", 1.0)
    _write_plan(
        run,
        [
            {
                "line_id": "L1",
                "speaker": "ALEX",
                "dialogue_stem_path": "stem.wav",
                "start_sec": 0.0,
                "end_sec": 1.0,
                "status": "matched",
            }
        ],
    )

    timeline, _ = build_timeline(run, fps=24.0, name="rel")

    clip = next(iter(timeline.find_clips()))
    assert clip.media_reference.target_url == (tmp_path / "stem.wav").resolve().as_uri()


def test_bad_plan_timing_is_skipped_not_fatal(tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()
    stem = run.dialogue_mapped_dir / "ok.wav"
    _wav(stem, 1.0)
    _write_plan(
        run,
        [
            {
                "line_id": "BAD",
                "speaker": "ALEX",
                "dialogue_stem_path": str(stem),
                "start_sec": "not-a-number",
                "end_sec": 1.0,
            },
            {
                "line_id": "OK",
                "speaker": "ALEX",
                "dialogue_stem_path": str(stem),
                "start_sec": 2.0,
                "end_sec": 3.0,
                "status": "matched",
            },
        ],
    )

    timeline, summary = build_timeline(run, fps=24.0, name="bad")

    assert summary.dialogue_clips == 1
    assert next(iter(timeline.find_clips())).name == "OK"


def test_shot_dx_track_is_a_single_layer(tmp_path: Path) -> None:
    run, shots_csv = _seed_run(tmp_path)
    dx_a = run.shot_dx_dir / "001_DX_stem.wav"
    dx_b = run.shot_dx_dir / "002_DX_stem.wav"
    _wav(dx_a, 2.0)
    _wav(dx_b, 1.0)
    run.shot_dx_plan_json.write_text(
        json.dumps(
            {
                "version": "1.0",
                "shots": [
                    {"shot_id": "001", "stem_path": str(dx_a), "start_sec": 10.0, "end_sec": 12.0},
                    {"shot_id": "002", "stem_path": str(dx_b), "start_sec": 13.0, "end_sec": 14.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    timeline, _ = build_timeline(
        run, shots_path=shots_csv, fps=24.0, name="dx", include_shot_dx=True
    )

    dx_tracks = [track for track in timeline.tracks if track.name.startswith("DX shots")]
    assert len(dx_tracks) == 1
    clips = list(dx_tracks[0].find_clips())
    assert [clip.name for clip in clips] == ["001 DX", "002 DX"]
    placed = clips[0].range_in_parent()
    assert placed.start_time.to_seconds() == pytest.approx(10.0, abs=0.05)
    assert placed.duration.to_seconds() == pytest.approx(2.0, abs=0.05)

    # Off by default.
    without, _ = build_timeline(run, shots_path=shots_csv, fps=24.0, name="dx2")
    assert not [track for track in without.tracks if track.name.startswith("DX shots")]


def test_media_prober_reads_wav_and_handles_missing(tmp_path: Path) -> None:
    prober = MediaProber()
    wav_path = tmp_path / "a.wav"
    _wav(wav_path, 2.0)

    info = prober.probe(wav_path)
    assert info is not None
    assert info.duration_sec == pytest.approx(2.0, abs=0.01)
    assert info.has_audio is True

    assert prober.probe(tmp_path / "missing.wav") is None


def _seed_scene_run(tmp_path: Path) -> tuple[RunPaths, Path, Path]:
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()
    rows = ["shot_id,video_path,start_sec,end_sec"]
    for index in range(4):
        video = tmp_path / f"shot_{index + 1:03d}.mov"
        video.write_bytes(b"\x00")
        rows.append(f"{index + 1:03d},{video},{float(index)},{float(index + 1)}")
    shots_csv = tmp_path / "shots.csv"
    shots_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    scenes_json = tmp_path / "scenes.json"
    scenes_json.write_text(
        json.dumps(
            {
                "version": "1.0",
                "scenes": [
                    {
                        "scene_id": "001",
                        "shot_ids": ["001", "002"],
                        "start_sec": 0.0,
                        "end_sec": 2.0,
                    },
                    {
                        "scene_id": "002",
                        "shot_ids": ["003", "004"],
                        "start_sec": 2.0,
                        "end_sec": 4.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return run, shots_csv, scenes_json


def test_scenes_add_markers_and_clip_metadata(tmp_path: Path) -> None:
    run, shots_csv, scenes_json = _seed_scene_run(tmp_path)
    timeline, summary = build_timeline(
        run, shots_path=shots_csv, scenes_path=scenes_json, fps=24.0, name="sc"
    )
    assert summary.scenes == 2

    video_tracks = [track for track in timeline.tracks if track.kind == "Video"]
    assert len(video_tracks) == 1  # flat layout preserved
    scene_markers = [m for m in video_tracks[0].markers if m.name.startswith("SC")]
    assert [m.name for m in scene_markers] == ["SC001", "SC002"]
    assert all(m.color == otio.schema.MarkerColor.PURPLE for m in scene_markers)

    clips = [c for c in video_tracks[0] if isinstance(c, otio.schema.Clip)]
    scene_ids = [c.metadata["unrender"]["scene_id"] for c in clips]
    assert scene_ids == ["001", "001", "002", "002"]

    enumerated = timeline.metadata["unrender"]["scenes"]
    assert [s["scene_id"] for s in enumerated] == ["001", "002"]
    assert list(enumerated[0]["shot_ids"]) == ["001", "002"]


def test_scenes_absent_manifest_is_noop(tmp_path: Path) -> None:
    run, shots_csv, _ = _seed_scene_run(tmp_path)
    timeline, summary = build_timeline(
        run, shots_path=shots_csv, scenes_path=tmp_path / "missing.json", fps=24.0, name="none"
    )
    assert summary.scenes == 0
    video_tracks = [track for track in timeline.tracks if track.kind == "Video"]
    assert not any(m.name.startswith("SC") for m in video_tracks[0].markers)
    assert "scenes" not in timeline.metadata["unrender"]
    clips = [c for c in video_tracks[0] if isinstance(c, otio.schema.Clip)]
    assert all("scene_id" not in (c.metadata.get("unrender") or {}) for c in clips)


def test_scene_stacks_nest_video_by_scene(tmp_path: Path) -> None:
    run, shots_csv, scenes_json = _seed_scene_run(tmp_path)
    timeline, summary = build_timeline(
        run,
        shots_path=shots_csv,
        scenes_path=scenes_json,
        scene_stacks=True,
        fps=24.0,
        name="stacks",
    )
    assert summary.scenes == 2
    video_tracks = [track for track in timeline.tracks if track.kind == "Video"]
    assert len(video_tracks) == 1
    stacks = [child for child in video_tracks[0] if isinstance(child, otio.schema.Stack)]
    assert [stack.name for stack in stacks] == ["SC001", "SC002"]
    for stack in stacks:
        inner_tracks = [t for t in stack if isinstance(t, otio.schema.Track)]
        assert len(inner_tracks) == 1
        clips = [c for c in inner_tracks[0] if isinstance(c, otio.schema.Clip)]
        assert len(clips) == 2  # two shots per scene
    assert summary.video_clips == 4
