from __future__ import annotations

from pathlib import Path

import pytest

from conftest import make_png_sequence, make_video, requires_ffmpeg
from unrender.editorial.shots.cut import cut_shots
from unrender.editorial.shots.manifest import plan_shot_media, shot_manifest_entries
from unrender.editorial.shots.sources import CutRange
from unrender.lib.sequence import detect_sequence
from unrender.lib.video import count_video_frames, probe_video
from unrender.project import RunPaths

FPS = 24.0


def _ranges() -> list[CutRange]:
    return [
        CutRange(shot_id="001", start_frame=0, end_frame=24),
        CutRange(shot_id="002", start_frame=24, end_frame=48),
    ]


@requires_ffmpeg
def test_cut_prores_master_stream_copy(tmp_path: Path) -> None:
    master = make_video(
        tmp_path / "master.mov", codec="prores", frames=48, audio=True, timecode="01:00:00:00"
    )
    ranges = _ranges()
    plans = plan_shot_media(
        ranges,
        shots_dir=tmp_path / "shots",
        shot_prefix="show",
        master_kind="video",
        master_ext=".mov",
    )
    outcomes = cut_shots(
        master=master,
        master_kind="video",
        ranges=ranges,
        plans=plans,
        fps=FPS,
        origin_frame=86_400,
    )
    assert all(outcome.status == "created" for outcome in outcomes)
    for range_, plan in zip(ranges, plans, strict=True):
        probe = probe_video(plan.plate_path)
        assert probe.codec == "prores"  # stream copy preserved the codec
        assert probe.has_audio
        assert count_video_frames(plan.plate_path) == range_.num_frames
    # Timecode track carries the shot's absolute start.
    assert probe_video(plans[0].plate_path).start_timecode == "01:00:00:00"
    assert probe_video(plans[1].plate_path).start_timecode == "01:00:01:00"

    # Second run skips without --force.
    outcomes = cut_shots(
        master=master, master_kind="video", ranges=ranges, plans=plans, fps=FPS, origin_frame=0
    )
    assert all(outcome.status == "skipped" for outcome in outcomes)


@requires_ffmpeg
def test_cut_h264_master_reencodes(tmp_path: Path) -> None:
    master = make_video(tmp_path / "master.mp4", codec="h264", frames=48)
    ranges = _ranges()
    plans = plan_shot_media(
        ranges,
        shots_dir=tmp_path / "shots",
        shot_prefix="show",
        master_kind="video",
        master_ext=".mp4",
    )
    outcomes = cut_shots(
        master=master, master_kind="video", ranges=ranges, plans=plans, fps=FPS, origin_frame=0
    )
    assert all(outcome.status == "created" for outcome in outcomes)
    for range_, plan in zip(ranges, plans, strict=True):
        assert probe_video(plan.plate_path).codec == "h264"
        assert count_video_frames(plan.plate_path) == range_.num_frames


@requires_ffmpeg
def test_cut_sequence_master_preserves_frame_files(tmp_path: Path) -> None:
    seq_dir = make_png_sequence(tmp_path / "seq", frames=48, start=1001, prefix="show.", padding=7)
    sequence = detect_sequence(seq_dir)
    ranges = _ranges()
    plans = plan_shot_media(
        ranges, shots_dir=tmp_path / "shots", shot_prefix="show", master_kind="sequence"
    )
    run = RunPaths.from_path(tmp_path / "run")
    proxy = make_video(run.media_dir / "seq_proxy.mov", codec="h264", frames=48)
    outcomes = cut_shots(
        master=seq_dir,
        master_kind="sequence",
        ranges=ranges,
        plans=plans,
        fps=FPS,
        origin_frame=0,
        sequence=sequence,
        proxy=proxy,
    )
    assert all(outcome.status == "created" for outcome in outcomes)
    # Original names preserved: shot 002 starts at master frame 1001 + 24.
    first_plate = plans[0].plate_path
    assert (first_plate / "show.0001001.png").exists()
    assert (first_plate / "show.0001024.png").exists()
    second_plate = plans[1].plate_path
    assert (second_plate / "show.0001025.png").exists()
    assert (second_plate / "show.0001048.png").exists()
    # Frame bytes are copied verbatim.
    source = sequence.frame_path(1001).read_bytes()
    assert (first_plate / "show.0001001.png").read_bytes() == source
    # Per-shot proxies were cut from the full-program proxy.
    for range_, plan in zip(ranges, plans, strict=True):
        assert plan.video_path.exists()
        assert count_video_frames(plan.video_path) == range_.num_frames


@requires_ffmpeg
def test_cut_sequence_anchors_timeline_zero_at_the_program_start(tmp_path: Path) -> None:
    """Head frames before the program must not shift the plates by a frame."""
    seq_dir = make_png_sequence(tmp_path / "seq", frames=50, start=1001, prefix="show.", padding=7)
    sequence = detect_sequence(seq_dir)
    ranges = _ranges()
    plans = plan_shot_media(
        ranges, shots_dir=tmp_path / "shots", shot_prefix="show", master_kind="sequence"
    )
    run = RunPaths.from_path(tmp_path / "run")
    proxy = make_video(run.media_dir / "seq_proxy.mov", codec="h264", frames=48)
    outcomes = cut_shots(
        master=seq_dir,
        master_kind="sequence",
        ranges=ranges,
        plans=plans,
        fps=FPS,
        origin_frame=0,
        sequence=sequence,
        program_start=1003,
        proxy=proxy,
    )
    assert all(outcome.status == "created" for outcome in outcomes)
    first_plate = plans[0].plate_path
    assert (first_plate / "show.0001003.png").exists()
    assert not (first_plate / "show.0001001.png").exists()
    assert (plans[1].plate_path / "show.0001027.png").exists()


@requires_ffmpeg
def test_cut_sequence_renumber_and_gaps(tmp_path: Path) -> None:
    seq_dir = make_png_sequence(tmp_path / "seq", frames=48, start=1001, prefix="show.", padding=7)
    (seq_dir / "show.0001010.png").unlink()
    sequence = detect_sequence(seq_dir)
    ranges = [CutRange(shot_id="001", start_frame=0, end_frame=24)]
    plans = plan_shot_media(
        ranges, shots_dir=tmp_path / "shots", shot_prefix="show", master_kind="sequence"
    )

    failed = cut_shots(
        master=seq_dir,
        master_kind="sequence",
        ranges=ranges,
        plans=plans,
        fps=FPS,
        origin_frame=0,
        sequence=sequence,
        proxy=None,
    )
    assert failed[0].status == "failed"
    assert "missing" in failed[0].message

    outcomes = cut_shots(
        master=seq_dir,
        master_kind="sequence",
        ranges=ranges,
        plans=plans,
        fps=FPS,
        origin_frame=0,
        sequence=sequence,
        proxy=None,
        allow_gaps=True,
        renumber=True,
        renumber_start=1001,
        force=True,
    )
    assert outcomes[0].status == "created"
    assert "1 missing frame(s)" in outcomes[0].message
    plate = plans[0].plate_path
    names = sorted(p.name for p in plate.iterdir())
    assert len(names) == 23  # 24 frames minus the missing one
    assert names[0] == "show_001.0001001.png"
    assert "show_001.0001010.png" not in names  # gap is skipped, not shifted


def test_cut_sequence_hardlink_mode(tmp_path: Path) -> None:
    seq_dir = tmp_path / "seq"
    seq_dir.mkdir()
    for frame in range(1, 5):
        (seq_dir / f"a.{frame:04d}.png").write_bytes(b"frame%d" % frame)
    sequence = detect_sequence(seq_dir)
    ranges = [CutRange(shot_id="001", start_frame=0, end_frame=4)]
    plans = plan_shot_media(
        ranges, shots_dir=tmp_path / "shots", shot_prefix="x", master_kind="sequence"
    )
    outcomes = cut_shots(
        master=seq_dir,
        master_kind="sequence",
        ranges=ranges,
        plans=plans,
        fps=FPS,
        origin_frame=0,
        sequence=sequence,
        proxy=None,
        link_mode="hardlink",
    )
    assert outcomes[0].status == "created"
    copied = plans[0].plate_path / "a.0001.png"
    assert copied.stat().st_ino == (seq_dir / "a.0001.png").stat().st_ino


def test_dry_run_touches_nothing(tmp_path: Path) -> None:
    ranges = _ranges()
    plans = plan_shot_media(
        ranges, shots_dir=tmp_path / "shots", shot_prefix="show", master_kind="video"
    )
    outcomes = cut_shots(
        master=tmp_path / "missing.mov",
        master_kind="video",
        ranges=ranges,
        plans=plans,
        fps=FPS,
        origin_frame=0,
        dry_run=True,
    )
    assert all(outcome.status == "planned" for outcome in outcomes)
    assert not (tmp_path / "shots").exists()


def test_manifest_entries_are_loader_compatible(tmp_path: Path) -> None:
    from unrender.manifests import load_shot_manifest, write_json

    ranges = _ranges()
    plans = plan_shot_media(
        ranges, shots_dir=tmp_path / "shots", shot_prefix="show", master_kind="sequence"
    )
    entries = shot_manifest_entries(ranges, plans, fps=23.976, origin_frame=86_400)
    assert entries[0]["start_tc"] == "01:00:00:00"
    assert entries[0]["end_tc"] == "01:00:01:00"
    assert entries[1]["end_frame"] == 48
    manifest = tmp_path / "shots.json"
    write_json(manifest, {"shots": entries})
    records = load_shot_manifest(manifest)
    assert [record.shot_id for record in records] == ["001", "002"]
    assert records[0].start_sec == pytest.approx(0.0)
    assert records[1].end_sec == pytest.approx(48 / 23.976)
    assert records[0].video_path.name == "show_001_proxy.mov"


def test_cut_shots_requires_matching_plans() -> None:
    with pytest.raises(ValueError, match="no shots to cut"):
        cut_shots(
            master=Path("x.mov"),
            master_kind="video",
            ranges=[],
            plans=[],
            fps=FPS,
            origin_frame=0,
        )


@requires_ffmpeg
def test_intra_cut_timecode_keeps_drop_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import unrender.editorial.shots.cut as cut_module

    master = make_video(tmp_path / "master.mov", codec="prores", frames=30, fps=30.0)
    captured: list[list[str]] = []
    monkeypatch.setattr(cut_module, "run_video_tool", lambda cmd, **_: captured.append(cmd))
    ranges = [CutRange(shot_id="001", start_frame=0, end_frame=15)]
    plans = plan_shot_media(
        ranges,
        shots_dir=tmp_path / "shots",
        shot_prefix="show",
        master_kind="video",
        master_ext=".mov",
    )
    cut_shots(
        master=master,
        master_kind="video",
        ranges=ranges,
        plans=plans,
        fps=29.97,
        origin_frame=0,
        drop_frame=True,
    )
    cmd = captured[0]
    timecode = cmd[cmd.index("-timecode") + 1]
    assert ";" in timecode  # drop-frame separator must survive to ffmpeg


@requires_ffmpeg
def test_inter_cut_seeks_before_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import unrender.editorial.shots.cut as cut_module

    master = make_video(tmp_path / "master.mp4", codec="h264", frames=48)
    captured: list[list[str]] = []
    monkeypatch.setattr(cut_module, "run_video_tool", lambda cmd, **_: captured.append(cmd))
    ranges = [CutRange(shot_id="001", start_frame=24, end_frame=48)]
    plans = plan_shot_media(
        ranges,
        shots_dir=tmp_path / "shots",
        shot_prefix="show",
        master_kind="video",
        master_ext=".mp4",
    )
    cut_shots(
        master=master,
        master_kind="video",
        ranges=ranges,
        plans=plans,
        fps=FPS,
        origin_frame=0,
    )
    cmd = captured[0]
    assert cmd.index("-ss") < cmd.index("-i")  # input seeking, not output seeking
