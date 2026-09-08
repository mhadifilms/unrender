from __future__ import annotations

from pathlib import Path

import pytest

from conftest import make_png_sequence, make_video, requires_ffmpeg
from unrender.editorial.shots.detect import (
    DetectOptions,
    build_shots,
    detect_cut_frames,
    resolve_detection_input,
)
from unrender.project import RunPaths

scenedetect = pytest.importorskip("scenedetect")


@pytest.fixture
def run(tmp_path: Path) -> RunPaths:
    return RunPaths.from_path(tmp_path / "run")


# ---------------------------------------------------------------------------
# build_shots post-processing (pure math, no media)


def test_build_shots_basic() -> None:
    ranges = build_shots([24, 48], total_frames=96, fps=24.0)
    assert [(r.start_frame, r.end_frame) for r in ranges] == [(0, 24), (24, 48), (48, 96)]
    assert [r.shot_id for r in ranges] == ["001", "002", "003"]


def test_build_shots_zero_cuts_yields_full_program() -> None:
    ranges = build_shots([], total_frames=48, fps=24.0)
    assert [(r.start_frame, r.end_frame) for r in ranges] == [(0, 48)]


def test_build_shots_coalesces_near_duplicates() -> None:
    # 0.33s at 24fps = 8 frames; 26 is within the gap of 24 and dropped.
    ranges = build_shots([24, 26, 48], total_frames=96, fps=24.0)
    assert [r.start_frame for r in ranges] == [0, 24, 48]


def test_build_shots_drops_cuts_near_edges() -> None:
    ranges = build_shots([2, 24, 94], total_frames=96, fps=24.0)
    assert [(r.start_frame, r.end_frame) for r in ranges] == [(0, 24), (24, 96)]


def test_build_shots_rejects_empty_master() -> None:
    with pytest.raises(ValueError, match="no frames"):
        build_shots([], total_frames=0, fps=24.0)


# ---------------------------------------------------------------------------
# The "never on the master" guard


@requires_ffmpeg
def test_detect_refuses_prores(tmp_path: Path) -> None:
    prores = make_video(tmp_path / "master.mov", codec="prores", frames=8)
    with pytest.raises(ValueError, match="must not run on a full-resolution prores"):
        detect_cut_frames(prores, options=DetectOptions())


def test_detect_refuses_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="image-sequence directory"):
        detect_cut_frames(tmp_path, options=DetectOptions())


@requires_ffmpeg
def test_resolve_input_uses_configured_proxy(tmp_path: Path, run: RunPaths) -> None:
    proxy = make_video(tmp_path / "proxy.mov", codec="h264", frames=8)
    path, created = resolve_detection_input(
        master=None,
        proxy=proxy,
        run=run,
        width=640,
        crf=23,
        exr_transfer="bt709",
        timecode_start=None,
    )
    assert path == proxy and created is None


def test_resolve_input_missing_configured_proxy(tmp_path: Path, run: RunPaths) -> None:
    with pytest.raises(ValueError, match="configured proxy does not exist"):
        resolve_detection_input(
            master=None,
            proxy=tmp_path / "nope.mov",
            run=run,
            width=640,
            crf=23,
            exr_transfer="bt709",
            timecode_start=None,
        )


@requires_ffmpeg
def test_resolve_input_rejects_unsafe_configured_proxy(tmp_path: Path, run: RunPaths) -> None:
    prores = make_video(tmp_path / "proxy.mov", codec="prores", frames=8)
    with pytest.raises(ValueError, match="must not run on a full-resolution"):
        resolve_detection_input(
            master=None,
            proxy=prores,
            run=run,
            width=640,
            crf=23,
            exr_transfer="bt709",
            timecode_start=None,
        )


@requires_ffmpeg
def test_resolve_input_compressed_master_used_directly(tmp_path: Path, run: RunPaths) -> None:
    h264 = make_video(tmp_path / "master.mp4", codec="h264", frames=8)
    path, created = resolve_detection_input(
        master=h264,
        proxy=None,
        run=run,
        width=640,
        crf=23,
        exr_transfer="bt709",
        timecode_start=None,
    )
    assert path == h264 and created is None


@requires_ffmpeg
def test_resolve_input_auto_creates_proxy_for_prores(tmp_path: Path, run: RunPaths) -> None:
    prores = make_video(tmp_path / "master.mov", codec="prores", frames=8)
    path, created = resolve_detection_input(
        master=prores,
        proxy=None,
        run=run,
        width=64,
        crf=28,
        exr_transfer="bt709",
        timecode_start=None,
    )
    assert created is not None and created.created
    assert path == created.path
    assert path.exists() and path != prores


@requires_ffmpeg
def test_resolve_input_no_proxy_create_errors(tmp_path: Path, run: RunPaths) -> None:
    prores = make_video(tmp_path / "master.mov", codec="prores", frames=8)
    with pytest.raises(ValueError, match="detection needs a proxy"):
        resolve_detection_input(
            master=prores,
            proxy=None,
            run=run,
            auto_create=False,
            width=640,
            crf=23,
            exr_transfer="bt709",
            timecode_start=None,
        )


@requires_ffmpeg
def test_resolve_input_sequence_master_auto_proxied(tmp_path: Path, run: RunPaths) -> None:
    seq = make_png_sequence(tmp_path / "seq", frames=8)
    path, created = resolve_detection_input(
        master=seq,
        proxy=None,
        run=run,
        fps=24.0,
        width=64,
        crf=28,
        exr_transfer="bt709",
        timecode_start=None,
    )
    assert created is not None and path.suffix == ".mov"


def test_resolve_input_nothing_to_detect(run: RunPaths) -> None:
    with pytest.raises(ValueError, match="nothing to detect on"):
        resolve_detection_input(
            master=None,
            proxy=None,
            run=run,
            width=640,
            crf=23,
            exr_transfer="bt709",
            timecode_start=None,
        )


# ---------------------------------------------------------------------------
# Real detection on a synthetic two-scene video


def test_default_options_are_recall_safe() -> None:
    # Default detector is `content` (best hard-cut recall for the cutting
    # pipeline) and min length matches scenedetect's flash-merge default (15),
    # so brief flash frames don't survive as false sub-shot slivers.
    opts = DetectOptions()
    assert opts.detector == "content"
    assert opts.min_shot_frames == 15


@requires_ffmpeg
def test_detect_finds_hard_cut(tmp_path: Path) -> None:
    video = make_video(tmp_path / "twoscene.mp4", codec="h264", frames=48, two_scenes=True)
    # Default (content) finds the hard cut.
    cuts = detect_cut_frames(video, options=DetectOptions(min_shot_frames=4))
    assert cuts == [24]
    ranges = build_shots(cuts, total_frames=48, fps=24.0)
    assert [(r.start_frame, r.end_frame) for r in ranges] == [(0, 24), (24, 48)]


@requires_ffmpeg
def test_detect_content_mode_finds_hard_cut(tmp_path: Path) -> None:
    video = make_video(tmp_path / "twoscene_c.mp4", codec="h264", frames=48, two_scenes=True)
    cuts = detect_cut_frames(video, options=DetectOptions(detector="content", min_shot_frames=4))
    assert cuts == [24]


@requires_ffmpeg
def test_detect_adaptive_mode_finds_hard_cut(tmp_path: Path) -> None:
    video = make_video(tmp_path / "twoscene_a.mp4", codec="h264", frames=48, two_scenes=True)
    cuts = detect_cut_frames(video, options=DetectOptions(detector="adaptive", min_shot_frames=4))
    assert cuts == [24]


def test_unknown_detector_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import unrender.editorial.shots.detect as detect_module

    monkeypatch.setattr(detect_module, "_ensure_detection_safe", lambda *a, **k: None)
    with pytest.raises(ValueError, match="unknown detector"):
        detect_cut_frames(tmp_path / "x.mp4", options=DetectOptions(detector="bogus"))
