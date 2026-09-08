"""Restartability of `shots detect` and downstream fallback wiring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import make_video, requires_ffmpeg, write_project_config, write_wav
from unrender.cli import main
from unrender.manifests import read_json
from unrender.project import RunPaths

pytest.importorskip("scenedetect")


def _project(config_dir: Path, run: RunPaths, *, paths: dict[str, str]) -> Path:
    path = config_dir / "show.json"
    write_project_config(path, run.root, paths=paths, shots={"min_shot_frames": 4})
    return path


@requires_ffmpeg
def test_detect_skips_redetection_and_resumes_cut(
    tmp_path: Path,
    run_paths: RunPaths,
    project_config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    master = make_video(tmp_path / "master.mp4", codec="h264", frames=48, two_scenes=True)
    config = _project(project_config_dir, run_paths, paths={"master": str(master)})

    # First run: manifest only.
    assert main(["shots", "detect", "-p", str(config), "--no-cut"]) == 0
    data = read_json(run_paths.shots_manifest_json)
    assert all(s["media_status"] == "planned" for s in data["shots"])
    capsys.readouterr()

    # Second run without --no-cut must NOT re-detect, only cut the planned media.
    import unrender.cli.commands.shots as shots_cmd

    def explode(*args, **kwargs):
        raise AssertionError("detection ran again despite existing manifest")

    monkeypatch.setattr(shots_cmd, "detect_cut_frames", explode)
    assert main(["shots", "detect", "-p", str(config)]) == 0
    out = capsys.readouterr().out
    assert "skipping detection" in out
    data = read_json(run_paths.shots_manifest_json)
    assert [s["media_status"] for s in data["shots"]] == ["created", "created"]
    for shot in data["shots"]:
        assert Path(shot["video_path"]).exists()

    # Third run: everything exists; cutting skips shot by shot.
    assert main(["shots", "detect", "-p", str(config)]) == 0
    assert "plate exists, skipping" in capsys.readouterr().out


@requires_ffmpeg
def test_detect_force_redetects(
    tmp_path: Path,
    run_paths: RunPaths,
    project_config_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    master = make_video(tmp_path / "master.mp4", codec="h264", frames=48, two_scenes=True)
    config = _project(project_config_dir, run_paths, paths={"master": str(master)})
    assert main(["shots", "detect", "-p", str(config), "--no-cut"]) == 0
    capsys.readouterr()
    assert main(["shots", "detect", "-p", str(config), "--no-cut", "--force"]) == 0
    out = capsys.readouterr().out
    assert "skipping detection" not in out
    assert "raw cut(s) detected" in out


@requires_ffmpeg
def test_detect_warns_when_master_changed(
    tmp_path: Path,
    run_paths: RunPaths,
    project_config_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    master = make_video(tmp_path / "master.mp4", codec="h264", frames=48, two_scenes=True)
    config = _project(project_config_dir, run_paths, paths={"master": str(master)})
    assert main(["shots", "detect", "-p", str(config), "--no-cut"]) == 0
    make_video(tmp_path / "master.mp4", codec="h264", frames=72, two_scenes=True)
    capsys.readouterr()
    assert main(["shots", "detect", "-p", str(config), "--no-cut"]) == 0
    assert "changed since" in capsys.readouterr().out


def test_foreign_manifest_without_frames_falls_through(
    tmp_path: Path,
    run_paths: RunPaths,
    project_config_dir: Path,
) -> None:
    """A hand-written manifest lacking frame fields does not satisfy resume."""
    run_paths.shots_manifest_json.parent.mkdir(parents=True, exist_ok=True)
    run_paths.shots_manifest_json.write_text(
        json.dumps({"shots": [{"shot_id": "001", "video_path": "x.mov", "start_sec": 0.0}]}),
        encoding="utf-8",
    )
    config = _project(project_config_dir, run_paths, paths={})
    # Falls through to real detection, which then fails on the missing master.
    assert main(["shots", "detect", "-p", str(config)]) == 1


@requires_ffmpeg
def test_shot_dx_falls_back_to_run_manifest(
    tmp_path: Path,
    run_paths: RunPaths,
    project_config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`audio shot-dx` finds run/shots.json without paths.shots configured."""
    master = make_video(tmp_path / "master.mp4", codec="h264", frames=48, two_scenes=True)
    config = _project(project_config_dir, run_paths, paths={"master": str(master)})
    assert main(["shots", "detect", "-p", str(config)]) == 0

    captured: dict[str, object] = {}

    def fake_build_shot_dx(**kwargs):
        captured["shots"] = kwargs["shots"]
        return []

    import unrender.cli.commands.audio as audio_cmd

    monkeypatch.setattr(audio_cmd, "build_shot_dx", fake_build_shot_dx)
    dx = tmp_path / "dx.wav"
    write_wav(dx, np.zeros(16_000))
    assert main(["audio", "shot-dx", "-p", str(config), "--dx-stem", str(dx)]) == 0
    shots = captured["shots"]
    assert [record.shot_id for record in shots] == ["001", "002"]


def test_shot_dx_without_any_manifest_mentions_detect(
    run_paths: RunPaths,
    project_config_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _project(project_config_dir, run_paths, paths={})
    assert main(["audio", "shot-dx", "-p", str(config)]) == 1
    assert "unrender shots detect" in capsys.readouterr().err


@requires_ffmpeg
def test_face_build_falls_back_to_created_proxy(
    tmp_path: Path,
    run_paths: RunPaths,
    project_config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`face build` finds the auto-created proxy without paths.proxy_master."""
    master = make_video(tmp_path / "master.mov", codec="prores", frames=8)
    config = _project(project_config_dir, run_paths, paths={"master": str(master)})
    assert main(["shots", "proxy", "-p", str(config)]) == 0

    captured: dict[str, object] = {}

    def fake_build_face_db(**kwargs):
        captured["video_path"] = kwargs["video_path"]

    import unrender.cli.commands.face as face_cmd

    monkeypatch.setattr(face_cmd, "build_face_db", fake_build_face_db)
    assert main(["face", "build", "-p", str(config)]) == 0
    assert str(captured["video_path"]).endswith("master_proxy.mov")


def test_face_build_without_proxy_mentions_shots_proxy(
    run_paths: RunPaths,
    project_config_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _project(project_config_dir, run_paths, paths={})
    assert main(["face", "build", "-p", str(config)]) == 1
    assert "unrender shots proxy" in capsys.readouterr().err


@requires_ffmpeg
def test_per_shot_proxy_for_video_master(
    tmp_path: Path,
    run_paths: RunPaths,
    project_config_dir: Path,
) -> None:
    master = make_video(tmp_path / "master.mov", codec="prores", frames=48)
    shot_list = tmp_path / "list.csv"
    shot_list.write_text(
        "shot_id,start,end\n1,0,1.0\n2,1.0,2.0\n",
        encoding="utf-8",
    )
    config = _project(
        project_config_dir,
        run_paths,
        paths={"master": str(master), "shots": str(shot_list)},
    )
    assert main(["shots", "cut", "-p", str(config), "--per-shot-proxy"]) == 0
    data = read_json(run_paths.shots_manifest_json)
    for shot in data["shots"]:
        plate = Path(shot["plate_path"])
        assert plate.exists()
        # The plate stays the manifest video; the proxy sits beside it.
        assert shot["video_path"] == shot["plate_path"]
        proxy = plate.parent / f"{plate.stem}_proxy.mov"
        assert proxy.exists()
