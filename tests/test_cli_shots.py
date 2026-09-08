from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import make_png_sequence, make_video, requires_ffmpeg, write_project_config
from unrender.cli import main
from unrender.manifests import load_shot_manifest, read_json
from unrender.project import RunPaths

pytest.importorskip("scenedetect")


def _project(
    config_dir: Path,
    run: RunPaths,
    *,
    paths: dict[str, str] | None = None,
    shots: dict[str, object] | None = None,
    fps: float | None = None,
) -> Path:
    path = config_dir / "show.json"
    write_project_config(path, run.root, paths=paths, shots=shots)
    if fps is not None:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["fps"] = fps
        path.write_text(json.dumps(data), encoding="utf-8")
    return path


@requires_ffmpeg
def test_shots_proxy_command(tmp_path: Path, run_paths: RunPaths, project_config_dir: Path) -> None:
    master = make_video(tmp_path / "master.mov", codec="prores", frames=8)
    config = _project(project_config_dir, run_paths, paths={"master": str(master)})
    assert main(["shots", "proxy", "-p", str(config)]) == 0
    proxies = list(run_paths.media_dir.glob("*_proxy.mov"))
    assert len(proxies) == 1


@requires_ffmpeg
def test_shots_detect_no_cut_writes_manifest_only(
    tmp_path: Path, run_paths: RunPaths, project_config_dir: Path
) -> None:
    master = make_video(tmp_path / "master.mp4", codec="h264", frames=48, two_scenes=True)
    config = _project(
        project_config_dir,
        run_paths,
        paths={"master": str(master)},
        shots={"min_shot_frames": 4},
    )
    assert main(["shots", "detect", "-p", str(config), "--no-cut"]) == 0

    manifest = run_paths.shots_manifest_json
    assert manifest.exists()
    data = read_json(manifest)
    assert data["source"] == "detect"
    assert data["detector"]["mode"] == "content"
    assert [s["shot_id"] for s in data["shots"]] == ["001", "002"]
    assert all(s["media_status"] == "planned" for s in data["shots"])
    # No media was cut.
    assert not run_paths.shots_media_dir.exists()
    # The manifest is loadable by the downstream loader.
    records = load_shot_manifest(manifest)
    assert records[0].start_sec == 0.0
    assert records[1].end_sec == pytest.approx(2.0)


@requires_ffmpeg
def test_shots_detect_default_cuts_media(
    tmp_path: Path, run_paths: RunPaths, project_config_dir: Path
) -> None:
    master = make_video(tmp_path / "master.mp4", codec="h264", frames=48, two_scenes=True)
    config = _project(
        project_config_dir,
        run_paths,
        paths={"master": str(master)},
        shots={"min_shot_frames": 4},
    )
    assert main(["shots", "detect", "-p", str(config), "--csv"]) == 0

    data = read_json(run_paths.shots_manifest_json)
    assert [s["media_status"] for s in data["shots"]] == ["created", "created"]
    for record in load_shot_manifest(run_paths.shots_manifest_json):
        assert record.video_path.exists()
    assert run_paths.shots_manifest_json.with_suffix(".csv").exists()


@requires_ffmpeg
def test_shots_cut_from_csv(tmp_path: Path, run_paths: RunPaths, project_config_dir: Path) -> None:
    master = make_video(tmp_path / "master.mov", codec="prores", frames=48, audio=True)
    shot_list = tmp_path / "list.csv"
    shot_list.write_text(
        "shot_id,start_tc,end_tc\n1,00:00:00:00,00:00:01:00\n2,00:00:01:00,00:00:02:00\n",
        encoding="utf-8",
    )
    config = _project(
        project_config_dir,
        run_paths,
        paths={"master": str(master), "shots": str(shot_list)},
    )
    assert main(["shots", "cut", "-p", str(config)]) == 0
    records = load_shot_manifest(run_paths.shots_manifest_json)
    assert [record.shot_id for record in records] == ["001", "002"]
    for record in records:
        assert record.video_path.exists()
        assert record.video_path.suffix == ".mov"


@requires_ffmpeg
def test_shots_cut_sequence_master_from_txt(
    tmp_path: Path, run_paths: RunPaths, project_config_dir: Path
) -> None:
    seq = make_png_sequence(tmp_path / "seq", frames=48, start=1001, prefix="s.", padding=7)
    shot_list = tmp_path / "cuts.txt"
    shot_list.write_text("00:00:01:00\n", encoding="utf-8")
    config = _project(
        project_config_dir,
        run_paths,
        paths={"master": str(seq), "shots": str(shot_list)},
        fps=24.0,
    )
    assert main(["shots", "cut", "-p", str(config)]) == 0
    records = load_shot_manifest(run_paths.shots_manifest_json)
    assert len(records) == 2
    for record in records:
        assert record.video_path.exists()  # per-shot proxy, auto-created chain
        plate = Path(str(record.video_path).replace("_proxy.mov", "_plate"))
        assert plate.is_dir()


@requires_ffmpeg
def test_shots_cut_dry_run(tmp_path: Path, run_paths: RunPaths, project_config_dir: Path) -> None:
    master = make_video(tmp_path / "master.mp4", codec="h264", frames=24)
    shot_list = tmp_path / "list.csv"
    shot_list.write_text("shot_id,start,end\n1,0,1.0\n", encoding="utf-8")
    config = _project(
        project_config_dir,
        run_paths,
        paths={"master": str(master), "shots": str(shot_list)},
    )
    assert main(["shots", "cut", "-p", str(config), "--dry-run"]) == 0
    assert not run_paths.shots_manifest_json.exists()
    assert not run_paths.shots_media_dir.exists()


def test_shots_cut_requires_master(run_paths: RunPaths, project_config_dir: Path) -> None:
    config = _project(project_config_dir, run_paths)
    assert main(["shots", "cut", "-p", str(config)]) == 1


def test_shots_detect_missing_configured_proxy(
    tmp_path: Path, run_paths: RunPaths, project_config_dir: Path
) -> None:
    config = _project(
        project_config_dir,
        run_paths,
        paths={"proxy_master": str(tmp_path / "nope.mov")},
    )
    assert main(["shots", "detect", "-p", str(config), "--no-cut"]) == 1


@requires_ffmpeg
def test_shots_detect_config_overrides_and_flags(
    tmp_path: Path,
    run_paths: RunPaths,
    project_config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config section values flow into DetectOptions; CLI flags win."""
    import unrender.cli.commands.shots as shots_cmd

    master = make_video(tmp_path / "master.mp4", codec="h264", frames=48, two_scenes=True)
    config = _project(
        project_config_dir,
        run_paths,
        paths={"master": str(master)},
        shots={"detector": "content", "content_threshold": 90.0, "min_shot_frames": 4},
    )

    captured: dict[str, object] = {}
    real = shots_cmd.detect_cut_frames

    def spy(video, *, options, ffprobe="ffprobe"):
        captured["options"] = options
        return real(video, options=options, ffprobe=ffprobe)

    monkeypatch.setattr(shots_cmd, "detect_cut_frames", spy)
    assert main(["shots", "detect", "-p", str(config), "--no-cut", "--threshold", "10"]) == 0
    options = captured["options"]
    assert options.content_threshold == 10.0  # flag beats config
    assert options.min_shot_frames == 4  # config beats default
