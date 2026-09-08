from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import write_project_config
from unrender.audio.separation.audioshake import SourceSeparationResult
from unrender.cli import commands as cli_commands
from unrender.cli import main
from unrender.cli.parser import build_parser
from unrender.project import RunPaths
from unrender.project.config import load_project_config


def test_cli_version_returns_success() -> None:
    assert main(["--version"]) == 0


def test_cli_status_reports_run_artifacts(capsys, tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()
    run.speaker_db.write_text(json.dumps({"version": "1.0"}), encoding="utf-8")

    assert main(["status", "--run-dir", str(run.root)]) == 0

    output = capsys.readouterr().out
    assert "speaker_db" in output
    assert "dialogue_stem_plan" in output
    assert "source_separation" in output
    assert "missing" in output


def test_run_location_aliases_normalize_to_run_dir(tmp_path: Path) -> None:
    parser = build_parser()
    run_dir = tmp_path / "run"

    for argv in (
        ["audio", "analyze", "--run-dir", str(run_dir)],
        ["audio", "analyze", "--out", str(run_dir)],
        ["status", "--run", str(run_dir)],
    ):
        args = parser.parse_args(argv)
        assert args.run_dir == run_dir
        assert not hasattr(args, "run")
        assert not hasattr(args, "out")


def test_audio_separate_parser_accepts_external_mono_pair(tmp_path: Path) -> None:
    parser = build_parser()
    run_dir = tmp_path / "run"
    left = tmp_path / "mix_L.wav"
    right = tmp_path / "mix_R.wav"

    args = parser.parse_args(
        [
            "audio",
            "separate",
            "--run-dir",
            str(run_dir),
            "--full-audio-pair",
            str(left),
            str(right),
        ]
    )

    assert args.full_audio_pair == [str(left), str(right)]
    assert args.handler is cli_commands._audio_separate


def test_cli_doctor_reports_missing_project_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    write_project_config(
        config_dir / "show-a.json",
        tmp_path / "runs" / "show-a",
        paths={"shots": str(tmp_path / "missing-shots.csv")},
    )

    assert main(["doctor", "-p", "show-a"]) == 1


def test_cli_doctor_reports_missing_source_audio_asset(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    missing = tmp_path / "original_en" / "_dme" / "show_DX_en_stem.wav"
    write_project_config(
        config_dir / "show-a.json",
        tmp_path / "runs" / "show-a",
        audio_assets={"original": {"dme": {"dx": str(missing)}}},
    )

    assert main(["doctor", "-p", "show-a"]) == 1

    assert f"audio_assets.original.dme.dx does not exist: {missing}" in capsys.readouterr().out


def test_shots_match_uses_shots_config_section(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    run = RunPaths.from_path(tmp_path / "runs" / "show-a")
    run.ensure()
    run.speaker_db.write_text(json.dumps({"version": "1.0"}), encoding="utf-8")
    shots_csv = tmp_path / "shots.csv"
    shots_csv.write_text(
        "shot_id,video_path,start_sec,end_sec\n001,/tmp/001.mov,0,1\n",
        encoding="utf-8",
    )
    write_project_config(
        config_dir / "show-a.json",
        run.root,
        paths={"shots": str(shots_csv)},
        shots={"samples_per_shot": 3, "sim_threshold": 0.7, "min_confidence": 0.8},
    )
    calls = {}

    def fake_match(**kwargs):
        calls.update(kwargs)
        return []

    monkeypatch.setattr("unrender.cli.commands.face.match_shots", fake_match)

    assert main(["shots", "match", "-p", "show-a"]) == 0

    assert calls["samples_per_shot"] == 3
    assert calls["sim_threshold"] == 0.7
    assert calls["min_confidence"] == 0.8


def test_project_config_loader_does_not_mutate_source(tmp_path: Path) -> None:
    script = tmp_path / "script.csv"
    script.write_text("speaker,text,start_sec,end_sec\nALEX,Hello,0,1\n", encoding="utf-8")
    config_path = tmp_path / "show-a.json"
    config_path.write_text(
        json.dumps({"paths": {"run_dir": str(tmp_path / "run"), "transcript": str(script)}}),
        encoding="utf-8",
    )

    project = load_project_config(config_path)

    assert "ALEX" in project.data["speakers"]
    assert "speakers" not in json.loads(config_path.read_text(encoding="utf-8"))


def test_audio_separate_uses_full_audio_from_project_config(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    run = RunPaths.from_path(tmp_path / "runs" / "show-a")
    full_audio = tmp_path / "full_mix.wav"
    full_audio.write_bytes(b"audio")
    write_project_config(
        config_dir / "show-a.json",
        run.root,
        audio_assets={"original": {"full_mix": str(full_audio)}},
    )
    calls = {}

    def fake_source(**kwargs):
        calls["source"] = kwargs
        dx = run.source_stems_dir / "show-a_DX_stem.wav"
        dx.parent.mkdir(parents=True, exist_ok=True)
        dx.write_bytes(b"dx")
        return SourceSeparationResult(
            source=str(kwargs["full_audio"]),
            output_dir=run.source_stems_dir,
            dialogue_stem=dx,
            stems={"dialogue": dx},
        )

    def fake_speakers(**kwargs):
        calls["speakers"] = kwargs
        return None

    monkeypatch.setattr("unrender.cli.commands.audio.separate_full_audio_source", fake_source)
    monkeypatch.setattr("unrender.cli.commands.audio.separate_global_dx_stem", fake_speakers)

    assert main(["audio", "separate", "-p", "show-a"]) == 0

    assert calls["source"]["full_audio"] == str(full_audio)
    assert calls["source"]["prefix"] == "show-a"
    assert calls["speakers"]["prefix"] == "show-a"
    assert calls["speakers"]["dx_stem"] == run.source_stems_dir / "show-a_DX_stem.wav"


def test_audio_separate_uses_explicit_full_audio_pair(monkeypatch, tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    left = tmp_path / "mix_L.wav"
    right = tmp_path / "mix_R.wav"
    left.write_bytes(b"left")
    right.write_bytes(b"right")
    calls = {}

    def fake_source(**kwargs):
        calls["source"] = kwargs
        dx = run.source_stems_dir / "khs_DX_stem.wav"
        dx.parent.mkdir(parents=True, exist_ok=True)
        dx.write_bytes(b"dx")
        return SourceSeparationResult(
            source=str(kwargs["full_audio"]),
            output_dir=run.source_stems_dir,
            dialogue_stem=dx,
            stems={"dialogue": dx},
        )

    def fake_speakers(**kwargs):
        calls["speakers"] = kwargs
        return None

    monkeypatch.setattr("unrender.cli.commands.audio.separate_full_audio_source", fake_source)
    monkeypatch.setattr("unrender.cli.commands.audio.separate_global_dx_stem", fake_speakers)

    assert (
        main(
            [
                "audio",
                "separate",
                "--run-dir",
                str(run.root),
                "--full-audio-pair",
                str(left),
                str(right),
                "--prefix",
                "khs",
            ]
        )
        == 0
    )

    assert calls["source"]["full_audio"] == (str(left), str(right))
    assert calls["speakers"]["dx_stem"] == run.source_stems_dir / "khs_DX_stem.wav"


def test_audio_separate_selects_configured_source_audio(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    run = RunPaths.from_path(tmp_path / "runs" / "show-a")
    original_dx = tmp_path / "original_dx.wav"
    original_dx.write_bytes(b"original")
    write_project_config(
        config_dir / "show-a.json",
        run.root,
        audio_assets={
            "original": {
                "language": "en",
                "dme": {"dx": str(original_dx)},
            },
        },
    )
    calls: list[dict[str, object]] = []

    def fake_speakers(**kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr("unrender.cli.commands.audio.separate_global_dx_stem", fake_speakers)

    assert main(["audio", "separate", "-p", "show-a"]) == 0

    assert calls[0]["dx_stem"] == str(original_dx)
    assert calls[0]["prefix"] == "show-a"


def test_audio_separate_rejects_flat_legacy_audio_config(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    write_project_config(
        config_dir / "show-a.json",
        tmp_path / "runs" / "show-a",
        paths={
            "full_audio": str(tmp_path / "legacy_mix.wav"),
            "dx_stem": str(tmp_path / "legacy_dx.wav"),
        },
    )

    assert main(["audio", "separate", "-p", "show-a"]) == 1


def test_audio_separate_explicit_dx_overrides_config_full_audio(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    run = RunPaths.from_path(tmp_path / "runs" / "show-a")
    full_audio = tmp_path / "full_mix.wav"
    dx_stem = tmp_path / "dx.wav"
    full_audio.write_bytes(b"audio")
    dx_stem.write_bytes(b"dx")
    write_project_config(
        config_dir / "show-a.json",
        run.root,
        audio_assets={"original": {"full_mix": str(full_audio)}},
    )
    calls = {"source": 0}

    def fake_source(**kwargs):
        calls["source"] += 1
        return None

    def fake_speakers(**kwargs):
        calls["speakers"] = kwargs
        return None

    monkeypatch.setattr("unrender.cli.commands.audio.separate_full_audio_source", fake_source)
    monkeypatch.setattr("unrender.cli.commands.audio.separate_global_dx_stem", fake_speakers)

    assert main(["audio", "separate", "-p", "show-a", "--dx-stem", str(dx_stem)]) == 0

    assert calls["source"] == 0
    assert calls["speakers"]["dx_stem"] == str(dx_stem)


def test_audio_separate_rejects_full_audio_and_dx_stem(tmp_path: Path) -> None:
    run = tmp_path / "run"

    assert (
        main(
            [
                "audio",
                "separate",
                "--run-dir",
                str(run),
                "--full-audio",
                str(tmp_path / "full.wav"),
                "--dx-stem",
                str(tmp_path / "dx.wav"),
            ]
        )
        == 1
    )


@pytest.mark.parametrize("conflicting_flag", ["--full-audio", "--dx-stem"])
def test_audio_separate_rejects_full_audio_pair_conflicts(
    tmp_path: Path,
    conflicting_flag: str,
) -> None:
    run = tmp_path / "run"

    assert (
        main(
            [
                "audio",
                "separate",
                "--run-dir",
                str(run),
                "--full-audio-pair",
                str(tmp_path / "left.wav"),
                str(tmp_path / "right.wav"),
                conflicting_flag,
                str(tmp_path / "conflict.wav"),
            ]
        )
        == 1
    )


def test_transcribe_lines_uses_generated_dx_from_source_separation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    run = RunPaths.from_path(tmp_path / "runs" / "show-a")
    run.ensure()
    dx = run.source_stems_dir / "show-a_DX_stem.wav"
    dx.write_bytes(b"dx")
    run.source_separation_json.write_text(
        json.dumps({"version": "1.0", "dialogue_stem": str(dx)}),
        encoding="utf-8",
    )
    write_project_config(config_dir / "show-a.json", run.root)
    calls = {}

    def fake_transcribe(**kwargs):
        calls.update(kwargs)
        return []

    monkeypatch.setattr("unrender.cli.commands.audio.transcribe_dialogue_lines", fake_transcribe)

    assert main(["audio", "transcribe-lines", "-p", "show-a"]) == 0

    assert calls["audio_path"] == dx


def test_audio_resolve_clips_defaults_to_voice_db_membership(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()
    clip = tmp_path / "DL_000001_speaker_01_stem.wav"
    clip.write_bytes(b"audio")
    run.voice_clips_json.write_text(
        json.dumps(
            {
                "clips": [
                    {
                        "path": str(clip),
                        "clip_id": "DL_000001_speaker_01",
                        "clip_type": "dialogue_line",
                        "line_id": "DL_000001",
                        "kept": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    calls = {}

    def fake_resolve(**kwargs):
        calls.update(kwargs)
        return []

    def fail_recluster(**kwargs):
        raise AssertionError("default resolver must not recluster clips")

    monkeypatch.setattr(
        "unrender.cli.commands.audio.resolve_voice_clips_from_voice_db", fake_resolve
    )
    monkeypatch.setattr("unrender.cli.commands.audio.resolve_voice_clips_clustered", fail_recluster)

    assert main(["audio", "resolve-clips", "--run-dir", str(run.root)]) == 0

    assert calls["voice_db_path"] == run.voice_db
    assert calls["clips"][0].clip_id == "DL_000001_speaker_01"


def test_audio_resolve_clips_cluster_resolver_is_explicit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    run.ensure()
    clip = tmp_path / "DL_000001_speaker_01_stem.wav"
    clip.write_bytes(b"audio")
    run.voice_clips_json.write_text(
        json.dumps(
            {
                "clips": [
                    {
                        "path": str(clip),
                        "clip_id": "DL_000001_speaker_01",
                        "clip_type": "dialogue_line",
                        "line_id": "DL_000001",
                        "kept": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    calls = {}

    def fake_cluster(**kwargs):
        calls.update(kwargs)
        return []

    monkeypatch.setattr("unrender.cli.commands.audio.resolve_voice_clips_clustered", fake_cluster)

    assert (
        main(["audio", "resolve-clips", "--run-dir", str(run.root), "--resolver", "cluster"]) == 0
    )

    assert calls["clips"][0].clip_id == "DL_000001_speaker_01"


def test_map_dialogue_uses_configured_clip_plan_and_shots(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    run = RunPaths.from_path(tmp_path / "runs" / "show-a")
    run.ensure()
    run.shot_matches_json.write_text(
        json.dumps(
            {
                "shots": [
                    {"shot_id": "001", "accepted": ["FACE"]},
                    {"shot_id": "002", "accepted": ["FALLBACK"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    run.dialogue_lines_json.write_text(
        json.dumps(
            {
                "version": "1.0",
                "lines": [{"line_id": "DL_000001", "start_sec": 0.0, "end_sec": 1.0, "text": "hi"}],
            }
        ),
        encoding="utf-8",
    )
    shots = tmp_path / "shots.csv"
    shots.write_text(
        "shot_id,video_path,start_sec,end_sec\n001,/tmp/001.mov,0,1\n",
        encoding="utf-8",
    )
    clip_plan = tmp_path / "custom_clip_plan.json"
    clip_plan.write_text(json.dumps({"clips": [{"line_id": "DL_000001"}]}), encoding="utf-8")
    write_project_config(
        config_dir / "show-a.json",
        run.root,
        paths={"shots": str(shots), "clip_plan": str(clip_plan)},
    )
    calls = {}

    def fake_materialize(**kwargs):
        calls["materialize"] = kwargs
        return kwargs["clip_plan"]

    def fake_map(**kwargs):
        calls["map"] = kwargs
        return []

    monkeypatch.setattr(
        "unrender.cli.commands.audio.materialize_dialogue_line_stems", fake_materialize
    )
    monkeypatch.setattr("unrender.cli.commands.audio.map_dialogue_to_shots", fake_map)

    assert main(["audio", "map-dialogue", "-p", "show-a"]) == 0

    assert calls["materialize"]["clip_plan"] == [{"line_id": "DL_000001"}]
    assert calls["map"]["shots"][0].shot_id == "001"
    assert calls["map"]["shot_speakers_by_id"] == {
        "001": ["FACE"],
        "002": ["FALLBACK"],
    }
