from __future__ import annotations

import argparse
import csv
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from conftest import write_wav
from unrender.cli import main
from unrender.cli.commands.context import _dx_stem_arg, _stem_sources_arg
from unrender.cli.commands.mne import _source_stem as _mne_source_stem
from unrender.manifests import read_json
from unrender.project import RunPaths
from unrender.project.config import (
    load_project_config,
    load_speaker_config,
    resolve_project_configs,
)


def test_resolve_project_configs_supports_names_and_globs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_project_config(config_dir / "show-a.json", tmp_path / "runs" / "show-a")
    _write_project_config(config_dir / "show-b.json", tmp_path / "runs" / "show-b")

    named = resolve_project_configs(["show-a"])
    globbed = resolve_project_configs(["show-*"])

    assert [project.name for project in named] == ["show-a"]
    assert [project.name for project in globbed] == ["show-a", "show-b"]


def test_project_config_exposes_source_audio_assets(tmp_path: Path) -> None:
    config_path = tmp_path / "show-a.json"
    original_dx = tmp_path / "original_dx.wav"
    source_stems = tmp_path / "source" / "*.wav"
    _write_project_config(
        config_path,
        tmp_path / "run",
        audio_assets={
            "original": {
                "language": "en",
                "full_mix": str(tmp_path / "original.wav"),
                "dme": {"dx": str(original_dx)},
                "separated_stems": str(source_stems),
            },
        },
    )

    project = load_project_config(config_path)

    assert project.audio_asset_string("language") == "en"
    assert project.audio_asset_path("dme.dx") == original_dx
    assert project.audio_asset_string("separated_stems") == str(source_stems)
    assert project.audio_asset_string("dme.fx") is None


def test_project_config_allows_anonymous_audio_workflows(tmp_path: Path) -> None:
    config_path = tmp_path / "anonymous.json"
    config_path.write_text(
        json.dumps(
            {
                "audio": {"speaker_strategy": "audioshake"},
                "audio_assets": {"original": {"full_mix": str(tmp_path / "master.mov")}},
            }
        ),
        encoding="utf-8",
    )

    project = load_project_config(config_path)

    assert project.data.get("speakers") is None
    assert project.audio_value("speaker_strategy") == "audioshake"
    with pytest.raises(ValueError, match="non-empty 'speakers'"):
        load_speaker_config(config_path)


def test_project_config_uses_original_separated_stem_glob(tmp_path: Path) -> None:
    stems_dir = tmp_path / "original_en" / "_separated"
    stems_dir.mkdir(parents=True)
    stems = [
        stems_dir / "show_speaker_01_en_stem.wav",
        stems_dir / "show_speaker_02_en_stem.wav",
    ]
    for stem in stems:
        stem.write_bytes(b"audio")
    config_path = tmp_path / "show-a.json"
    _write_project_config(
        config_path,
        tmp_path / "run",
        audio_assets={
            "original": {
                "language": "en",
                "separated_stems": str(stems_dir / "*.wav"),
            }
        },
    )
    project = load_project_config(config_path)
    args = argparse.Namespace(stems=None, project_config=project)

    sources = _stem_sources_arg(args, RunPaths.from_path(tmp_path / "run"))

    assert [source.path for source in sources] == stems


def test_mne_source_stems_use_original_dme_assets(tmp_path: Path) -> None:
    config_path = tmp_path / "show-a.json"
    fx = tmp_path / "original_en" / "_dme" / "show_FX_en_stem.wav"
    mx = tmp_path / "original_en" / "_dme" / "show_MX_en_stem.wav"
    _write_project_config(
        config_path,
        tmp_path / "run",
        audio_assets={
            "original": {
                "dme": {
                    "fx": str(fx),
                    "mx": str(mx),
                }
            }
        },
    )
    project = load_project_config(config_path)
    args = argparse.Namespace(project_config=project)
    run = RunPaths.from_path(tmp_path / "run")

    assert _mne_source_stem(args, "fx_stem", None, run, suffix="FX") == fx
    assert _mne_source_stem(args, "mx_stem", None, run, suffix="MX") == mx


def _declared_language_project(tmp_path: Path, dx: Path, *, language: str = "de") -> object:
    config_path = tmp_path / "show-a.json"
    _write_project_config(
        config_path,
        tmp_path / "run",
        audio_assets={"original": {"language": language, "dme": {"dx": str(dx)}}},
    )
    return load_project_config(config_path)


def _detects(monkeypatch, detected: str) -> list[Path]:
    """Stand in for the WhisperX probe, recording what it was asked about."""
    from unrender.audio import language as audio_language

    seen: list[Path] = []

    def probe(path: Path, *, expected: str | None = None, **kwargs: object):
        seen.append(path)
        return audio_language.LanguageProbe(
            path=path,
            detected=detected,
            expected=expected,
            confidence=1.0,
            windows=((0.0, detected),),
        )

    monkeypatch.setattr(audio_language, "probe_audio_language", probe)
    return seen


def test_dx_resolution_refuses_a_stem_in_the_wrong_language(monkeypatch, tmp_path: Path) -> None:
    # Every command that consumes a dialogue stem resolves it here, so this is
    # the one place a wrong-language source cannot slip past the declared language.
    dx = tmp_path / "dialogue.wav"
    dx.write_bytes(b"audio")
    args = argparse.Namespace(
        project_config=_declared_language_project(tmp_path, dx), skip_language_check=False
    )
    _detects(monkeypatch, "en")

    with pytest.raises(ValueError, match="spoken in 'en' but 'de' was expected"):
        _dx_stem_arg(args, RunPaths.from_path(tmp_path / "run"), explicit=None, required=True)


def test_dx_resolution_can_be_told_to_accept_the_mismatch(monkeypatch, tmp_path: Path) -> None:
    dx = tmp_path / "dialogue.wav"
    dx.write_bytes(b"audio")
    args = argparse.Namespace(
        project_config=_declared_language_project(tmp_path, dx), skip_language_check=True
    )
    seen = _detects(monkeypatch, "en")

    resolved = _dx_stem_arg(
        args, RunPaths.from_path(tmp_path / "run"), explicit=None, required=True
    )

    assert resolved == str(dx)
    assert seen == [], "the probe is skipped entirely, not just its verdict"


def test_dx_resolution_does_not_probe_a_project_that_declares_nothing(
    monkeypatch, tmp_path: Path
) -> None:
    # Detection costs real seconds per command, so an undeclared project pays
    # nothing for a check it never asked for.
    dx = tmp_path / "dialogue.wav"
    dx.write_bytes(b"audio")
    config_path = tmp_path / "show-a.json"
    _write_project_config(
        config_path, tmp_path / "run", audio_assets={"original": {"dme": {"dx": str(dx)}}}
    )
    args = argparse.Namespace(
        project_config=load_project_config(config_path), skip_language_check=False
    )
    seen = _detects(monkeypatch, "en")

    _dx_stem_arg(args, RunPaths.from_path(tmp_path / "run"), explicit=None, required=True)

    assert seen == []


def test_audio_language_command_reports_a_mismatch_as_a_failure(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "dialogue.wav"
    source.write_bytes(b"audio")
    _detects(monkeypatch, "en")

    assert main(["audio", "language", str(source), "--expect", "de"]) == 1
    assert main(["audio", "language", str(source), "--expect", "en"]) == 0


def test_cli_project_config_runs_multiple_projects(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    runs = [tmp_path / "runs" / "show-a", tmp_path / "runs" / "show-b"]
    for index, run_dir in enumerate(runs, 1):
        _write_project_config(config_dir / f"show-{index}.json", run_dir)

    assert main(["labels", "template", "-p", "show-*"]) == 0

    assert all((run_dir / "labels.csv").exists() for run_dir in runs)


def test_cli_project_config_supplies_label_config_and_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    run = RunPaths.from_path(tmp_path / "runs" / "show-a")
    run.ensure()
    _write_project_config(config_dir / "show-a.json", run.root)
    run.face_db.write_text(
        json.dumps(
            {
                "version": "1.0",
                "clusters": [
                    {
                        "cluster_id": 7,
                        "name": "",
                        "face_count": 3,
                        "centroid": [1.0, 0.0],
                        "grid_path": str(run.face_review_dir / "cluster_007.jpg"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert main(["labels", "template", "-p", "show-a"]) == 0
    rows = list(csv.DictReader(run.labels_csv.open(newline="", encoding="utf-8")))
    rows[0]["speaker"] = "RENEE"
    with run.labels_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    assert main(["labels", "apply", "-p", "show-a"]) == 0

    speaker_db = read_json(run.speaker_db)
    assert speaker_db["face_clusters"][0]["speaker"] == "RENÉE"


def test_cli_project_config_uses_stem_map_as_native_split_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    run = RunPaths.from_path(tmp_path / "runs" / "show-a")
    stems_dir = tmp_path / "stems"
    stems_dir.mkdir()
    alex = stems_dir / "show_ALEX_stem.wav"
    jordan = stems_dir / "show_JORDAN_stem.wav"
    rate = 16_000
    write_wav(alex, np.full(rate * 13, 100, dtype=np.int16))
    write_wav(jordan, np.full(rate * 13, 50, dtype=np.int16))
    shots = tmp_path / "shots.csv"
    shots.write_text(
        "shot_id,video_path,speaker,start_sec,end_sec\n001,/tmp/001.mov,ALEX,10,12\n",
        encoding="utf-8",
    )
    _write_project_config(
        config_dir / "show-a.json",
        run.root,
        paths={"shots": str(shots)},
        stem_map={"ALEX": str(alex), "JORDAN": str(jordan)},
    )

    # No DX stem anywhere: shot-dx merges the stem_map speaker stems.
    assert main(["audio", "shot-dx", "-p", "show-a"]) == 0

    assert run.merged_dx_stem.exists()
    plan = read_json(run.shot_dx_plan_json)["shots"]
    assert plan[0]["shot_id"] == "001"
    shot_dx = Path(plan[0]["stem_path"])
    assert shot_dx.exists()
    with wave.open(str(shot_dx), "rb") as handle:
        assert handle.getnframes() == rate * 2  # exactly the 10-12s shot window
        merged_value = np.frombuffer(handle.readframes(1), dtype=np.int16)[0]
    assert merged_value == 150  # 100 + 50: summed speaker stems


def test_cli_project_config_rejects_unconfigured_stem_map_speaker(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    run = RunPaths.from_path(tmp_path / "runs" / "show-a")
    stem = tmp_path / "show_RANDOM_stem.wav"
    stem.write_bytes(b"source")
    shots = tmp_path / "shots.csv"
    shots.write_text(
        "shot_id,video_path,speaker,start_sec,end_sec\n001,/tmp/001.mov,ALEX,10,12\n",
        encoding="utf-8",
    )
    _write_project_config(
        config_dir / "show-a.json",
        run.root,
        paths={"shots": str(shots)},
        stem_map={"RANDOM": str(stem)},
    )

    assert main(["audio", "shot-dx", "-p", "show-a"]) == 1


def test_cli_project_config_supplies_transcribe_line_args(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    run = RunPaths.from_path(tmp_path / "runs" / "show-a")
    dx_stem = tmp_path / "dx.wav"
    dx_stem.write_bytes(b"audio")
    shots = tmp_path / "shots.csv"
    shots.write_text(
        "shot_id,video_path,start_sec,end_sec\n001,/tmp/001.mov,10,12\n",
        encoding="utf-8",
    )
    _write_project_config(
        config_dir / "show-a.json",
        run.root,
        paths={"shots": str(shots)},
        audio_assets={"original": {"dme": {"dx": str(dx_stem)}}},
        audio={"max_gap_sec": 0.25, "handle_sec": 0.2, "whisper_model": "tiny"},
    )
    calls = {}

    def fake_transcribe(**kwargs) -> list[dict[str, object]]:
        calls.update(kwargs)
        return []

    monkeypatch.setattr("unrender.cli.commands.audio.transcribe_dialogue_lines", fake_transcribe)

    assert main(["audio", "transcribe-lines", "-p", "show-a"]) == 0

    assert calls["audio_path"] == dx_stem
    assert calls["whisper_model"] == "tiny"
    assert calls["max_gap_sec"] == 0.25
    assert calls["handle_sec"] == 0.2


def test_cli_transcribe_selects_source_dx_and_separated_stems(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    run = RunPaths.from_path(tmp_path / "runs" / "show-a")
    source_dx = tmp_path / "source_de" / "_dme" / "show_DX_de_stem.wav"
    source_stems = tmp_path / "source_de" / "_separated"
    source_dx.parent.mkdir(parents=True)
    source_stems.mkdir(parents=True)
    source_dx.write_bytes(b"dx")
    speaker = source_stems / "show_speaker_01_de_stem.wav"
    speaker.write_bytes(b"speaker")
    _write_project_config(
        config_dir / "show-a.json",
        run.root,
        audio_assets={
            "original": {
                "language": "de",
                "dme": {"dx": str(source_dx)},
                "separated_stems": str(source_stems / "*.wav"),
            }
        },
    )
    calls = {}

    def fake_transcribe(**kwargs) -> list[dict[str, object]]:
        calls.update(kwargs)
        return []

    monkeypatch.setattr("unrender.cli.commands.audio.transcribe_dialogue_lines", fake_transcribe)

    assert main(["audio", "transcribe-lines", "-p", "show-a"]) == 0

    assert calls["audio_path"] == source_dx
    assert [source.path for source in calls["source_stems"]] == [speaker]


def test_cli_project_config_imports_dialogue_lines_from_transcript(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    run = RunPaths.from_path(tmp_path / "runs" / "show-a")
    script = tmp_path / "TRANSCRIPT.csv"
    script.write_text(
        "start_sec,end_sec,speaker,text\n1,2,Alex,hello\n",
        encoding="utf-8",
    )
    _write_project_config(
        config_dir / "show-a.json",
        run.root,
        paths={"transcript": str(script)},
        audio={"transcript_fps": 24.0, "handle_sec": 0.2},
        speakers={"ALEX": {"aliases": []}},
    )
    calls = {}

    def fake_import(**kwargs) -> list[dict[str, object]]:
        calls.update(kwargs)
        return []

    monkeypatch.setattr("unrender.cli.commands.audio.import_transcript_lines", fake_import)

    assert main(["audio", "transcribe-lines", "-p", "show-a"]) == 0

    assert calls["transcript_path"] == script
    assert calls["fps"] == 24.0
    assert calls["handle_sec"] == 0.2


def test_cli_project_config_builds_native_clips_from_stem_map(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    run = RunPaths.from_path(tmp_path / "runs" / "show-a")
    run.ensure()
    run.dialogue_lines_json.write_text(
        json.dumps(
            {
                "version": "1.0",
                "lines": [
                    {
                        "line_id": "DL_000001",
                        "start_sec": 1.0,
                        "end_sec": 2.0,
                        "cut_start_sec": 0.8,
                        "cut_end_sec": 2.2,
                        "text": "hello",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    stem = tmp_path / "ALEX_stem.wav"
    stem.write_bytes(b"audio")
    _write_project_config(
        config_dir / "show-a.json",
        run.root,
        stem_map={"ALEX": str(stem)},
        audio={"clip_granularity": "dialogue-lines"},
    )
    calls = {}

    def fake_build(**kwargs) -> list[dict[str, object]]:
        calls.update(kwargs)
        return []

    monkeypatch.setattr("unrender.cli.commands.audio.build_voice_clips", fake_build)

    assert main(["audio", "build-clips", "-p", "show-a"]) == 0

    assert calls["clip_type"] == "dialogue_line"
    assert calls["lines"][0].line_id == "DL_000001"
    assert calls["stems"][0].speaker == "ALEX"


def _write_project_config(
    path: Path,
    run_dir: Path,
    *,
    paths: dict[str, str] | None = None,
    stem_map: dict[str, str] | None = None,
    audio: dict[str, object] | None = None,
    audio_assets: dict[str, object] | None = None,
    speakers: dict[str, object] | None = None,
) -> None:
    config_paths = {"run_dir": str(run_dir)}
    config_paths.update(paths or {})
    data = {
        "speakers": speakers
        or {
            "RENÉE": {"aliases": ["RENEE"]},
            "ALEX": {"aliases": []},
            "JORDAN": {"aliases": []},
        },
        "paths": config_paths,
    }
    if stem_map is not None:
        data["stem_map"] = stem_map
    if audio is not None:
        data["audio"] = audio
    if audio_assets is not None:
        data["audio_assets"] = audio_assets
    path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )
