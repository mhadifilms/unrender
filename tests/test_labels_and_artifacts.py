from __future__ import annotations

import csv
import json
from pathlib import Path

from unrender.manifests import (
    load_dialogue_lines,
    load_shot_manifest,
    load_voice_inputs,
    read_json,
)
from unrender.project import RunPaths
from unrender.project.config import load_speaker_config
from unrender.speakers.labeling import apply_labels, label_interactive, write_label_template


def test_labels_template_and_apply_updates_shared_db(tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path)
    run.ensure()
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
    run.voice_db.write_text(
        json.dumps(
            {
                "version": "1.0",
                "clusters": [
                    {
                        "cluster_id": 2,
                        "name": "",
                        "sample_count": 4,
                        "centroid": [0.0, 1.0],
                        "review_dir": str(run.voice_review_dir / "cluster_002"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    labels_path = write_label_template(run)
    rows = list(csv.DictReader(labels_path.open(newline="", encoding="utf-8")))
    assert {row["cluster_ref"] for row in rows} == {"face:7", "voice:2"}
    for row in rows:
        row["speaker"] = "RENEE" if row["type"] == "face" else "ALEX"
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    db = apply_labels(
        run,
        labels_path,
        config={"speakers": {"RENÉE": {"aliases": ["RENEE"]}, "ALEX": {"aliases": []}}},
    )

    assert db["face_clusters"][0]["speaker"] == "RENÉE"
    assert db["voice_clusters"][0]["speaker"] == "ALEX"
    assert db["speakers"]["RENEE"]["face_clusters"] == [7]
    assert db["speakers"]["ALEX"]["voice_clusters"] == [2]


def test_apply_labels_rejects_labels_outside_configured_speakers(tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path)
    run.ensure()
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
    labels_path = write_label_template(run)
    rows = list(csv.DictReader(labels_path.open(newline="", encoding="utf-8")))
    rows[0]["speaker"] = "RANDOM"
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    db = apply_labels(run, labels_path, config={"speakers": ["ALEX"]})

    assert db["face_clusters"][0].get("speaker", "") == ""
    assert "RANDOM" not in db["speakers"]


def test_interactive_labels_choose_from_configured_speakers(monkeypatch, tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path)
    run.ensure()
    run.voice_db.write_text(
        json.dumps(
            {
                "version": "1.0",
                "clusters": [
                    {
                        "cluster_id": 2,
                        "name": "",
                        "sample_count": 4,
                        "centroid": [0.0, 1.0],
                        "review_dir": str(run.voice_review_dir / "cluster_002"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "2")

    db = label_interactive(
        run,
        config={"speakers": {"ALEX": {}, "RENÉE": {"aliases": ["RENEE"]}}},
        cluster_type="voice",
    )

    assert db["voice_clusters"][0]["speaker"] == "RENÉE"
    assert db["speakers"]["RENEE"]["voice_clusters"] == [2]


def test_interactive_labels_reject_custom_names_and_reprompt(monkeypatch, tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path)
    run.ensure()
    run.voice_db.write_text(
        json.dumps(
            {
                "version": "1.0",
                "clusters": [
                    {
                        "cluster_id": 2,
                        "name": "",
                        "sample_count": 4,
                        "centroid": [0.0, 1.0],
                        "review_dir": str(run.voice_review_dir / "cluster_002"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    answers = iter(["CUSTOM", "1"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    db = label_interactive(run, config={"speakers": {"ALEX": {}}}, cluster_type="voice")

    assert db["voice_clusters"][0]["speaker"] == "ALEX"
    assert "CUSTOM" not in db["speakers"]


def test_interactive_labels_confirm_and_persist_skip(monkeypatch, tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path)
    run.ensure()
    run.voice_db.write_text(
        json.dumps(
            {
                "version": "1.0",
                "clusters": [
                    {
                        "cluster_id": 2,
                        "name": "",
                        "sample_count": 4,
                        "centroid": [0.0, 1.0],
                        "review_dir": str(run.voice_review_dir / "cluster_002"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    answers = iter(["", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    db = label_interactive(run, config={"speakers": {"ALEX": {}}}, cluster_type="voice")
    stored = read_json(run.voice_db)["clusters"][0]

    assert stored["skipped"] is True
    assert stored.get("speaker", "") == ""
    assert db["voice_clusters"][0]["skipped"] is True

    def fail_input(_prompt: str) -> str:
        raise AssertionError("skipped clusters should not be prompted without relabel_all")

    monkeypatch.setattr("builtins.input", fail_input)
    label_interactive(run, config={"speakers": {"ALEX": {}}}, cluster_type="voice")


def test_interactive_labels_relabel_all_can_label_skipped(monkeypatch, tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path)
    run.ensure()
    run.voice_db.write_text(
        json.dumps(
            {
                "version": "1.0",
                "clusters": [
                    {
                        "cluster_id": 2,
                        "name": "",
                        "sample_count": 4,
                        "skipped": True,
                        "centroid": [0.0, 1.0],
                        "review_dir": str(run.voice_review_dir / "cluster_002"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "1")

    db = label_interactive(
        run, config={"speakers": {"ALEX": {}}}, cluster_type="voice", relabel_all=True
    )
    stored = read_json(run.voice_db)["clusters"][0]

    assert stored["speaker"] == "ALEX"
    assert "skipped" not in stored
    assert db["voice_clusters"][0]["speaker"] == "ALEX"


def test_interactive_labels_label_skipped_without_relabeling_all(
    monkeypatch, tmp_path: Path
) -> None:
    run = RunPaths.from_path(tmp_path)
    run.ensure()
    run.voice_db.write_text(
        json.dumps(
            {
                "version": "1.0",
                "clusters": [
                    {
                        "cluster_id": 2,
                        "name": "",
                        "sample_count": 4,
                        "skipped": True,
                        "centroid": [0.0, 1.0],
                        "review_dir": str(run.voice_review_dir / "cluster_002"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "1")

    db = label_interactive(
        run, config={"speakers": {"ALEX": {}}}, cluster_type="voice", label_skipped=True
    )

    assert db["voice_clusters"][0]["speaker"] == "ALEX"
    assert "skipped" not in read_json(run.voice_db)["clusters"][0]


def test_interactive_labels_preview_opens_review_path(monkeypatch, tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path)
    run.ensure()
    review = tmp_path / "cluster_001.jpg"
    review.write_bytes(b"jpg")
    run.face_db.write_text(
        json.dumps(
            {
                "version": "1.0",
                "clusters": [
                    {
                        "cluster_id": 1,
                        "name": "",
                        "face_count": 3,
                        "grid_path": str(review),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    opened = []

    class DummyProcess:
        pass

    monkeypatch.setattr("builtins.input", lambda _prompt: "1")
    monkeypatch.setattr(
        "unrender.speakers.labeling.subprocess.Popen",
        lambda argv: opened.append(argv) or DummyProcess(),
    )

    label_interactive(run, config={"speakers": {"ALEX": {}}}, cluster_type="face", preview=True)

    assert opened
    assert str(review) in opened[0]


def test_load_speaker_config_requires_speaker_set(tmp_path: Path) -> None:
    config_path = tmp_path / "speakers.json"
    config_path.write_text(json.dumps({"speakers": ["ALEX"]}), encoding="utf-8")

    assert load_speaker_config(config_path)["speakers"] == ["ALEX"]


def test_load_shot_manifest_csv(tmp_path: Path) -> None:
    manifest = tmp_path / "shots.csv"
    manifest.write_text(
        "shot_id,video_path,existing_speaker,target_face_box\n"
        '001,/tmp/shot001.mov,ALEX,"[10, 20, 30, 40]"\n',
        encoding="utf-8",
    )

    shots = load_shot_manifest(manifest)

    assert shots[0].shot_id == "001"
    assert str(shots[0].video_path) == "/tmp/shot001.mov"
    assert shots[0].existing_speaker == "ALEX"
    assert shots[0].target_face_box == (10, 20, 30, 40)


def test_load_voice_inputs_manifest_with_embedding(tmp_path: Path) -> None:
    manifest = tmp_path / "clips.csv"
    row = (
        "/tmp/001_speaker_01.wav,"
        "DL_000001_speaker_01,"
        "dialogue_line,"
        "DL_000001,"
        "001,"
        "speaker_01,"
        "/tmp/full_speaker_01.wav,"
        '"hello",'
        '"[1.0, 0.0]"\n'
    )
    manifest.write_text(
        "path,clip_id,clip_type,line_id,shot_id,source_group,source_stem,text,embedding\n" + row,
        encoding="utf-8",
    )

    clips = load_voice_inputs(str(manifest))

    assert clips[0].clip_id == "DL_000001_speaker_01"
    assert clips[0].clip_type == "dialogue_line"
    assert clips[0].line_id == "DL_000001"
    assert clips[0].shot_id == "001"
    assert clips[0].source_group == "speaker_01"
    assert clips[0].source_stem == "/tmp/full_speaker_01.wav"
    assert clips[0].text == "hello"
    assert clips[0].embedding == (1.0, 0.0)


def test_load_voice_inputs_skips_pruned_voice_clip_rows(tmp_path: Path) -> None:
    kept = tmp_path / "kept.wav"
    dropped = tmp_path / "dropped.wav"
    kept.write_bytes(b"wav")
    manifest = tmp_path / "voice_clips.json"
    manifest.write_text(
        json.dumps(
            {
                "clips": [
                    {"path": str(kept), "clip_id": "kept", "kept": True},
                    {"path": str(dropped), "clip_id": "dropped", "kept": False},
                ]
            }
        ),
        encoding="utf-8",
    )

    clips = load_voice_inputs(str(manifest))

    assert [clip.clip_id for clip in clips] == ["kept"]


def test_load_dialogue_lines_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "dialogue_lines.json"
    manifest.write_text(
        json.dumps(
            {
                "version": "1.0",
                "lines": [
                    {
                        "line_id": "DL_000001",
                        "start_sec": 1.0,
                        "end_sec": 2.0,
                        "cut_start_sec": 0.82,
                        "cut_end_sec": 2.18,
                        "diarized_speaker": "SPEAKER_00",
                        "text": "hello",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    lines = load_dialogue_lines(manifest)

    assert lines[0].line_id == "DL_000001"
    assert lines[0].cut_start_sec == 0.82
    assert lines[0].text == "hello"


def test_load_voice_inputs_infers_full_shot_id_before_speaker_token(tmp_path: Path) -> None:
    clip_dir = tmp_path / "audio" / "_unmapped" / "demo_001"
    clip_dir.mkdir(parents=True)
    clip = clip_dir / "demo_001_speaker_02_stem.wav"
    clip.write_bytes(b"audio")

    clips = load_voice_inputs(str(tmp_path / "audio" / "_unmapped" / "**" / "*.wav"))

    assert clips[0].shot_id == "demo_001"
    assert clips[0].source_group == "speaker_02"


def test_load_voice_inputs_infers_named_unmapped_source_group(tmp_path: Path) -> None:
    clip_dir = tmp_path / "audio" / "_unmapped" / "demo_001"
    clip_dir.mkdir(parents=True)
    clip = clip_dir / "demo_001_JORDAN_stem.wav"
    clip.write_bytes(b"audio")

    clips = load_voice_inputs(str(tmp_path / "audio" / "_unmapped" / "**" / "*.wav"))

    assert clips[0].shot_id == "demo_001"
    assert clips[0].source_group == "JORDAN"
