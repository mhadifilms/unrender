"""Cluster-then-assign dialogue resolution and margin-based clip matching."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from unrender.analysis.identity.voice import match_voice_db
from unrender.editorial.dialogue.clusters import resolve_voice_clips_clustered
from unrender.manifests import VoiceInput, read_json


def _write_speaker_db(path: Path, centroids: dict[str, list[float]]) -> None:
    path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "speakers": {},
                "face_clusters": [],
                "voice_clusters": [
                    {"cluster_id": index, "name": name, "speaker": name, "centroid": centroid}
                    for index, (name, centroid) in enumerate(centroids.items())
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_wav(path: Path, amplitude: int, seconds: float = 0.5) -> None:
    rate = 16_000
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(np.full(int(rate * seconds), amplitude, dtype=np.int16).tobytes())


def _clip(
    line_id: str,
    group: str,
    embedding: list[float],
    path: Path,
    *,
    start: float = 0.0,
    end: float = 1.0,
) -> VoiceInput:
    return VoiceInput(
        path=path,
        clip_id=f"{line_id}_{group}",
        clip_type="dialogue_line",
        line_id=line_id,
        source_group=group,
        start_sec=start,
        end_sec=end,
        clip_start_sec=start,
        clip_end_sec=end,
        embedding=tuple(embedding),
    )


def _noisy(base: list[float], seed: int, scale: float = 0.05) -> list[float]:
    rng = np.random.default_rng(seed)
    vec = np.asarray(base, dtype=np.float64) + rng.normal(0.0, scale, len(base))
    return [float(v) for v in vec]


def test_clustered_resolver_assigns_lines_by_cluster(tmp_path: Path) -> None:
    db = tmp_path / "speaker_db.json"
    _write_speaker_db(db, {"ALEX": [1.0, 0.0, 0.0], "JORDAN": [0.0, 1.0, 0.0]})

    clips = []
    for i in range(4):
        clips.append(_clip(f"L{i:03d}", "speaker_01", _noisy([1, 0, 0], i), tmp_path / f"a{i}.wav"))
    for i in range(2):
        clips.append(
            _clip(f"L{i + 4:03d}", "speaker_02", _noisy([0, 1, 0], 10 + i), tmp_path / f"j{i}.wav")
        )

    plan = resolve_voice_clips_clustered(
        clips=clips,
        speaker_db_path=db,
        output_json=tmp_path / "plan.json",
        output_csv=tmp_path / "plan.csv",
    )

    by_line = {entry["line_id"]: entry for entry in plan}
    assert [by_line[f"L{i:03d}"]["speaker"] for i in range(4)] == ["ALEX"] * 4
    assert [by_line[f"L{i:03d}"]["speaker"] for i in (4, 5)] == ["JORDAN"] * 2
    assert all(entry["status"] == "matched" for entry in plan)
    assert all(entry["resolution_source"] == "cluster" for entry in plan)
    # Exactly two clusters, exclusively assigned.
    manifest = read_json(tmp_path / "plan.json")
    speakers = {cluster["speaker"] for cluster in manifest["clusters"]}
    assert speakers == {"ALEX", "JORDAN"}
    # Same-voice lines share a cluster id.
    assert len({by_line[f"L{i:03d}"]["cluster_id"] for i in range(4)}) == 1


def test_clustered_resolver_hungarian_prevents_double_assignment(tmp_path: Path) -> None:
    # Both clusters sit closer to ALEX than JORDAN; independent argmax would
    # give ALEX both, joint assignment must split them.
    db = tmp_path / "speaker_db.json"
    _write_speaker_db(db, {"ALEX": [1.0, 0.0, 0.0], "JORDAN": [0.6, 0.8, 0.0]})

    clips = [
        _clip("L000", "speaker_01", [1.0, 0.05, 0.0], tmp_path / "a.wav"),
        _clip("L001", "speaker_01", [1.0, 0.0, 0.05], tmp_path / "b.wav"),
        _clip("L002", "speaker_02", [0.9, 0.44, 0.0], tmp_path / "c.wav"),
        _clip("L003", "speaker_02", [0.9, 0.42, 0.05], tmp_path / "d.wav"),
    ]

    plan = resolve_voice_clips_clustered(
        clips=clips,
        speaker_db_path=db,
        output_json=tmp_path / "plan.json",
        output_csv=tmp_path / "plan.csv",
    )

    by_line = {entry["line_id"]: entry["speaker"] for entry in plan}
    assert by_line == {"L000": "ALEX", "L001": "ALEX", "L002": "JORDAN", "L003": "JORDAN"}


def test_long_confident_clip_overrides_cluster_label(tmp_path: Path) -> None:
    db = tmp_path / "speaker_db.json"
    _write_speaker_db(db, {"ALEX": [1.0, 0.0, 0.0], "JORDAN": [0.0, 1.0, 0.0]})

    clips = [
        _clip("L000", "speaker_01", _noisy([1, 0, 0], 0), tmp_path / "a.wav", end=0.5),
        _clip("L001", "speaker_01", _noisy([1, 0, 0], 1), tmp_path / "b.wav", end=0.5),
        # Rare speaker: one long line that clearly matches JORDAN.
        _clip("L002", "speaker_02", [0.0, 1.0, 0.0], tmp_path / "c.wav", end=2.0),
    ]

    # Force everything into one cluster so the rare speaker is folded in.
    plan = resolve_voice_clips_clustered(
        clips=clips,
        speaker_db_path=db,
        output_json=tmp_path / "plan.json",
        output_csv=tmp_path / "plan.csv",
        n_clusters=1,
    )

    by_line = {entry["line_id"]: entry for entry in plan}
    assert by_line["L000"]["speaker"] == "ALEX"
    assert by_line["L001"]["speaker"] == "ALEX"
    assert by_line["L002"]["speaker"] == "JORDAN"
    assert by_line["L002"]["resolution_source"] == "clip-override"
    # Short lines cannot override even if they matched someone else.
    assert by_line["L000"]["resolution_source"] == "cluster"


def test_stem_selection_fuses_identity_with_energy(tmp_path: Path) -> None:
    db = tmp_path / "speaker_db.json"
    _write_speaker_db(db, {"ALEX": [1.0, 0.0, 0.0]})

    loud = tmp_path / "loud.wav"
    quiet = tmp_path / "quiet.wav"
    _write_wav(loud, 12_000)
    _write_wav(quiet, 400)  # ~-30 dB below: bleed of the same voice

    clips = [
        _clip("L000", "speaker_02", [1.0, 0.0, 0.0], quiet),
        _clip("L000", "speaker_01", [1.0, 0.0, 0.0], loud),
    ]

    plan = resolve_voice_clips_clustered(
        clips=clips,
        speaker_db_path=db,
        output_json=tmp_path / "plan.json",
        output_csv=tmp_path / "plan.csv",
    )

    assert len(plan) == 1
    assert plan[0]["stem_path"] == str(loud)
    assert plan[0]["source_group"] == "speaker_01"


def test_energy_gate_skips_near_silent_slices_before_embedding(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "speaker_db.json"
    _write_speaker_db(db, {"ALEX": [1.0, 0.0]})

    loud = tmp_path / "loud.wav"
    quiet = tmp_path / "quiet.wav"
    _write_wav(loud, 12_000)
    _write_wav(quiet, 10)  # ~60 dB below: cannot win stem selection

    def clip_without_embedding(group: str, path: Path) -> VoiceInput:
        return VoiceInput(
            path=path,
            clip_id=f"L000_{group}",
            clip_type="dialogue_line",
            line_id="L000",
            source_group=group,
            start_sec=0.0,
            end_sec=1.0,
        )

    embedded_groups: list[str] = []

    def fake_embed(clips, *, backend="pyannote"):
        embedded_groups.extend(clip.source_group for clip in clips)
        return [
            {
                "path": clip.path,
                "clip_id": clip.clip_id,
                "clip_type": clip.clip_type,
                "line_id": clip.line_id,
                "shot_id": clip.shot_id,
                "source_group": clip.source_group,
                "source_stem": clip.source_stem,
                "start_sec": clip.start_sec,
                "end_sec": clip.end_sec,
                "clip_start_sec": clip.clip_start_sec,
                "clip_end_sec": clip.clip_end_sec,
                "text": clip.text,
                "embedding": (1.0, 0.0),
            }
            for clip in clips
        ]

    monkeypatch.setattr("unrender.editorial.dialogue.clusters.embed_voice_inputs", fake_embed)

    plan = resolve_voice_clips_clustered(
        clips=[
            clip_without_embedding("speaker_01", loud),
            clip_without_embedding("speaker_02", quiet),
        ],
        speaker_db_path=db,
        output_json=tmp_path / "plan.json",
        output_csv=tmp_path / "plan.csv",
    )

    assert embedded_groups == ["speaker_01"]
    assert len(plan) == 1
    assert plan[0]["stem_path"] == str(loud)


def test_match_voice_db_margin_abstains_on_ambiguous_clips(tmp_path: Path) -> None:
    db = tmp_path / "speaker_db.json"
    _write_speaker_db(db, {"ALEX": [1.0, 0.0, 0.0], "JORDAN": [0.0, 1.0, 0.0]})
    ambiguous = _clip("L000", "speaker_01", [0.707, 0.707, 0.0], tmp_path / "a.wav")

    matches = match_voice_db(
        clips=[ambiguous],
        speaker_db_path=db,
        output_json=tmp_path / "matches.json",
        sim_threshold=0.65,
        min_margin=0.05,
    )

    assert matches[0]["accepted"] == []
    assert abs(matches[0]["margin"]) < 0.01
    assert len(matches[0]["matches"]) == 2

    accepted = match_voice_db(
        clips=[ambiguous],
        speaker_db_path=db,
        output_json=tmp_path / "matches2.json",
        sim_threshold=0.65,
        min_margin=0.0,
    )
    assert accepted[0]["accepted"] != []


def test_resolve_clips_parser_flags() -> None:
    from unrender.cli.parser import build_parser

    args = build_parser().parse_args(
        ["audio", "resolve-clips", "--resolver", "clip", "--min-margin", "0.1"]
    )
    assert args.resolver == "clip"
    assert args.min_margin == 0.1

    # All tuning flags default to None so project-config values can apply.
    default = build_parser().parse_args(["audio", "resolve-clips"])
    assert default.resolver is None
    assert default.min_margin is None
    assert default.override_sec is None
    assert default.override_margin is None
    assert default.sim_threshold is None
    assert default.backend is None
