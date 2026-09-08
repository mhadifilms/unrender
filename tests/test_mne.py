from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import write_wav
from unrender.audio.mne.fx_classify import FX_CATEGORIES, FxClassifySettings, classify_fx
from unrender.audio.mne.music import MusicSettings, process_music
from unrender.audio.mne.room_tone import RoomToneSettings, extract_room_tone
from unrender.audio.separation.audioshake_client import AudioShakeClient
from unrender.lib import audio_features as features
from unrender.project import RunPaths

SR = 16_000


def _tone(freq: float, dur: float, *, amp: int = 4000, sr: int = SR) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.int16)


def _noise(dur: float, *, amp: int, seed: int, sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(dur * sr)) * amp).astype(np.int16)


def _silence(dur: float, *, sr: int = SR) -> np.ndarray:
    return np.zeros(int(dur * sr), dtype=np.int16)


# --------------------------------------------------------------------------- #
# features: granular loop seam continuity
# --------------------------------------------------------------------------- #


def test_granular_loop_has_no_seam_discontinuities() -> None:
    # A smooth donor makes any grain-join jump obvious: naive concatenation of
    # random grains would spike, equal-power overlap-add must not.
    t = np.arange(2 * SR) / SR
    donor = (0.5 * np.sin(2 * np.pi * 110.0 * t)).astype(np.float32)
    donor_max_step = float(np.max(np.abs(np.diff(donor))))

    rng = np.random.default_rng(0)
    loop = features.granular_loop(
        donor,
        loop_samples=SR,
        grain_samples=SR // 4,
        crossfade_samples=SR // 20,
        rng=rng,
    )

    assert loop.size == SR
    max_step = float(np.max(np.abs(np.diff(loop))))
    # Overlap-add of two smooth grains at most doubles the local slope.
    assert max_step < donor_max_step * 3.0


def test_activity_segments_merge_and_min_duration() -> None:
    mono = np.concatenate(
        [
            np.zeros(SR, dtype=np.float32),
            (0.5 * np.ones(SR)).astype(np.float32),
            np.zeros(SR, dtype=np.float32),
        ]
    )
    segments = features.activity_segments(
        mono, sample_rate=SR, threshold_db=-40.0, merge_gap_sec=0.2, min_duration_sec=0.5
    )
    assert len(segments) == 1
    assert segments[0].start_sec == pytest.approx(1.0, abs=0.1)
    assert segments[0].end_sec == pytest.approx(2.0, abs=0.1)


# --------------------------------------------------------------------------- #
# FX classification (CLAP mocked)
# --------------------------------------------------------------------------- #


def test_classify_fx_segments_events_and_writes_tracks(run_paths: RunPaths, monkeypatch) -> None:
    fx = run_paths.source_stems_dir / "show_FX_stem.wav"
    stem = np.concatenate(
        [
            _silence(0.6),
            _tone(500, 1.0, amp=3000),
            _silence(0.6),
            _tone(900, 1.0, amp=3000),
            _silence(0.6),
        ]
    )
    write_wav(fx, stem)

    labels = ["foley", "hard_fx"]

    def fake_clap(clips, *, sample_rate, model_name, device):
        assert sample_rate == SR
        out = []
        for index in range(len(clips)):
            label = labels[index % len(labels)]
            scores = {key: 0.02 for key, _suffix, _prompts in FX_CATEGORIES}
            scores[label] = 0.9
            out.append(scores)
        return out

    monkeypatch.setattr("unrender.audio.mne.fx_classify.classify_events_clap", fake_clap)

    report = classify_fx(
        run=run_paths,
        fx_stem=fx,
        settings=FxClassifySettings(activity_threshold_db=-50.0, min_event_sec=0.5),
        prefix="show",
    )

    assert len(report["events"]) == 2
    assert {event["label"] for event in report["events"]} == {"foley", "hard_fx"}
    for suffix in ("AMB", "FOLEY", "HFX", "DSGN"):
        assert (run_paths.mne_fx_dir / f"show_{suffix}_stem.wav").exists()
    # Each classified burst routes energy into its own category track; the
    # unused design track stays silent.
    from unrender.lib.audio import read_wav_float

    foley, _r1, _w1 = read_wav_float(run_paths.mne_fx_dir / "show_FOLEY_stem.wav")
    hfx, _r2, _w2 = read_wav_float(run_paths.mne_fx_dir / "show_HFX_stem.wav")
    dsgn, _r3, _w3 = read_wav_float(run_paths.mne_fx_dir / "show_DSGN_stem.wav")
    assert float(np.max(np.abs(foley))) > 0.0
    assert float(np.max(np.abs(hfx))) > 0.0
    assert float(np.max(np.abs(dsgn))) == 0.0


def test_classify_fx_manifest_round_trip_and_skip(run_paths: RunPaths, monkeypatch) -> None:
    fx = run_paths.source_stems_dir / "show_FX_stem.wav"
    write_wav(fx, np.concatenate([_silence(0.5), _tone(500, 1.0, amp=3000), _silence(0.5)]))

    calls = {"n": 0}

    def fake_clap(clips, **kwargs):
        calls["n"] += 1
        return [{"foley": 0.9, "ambience": 0.03, "hard_fx": 0.03, "design_fx": 0.04}]

    monkeypatch.setattr("unrender.audio.mne.fx_classify.classify_events_clap", fake_clap)

    first = classify_fx(run=run_paths, fx_stem=fx, prefix="show")
    second = classify_fx(run=run_paths, fx_stem=fx, prefix="show")
    assert calls["n"] == 1  # second run is skipped (manifest exists)
    assert first["events"] == second["events"]
    assert run_paths.mne_fx_classification_json.exists()


# --------------------------------------------------------------------------- #
# Music cue detection
# --------------------------------------------------------------------------- #


def test_music_cue_detection_boundaries(run_paths: RunPaths) -> None:
    mx = run_paths.source_stems_dir / "show_MX_stem.wav"
    stem = np.concatenate(
        [
            _silence(1.0),
            _tone(440, 3.0, amp=4000),
            _silence(1.5),
            _tone(440, 2.5, amp=4000),
            _silence(1.0),
        ]
    )
    write_wav(mx, stem)

    report = process_music(
        run=run_paths,
        mx_stem=mx,
        settings=MusicSettings(
            denoise=False,
            silence_db=-50.0,
            min_cue_sec=2.0,
            min_silence_sec=1.0,
            spectral_split=False,
            pad_sec=0.5,
        ),
        prefix="show",
    )

    cues = report["cues"]
    assert len(cues) == 2
    assert cues[0]["start_sec"] == pytest.approx(1.0, abs=0.15)
    assert cues[0]["end_sec"] == pytest.approx(4.0, abs=0.15)
    assert cues[1]["start_sec"] == pytest.approx(5.5, abs=0.2)
    assert cues[0]["pad_start_sec"] == pytest.approx(0.5, abs=0.15)
    for cue in cues:
        assert Path(cue["path"]).exists()


def test_music_spectral_split_of_butt_joined_cues(run_paths: RunPaths) -> None:
    mx = run_paths.source_stems_dir / "show_MX_stem.wav"
    stem = np.concatenate(
        [
            _silence(0.5),
            _tone(200, 3.0, amp=4000),
            _tone(4000, 3.0, amp=4000),
            _silence(0.5),
        ]
    )
    write_wav(mx, stem)

    report = process_music(
        run=run_paths,
        mx_stem=mx,
        settings=MusicSettings(
            denoise=False,
            silence_db=-50.0,
            min_cue_sec=1.0,
            min_silence_sec=1.0,
            spectral_split=True,
            spectral_threshold=0.1,
            spectral_window_sec=1.0,
        ),
        prefix="show",
    )

    assert len(report["cues"]) >= 2


# --------------------------------------------------------------------------- #
# Room tone gap harvesting and rejection
# --------------------------------------------------------------------------- #


def test_room_tone_harvests_and_rejects_gaps(run_paths: RunPaths) -> None:
    dx = run_paths.source_stems_dir / "show_DX_stem.wav"
    transient = _noise(1.0, amp=60, seed=7)
    transient[SR // 2] = 9000  # a click -> high crest factor -> rejected
    stem = np.concatenate(
        [
            _noise(1.0, amp=60, seed=1),  # gap: accepted room tone
            _tone(300, 1.0, amp=6000),  # speech island
            _noise(1.0, amp=60, seed=2),  # gap: accepted room tone
            _tone(300, 1.0, amp=6000),  # speech island
            _silence(1.0),  # gap: digital silence -> rejected
            _tone(300, 1.0, amp=6000),  # speech island
            transient,  # gap: transient -> rejected
        ]
    )
    write_wav(dx, stem)

    report = extract_room_tone(
        run=run_paths,
        dx_stem=dx,
        settings=RoomToneSettings(
            min_gap_sec=0.4,
            threshold_db=-45.0,
            digital_silence_db=-80.0,
            transient_ratio=6.0,
            loop_sec=1.0,
            grain_ms=200.0,
        ),
        prefix="show",
    )

    assert len(report["donor_gaps"]) == 2
    reasons = {row["reason"] for row in report["rejected_gaps"]}
    assert "digital_silence" in reasons
    assert "transient" in reasons

    rt = Path(report["rt_stem"])
    assert rt.exists()
    assert report["groups"]
    assert Path(report["groups"][0]["loop_path"]).exists()

    from unrender.lib.audio import probe_duration_sec

    assert probe_duration_sec(rt) == pytest.approx(probe_duration_sec(dx), abs=0.1)


def test_room_tone_groups_by_scene_when_manifest_present(run_paths: RunPaths) -> None:
    import json

    dx = run_paths.source_stems_dir / "show_DX_stem.wav"
    stem = np.concatenate(
        [
            _noise(1.5, amp=60, seed=1),  # scene A room tone
            _tone(300, 1.0, amp=6000),  # speech
            _noise(1.5, amp=60, seed=2),  # scene B room tone
            _tone(300, 1.0, amp=6000),  # speech
            _noise(1.5, amp=60, seed=3),  # scene B room tone
        ]
    )
    write_wav(dx, stem)
    scenes = run_paths.scenes_manifest_json
    scenes.write_text(
        json.dumps(
            {
                "scenes": [
                    {"scene_id": "A", "start_sec": 0.0, "end_sec": 3.0},
                    {"scene_id": "B", "start_sec": 3.0, "end_sec": 6.5},
                ]
            }
        ),
        encoding="utf-8",
    )

    report = extract_room_tone(
        run=run_paths,
        dx_stem=dx,
        settings=RoomToneSettings(loop_sec=1.0, grain_ms=200.0, edge_handle_sec=0.1),
        scenes_path=scenes,
        prefix="show",
    )

    assert report["grouping"] == "scene"
    group_ids = {group["group_id"] for group in report["groups"]}
    assert group_ids == {"A", "B"}
    span_ids = {span["group_id"] for span in report["spans"]}
    assert span_ids == {"A", "B"}
    for group in report["groups"]:
        assert Path(group["loop_path"]).exists()
    assert Path(report["rt_stem"]).exists()


def test_room_tone_manifest_round_trip_and_skip(run_paths: RunPaths) -> None:
    dx = run_paths.source_stems_dir / "show_DX_stem.wav"
    stem = np.concatenate(
        [
            _noise(1.0, amp=60, seed=1),
            _tone(300, 1.0, amp=6000),
            _noise(1.0, amp=60, seed=2),
        ]
    )
    write_wav(dx, stem)

    first = extract_room_tone(
        run=run_paths, dx_stem=dx, settings=RoomToneSettings(loop_sec=1.0), prefix="show"
    )
    second = extract_room_tone(
        run=run_paths, dx_stem=dx, settings=RoomToneSettings(loop_sec=1.0), prefix="show"
    )
    assert first["donor_gaps"] == second["donor_gaps"]
    assert run_paths.mne_room_tone_json.exists()


# --------------------------------------------------------------------------- #
# AudioShake separate_targets against a mocked API/session
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.status_code = 200

    def json(self) -> dict:
        return self._payload

    def iter_content(self, chunk_size: int = 8192):
        yield b"audio-bytes"

    def raise_for_status(self) -> None:
        return None


class _FakeSession:
    def __init__(self) -> None:
        self.headers: dict = {}
        self.requests: list[tuple[str, str, dict]] = []

    def request(self, method: str, url: str, **kwargs):
        self.requests.append((method, url, kwargs))
        if method == "POST" and url.endswith("/tasks"):
            return _FakeResponse({"id": "task-1"})
        if method == "GET" and "/tasks/" in url:
            return _FakeResponse(
                {
                    "status": "completed",
                    "targets": [
                        {
                            "status": "completed",
                            "output": [
                                {"link": "https://x/drums.wav", "stem": "drums"},
                                {"link": "https://x/bass.wav", "stem": "bass"},
                            ],
                        }
                    ],
                }
            )
        raise AssertionError(f"unexpected request {method} {url}")


def test_cli_mne_subcommands_dry_run(run_paths: RunPaths) -> None:
    from unrender.cli import main

    fx = run_paths.source_stems_dir / "show_FX_stem.wav"
    mx = run_paths.source_stems_dir / "show_MX_stem.wav"
    dx = run_paths.source_stems_dir / "show_DX_stem.wav"
    write_wav(fx, _tone(500, 1.0, amp=3000))
    write_wav(mx, _tone(440, 1.0, amp=3000))
    write_wav(dx, _tone(300, 1.0, amp=3000))

    run = str(run_paths.root)
    assert main(["mne", "classify-fx", "--run-dir", run, "--fx-stem", str(fx), "-n"]) == 0
    assert main(["mne", "music", "--run-dir", run, "--mx-stem", str(mx), "-n"]) == 0
    assert main(["mne", "room-tone", "--run-dir", run, "--dx-stem", str(dx), "-n"]) == 0
    # Dry runs write nothing.
    assert not run_paths.mne_fx_classification_json.exists()
    assert not run_paths.mne_music_cues_json.exists()
    assert not run_paths.mne_room_tone_json.exists()


def test_separate_targets_builds_targets_and_downloads(tmp_path: Path, monkeypatch) -> None:
    client = AudioShakeClient(api_key="test-key")
    session = _FakeSession()
    client.session = session

    def fake_get(link: str, **kwargs):
        return _FakeResponse({})

    monkeypatch.setattr("unrender.audio.separation.audioshake_client.requests.get", fake_get)

    out = tmp_path / "out"
    downloaded = client.separate_targets(
        "https://x/mx.wav",
        out,
        ["drums", "bass"],
        prefix="show",
    )

    # The task body carries one target per requested model.
    post = next(req for req in session.requests if req[0] == "POST")
    body = post[2]["json"]
    assert [target["model"] for target in body["targets"]] == ["drums", "bass"]

    assert {path.name for path in downloaded} == {"show_drums.wav", "show_bass.wav"}
    assert all(path.exists() for path in downloaded)
