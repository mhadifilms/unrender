from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unrender.audio import language as audio_language
from unrender.audio.audio_io import write_audio
from unrender.audio.language import (
    PROBE_SAMPLE_RATE,
    LanguageProbe,
    _candidate_starts,
    probe_audio_language,
)


def _speech(seconds: float, *, amplitude: float = 0.3) -> np.ndarray:
    count = int(seconds * PROBE_SAMPLE_RATE)
    time = np.arange(count, dtype=np.float32) / PROBE_SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 220.0 * time)).astype(np.float32)


def _source(
    monkeypatch: pytest.MonkeyPatch,
    *,
    duration_sec: float = 900.0,
    streams: int = 1,
    silent_before: float = 0.0,
) -> list[dict[str, object]]:
    """Fake a media file, returning the extraction calls ffmpeg is asked for.

    Windows starting before ``silent_before`` come back silent, which is how a
    sparse speaker stem behaves.
    """
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        audio_language, "probe_audio_streams", lambda path, **kwargs: tuple(range(streams))
    )
    monkeypatch.setattr(
        audio_language, "probe_media_duration_sec", lambda path, **kwargs: duration_sec
    )

    def extract(source: Path, target: Path, **kwargs: object) -> None:
        calls.append(kwargs)
        silent = float(kwargs["start_sec"]) < silent_before  # type: ignore[arg-type]
        samples = np.zeros(PROBE_SAMPLE_RATE * 30, dtype=np.float32) if silent else _speech(30.0)
        write_audio(target, samples, PROBE_SAMPLE_RATE, subtype="PCM_16")

    monkeypatch.setattr(audio_language, "extract_audio_window", extract)
    return calls


def _detects(monkeypatch: pytest.MonkeyPatch, codes: list[str | None]) -> None:
    remaining = list(codes)

    def detect(excerpt: Path, **kwargs: object) -> str | None:
        return remaining.pop(0) if remaining else None

    monkeypatch.setattr(audio_language, "_detect_window", detect)


@pytest.fixture(autouse=True)
def _fresh_detection_cache() -> None:
    audio_language._DETECTIONS.clear()


def test_probe_reuses_a_detection_across_stages_asking_about_one_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Several stages ask about the same 90-minute program in a single run, and
    # each answer costs half a minute of WhisperX.
    source = tmp_path / "dx.wav"
    source.write_bytes(b"audio")
    calls = _source(monkeypatch)
    _detects(monkeypatch, ["de", "de", "de", "en", "en", "en"])

    first = probe_audio_language(source, expected="de")
    second = probe_audio_language(source, expected="fr")

    assert second.detected == first.detected == "de"
    assert second.expected == "fr", "the verdict is per caller, the detection is not"
    assert second.issue is not None
    assert len(calls) == 3, "the second ask sampled nothing"


def test_probe_reports_the_agreed_language(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _source(monkeypatch)
    _detects(monkeypatch, ["de", "de", "de"])

    probe = probe_audio_language(tmp_path / "dx.wav", expected="German")

    assert probe.detected == "de"
    assert probe.expected == "de"
    assert probe.matches is True
    assert probe.confidence == 1.0
    assert probe.issue is None


def test_probe_flags_audio_in_an_unexpected_language(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source(monkeypatch)
    _detects(monkeypatch, ["en", "en", "en"])

    probe = probe_audio_language(tmp_path / "dialogue.wav", expected="de")

    assert probe.matches is False
    assert probe.issue is not None
    assert "'en'" in probe.issue and "'de'" in probe.issue


def test_probe_tolerates_a_single_disagreeing_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source(monkeypatch)
    _detects(monkeypatch, ["de", "de", "nn"])

    probe = probe_audio_language(tmp_path / "dx.wav", expected="de")

    assert probe.detected == "de"
    assert probe.issue is None


def test_probe_does_not_fail_a_run_on_a_low_confidence_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source(monkeypatch)
    _detects(monkeypatch, ["en", "de", "fr"])

    probe = probe_audio_language(tmp_path / "dx.wav", expected="de")

    assert probe.confidence < 0.6
    assert probe.issue is None


def test_probe_does_not_refuse_on_a_single_window_that_agreed_with_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two windows detect nothing and the third says English. That is one
    # sample, not a consensus, and must not fail a German project's run.
    _source(monkeypatch)
    _detects(monkeypatch, [None, None, "en"])

    probe = probe_audio_language(tmp_path / "dx.wav", expected="de")

    assert probe.detected == "en"
    assert probe.confidence == pytest.approx(1 / 3)
    assert probe.issue is None


def test_probe_skips_an_unseekable_window_instead_of_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _source(monkeypatch)
    inner = audio_language.extract_audio_window

    def extract(source: Path, target: Path, **kwargs: object) -> None:
        inner(source, target, **kwargs)  # type: ignore[arg-type]
        if len(calls) == 1:
            raise RuntimeError("ffmpeg failed to seek")

    monkeypatch.setattr(audio_language, "extract_audio_window", extract)
    _detects(monkeypatch, ["de", "de", "de"])

    probe = probe_audio_language(tmp_path / "dx.wav", expected="de")

    assert probe.detected == "de"
    assert len(probe.windows) == 3, "the failed offset was replaced, not fatal"


def test_probe_without_an_expectation_reports_but_never_complains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source(monkeypatch)
    _detects(monkeypatch, ["es", "es", "es"])

    probe = probe_audio_language(tmp_path / "mix.wav")

    assert probe.detected == "es"
    assert probe.matches is None
    assert probe.issue is None


def test_probe_samples_only_what_it_needs_from_a_long_program(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A 93-minute program must not be decoded end to end to read 90 seconds.
    calls = _source(monkeypatch, duration_sec=5587.0)
    _detects(monkeypatch, ["de", "de", "de"])

    probe = probe_audio_language(tmp_path / "program.mov", expected="de")

    starts = [call["start_sec"] for call in calls]
    assert len(starts) == 3
    assert starts == sorted(starts), "windows are sampled in program order"
    assert starts[0] > 0 and starts[-1] < 5587.0
    assert [start for start, _ in probe.windows] == starts
    assert all(call["duration_sec"] == 30.0 for call in calls)


def test_probe_skips_silence_and_keeps_looking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A separated speaker stem is silent until late in the program.
    calls = _source(monkeypatch, duration_sec=3600.0, silent_before=2400.0)
    _detects(monkeypatch, ["de", "de", "de"])

    probe = probe_audio_language(tmp_path / "speaker.wav", expected="de")

    assert probe.detected == "de"
    assert len(probe.windows) == 3
    assert all(start >= 2400.0 for start, _ in probe.windows)
    assert len(calls) > 3, "quiet windows were tried and skipped"


def test_probe_reports_total_silence_as_undetermined_rather_than_a_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source(monkeypatch, duration_sec=3600.0, silent_before=1e9)
    _detects(monkeypatch, [])

    probe = probe_audio_language(tmp_path / "silent.wav", expected="de")

    assert probe.windows == ()
    assert probe.detected is None
    assert probe.matches is None
    assert probe.issue is None


def test_probe_merges_every_track_of_a_multi_track_master(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A 5.1 + stereo master keeps dialogue on one discrete track, so decoding
    # ffmpeg's default pick alone can sample a channel with no speech in it.
    calls = _source(monkeypatch, streams=8)
    _detects(monkeypatch, ["de", "de", "de"])

    probe_audio_language(tmp_path / "program.mov", expected="de")

    assert all(call["audio_streams"] == 8 for call in calls)


def test_probe_reports_a_picture_with_no_audio_as_uncheckable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # probe_audio_streams raises ValueError, which callers reserve for "refuse
    # this run". A file we simply cannot check must not look like a refusal.
    def no_audio(path: Path, **kwargs: object) -> tuple[int, ...]:
        raise ValueError(f"media contains no audio streams: {path}")

    monkeypatch.setattr(audio_language, "probe_audio_streams", no_audio)

    with pytest.raises(RuntimeError, match="no audio"):
        probe_audio_language(tmp_path / "silent.mov")


def test_probe_rejects_nonsense_sampling_settings(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="window_sec must be positive"):
        probe_audio_language(tmp_path / "dx.wav", window_sec=0)


def test_candidate_starts_spread_across_the_program_before_refining() -> None:
    starts = _candidate_starts(3600.0, window_sec=30.0, count=3)

    assert starts[:3] == sorted(starts[:3])
    assert starts[0] < 1200.0 < starts[1] < 2400.0 < starts[2]
    assert len(starts) == len(set(starts)) == 12
    assert max(starts) <= 3570.0


def test_candidate_starts_handles_a_file_shorter_than_one_window() -> None:
    assert _candidate_starts(10.0, window_sec=30.0, count=3) == [0.0]


def test_issue_names_the_file_so_the_operator_can_act(tmp_path: Path) -> None:
    probe = LanguageProbe(
        path=tmp_path / "dme" / "dialogue.wav",
        detected="en",
        expected="de",
        confidence=1.0,
        windows=((0.0, "en"),),
    )

    assert "dialogue.wav" in (probe.issue or "")
