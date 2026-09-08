from __future__ import annotations

import shutil
import struct
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from unrender.lib import ffmpeg


def _wav_bytes(format_tag: int, bits: int, *, extensible: bool = False) -> bytes:
    channels, rate = 1, 16_000
    block_align = channels * bits // 8
    fmt = struct.pack(
        "<HHIIHH",
        0xFFFE if extensible else format_tag,
        channels,
        rate,
        rate * block_align,
        block_align,
        bits,
    )
    if extensible:
        guid = struct.pack("<H", format_tag) + bytes(14)
        fmt += struct.pack("<HHI", 22, bits, 1) + guid
    data = b"\x00" * (block_align * 4)
    chunks = (
        b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    )
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


def test_run_ffmpeg_slice_missing_binary_raises_friendly_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="ffmpeg executable not found"):
        ffmpeg.run_ffmpeg_slice(
            ffmpeg="unrender-nonexistent-ffmpeg",
            source=tmp_path / "in.wav",
            target=tmp_path / "out.wav",
            start_sec=0.0,
            end_sec=1.0,
        )


def test_run_ffmpeg_wraps_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1)

    monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="timed out"):
        ffmpeg._run_ffmpeg(["ffmpeg", "-i", "in.wav", "out.wav"], timeout=1)


def test_run_ffmpeg_wraps_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(returncode=1, cmd="ffmpeg")

    monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="status 1"):
        ffmpeg._run_ffmpeg(["ffmpeg", "-i", "in.wav", "out.wav"])


def test_pcm_codec_matches_source_sample_format(tmp_path: Path) -> None:
    float32 = tmp_path / "float32.wav"
    float32.write_bytes(_wav_bytes(3, 32))
    pcm24 = tmp_path / "pcm24.wav"
    pcm24.write_bytes(_wav_bytes(1, 24))
    extensible = tmp_path / "extensible.wav"
    extensible.write_bytes(_wav_bytes(3, 32, extensible=True))

    pcm16 = tmp_path / "pcm16.wav"
    with wave.open(str(pcm16), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(np.zeros(16, dtype=np.int16).tobytes())

    assert ffmpeg.pcm_codec_for_source(float32) == "pcm_f32le"
    assert ffmpeg.pcm_codec_for_source(pcm24) == "pcm_s24le"
    assert ffmpeg.pcm_codec_for_source(extensible) == "pcm_f32le"
    assert ffmpeg.pcm_codec_for_source(pcm16) == "pcm_s16le"


def test_pcm_codec_falls_back_for_non_wav_and_unreadable(tmp_path: Path) -> None:
    flac = tmp_path / "audio.flac"
    flac.write_bytes(b"fLaC")
    garbage = tmp_path / "garbage.wav"
    garbage.write_bytes(b"not a riff file")

    assert ffmpeg.pcm_codec_for_source(flac) == "pcm_s16le"
    assert ffmpeg.pcm_codec_for_source(garbage) == "pcm_s16le"
    assert ffmpeg.pcm_codec_for_source(tmp_path / "missing.wav") == "pcm_s16le"


def test_run_ffmpeg_slice_preserves_float_bit_depth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Float WAVs cannot take the native wave-module path, so this also
    # verifies the ffmpeg fallback engages for non-PCM sources.
    source = tmp_path / "stem.wav"
    source.write_bytes(_wav_bytes(3, 32))
    captured: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> None:
        captured["cmd"] = cmd

    monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)
    ffmpeg.run_ffmpeg_slice(
        ffmpeg="ffmpeg",
        source=source,
        target=tmp_path / "out.wav",
        start_sec=0.0,
        end_sec=1.0,
    )

    assert "pcm_f32le" in captured["cmd"]


def test_exact_slice_pads_and_trims_to_requested_duration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "stem.wav"
    source.write_bytes(_wav_bytes(1, 24))
    captured: dict[str, list[str]] = {}

    monkeypatch.setattr(
        ffmpeg,
        "_run_ffmpeg",
        lambda command: captured.update(command=command),
    )
    ffmpeg.run_ffmpeg_exact_slice(
        ffmpeg="ffmpeg",
        source=source,
        target=tmp_path / "out.wav",
        start_sec=10.25,
        end_sec=11.75,
    )

    command = captured["command"]
    assert command[command.index("-ss") + 1] == "10.250000"
    assert command[command.index("-t") + 1] == "1.500000"
    filter_graph = command[command.index("-af") + 1]
    assert "apad=whole_dur=1.500000" in filter_graph
    assert "atrim=start=0:end=1.500000" in filter_graph
    assert "pcm_s24le" in command


def test_timeline_mix_pads_dialogue_to_exact_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "stem.wav"
    source.write_bytes(_wav_bytes(1, 24))
    captured: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> None:
        captured["cmd"] = cmd

    monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)
    ffmpeg.run_ffmpeg_timeline_mix(
        ffmpeg="ffmpeg",
        segments=[
            (source, 10.5, 11.0, 0.5),
            (source, 11.4, 12.0, 1.4),
        ],
        target=tmp_path / "out.wav",
        duration_sec=2.0,
    )

    filter_complex = captured["cmd"][captured["cmd"].index("-filter_complex") + 1]
    assert "adelay=500:all=1" in filter_complex
    assert "adelay=1400:all=1" in filter_complex
    assert "anullsrc=r=48000:cl=mono:d=2.000000" in filter_complex
    assert "apad=whole_dur=2.000000" in filter_complex
    assert "atrim=start=0:end=2.000000" in filter_complex
    assert "pcm_s24le" in captured["cmd"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_timeline_mix_integration_preserves_full_duration(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    rate = 48_000
    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(np.ones(rate * 3, dtype=np.int16).tobytes())

    target = tmp_path / "timeline.wav"
    ffmpeg.run_ffmpeg_timeline_mix(
        ffmpeg="ffmpeg",
        segments=[
            (source, 0.0, 0.5, 0.5),
            (source, 1.0, 1.6, 1.4),
        ],
        target=target,
        duration_sec=2.5,
    )

    with wave.open(str(target), "rb") as handle:
        assert handle.getnframes() / handle.getframerate() == pytest.approx(2.5, abs=1e-4)


def test_native_wav_slice_is_sample_accurate_without_ffmpeg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rate = 16_000
    samples = np.arange(rate * 2, dtype=np.int16)  # 2s ramp, stereo below
    stereo = np.column_stack([samples, -samples]).reshape(-1)
    source = tmp_path / "src.wav"
    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(stereo.tobytes())

    def no_ffmpeg(*args: object, **kwargs: object) -> None:
        raise AssertionError("ffmpeg must not be spawned for PCM WAV slices")

    monkeypatch.setattr(ffmpeg, "_run_ffmpeg", no_ffmpeg)
    target = tmp_path / "out.wav"
    ffmpeg.run_ffmpeg_slice(
        ffmpeg="ffmpeg", source=source, target=target, start_sec=0.25, end_sec=0.75
    )

    with wave.open(str(target), "rb") as handle:
        assert handle.getnchannels() == 2
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == rate
        out = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)

    start_frame = round(0.25 * rate)
    count = round(0.5 * rate)
    expected = stereo[start_frame * 2 : (start_frame + count) * 2]
    assert np.array_equal(out, expected)


def _multi_track_source(path: Path, *, duration: float = 6.0) -> Path:
    """A file shaped like a delivery master: separate tones on separate tracks.

    ffmpeg picks one audio stream by default, so a master that carries dialogue
    on a discrete track can be sampled without ever hearing speech. The two
    tones stand in for that, one per track.
    """
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=48000:duration={duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=1760:sample_rate=48000:duration={duration}",
            "-map",
            "0:a",
            "-map",
            "1:a",
            "-c:a",
            "pcm_s16le",
            str(path),
        ],
        check=True,
    )
    return path


def _tones(path: Path, *, sample_rate: int) -> set[int]:
    """The dominant frequencies present, rounded to the nearest 10 Hz."""
    with wave.open(str(path), "rb") as handle:
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    spectrum = np.abs(np.fft.rfft(samples.astype(np.float64)))
    frequencies = np.fft.rfftfreq(samples.size, 1 / sample_rate)
    peaks = frequencies[spectrum > spectrum.max() * 0.25]
    return {int(round(peak / 10.0) * 10) for peak in peaks}


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_extract_audio_window_merges_every_track_when_asked(tmp_path: Path) -> None:
    source = _multi_track_source(tmp_path / "master.mkv")

    one = tmp_path / "one.wav"
    ffmpeg.extract_audio_window(
        source, one, start_sec=1.0, duration_sec=2.0, sample_rate=16_000, audio_streams=1
    )
    both = tmp_path / "both.wav"
    ffmpeg.extract_audio_window(
        source, both, start_sec=1.0, duration_sec=2.0, sample_rate=16_000, audio_streams=2
    )

    assert _tones(one, sample_rate=16_000) == {440}, "default pick hears one track only"
    assert _tones(both, sample_rate=16_000) == {440, 1760}


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_extract_audio_window_seeks_and_writes_mono_at_the_asked_rate(tmp_path: Path) -> None:
    source = _multi_track_source(tmp_path / "master.mkv")

    target = tmp_path / "window.wav"
    ffmpeg.extract_audio_window(source, target, start_sec=4.0, duration_sec=2.0, sample_rate=16_000)

    with wave.open(str(target), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == 16_000
        assert handle.getnframes() / 16_000 == pytest.approx(2.0, abs=0.05)


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is required")
def test_probe_media_duration_reads_a_real_file(tmp_path: Path) -> None:
    source = _multi_track_source(tmp_path / "master.mkv", duration=6.0)

    assert ffmpeg.probe_media_duration_sec(source) == pytest.approx(6.0, abs=0.1)


def test_probe_media_duration_reports_a_missing_binary(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        ffmpeg.probe_media_duration_sec(
            tmp_path / "missing.mov", ffprobe="unrender-nonexistent-ffprobe"
        )


def test_native_wav_slice_clamps_to_file_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rate = 16_000
    source = tmp_path / "src.wav"
    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(np.ones(rate, dtype=np.int16).tobytes())

    monkeypatch.setattr(
        ffmpeg, "_run_ffmpeg", lambda *a, **k: (_ for _ in ()).throw(AssertionError)
    )
    target = tmp_path / "out.wav"
    ffmpeg.run_ffmpeg_slice(
        ffmpeg="ffmpeg", source=source, target=target, start_sec=0.5, end_sec=5.0
    )

    with wave.open(str(target), "rb") as handle:
        assert handle.getnframes() == rate // 2
