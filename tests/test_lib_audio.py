"""IEEE-float WAV support in unrender.lib.audio (BandIt float32 stems)."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from conftest import write_wav
from unrender.lib.audio import (
    WavReader,
    measure_activity,
    measure_peak_dbfs,
    mix_wav_files,
    probe_duration_sec,
    probe_rms_db,
    read_audio_any,
    read_wav_float,
)

_KSDATAFORMAT_SUBTYPE_IEEE_FLOAT = (
    b"\x03\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
)


def _write_float_wav(
    path: Path,
    data: np.ndarray,
    *,
    sample_rate: int = 16_000,
    bits: int = 32,
    extensible: bool = False,
) -> None:
    """Minimal RIFF writer for IEEE-float WAVs (format tag 3 / extensible)."""
    array = np.asarray(data, dtype=np.float32 if bits == 32 else np.float64)
    if array.ndim == 1:
        array = array[:, None]
    channels = array.shape[1]
    payload = array.astype("<f4" if bits == 32 else "<f8").tobytes()
    block_align = channels * bits // 8
    byte_rate = sample_rate * block_align
    if extensible:
        fmt = struct.pack(
            "<HHIIHHHHI",
            0xFFFE,
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits,
            22,
            bits,
            0,
        )
        fmt += _KSDATAFORMAT_SUBTYPE_IEEE_FLOAT
    else:
        fmt = struct.pack("<HHIIHH", 3, channels, sample_rate, byte_rate, block_align, bits)
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(payload)) + payload
    path.write_bytes(b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body)


def test_stdlib_wave_rejects_float_wav(tmp_path: Path) -> None:
    # Guard: this suite exists because wave.open cannot read format tag 3.
    path = tmp_path / "float.wav"
    _write_float_wav(path, np.zeros(8, dtype=np.float32))
    with pytest.raises(wave.Error, match="unknown format: 3"), wave.open(str(path), "rb"):
        pass


def test_probe_duration_sec_float32(tmp_path: Path) -> None:
    path = tmp_path / "float.wav"
    _write_float_wav(path, np.zeros(24_000, dtype=np.float32), sample_rate=16_000)
    assert probe_duration_sec(path) == pytest.approx(1.5)


def test_wav_reader_metadata_and_streaming_float32(tmp_path: Path) -> None:
    path = tmp_path / "float.wav"
    samples = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    _write_float_wav(path, np.stack([samples, -samples], axis=1), sample_rate=48_000)
    with WavReader(path) as reader:
        assert reader.is_float
        assert reader.channels == 2
        assert reader.sample_rate == 48_000
        assert reader.frames == 1000
        assert reader.sample_width == 4
        assert reader.pcm_sample_width == 4
        chunks = []
        while True:
            block = reader.read(300)
            if not block.size:
                break
            chunks.append(block)
    interleaved = np.concatenate(chunks)
    assert interleaved.size == 2000
    np.testing.assert_allclose(interleaved.reshape(-1, 2)[:, 0], samples, atol=0.0)


def test_measurements_match_between_float_and_int16(tmp_path: Path) -> None:
    rate = 16_000
    tone = 0.25 * np.sin(2 * np.pi * 440 * np.arange(rate, dtype=np.float32) / rate)
    signal = np.concatenate([tone, np.zeros(rate, dtype=np.float32)])
    float_path = tmp_path / "float.wav"
    int_path = tmp_path / "int16.wav"
    _write_float_wav(float_path, signal, sample_rate=rate)
    write_wav(int_path, (signal * 32767.0).astype(np.int16), sample_rate=rate)

    assert measure_peak_dbfs(float_path) == pytest.approx(20 * np.log10(0.25), abs=1e-3)
    assert measure_peak_dbfs(float_path) == pytest.approx(measure_peak_dbfs(int_path), abs=0.01)
    assert probe_rms_db(float_path) == pytest.approx(probe_rms_db(int_path), abs=0.01)

    float_activity = measure_activity(float_path)
    int_activity = measure_activity(int_path)
    assert float_activity.voiced_ratio == pytest.approx(0.5, abs=0.02)
    assert float_activity.voiced_ratio == pytest.approx(int_activity.voiced_ratio, abs=0.01)
    assert float_activity.rms_dbfs == pytest.approx(int_activity.rms_dbfs, abs=0.01)


def test_read_wav_float_reads_float32_and_float64(tmp_path: Path) -> None:
    samples = np.asarray([[0.5, -0.5], [0.125, 0.75]], dtype=np.float32)
    f32 = tmp_path / "f32.wav"
    f64 = tmp_path / "f64.wav"
    _write_float_wav(f32, samples, sample_rate=22_050, bits=32)
    _write_float_wav(f64, samples.astype(np.float64), sample_rate=22_050, bits=64)

    data32, rate32, width32 = read_wav_float(f32)
    assert (rate32, width32) == (22_050, 4)
    np.testing.assert_allclose(data32, samples, atol=0.0)

    data64, rate64, width64 = read_wav_float(f64)
    assert (rate64, width64) == (22_050, 4)
    np.testing.assert_allclose(data64, samples, atol=1e-7)


def test_read_wav_float_reads_extensible_float(tmp_path: Path) -> None:
    samples = np.asarray([0.25, -0.25, 0.0], dtype=np.float32)
    path = tmp_path / "extensible.wav"
    _write_float_wav(path, samples, extensible=True)
    data, rate, width = read_wav_float(path)
    assert (rate, width) == (16_000, 4)
    np.testing.assert_allclose(data[:, 0], samples, atol=0.0)


def test_read_audio_any_reads_float_wav_natively(tmp_path: Path) -> None:
    samples = np.asarray([0.5, -0.5, 0.25], dtype=np.float32)
    path = tmp_path / "float.wav"
    _write_float_wav(path, samples, sample_rate=16_000)
    data, rate = read_audio_any(path)
    assert rate == 16_000
    np.testing.assert_allclose(data[:, 0], samples, atol=0.0)


def test_mix_wav_files_sums_float32_sources(tmp_path: Path) -> None:
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    _write_float_wav(a, np.asarray([0.25, -0.25, 0.9], dtype=np.float32))
    _write_float_wav(b, np.asarray([0.25, -0.5, 0.9], dtype=np.float32))

    target = mix_wav_files([a, b], tmp_path / "mix.wav")

    with wave.open(str(target), "rb") as handle:
        assert handle.getsampwidth() == 4  # float sources land as int32 PCM
        raw = handle.readframes(handle.getnframes())
    mixed = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    np.testing.assert_allclose(mixed[:2], [0.5, -0.75], atol=1e-9)
    assert mixed[2] == pytest.approx(1.0, abs=1e-6)  # clipped at full scale


def test_mix_wav_files_pads_shorter_float_source(tmp_path: Path) -> None:
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    _write_float_wav(a, np.full(6, 0.5, dtype=np.float32))
    _write_float_wav(b, np.full(3, 0.25, dtype=np.float32))

    target = mix_wav_files([a, b], tmp_path / "mix.wav")

    with wave.open(str(target), "rb") as handle:
        raw = handle.readframes(handle.getnframes())
    mixed = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    np.testing.assert_allclose(mixed, [0.75] * 3 + [0.5] * 3, atol=1e-9)


def test_mix_wav_files_rejects_float_and_int16_mix(tmp_path: Path) -> None:
    a = tmp_path / "float.wav"
    b = tmp_path / "int16.wav"
    _write_float_wav(a, np.zeros(10, dtype=np.float32))
    write_wav(b, np.zeros(10, dtype=np.int16))
    with pytest.raises(ValueError, match="disagree"):
        mix_wav_files([a, b], tmp_path / "mix.wav")


def test_wav_reader_raises_wave_error_for_malformed_files(tmp_path: Path) -> None:
    not_riff = tmp_path / "junk.wav"
    not_riff.write_bytes(b"this is not a wav file at all")
    with pytest.raises(wave.Error):
        WavReader(not_riff)

    no_data = tmp_path / "nodata.wav"
    fmt = struct.pack("<HHIIHH", 3, 1, 16_000, 64_000, 4, 32)
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    no_data.write_bytes(b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body)
    with pytest.raises(wave.Error, match="no data chunk"):
        WavReader(no_data)


def test_wav_reader_clamps_overstated_data_size(tmp_path: Path) -> None:
    path = tmp_path / "truncated.wav"
    _write_float_wav(path, np.full(100, 0.5, dtype=np.float32))
    raw = bytearray(path.read_bytes())
    data_index = raw.index(b"data")
    struct.pack_into("<I", raw, data_index + 4, 0xFFFFFFFF)  # streamed placeholder
    path.write_bytes(bytes(raw))

    with WavReader(path) as reader:
        assert reader.frames == 100
        values = reader.read(1000)
    np.testing.assert_allclose(values, np.full(100, 0.5, dtype=np.float32), atol=0.0)
