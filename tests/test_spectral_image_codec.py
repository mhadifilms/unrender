from __future__ import annotations

import numpy as np
import pytest

from unrender.audio.spectral import (
    StftParams,
    decode_image_to_audio,
    encode_audio_to_image,
    read_spectral_image,
    roundtrip_metrics,
    write_preview_png,
    write_spectral_image,
)


def _tone(freq: float, seconds: float, rate: int, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def test_sine_roundtrip_high_snr():
    rate = 22050
    audio = _tone(440.0, 0.5, rate) + _tone(1320.0, 0.5, rate, amp=0.2)
    image = encode_audio_to_image(audio, rate, params=StftParams(phase_bits=16))
    recon, out_rate = decode_image_to_audio(image)
    assert out_rate == rate
    assert recon.shape[0] == audio.shape[0]
    metrics = roundtrip_metrics(audio, recon)
    assert metrics["snr_db"] > 80.0
    assert metrics["correlation"] > 0.9999


def test_length_is_preserved_exactly():
    rate = 16000
    for n in (1, 100, 1024, 4096, 5000):
        audio = _tone(300.0, n / rate, rate)[:n]
        image = encode_audio_to_image(audio, rate)
        recon, _ = decode_image_to_audio(image)
        assert recon.shape[0] == n


def test_16bit_phase_beats_8bit_phase():
    rate = 22050
    rng = np.random.default_rng(0)
    audio = (0.3 * rng.standard_normal(rate)).astype(np.float32)
    hi = encode_audio_to_image(audio, rate, params=StftParams(phase_bits=16))
    lo = encode_audio_to_image(audio, rate, params=StftParams(phase_bits=8))
    snr_hi = roundtrip_metrics(audio, decode_image_to_audio(hi)[0])["snr_db"]
    snr_lo = roundtrip_metrics(audio, decode_image_to_audio(lo)[0])["snr_db"]
    assert snr_hi > snr_lo


def test_stereo_roundtrip_and_shape():
    rate = 22050
    left = _tone(220.0, 0.4, rate)
    right = _tone(660.0, 0.4, rate, amp=0.3)
    audio = np.stack([left, right], axis=1)
    image = encode_audio_to_image(audio, rate)
    assert image.channels == 2
    assert image.pixels.shape[0] == 2 * image.params.bins
    recon, _ = decode_image_to_audio(image)
    assert recon.shape[1] == 2
    assert roundtrip_metrics(audio, recon)["snr_db"] > 75.0


def test_absolute_level_preserved_no_normalization():
    rate = 16000
    quiet = _tone(440.0, 0.3, rate, amp=0.02)
    image = encode_audio_to_image(quiet, rate)
    recon, _ = decode_image_to_audio(image)
    assert np.max(np.abs(recon)) < 0.05  # not blown up to full scale


def test_png_write_read_roundtrip(tmp_path):
    rate = 22050
    audio = _tone(500.0, 0.3, rate)
    image = encode_audio_to_image(audio, rate)
    png = tmp_path / "clip.spectral.png"
    write_spectral_image(png, image)
    loaded = read_spectral_image(png)
    assert loaded.sample_rate == rate
    assert loaded.length == audio.shape[0]
    assert loaded.params.n_fft == image.params.n_fft
    recon, _ = decode_image_to_audio(loaded)
    assert roundtrip_metrics(audio, recon)["snr_db"] > 80.0


def test_preview_png_is_grayscale(tmp_path):
    from PIL import Image

    rate = 22050
    audio = _tone(500.0, 0.3, rate)
    image = encode_audio_to_image(audio, rate)
    preview = write_preview_png(tmp_path / "preview.png", image)
    with Image.open(preview) as handle:
        assert handle.mode == "L"


def test_lossless_mode_is_transparent():
    rate = 22050
    rng = np.random.default_rng(1)
    audio = (0.4 * rng.standard_normal((rate, 2))).astype(np.float32)
    image = encode_audio_to_image(audio, rate, params=StftParams(mode="lossless"))
    recon, _ = decode_image_to_audio(image)
    assert recon.shape == audio.shape
    metrics = roundtrip_metrics(audio, recon)
    # float32 coefficient storage -> reconstruction limited only by float round-off
    assert metrics["snr_db"] > 120.0
    assert metrics["max_abs_error"] < 1e-4


def test_lossless_beats_visual_snr():
    rate = 22050
    rng = np.random.default_rng(2)
    audio = (0.3 * rng.standard_normal(rate)).astype(np.float32)
    visual = decode_image_to_audio(encode_audio_to_image(audio, rate))[0]
    lossless = decode_image_to_audio(
        encode_audio_to_image(audio, rate, params=StftParams(mode="lossless"))
    )[0]
    assert roundtrip_metrics(audio, lossless)["snr_db"] > roundtrip_metrics(audio, visual)["snr_db"]


def test_lossless_png_roundtrip(tmp_path):
    rate = 22050
    audio = _tone(500.0, 0.3, rate)
    image = encode_audio_to_image(audio, rate, params=StftParams(mode="lossless"))
    png = tmp_path / "clip.spectral.png"
    write_spectral_image(png, image)
    loaded = read_spectral_image(png)
    assert loaded.params.mode == "lossless"
    recon, _ = decode_image_to_audio(loaded)
    assert roundtrip_metrics(audio, recon)["snr_db"] > 120.0


def test_write_wav_float32_is_ieee_float(tmp_path):
    from scipy.io import wavfile

    from unrender.lib.audio import write_wav_float32

    rate = 22050
    audio = _tone(440.0, 0.2, rate).reshape(-1, 1)
    out = tmp_path / "out.wav"
    write_wav_float32(out, audio, rate)
    read_rate, data = wavfile.read(out)
    assert read_rate == rate
    assert data.dtype == np.float32


def test_read_rejects_png_without_metadata(tmp_path):
    from PIL import Image

    plain = tmp_path / "plain.png"
    Image.fromarray(np.zeros((8, 8, 4), dtype=np.uint8), mode="RGBA").save(plain)
    with pytest.raises(ValueError, match="metadata"):
        read_spectral_image(plain)


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        StftParams(n_fft=7).validate()
    with pytest.raises(ValueError):
        StftParams(hop=0).validate()
    with pytest.raises(ValueError):
        StftParams(phase_bits=12).validate()
