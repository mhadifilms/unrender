from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unrender.audio.audio_io import fit_length, resample_to
from unrender.audio.restoration.reconstruction import (
    ReferenceReconstructionSettings,
    ReferenceSourceReconstructor,
    _gate_from_raw_activity,
    project_lowband_guided_spectrum,
    project_original_spectrum,
)
from unrender.audio.restoration.upsamplers import (
    AudioUpsampler,
    UpsamplerSettings,
    _ensure_flashsr_model,
    _reuse_model_files,
    _reuse_runner_script,
)


def test_resample_and_fit_length_helpers() -> None:
    audio = np.arange(8, dtype=np.float32)[:, None]
    resampled = resample_to(audio, 8, 16)

    assert resampled.shape[0] == 16
    assert fit_length(audio, 4).shape == (4, 1)
    assert fit_length(audio, 10).shape == (10, 1)


class _FakeFlashSRSession:
    """Stand-in ONNX session that upsamples 16k -> 48k by tripling length."""

    class _Input:
        name = "x"

    def get_inputs(self):
        return [self._Input()]

    def run(self, _outputs, feed):
        x = next(iter(feed.values()))
        return [np.full((1, 1, x.shape[-1] * 3), 0.5, dtype=np.float32)]


def test_flashsr_upsampler_runs_onnx_session(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(AudioUpsampler, "_get_flashsr_session", lambda self: _FakeFlashSRSession())
    upsampler = AudioUpsampler(UpsamplerSettings(backend="flashsr", target_sample_rate=48_000))

    out, sr = upsampler.upsample(
        np.full((100, 1), 0.2, dtype=np.float32), 16_000, work_dir=tmp_path
    )

    assert sr == 48_000
    assert out.shape[0] == 300  # 16k -> 48k tripling from the fake session
    assert np.allclose(out, 0.5)


def test_flashsr_off_switch_uses_plain_resample(monkeypatch, tmp_path: Path) -> None:
    def _fail(_self):
        raise AssertionError("resample backend must not load the FlashSR model")

    monkeypatch.setattr(AudioUpsampler, "_get_flashsr_session", _fail)
    upsampler = AudioUpsampler(UpsamplerSettings(backend="resample", target_sample_rate=48_000))

    out, sr = upsampler.upsample(
        np.full((100, 1), 0.2, dtype=np.float32), 16_000, work_dir=tmp_path
    )

    assert sr == 48_000
    assert out.shape[0] == 300


def test_audioshake_denoise_upsampler_calls_cloud_and_resamples(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import unrender.audio.separation.audioshake as audioshake_mod
    from unrender.audio.audio_io import read_audio, write_audio

    calls = {"count": 0}

    def fake_denoise(source, output_dir, **kwargs):
        calls["count"] += 1
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cleaned = out_dir / "clean_speech_denoise.wav"
        audio, sample_rate = read_audio(Path(source))
        write_audio(cleaned, audio * 0.5, sample_rate, subtype="FLOAT")
        return cleaned

    monkeypatch.setattr(audioshake_mod, "speech_denoise_file", fake_denoise)
    upsampler = AudioUpsampler(
        UpsamplerSettings(backend="audioshake-denoise", target_sample_rate=48_000)
    )

    out, sr = upsampler.upsample(
        np.full((16_000, 1), 0.2, dtype=np.float32), 16_000, work_dir=tmp_path
    )

    assert calls["count"] == 1
    assert sr == 48_000
    # speech_denoise cleans at 16k; the resample tail reaches the 48k target.
    assert out.shape[0] == 48_000


def test_ensure_flashsr_model_prefers_explicit_path(tmp_path: Path) -> None:
    model = tmp_path / "flashsr.onnx"
    model.write_bytes(b"onnx")

    resolved = _ensure_flashsr_model(UpsamplerSettings(flashsr_model_path=model))

    assert resolved == model


def test_original_reference_projection_is_stereo_and_mixture_consistent() -> None:
    sample_rate = 8_000
    time = np.arange(4_096, dtype=np.float32) / sample_rate
    left = 0.4 * np.sin(2 * np.pi * 220 * time)
    right = 0.3 * np.sin(2 * np.pi * 220 * time + 0.3)
    original = np.stack([left, right], axis=1)
    proposals = [
        np.sin(2 * np.pi * 220 * time)[:, None],
        np.sin(2 * np.pi * 880 * time)[:, None] * 0.01,
    ]

    sources, residual = project_original_spectrum(
        original,
        proposals,
        sample_rate,
        n_fft=256,
        hop_length=64,
        residual_floor=0.01,
    )

    assert len(sources) == 2
    assert all(source.shape == original.shape for source in sources)
    assert np.max(np.abs(sources[1])) < np.max(np.abs(sources[0]))
    reconstructed = sources[0] + sources[1] + residual
    assert np.allclose(reconstructed, original, atol=1e-6)


def test_original_reference_projection_preserves_separator_low_band() -> None:
    sample_rate = 8_000
    time = np.arange(8_192, dtype=np.float32) / sample_rate
    source_a = (0.2 * np.sin(2 * np.pi * 220 * time))[:, None]
    source_b = (0.15 * np.sin(2 * np.pi * 630 * time + 0.2))[:, None]
    original = source_a + source_b

    sources, residual = project_original_spectrum(
        original,
        [source_a, source_b],
        sample_rate,
        n_fft=256,
        hop_length=64,
        residual_floor=0.01,
        preserve_low_band_hz=3_800,
    )

    assert np.corrcoef(source_a[:, 0], sources[0][:, 0])[0, 1] > 0.999
    assert np.corrcoef(source_b[:, 0], sources[1][:, 0])[0, 1] > 0.999
    assert np.allclose(sources[0] + sources[1] + residual, original, atol=1e-6)


def test_lowband_guided_projection_assigns_harmonic_high_band() -> None:
    sample_rate = 48_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    source_a = (0.2 * np.sin(2 * np.pi * 1_000 * time) + 0.04 * np.sin(2 * np.pi * 7_000 * time))[
        :, None
    ]
    source_b = (0.2 * np.sin(2 * np.pi * 3_100 * time) + 0.04 * np.sin(2 * np.pi * 9_300 * time))[
        :, None
    ]
    original = source_a + source_b
    estimates = [
        resample_to(source_a, sample_rate, 8_000),
        resample_to(source_b, sample_rate, 8_000),
    ]

    sources, residual, metadata = project_lowband_guided_spectrum(
        original,
        estimates,
        8_000,
        sample_rate,
        n_fft=1024,
        hop_length=256,
    )

    spectra = [np.abs(np.fft.rfft(source[:, 0])) for source in sources]
    frequencies = np.fft.rfftfreq(len(original), 1.0 / sample_rate)
    index_7k = int(np.argmin(np.abs(frequencies - 7_000)))
    index_9k = int(np.argmin(np.abs(frequencies - 9_300)))
    assert spectra[0][index_7k] > spectra[1][index_7k] * 2.0
    assert spectra[1][index_9k] > spectra[0][index_9k] * 2.0
    assert metadata["lowband_cutoff_hz"] == 3_800.0
    assert np.allclose(sources[0] + sources[1] + residual, original, atol=1e-6)


def test_reference_reconstructor_uses_configured_proposal_rate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    seen = {}

    def fake_upsample(self, audio, sample_rate, *, work_dir):
        seen["target_rate"] = self.settings.target_sample_rate
        return np.repeat(audio, 3, axis=0), 48_000

    monkeypatch.setattr(AudioUpsampler, "upsample", fake_upsample)
    reconstructor = ReferenceSourceReconstructor(
        ReferenceReconstructionSettings(
            restorer="flashsr",
            target_sample_rate=48_000,
            n_fft=256,
            hop_length=64,
        )
    )
    original = np.ones((16_000, 1), dtype=np.float32) * 0.1

    sources, residual, _metadata = reconstructor.reconstruct(
        original,
        16_000,
        [original],
        16_000,
        work_dir=tmp_path,
    )

    assert seen["target_rate"] == 48_000
    assert sources[0].shape == original.shape
    assert np.allclose(sources[0] + residual, original, atol=1e-6)


def test_two_speaker_reconstruction_gates_restorer_pause_leakage(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_upsample(self, audio, sample_rate, *, work_dir):
        restored = np.repeat(audio, 3, axis=0) + 0.01
        return restored, 48_000

    monkeypatch.setattr(AudioUpsampler, "upsample", fake_upsample)
    target_a = np.concatenate(
        [
            np.full((8_000, 1), 0.1, dtype=np.float32),
            np.zeros((8_000, 1), dtype=np.float32),
        ]
    )
    target_b = np.concatenate(
        [
            np.zeros((8_000, 1), dtype=np.float32),
            np.full((8_000, 1), 0.1, dtype=np.float32),
        ]
    )
    original = np.repeat(target_a + target_b, 3, axis=0)
    reconstructor = ReferenceSourceReconstructor(
        ReferenceReconstructionSettings(
            restorer="flashsr",
            target_sample_rate=48_000,
        )
    )

    sources, residual, metadata = reconstructor.reconstruct(
        original,
        48_000,
        [target_a, target_b],
        16_000,
        work_dir=tmp_path,
    )

    assert metadata["backend"] == "isolated_source_gated"
    assert np.sqrt(np.mean(sources[0][30_000:45_000] ** 2)) < 1e-4
    assert np.sqrt(np.mean(sources[1][3_000:15_000] ** 2)) < 1e-4
    assert np.allclose(sources[0] + sources[1] + residual, original, atol=1e-6)


def test_two_speaker_reconstruction_keeps_low_scale_source_activity(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_upsample(self, audio, sample_rate, *, work_dir):
        del self, sample_rate, work_dir
        return np.repeat(audio, 3, axis=0) + 0.01, 48_000

    monkeypatch.setattr(AudioUpsampler, "upsample", fake_upsample)
    quiet_source = np.concatenate(
        [
            np.full((8_000, 1), 1e-4, dtype=np.float32),
            np.zeros((8_000, 1), dtype=np.float32),
        ]
    )
    loud_source = np.concatenate(
        [
            np.zeros((8_000, 1), dtype=np.float32),
            np.full((8_000, 1), 0.1, dtype=np.float32),
        ]
    )
    original = np.repeat(quiet_source + loud_source, 3, axis=0)
    reconstructor = ReferenceSourceReconstructor(
        ReferenceReconstructionSettings(
            restorer="flashsr",
            target_sample_rate=48_000,
        )
    )

    sources, residual, metadata = reconstructor.reconstruct(
        original,
        48_000,
        [quiet_source, loud_source],
        16_000,
        work_dir=tmp_path,
    )

    assert metadata["activity_gate"] == "source_relative_20db"
    assert np.sqrt(np.mean(sources[0][3_000:15_000] ** 2)) > 5e-5
    assert np.sqrt(np.mean(sources[0][30_000:45_000] ** 2)) < 1e-6
    assert np.allclose(sources[0] + sources[1] + residual, original, atol=1e-6)


def test_source_relative_gate_keeps_one_frame_quiet_burst() -> None:
    raw = np.zeros(8_000, dtype=np.float32)
    raw[:160] = 1e-4
    restored = np.full((48_000, 1), 0.01, dtype=np.float32)

    gated = _gate_from_raw_activity(raw, restored, 8_000, 48_000)

    assert np.sqrt(np.mean(gated[200:700] ** 2)) > 1e-3
    assert np.max(np.abs(gated[6_000:])) == 0.0


def test_two_source_reconstruction_prefers_reuse_in_auto_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    seen = {}

    def fake_upsample(self, audio, sample_rate, *, work_dir):
        seen["backend"] = self.settings.backend
        return np.repeat(audio, 6, axis=0), 48_000

    monkeypatch.setattr(AudioUpsampler, "upsample", fake_upsample)
    source_a = np.full((8_000, 1), 0.1, dtype=np.float32)
    source_b = np.full((8_000, 1), -0.1, dtype=np.float32)
    original = np.repeat(source_a + source_b, 6, axis=0)
    reconstructor = ReferenceSourceReconstructor(
        ReferenceReconstructionSettings(
            restorer="auto",
            target_sample_rate=48_000,
        )
    )

    sources, residual, metadata = reconstructor.reconstruct(
        original,
        48_000,
        [source_a, source_b],
        8_000,
        work_dir=tmp_path,
    )

    assert seen["backend"] == "reuse"
    assert metadata["backend"] == "isolated_source_gated"
    assert metadata["restorer"] == "reuse"
    assert np.allclose(sources[0] + sources[1] + residual, original, atol=1e-6)


def test_isolated_restoration_recovers_separator_program_level(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_upsample(self, audio, sample_rate, *, work_dir):
        del self, sample_rate, work_dir
        return np.repeat(audio * 20.0, 6, axis=0), 48_000

    monkeypatch.setattr(AudioUpsampler, "upsample", fake_upsample)
    source_a = np.full((8_000, 1), 0.02, dtype=np.float32)
    source_b = np.full((8_000, 1), -0.01, dtype=np.float32)
    original = np.repeat(source_a + source_b, 6, axis=0)
    reconstructor = ReferenceSourceReconstructor(
        ReferenceReconstructionSettings(restorer="reuse", target_sample_rate=48_000)
    )

    sources, residual, metadata = reconstructor.reconstruct(
        original,
        48_000,
        [source_a, source_b],
        8_000,
        work_dir=tmp_path,
    )

    assert metadata["source_level_reference"] == (
        "separator_relative_lowband_rms_then_original_mix_program_energy"
    )
    assert np.allclose(metadata["source_level_gains"], [0.05, 0.05], atol=2e-3)
    assert 0.0 < metadata["program_level_gain"] < 1.0
    source_energy = np.sqrt(sum(float(np.mean(source**2)) for source in sources))
    assert source_energy == pytest.approx(np.sqrt(np.mean(original**2)), abs=1e-3)
    assert max(float(np.max(np.abs(source))) for source in sources) < 0.02
    assert np.allclose(sources[0] + sources[1] + residual, original, atol=1e-6)


def test_isolated_restoration_does_not_mute_sources_for_mixture_transient(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_upsample(self, audio, sample_rate, *, work_dir):
        del self, sample_rate, work_dir
        return np.repeat(audio, 6, axis=0), 48_000

    monkeypatch.setattr(AudioUpsampler, "upsample", fake_upsample)
    source_a = np.full((8_000, 1), 0.02, dtype=np.float32)
    source_b = np.full((8_000, 1), 0.01, dtype=np.float32)
    original = np.repeat(source_a + source_b, 6, axis=0)
    original[24_000] = 0.99
    reconstructor = ReferenceSourceReconstructor(
        ReferenceReconstructionSettings(restorer="reuse", target_sample_rate=48_000)
    )

    sources, residual, metadata = reconstructor.reconstruct(
        original,
        48_000,
        [source_a, source_b],
        8_000,
        work_dir=tmp_path,
    )

    assert metadata["program_level_gain"] > 0.9
    assert np.sqrt(np.mean(sources[0] ** 2)) > 0.015
    assert np.sqrt(np.mean(sources[1] ** 2)) > 0.007
    assert np.allclose(sources[0] + sources[1] + residual, original, atol=1e-6)


def test_reuse_model_files_support_current_safetensors_snapshot(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    config = tmp_path / "config.json"
    checkpoint.write_bytes(b"weights")
    config.write_text("{}", encoding="utf-8")

    resolved_checkpoint, resolved_config = _reuse_model_files(
        tmp_path,
        UpsamplerSettings(backend="reuse"),
    )

    assert resolved_checkpoint == checkpoint
    assert resolved_config == config


def test_reuse_runner_patches_torchcodec_io_and_model_namespaces(tmp_path: Path) -> None:
    script = _reuse_runner_script(
        inference_script=tmp_path / "inference.py",
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        checkpoint=tmp_path / "model.safetensors",
        config=tmp_path / "config.json",
        target_sample_rate=48_000,
    )

    assert "torchaudio.load = _load" in script
    assert "torchaudio.save = _save" in script
    assert 'for _package_name in ("models", "utils")' in script
    assert "\"--BWE\", '48000'" in script
