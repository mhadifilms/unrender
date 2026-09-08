from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.request import urlopen

import numpy as np

from unrender.audio.audio_io import (
    peak_normalize,
    read_audio,
    resample_to,
    to_mono,
    write_audio,
)

UpsamplerBackend = Literal[
    "resample",
    "flashsr",
    "audiosr",
    "resemble-enhance",
    "audioshake-denoise",
    "reuse",
]

# FlashSR: a 2 MB HierSpeech++-based 16 kHz -> 48 kHz speech upsampler exported to
# ONNX. It runs faster than realtime on CPU and restores high-frequency detail
# more conservatively (i.e. less "bright"/artefact-prone) than the diffusion
# AudioSR model, which makes it the default speaker-stem upsampler. The weight
# is pinned to a specific commit and checksum-verified after download.
FLASHSR_MODEL_URL = (
    "https://raw.githubusercontent.com/ysharma3501/FlashSR/"
    "2a69326250613c0a0f6c1c8d9f0c48cb779842b8/models/model.onnx"
)
FLASHSR_MODEL_SHA256 = "db8f28d1babc905ab8f69b290e950cf50164b8c6656f40a67a3b92ebe66ceb3d"
FLASHSR_MODEL_FILENAME = "flashsr_hierspeechpp_16k_48k.onnx"
FLASHSR_INPUT_RATE = 16_000
FLASHSR_OUTPUT_RATE = 48_000
# Socket timeout (seconds) so a stalled download cannot hang the pipeline.
DOWNLOAD_TIMEOUT_SEC = 120
REUSE_MODEL_ID = "nvidia/RE-USE"
REUSE_MODEL_REVISION = "761905064ea1ea882e015e20a64e2e9d28458890"


@dataclass(frozen=True)
class UpsamplerSettings:
    backend: UpsamplerBackend = "flashsr"
    target_sample_rate: int = 48_000
    device: str = "auto"
    model_name: str = "speech"
    ddim_steps: int = 50
    guidance_scale: float = 3.5
    seed: int | None = 42
    audiosr_python: str | None = None
    hf_home: str | None = None
    flashsr_model_path: Path | None = None
    resemble_solver: str = "midpoint"
    resemble_nfe: int = 64
    resemble_tau: float = 0.5
    resemble_denoise: bool = False
    reuse_python: str | None = None
    reuse_model_dir: Path | None = None
    reuse_checkpoint_path: Path | None = None
    reuse_config_path: Path | None = None


class AudioUpsampler:
    def __init__(self, settings: UpsamplerSettings) -> None:
        self.settings = settings
        self._audiosr_model = None
        self._flashsr_session = None

    def upsample(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        work_dir: Path,
    ) -> tuple[np.ndarray, int]:
        if self.settings.backend == "resample":
            return (
                resample_to(audio, sample_rate, self.settings.target_sample_rate),
                self.settings.target_sample_rate,
            )
        if self.settings.backend == "flashsr":
            enhanced, enhanced_sr = self._flashsr(audio, sample_rate)
        elif self.settings.backend == "audiosr":
            enhanced, enhanced_sr = self._audiosr(audio, sample_rate, work_dir=work_dir)
        elif self.settings.backend == "resemble-enhance":
            enhanced, enhanced_sr = self._resemble_enhance(audio, sample_rate)
        elif self.settings.backend == "audioshake-denoise":
            enhanced, enhanced_sr = self._audioshake_denoise(audio, sample_rate, work_dir=work_dir)
        elif self.settings.backend == "reuse":
            enhanced, enhanced_sr = self._reuse(audio, sample_rate, work_dir=work_dir)
        else:
            raise ValueError(f"unsupported upsampler backend: {self.settings.backend}")
        if enhanced_sr != self.settings.target_sample_rate:
            enhanced = resample_to(enhanced, enhanced_sr, self.settings.target_sample_rate)
            enhanced_sr = self.settings.target_sample_rate
        return peak_normalize(enhanced), enhanced_sr

    def _flashsr(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        session = self._get_flashsr_session()
        mono = to_mono(audio)
        if sample_rate != FLASHSR_INPUT_RATE:
            mono = to_mono(resample_to(mono, sample_rate, FLASHSR_INPUT_RATE))
        # The exported graph expects a (batch, channel, samples) float32 tensor.
        x = np.ascontiguousarray(mono.reshape(1, 1, -1), dtype=np.float32)
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: x})[0]
        return np.asarray(output, dtype=np.float32).reshape(-1, 1), FLASHSR_OUTPUT_RATE

    def _get_flashsr_session(self):
        if self._flashsr_session is not None:
            return self._flashsr_session
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "FlashSR upsampling requires onnxruntime: pip install '.[flashsr]'"
            ) from exc
        model_path = _ensure_flashsr_model(self.settings)
        self._flashsr_session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        return self._flashsr_session

    def _audiosr(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        work_dir: Path,
    ) -> tuple[np.ndarray, int]:
        external_python = self.settings.audiosr_python or os.environ.get("UNRENDER_AUDIOSR_PYTHON")
        if external_python:
            return self._audiosr_external(
                audio,
                sample_rate,
                work_dir=work_dir,
                python=external_python,
            )
        try:
            import audiosr.pipeline as audiosr_pipeline
            from audiosr import build_model, super_resolution
        except ImportError as exc:
            raise ImportError(
                "AudioSR upsampling requires `pip install audiosr==0.0.7` "
                "or --audiosr-python pointing at an AudioSR Python 3.10 environment"
            ) from exc

        work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            dir=work_dir,
            delete=False,
        ) as handle:
            input_path = Path(handle.name)
        try:
            audiosr_pipeline.lowpass_filtering_prepare_inference = _audiosr_lowpass_passthrough
            input_audio = resample_to(audio, sample_rate, self.settings.target_sample_rate)
            write_audio(
                input_path,
                to_mono(input_audio),
                self.settings.target_sample_rate,
                subtype="FLOAT",
            )
            if self._audiosr_model is None:
                self._audiosr_model = build_model(
                    model_name=self.settings.model_name,
                    device=self.settings.device,
                )
            waveform = super_resolution(
                self._audiosr_model,
                str(input_path),
                seed=self.settings.seed,
                guidance_scale=self.settings.guidance_scale,
                ddim_steps=self.settings.ddim_steps,
                latent_t_per_second=12.8,
            )
            arr = np.asarray(waveform, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[0]
            if arr.ndim == 2 and arr.shape[0] <= arr.shape[1]:
                arr = arr.T
            return arr.reshape(-1, 1), 48_000
        finally:
            input_path.unlink(missing_ok=True)

    def _audiosr_external(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        work_dir: Path,
        python: str,
    ) -> tuple[np.ndarray, int]:
        work_dir.mkdir(parents=True, exist_ok=True)
        input_path = work_dir / "audiosr_input.wav"
        output_path = work_dir / "audiosr_output.wav"
        input_audio = resample_to(audio, sample_rate, self.settings.target_sample_rate)
        write_audio(
            input_path,
            to_mono(input_audio),
            self.settings.target_sample_rate,
            subtype="FLOAT",
        )
        script = f"""
from pathlib import Path
import numpy as np
import soundfile as sf
import audiosr.pipeline as audiosr_pipeline
from audiosr import build_model, super_resolution

audiosr_pipeline.lowpass_filtering_prepare_inference = (
    lambda batch: {{'waveform_lowpass': batch['waveform']}}
)
model = build_model(model_name={self.settings.model_name!r}, device={self.settings.device!r})
waveform = super_resolution(
    model,
    {str(input_path)!r},
    seed={self.settings.seed!r},
    guidance_scale={self.settings.guidance_scale!r},
    ddim_steps={self.settings.ddim_steps!r},
    latent_t_per_second=12.8,
)
arr = np.asarray(waveform, dtype=np.float32)
if arr.ndim == 3:
    arr = arr[0]
if arr.ndim == 2 and arr.shape[0] <= arr.shape[1]:
    arr = arr.T
sf.write({str(output_path)!r}, arr.reshape(-1), 48000)
"""
        env = os.environ.copy()
        if self.settings.hf_home:
            env["HF_HOME"] = self.settings.hf_home
        subprocess.run([python, "-c", script], check=True, env=env)
        return read_audio(output_path)

    def _resemble_enhance(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        try:
            import torch
            from resemble_enhance.enhancer.inference import denoise, enhance
        except ImportError as exc:
            raise ImportError(
                "Resemble Enhance upsampling requires `pip install resemble-enhance`"
            ) from exc

        mono = to_mono(audio)
        device = self.settings.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        wav = torch.from_numpy(mono.astype("float32"))
        if self.settings.resemble_denoise:
            wav, sample_rate = denoise(wav, sample_rate, device)
        enhanced, new_sr = enhance(
            wav,
            sample_rate,
            device,
            nfe=self.settings.resemble_nfe,
            solver=self.settings.resemble_solver.lower(),
            lambd=0.9 if self.settings.resemble_denoise else 0.1,
            tau=self.settings.resemble_tau,
        )
        if hasattr(enhanced, "detach"):
            enhanced = enhanced.detach().cpu().numpy()
        return np.asarray(enhanced, dtype=np.float32).reshape(-1, 1), int(new_sr)

    def _audioshake_denoise(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        work_dir: Path,
    ) -> tuple[np.ndarray, int]:
        # speech_denoise cleans (hum/hiss/crowd/wind) but does not upsample, so
        # the caller's resample tail restores the target rate afterwards.
        from unrender.audio.separation.audioshake import speech_denoise_file

        work_dir.mkdir(parents=True, exist_ok=True)
        input_path = work_dir / "speech_denoise_input.wav"
        write_audio(input_path, to_mono(audio), sample_rate, subtype="FLOAT")
        cleaned = speech_denoise_file(input_path, work_dir / "denoise", prefix="speech_denoise")
        denoised, denoised_sr = read_audio(cleaned)
        return np.asarray(denoised, dtype=np.float32).reshape(-1, 1), int(denoised_sr)

    def _reuse(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        work_dir: Path,
    ) -> tuple[np.ndarray, int]:
        work_dir = Path(work_dir).expanduser().resolve()
        model_dir = _ensure_reuse_model(self.settings)
        inference_script = model_dir / "inference.py"
        if not inference_script.exists():
            raise FileNotFoundError(f"RE-USE inference.py not found under {model_dir}")
        checkpoint, config = _reuse_model_files(model_dir, self.settings)
        input_dir = work_dir / "reuse_input"
        output_dir = work_dir / "reuse_output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_path = input_dir / "source.wav"
        write_audio(input_path, to_mono(audio)[:, None], sample_rate, subtype="FLOAT")
        command = [
            self.settings.reuse_python or os.environ.get("UNRENDER_REUSE_PYTHON") or sys.executable,
            "-c",
            _reuse_runner_script(
                inference_script=inference_script,
                input_dir=input_dir,
                output_dir=output_dir,
                checkpoint=checkpoint,
                config=config,
                target_sample_rate=self.settings.target_sample_rate,
            ),
        ]
        subprocess.run(command, check=True, cwd=model_dir)
        outputs = sorted([*output_dir.rglob("*.wav"), *output_dir.rglob("*.flac")])
        if not outputs:
            raise RuntimeError(f"RE-USE produced no audio output in {output_dir}")
        return read_audio(outputs[0])


def _audiosr_lowpass_passthrough(batch):
    return {"waveform_lowpass": batch["waveform"]}


def _flashsr_cache_dir() -> Path:
    value = os.environ.get("UNRENDER_FLASHSR_CACHE_DIR")
    if value:
        return Path(value).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser() / "unrender" / "flashsr"
    if os.name == "posix" and Path.home().joinpath("Library").exists():
        return Path.home() / "Library" / "Caches" / "unrender" / "flashsr"
    return Path.home() / ".cache" / "unrender" / "flashsr"


def _ensure_flashsr_model(settings: UpsamplerSettings) -> Path:
    if settings.flashsr_model_path is not None:
        override = Path(settings.flashsr_model_path).expanduser()
        if not override.exists():
            raise FileNotFoundError(f"FlashSR model not found: {override}")
        return override
    model_path = _flashsr_cache_dir() / FLASHSR_MODEL_FILENAME
    if model_path.exists():
        _verify_sha256(model_path, FLASHSR_MODEL_SHA256)
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_suffix(model_path.suffix + ".part")
    print(f"  Downloading FlashSR model to {model_path}...", flush=True)
    _download_file(FLASHSR_MODEL_URL, temporary)
    _verify_sha256(temporary, FLASHSR_MODEL_SHA256)
    temporary.replace(model_path)
    return model_path


def _ensure_reuse_model(settings: UpsamplerSettings) -> Path:
    explicit = settings.reuse_model_dir or (
        Path(os.environ["UNRENDER_REUSE_MODEL_DIR"]).expanduser()
        if os.environ.get("UNRENDER_REUSE_MODEL_DIR")
        else None
    )
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"RE-USE model directory not found: {path}")
        return path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "RE-USE download requires huggingface-hub or UNRENDER_REUSE_MODEL_DIR"
        ) from exc
    return Path(
        snapshot_download(
            repo_id=REUSE_MODEL_ID,
            revision=REUSE_MODEL_REVISION,
        )
    )


def _reuse_model_files(
    model_dir: Path,
    settings: UpsamplerSettings,
) -> tuple[Path, Path]:
    checkpoint = (
        Path(settings.reuse_checkpoint_path).expanduser()
        if settings.reuse_checkpoint_path is not None
        else None
    )
    config = (
        Path(settings.reuse_config_path).expanduser()
        if settings.reuse_config_path is not None
        else None
    )
    if checkpoint is None:
        checkpoints = sorted(
            [*model_dir.rglob("*.pth"), *model_dir.rglob("*.safetensors")],
            key=lambda path: path.stat().st_size,
        )
        checkpoint = checkpoints[-1] if checkpoints else None
    if config is None:
        json_config = model_dir / "config.json"
        configs = sorted(model_dir.rglob("*.yaml"))
        config = (
            json_config
            if json_config.exists()
            else configs[0] if len(configs) == 1 else _matching_reuse_config(configs)
        )
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(f"RE-USE checkpoint not found under {model_dir}")
    if config is None or not config.exists():
        raise FileNotFoundError(f"RE-USE config not found under {model_dir}")
    return checkpoint, config


def _matching_reuse_config(configs: list[Path]) -> Path | None:
    preferred = [
        path for path in configs if "30x1" in path.name.lower() or "early" in path.name.lower()
    ]
    if len(preferred) == 1:
        return preferred[0]
    return configs[0] if configs else None


def _reuse_runner_script(
    *,
    inference_script: Path,
    input_dir: Path,
    output_dir: Path,
    checkpoint: Path,
    config: Path,
    target_sample_rate: int,
) -> str:
    """Run RE-USE with soundfile I/O so TorchCodec is not required on ARM64."""
    return f"""
import runpy
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, {str(inference_script.parent)!r})
for _package_name in ("models", "utils"):
    _package = types.ModuleType(_package_name)
    _package.__path__ = [str(Path({str(inference_script.parent)!r}) / _package_name)]
    sys.modules[_package_name] = _package

def _load(path):
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(np.ascontiguousarray(audio.T)), int(sample_rate)

def _save(path, waveform, sample_rate):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    values = waveform.detach().cpu().numpy()
    sf.write(str(target), np.asarray(values, dtype=np.float32).T, int(sample_rate))

torchaudio.load = _load
torchaudio.save = _save
sys.argv = [
    {str(inference_script)!r},
    "--input_folder", {str(input_dir)!r},
    "--output_folder", {str(output_dir)!r},
    "--checkpoint_file", {str(checkpoint)!r},
    "--config", {str(config)!r},
    "--BWE", {str(target_sample_rate)!r},
]
runpy.run_path({str(inference_script)!r}, run_name="__main__")
"""


def _download_file(url: str, target: Path) -> None:
    with urlopen(url, timeout=DOWNLOAD_TIMEOUT_SEC) as response, target.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def _verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual.lower() != expected.lower():
        raise ValueError(f"FlashSR model checksum mismatch: expected {expected}, got {actual}")
