from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from unrender.audio.spectral import (
    StftParams,
    decode_image_to_audio,
    encode_audio_to_image,
    load_audio_file,
    read_spectral_image,
    roundtrip_metrics,
    write_preview_png,
    write_spectral_image,
)
from unrender.lib.audio import write_wav_float, write_wav_float32

# Output audio formats: name -> (is_float, pcm sample width in bytes).
_AUDIO_FORMATS: dict[str, tuple[bool, int]] = {
    "f32": (True, 4),
    "s16": (False, 2),
    "s24": (False, 3),
    "s32": (False, 4),
}


def _load_audio(source: Path, *, ffmpeg: str) -> tuple[np.ndarray, int]:
    return load_audio_file(source, ffmpeg=ffmpeg)


def _params_from_args(args: argparse.Namespace) -> StftParams:
    return StftParams(
        n_fft=args.n_fft,
        hop=args.hop or args.n_fft // 4,
        phase_bits=args.phase_bits,
        mode=args.mode,
    )


def _write_audio(path: Path, audio: np.ndarray, rate: int, fmt: str) -> None:
    is_float, width = _AUDIO_FORMATS[fmt]
    if is_float:
        write_wav_float32(path, audio, rate)
    else:
        write_wav_float(path, audio, rate, sample_width=width)


def _spectral_encode(args: argparse.Namespace) -> int:
    """Encode an audio file into an invertible STFT phase image (PNG)."""
    source = args.input.expanduser()
    if not source.exists():
        raise ValueError(f"input audio not found: {source}")
    out = (
        args.output.expanduser() if args.output is not None else source.with_suffix(".spectral.png")
    )

    audio, rate = _load_audio(source, ffmpeg=args.ffmpeg)
    params = _params_from_args(args)
    image = encode_audio_to_image(audio, rate, params=params)
    write_spectral_image(out, image)
    print(
        f"Encoded {source.name} -> {out} "
        f"({image.pixels.shape[1]}x{image.pixels.shape[0]} px, "
        f"{image.channels} ch, {rate} Hz, n_fft={params.n_fft}, "
        f"mode={params.mode}, phase={params.phase_bits}b)"
    )

    if args.preview is not None:
        preview = write_preview_png(args.preview.expanduser(), image)
        print(f"Wrote preview spectrogram -> {preview}")

    if args.verify:
        recon, _ = decode_image_to_audio(image)
        metrics = roundtrip_metrics(audio, recon)
        print(
            f"Round-trip: SNR {metrics['snr_db']:.1f} dB, "
            f"corr {metrics['correlation']:.6f}, "
            f"max abs err {metrics['max_abs_error']:.2e}"
        )
    return 0


def _spectral_decode(args: argparse.Namespace) -> int:
    """Decode a spectral phase image (PNG) back into a WAV file."""
    source = args.input.expanduser()
    if not source.exists():
        raise ValueError(f"input image not found: {source}")
    out = (
        args.output.expanduser()
        if args.output is not None
        else source.with_suffix("").with_suffix(".wav")
    )
    image = read_spectral_image(source)
    audio, rate = decode_image_to_audio(image)
    _write_audio(out, audio, rate, args.format)
    print(
        f"Decoded {source.name} -> {out} "
        f"({audio.shape[0]} samples, {audio.shape[1]} ch, {rate} Hz, {args.format})"
    )
    return 0


def _spectral_roundtrip(args: argparse.Namespace) -> int:
    """Encode then decode an audio file and report reconstruction fidelity."""
    source = args.input.expanduser()
    if not source.exists():
        raise ValueError(f"input audio not found: {source}")
    audio, rate = _load_audio(source, ffmpeg=args.ffmpeg)
    params = _params_from_args(args)
    image = encode_audio_to_image(audio, rate, params=params)
    recon, _ = decode_image_to_audio(image)
    metrics = roundtrip_metrics(audio, recon)

    if args.write_image is not None:
        write_spectral_image(args.write_image.expanduser(), image)
    if args.write_audio is not None:
        _write_audio(args.write_audio.expanduser(), recon, rate, args.format)
    if args.write_preview is not None:
        write_preview_png(args.write_preview.expanduser(), image)

    print(
        f"{source.name}: {rate} Hz, mode={params.mode}, n_fft={params.n_fft} "
        f"hop={params.hop} phase={params.phase_bits}b -> "
        f"SNR {metrics['snr_db']:.1f} dB, corr {metrics['correlation']:.6f}, "
        f"max abs err {metrics['max_abs_error']:.2e}"
    )
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-fft", type=int, default=1024, help="FFT window size (default: 1024)")
    parser.add_argument(
        "--hop", type=int, default=0, help="Hop size in samples (default: n_fft // 4)"
    )
    parser.add_argument(
        "--mode",
        choices=("visual", "lossless"),
        default="visual",
        help=(
            "visual: dB magnitude + quantized phase, ~89 dB, reads like a "
            "spectrogram (default); lossless: raw float32 coefficients, "
            "transparent to 32-bit float, image looks like noise"
        ),
    )
    parser.add_argument(
        "--phase-bits",
        type=int,
        choices=(8, 16),
        default=16,
        help="Phase precision for visual mode: 16 (uses alpha) or 8 (default: 16)",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg binary for non-WAV inputs")


def register(subparsers: argparse._SubParsersAction) -> None:
    spectral = subparsers.add_parser(
        "spectral",
        help="Encode audio as an invertible STFT phase image and back",
    )
    spectral_sub = spectral.add_subparsers(dest="spectral_command")

    encode = spectral_sub.add_parser("encode", help="Audio -> spectral phase image (PNG)")
    encode.add_argument("input", type=Path, help="Input audio (wav/mp3/flac/m4a/...)")
    encode.add_argument(
        "--output", "-o", type=Path, help="Output PNG (default: <input>.spectral.png)"
    )
    encode.add_argument("--preview", type=Path, help="Also write a legible dB spectrogram PNG")
    encode.add_argument(
        "--verify", action="store_true", help="Decode in memory and report round-trip fidelity"
    )
    _add_common(encode)
    encode.set_defaults(handler=_spectral_encode)

    decode = spectral_sub.add_parser("decode", help="Spectral phase image (PNG) -> WAV")
    decode.add_argument("input", type=Path, help="Input spectral PNG")
    decode.add_argument("--output", "-o", type=Path, help="Output WAV (default: <input>.wav)")
    decode.add_argument(
        "--format",
        choices=tuple(_AUDIO_FORMATS),
        default="f32",
        help="Output WAV format: f32 (32-bit float, default), s16, s24, s32",
    )
    decode.set_defaults(handler=_spectral_decode)

    roundtrip = spectral_sub.add_parser(
        "roundtrip", help="Encode+decode and report reconstruction fidelity"
    )
    roundtrip.add_argument("input", type=Path, help="Input audio")
    roundtrip.add_argument("--write-image", type=Path, help="Save the intermediate spectral PNG")
    roundtrip.add_argument("--write-audio", type=Path, help="Save the reconstructed WAV")
    roundtrip.add_argument("--write-preview", type=Path, help="Save the dB spectrogram preview PNG")
    roundtrip.add_argument(
        "--format",
        choices=tuple(_AUDIO_FORMATS),
        default="f32",
        help="Output WAV format for --write-audio (default: f32)",
    )
    _add_common(roundtrip)
    roundtrip.set_defaults(handler=_spectral_roundtrip)
