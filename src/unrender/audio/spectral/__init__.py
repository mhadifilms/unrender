"""Invertible audio <-> image (STFT phase-image) codec.

Represents audio as a picture: a short-time Fourier transform is stored in an
RGBA PNG with per-bin magnitude packed into the R/G channels and phase into
B (and, in high-fidelity mode, A). Because phase is preserved, decoding the
image reconstructs the waveform via a windowed overlap-add inverse STFT, so
the round-trip is near-lossless rather than the phase-discarding spectrogram
used elsewhere in :mod:`unrender.audio.mne`.

Inspired by the ``meyda`` handler in https://github.com/p2r3/convert, but
reworked for fidelity: proper Hann-windowed analysis, WOLA reconstruction,
a stored magnitude scale (no output normalization that destroys absolute
level), all FFT bins retained, optional 16-bit phase, and self-describing
metadata embedded in the PNG.
"""

from __future__ import annotations

from unrender.audio.spectral.audio_io import load_audio_file
from unrender.audio.spectral.image_codec import (
    SpectralAxes,
    SpectralImage,
    StftParams,
    decode_image_to_audio,
    encode_audio_to_image,
    preview_array,
    read_spectral_image,
    roundtrip_metrics,
    write_preview_png,
    write_spectral_image,
)

__all__ = [
    "SpectralAxes",
    "SpectralImage",
    "StftParams",
    "decode_image_to_audio",
    "encode_audio_to_image",
    "load_audio_file",
    "preview_array",
    "read_spectral_image",
    "roundtrip_metrics",
    "write_preview_png",
    "write_spectral_image",
]
