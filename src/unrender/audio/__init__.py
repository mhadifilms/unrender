"""AudioShake separation, editable audio, and voice embeddings."""

from unrender.audio.embeddings import (
    PyannoteVoiceEmbedder,
    SpeechBrainEcapaEmbedder,
    VoiceEmbedder,
    build_voice_embedder,
)

__all__ = [
    "AudioStream",
    "NormalizedAudioInput",
    "PyannoteVoiceEmbedder",
    "SpeechBrainEcapaEmbedder",
    "VoiceEmbedder",
    "build_voice_embedder",
    "normalize_full_audio_input",
    "probe_audio_streams",
    "separate_full_audio_source",
    "separate_global_dx_stem",
]


def __getattr__(name: str):
    if name in {
        "AudioStream",
        "NormalizedAudioInput",
        "normalize_full_audio_input",
        "probe_audio_streams",
        "separate_full_audio_source",
        "separate_global_dx_stem",
    }:
        from unrender.audio import separation

        return getattr(separation, name)
    raise AttributeError(name)
