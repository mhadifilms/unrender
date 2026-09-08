from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from unrender.audio.asr import load_whisperx_model, transcribe_audio
from unrender.editorial.dialogue.transcription import SpeechIsland, detect_speech_islands
from unrender.lib.fingerprints import input_fingerprints, warn_if_inputs_changed
from unrender.manifests import read_json, write_json


def audit_stems(
    *,
    stems: dict[str, Path],
    output_json: Path,
    threshold_db: float = -45.0,
    max_allowed_speech_sec: float = 1.0,
    verify_asr: bool = True,
    asr_model: str = "small",
    asr_device: str = "cpu",
    asr_compute_type: str = "int8",
    language: str | None = "en",
    force: bool = False,
) -> dict[str, Any]:
    """Measure residual dialogue in FX/MX stems before trusting them in a mix.

    An RMS gate alone cannot tell speech from music, so flagged islands are
    verified with ASR: only islands that transcribe to words count as bleed.
    Falls back to RMS-only (method="rms") when whisperx is unavailable.
    """
    inputs = {"stems": list(stems.values())}
    if output_json.exists() and not force:
        data = read_json(output_json)
        warn_if_inputs_changed(data, inputs, output_json)
        print(f"Stem audit already exists, skipping: {output_json}", flush=True)
        return data

    transcriber = (
        _island_transcriber(
            asr_model=asr_model,
            asr_device=asr_device,
            asr_compute_type=asr_compute_type,
            language=language,
        )
        if verify_asr
        else None
    )
    method = "rms+asr" if transcriber is not None else "rms"
    if verify_asr and transcriber is None:
        print("  WARNING: whisperx unavailable; falling back to RMS-only audit", flush=True)

    rows = []
    total_speech = 0.0
    for role, path in stems.items():
        if not path.exists():
            raise FileNotFoundError(f"{role} stem not found: {path}")
        islands = detect_speech_islands(path, threshold_db=threshold_db)
        island_rows: list[dict[str, Any]] = []
        speech_sec = 0.0
        for island in islands:
            row: dict[str, Any] = {
                "start_sec": round(island.start_sec, 3),
                "end_sec": round(island.end_sec, 3),
            }
            if transcriber is not None:
                transcript = transcriber(path, island)
                is_speech = _plausible_speech(transcript, island.end_sec - island.start_sec)
                row["transcript"] = transcript
                row["is_speech"] = is_speech
                if not is_speech:
                    island_rows.append(row)
                    continue
            speech_sec += island.end_sec - island.start_sec
            island_rows.append(row)
        total_speech += speech_sec
        rows.append(
            {
                "role": role,
                "path": str(path),
                "speech_sec": round(speech_sec, 3),
                "flagged_islands": len(islands),
                "islands": island_rows,
            }
        )
        print(
            f"  {role}: {speech_sec:.2f}s {method} speech bleed "
            f"({len(islands)} island(s) checked)"
        )

    status = "pass" if total_speech <= max_allowed_speech_sec else "fail"
    data = {
        "version": "1.1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": method,
        "threshold_db": threshold_db,
        "max_allowed_speech_sec": max_allowed_speech_sec,
        "total_speech_sec": round(total_speech, 3),
        "status": status,
        "inputs": input_fingerprints(inputs),
        "stems": rows,
    }
    write_json(output_json, data)
    print(f"Stem audit saved: {output_json} ({status})", flush=True)
    if status == "fail":
        print(
            "  WARNING: M&E stems appear to contain residual speech; re-separate before final mix."
        )
    return data


# Whisper reliably hallucinates these on music and silence.
_HALLUCINATED_TRANSCRIPTS = {
    "you",
    "thank you",
    "thank you so much",
    "thanks for watching",
    "bye",
    "yeah",
    "the end",
    "so",
    "oh",
    "mm",
    "hmm",
}
_MIN_SPEECH_ISLAND_SEC = 0.3
_MIN_SPEECH_WORDS = 2
# Real dialogue lands around 10-20 chars/sec; far outside that means the
# transcript cannot belong to the island it was decoded from.
_SPEECH_DENSITY_RANGE = (2.0, 30.0)


def _plausible_speech(transcript: str, duration_sec: float) -> bool:
    """Reject ASR hallucinations: transcripts that cannot be real speech.

    Whisper produces fragments like "you" / "Thank you." on music and noise,
    often on islands far too short to contain them.
    """
    text = transcript.strip().strip(".!?,").lower()
    if not text:
        return False
    if text in _HALLUCINATED_TRANSCRIPTS:
        return False
    if duration_sec < _MIN_SPEECH_ISLAND_SEC:
        return False
    if len(text.split()) < _MIN_SPEECH_WORDS:
        return False
    density = len(text) / duration_sec
    low, high = _SPEECH_DENSITY_RANGE
    return low <= density <= high


def _island_transcriber(
    *,
    asr_model: str,
    asr_device: str,
    asr_compute_type: str,
    language: str | None,
):
    """Build a per-island transcriber, or None when whisperx is unavailable.

    Loads each stem's audio once (16 kHz mono via whisperx) and slices it per
    island rather than re-decoding the file for every island.
    """
    try:
        import whisperx
    except ImportError:
        return None
    model = load_whisperx_model(
        model_name=asr_model,
        device=asr_device,
        compute_type=asr_compute_type,
        language=language,
    )
    if model is None:
        return None
    audio_cache: dict[str, Any] = {}
    sample_rate = 16_000

    def transcribe(path: Path, island: SpeechIsland) -> str:
        key = str(path)
        if key not in audio_cache:
            audio_cache.clear()  # keep at most one stem in memory
            audio_cache[key] = whisperx.load_audio(str(path))
        audio = audio_cache[key]
        start = max(0, int(island.start_sec * sample_rate))
        end = min(audio.shape[0], int(island.end_sec * sample_rate))
        if end <= start:
            return ""
        return transcribe_audio(
            audio[start:end],
            model_name=asr_model,
            device=asr_device,
            compute_type=asr_compute_type,
            language=language,
            batch_size=8,
        ).strip()

    return transcribe
