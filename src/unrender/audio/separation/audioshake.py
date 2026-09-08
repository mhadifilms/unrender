from __future__ import annotations

import math
import re
import wave
from pathlib import Path
from typing import Any

from unrender.audio.separation.audioshake_client import (
    DEFAULT_POLL_INTERVAL,
    DEFAULT_TIMEOUT,
    AudioShakeClient,
    AudioShakeError,
    AudioShakeNoSpeakersError,
)
from unrender.audio.separation.input_audio import AudioInputSource, normalize_full_audio_input
from unrender.audio.separation.models import (
    AudioSeparationResult,
    SourceRole,
    SourceSeparationResult,
    SpeakerVariant,
)
from unrender.lib.audio import probe_duration_sec
from unrender.lib.fingerprints import input_fingerprints, warn_if_inputs_changed
from unrender.lib.names import safe_name
from unrender.manifests import read_json, write_json
from unrender.project import RunPaths


def separate_global_dx_stem(
    *,
    run: RunPaths,
    dx_stem: str | Path,
    client: AudioShakeClient | None = None,
    variant: SpeakerVariant = "n_speaker",
    fmt: str = "wav",
    prefix: str,
    timeout: int = DEFAULT_TIMEOUT,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    force: bool = False,
    dry_run: bool = False,
) -> AudioSeparationResult:
    run.ensure()
    output_prefix = _audio_prefix(prefix)
    inputs = {
        "source": dx_stem,
        "variant": variant,
        "format": fmt,
    }
    existing = existing_unmapped_speaker_stems(run, prefix=output_prefix)
    if existing and not force:
        print(
            f"Audio speaker stems already exist: {len(existing)} file(s) "
            f"in {run.unmapped_speakers_dir}",
            flush=True,
        )
        _write_manifest(
            run,
            source=str(dx_stem),
            stems=existing,
            variant=variant,
            fmt=fmt,
            skipped=True,
            inputs=_carried_inputs(run.audio_separation_json, inputs),
        )
        return AudioSeparationResult(
            str(dx_stem), run.unmapped_speakers_dir, existing, skipped=True
        )

    if not force and run.audio_separation_json.exists():
        previous = read_json(run.audio_separation_json)
        if isinstance(previous, dict) and previous.get("status") == "no_speakers_detected":
            message = str(previous.get("message") or "")
            print(f"Audio speaker separation already found no speakers: {dx_stem}", flush=True)
            _write_manifest(
                run,
                source=str(dx_stem),
                stems=[],
                variant=variant,
                fmt=fmt,
                skipped=True,
                inputs=_carried_inputs(run.audio_separation_json, inputs),
                status="no_speakers_detected",
                message=message,
            )
            return AudioSeparationResult(str(dx_stem), run.unmapped_speakers_dir, [], skipped=True)

    if dry_run:
        print(f"DRY RUN: would separate DX stem with AudioShake: {dx_stem}", flush=True)
        print(
            "DRY RUN: would write speaker stems to "
            f"{run.unmapped_speakers_dir}/{output_prefix}_speaker_##_stem.wav",
            flush=True,
        )
        return AudioSeparationResult(str(dx_stem), run.unmapped_speakers_dir, [], skipped=True)

    raw_dir = run.audio_dir / "audioshake_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for path in existing:
            path.unlink()

    active_client = client or AudioShakeClient()
    try:
        downloaded = active_client.separate_speakers(
            dx_stem,
            raw_dir,
            variant=variant,
            fmt=fmt,
            prefix=output_prefix,
            timeout=timeout,
            poll_interval=poll_interval,
        )
    except AudioShakeNoSpeakersError as exc:
        _write_manifest(
            run,
            source=str(dx_stem),
            stems=[],
            variant=variant,
            fmt=fmt,
            skipped=False,
            inputs=input_fingerprints(inputs),
            status="no_speakers_detected",
            message=str(exc),
        )
        print(f"Audio speaker separation returned no speakers: {dx_stem}", flush=True)
        return AudioSeparationResult(str(dx_stem), run.unmapped_speakers_dir, [])
    normalized = normalize_speaker_outputs(
        downloaded,
        run.unmapped_speakers_dir,
        prefix=output_prefix,
        force=force,
    )
    _write_manifest(
        run,
        source=str(dx_stem),
        stems=normalized,
        variant=variant,
        fmt=fmt,
        skipped=False,
        inputs=input_fingerprints(inputs),
    )
    print(f"Audio speaker stems ready: {len(normalized)} file(s)", flush=True)
    return AudioSeparationResult(str(dx_stem), run.unmapped_speakers_dir, normalized)


def separate_full_audio_source(
    *,
    run: RunPaths,
    full_audio: AudioInputSource,
    client: AudioShakeClient | None = None,
    fmt: str = "wav",
    prefix: str,
    timeout: int = DEFAULT_TIMEOUT,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    force: bool = False,
    dry_run: bool = False,
) -> SourceSeparationResult:
    run.ensure()
    output_prefix = _audio_prefix(prefix)
    normalized_input = normalize_full_audio_input(
        run=run,
        source=full_audio,
        prefix=output_prefix,
        force=force,
        dry_run=dry_run,
    )
    inputs = {
        "source": full_audio,
        "backend": "audioshake",
        "format": fmt,
    }
    existing = existing_source_stems(run, prefix=output_prefix)
    if existing.get("dialogue") and not force:
        print(
            f"Source stems already exist in {run.source_stems_dir}; "
            f"using dialogue stem: {existing['dialogue']}",
            flush=True,
        )
        _write_source_manifest(
            run,
            source=str(full_audio),
            stems=existing,
            fmt=fmt,
            backend="audioshake",
            skipped=True,
            input_audio=normalized_input.provenance,
            inputs=_carried_inputs(run.source_separation_json, inputs),
        )
        return SourceSeparationResult(
            str(full_audio),
            run.source_stems_dir,
            existing.get("dialogue"),
            existing,
            skipped=True,
        )

    if dry_run:
        print(
            "DRY RUN: would separate full audio into DME stems: "
            f"{normalized_input.prepared_source}",
            flush=True,
        )
        print(
            "DRY RUN: would write source stems to "
            f"{run.source_stems_dir}/{output_prefix}_DX/MX/FX_stem.wav",
            flush=True,
        )
        return SourceSeparationResult(str(full_audio), run.source_stems_dir, None, {}, skipped=True)

    raw_dir = run.audio_dir / "audioshake_source_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for path in existing.values():
            if path.exists():
                path.unlink()

    active_client = client or AudioShakeClient()
    downloaded = active_client.separate_dme(
        normalized_input.prepared_source,
        raw_dir,
        fmt=fmt,
        prefix=output_prefix,
        timeout=timeout,
        poll_interval=poll_interval,
    )
    normalized = normalize_source_outputs(
        downloaded,
        run.source_stems_dir,
        prefix=output_prefix,
        force=force,
    )
    dialogue_stem = normalized.get("dialogue")
    if dialogue_stem is None:
        names = ", ".join(path.name for path in downloaded)
        raise AudioShakeError(f"DME separation did not produce a dialogue stem: {names}")
    _write_source_manifest(
        run,
        source=str(full_audio),
        stems=normalized,
        fmt=fmt,
        backend="audioshake",
        skipped=False,
        input_audio=normalized_input.provenance,
        inputs=input_fingerprints(inputs),
    )
    print(f"Source stems ready: {len(normalized)} file(s)", flush=True)
    return SourceSeparationResult(str(full_audio), run.source_stems_dir, dialogue_stem, normalized)


SPEECH_DENOISE_MODEL = "speech_denoise"
SPEECH_DENOISE_CREDITS_PER_MIN = 1.5


def speech_denoise_file(
    source: str | Path,
    output_dir: Path,
    *,
    client: AudioShakeClient | None = None,
    fmt: str = "wav",
    prefix: str = "",
    timeout: int = DEFAULT_TIMEOUT,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
) -> Path:
    """Clean one audio file with AudioShake's ``speech_denoise`` model.

    Removes environmental interference (hum, hiss, crowd babble, wind) while
    keeping the natural ambience, returning the path to the downloaded clean
    file. This is the shared cloud primitive behind the edit-words prompt
    degrain backend and the SR upsampler denoise backend; it wraps the generic
    ``separate_targets`` task so neither caller reimplements the upload/poll
    flow.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _warn_speech_denoise_credits(source)
    active_client = client or AudioShakeClient()
    downloaded = active_client.separate_targets(
        source,
        output_dir,
        [SPEECH_DENOISE_MODEL],
        fmt=fmt,
        prefix=prefix,
        timeout=timeout,
        poll_interval=poll_interval,
    )
    return _pick_speech_denoise_output(downloaded)


def _pick_speech_denoise_output(downloaded: list[Path]) -> Path:
    if not downloaded:
        raise AudioShakeError("AudioShake returned no speech_denoise output")
    for path in downloaded:
        name = path.name.lower()
        if "speech" in name or "denoise" in name:
            return path
    return downloaded[0]


def _warn_speech_denoise_credits(source: str | Path) -> None:
    """AudioShake bills speech_denoise per minute rounded up, so short clips
    (e.g. seconds-long edit prompts) still cost a full minute; surface it."""
    text = str(source)
    if text.startswith(("http://", "https://")):
        return
    path = Path(source).expanduser()
    if not path.exists():
        return
    try:
        duration = probe_duration_sec(path)
    except (OSError, EOFError, wave.Error):
        return
    minutes = max(1, math.ceil(duration / 60.0))
    credits = minutes * SPEECH_DENOISE_CREDITS_PER_MIN
    print(
        f"  Speech denoise: ~{credits:.1f} AudioShake credit(s) "
        f"({minutes} min rounded up at {SPEECH_DENOISE_CREDITS_PER_MIN}/min).",
        flush=True,
    )


def existing_unmapped_speaker_stems(run: RunPaths, *, prefix: str) -> list[Path]:
    return sorted(run.unmapped_speakers_dir.glob(f"{_audio_prefix(prefix)}_speaker_*_stem.*"))


def existing_source_stems(run: RunPaths, *, prefix: str) -> dict[str, Path]:
    stems: dict[str, Path] = {}
    for role, suffix in _TOOLKIT_DME_SUFFIX_BY_ROLE.items():
        matches = sorted(run.source_stems_dir.glob(f"{_audio_prefix(prefix)}_{suffix}_stem.*"))
        if matches:
            stems[role] = matches[0]
    return dict(sorted(stems.items()))


def normalize_source_outputs(
    downloaded: list[Path], output_dir: Path, *, prefix: str, force: bool
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized: dict[str, Path] = {}
    output_prefix = _audio_prefix(prefix)
    for source in sorted(downloaded):
        role = _source_role(source, prefix=output_prefix)
        if role is None:
            continue
        suffix = source.suffix or ".wav"
        toolkit_suffix = _TOOLKIT_DME_SUFFIX_BY_ROLE[role]
        target = output_dir / f"{output_prefix}_{toolkit_suffix}_stem{suffix}"
        if source.resolve() == target.resolve():
            normalized[role] = target
            continue
        if target.exists():
            if not force:
                normalized[role] = target
                continue
            target.unlink()
        source.replace(target)
        normalized[role] = target
    return dict(sorted(normalized.items()))


def normalize_speaker_outputs(
    downloaded: list[Path], output_dir: Path, *, prefix: str, force: bool
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized: list[Path] = []
    output_prefix = _audio_prefix(prefix)
    for index, source in enumerate(sorted(downloaded), 1):
        number = _speaker_number(_strip_output_prefix(source.stem, output_prefix), index)
        suffix = source.suffix or ".wav"
        target = output_dir / f"{output_prefix}_speaker_{number:02d}_stem{suffix}"
        if source.resolve() == target.resolve():
            normalized.append(target)
            continue
        if target.exists():
            if not force:
                normalized.append(target)
                continue
            target.unlink()
        source.replace(target)
        normalized.append(target)
    return sorted(normalized)


def _speaker_number(name: str, fallback: int) -> int:
    match = re.search(r"speaker[_-]?(\d+)", name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else fallback


_TOOLKIT_DME_SUFFIX_BY_ROLE: dict[SourceRole, str] = {
    "dialogue": "DX",
    "music_fx": "MX",
    "effects": "FX",
}


def _audio_prefix(value: str) -> str:
    return safe_name(value, fallback="audio")


def _source_role(path: Path, *, prefix: str = "") -> SourceRole | None:
    # Classify only the downloaded stem name, never the user-supplied prefix:
    # a project called e.g. "Redux" must not make every stem look like dialogue.
    name = _strip_output_prefix(path.stem, prefix).lower()
    if "dialogue" in name or "dialog" in name or "vocal" in name or _has_token(name, "dx"):
        return "dialogue"
    if "music" in name or _has_token(name, "mx"):
        return "music_fx"
    if "effect" in name or _has_token(name, "fx"):
        return "effects"
    return None


def _strip_output_prefix(stem: str, prefix: str) -> str:
    if prefix and stem.lower().startswith(f"{prefix.lower()}_"):
        return stem[len(prefix) + 1 :]
    return stem


def _has_token(name: str, token: str) -> bool:
    return re.search(rf"(^|[_-]){token}($|[_-])", name) is not None


def _carried_inputs(manifest_path: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    """Preserve the fingerprints recorded when the stems were actually made.

    A skipped run rewrites the manifest; refreshing the fingerprints there
    would hide the fact that the cached stems came from a different input.
    Warns when the current inputs no longer match the recorded ones.
    """
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if isinstance(previous, dict):
            changed = warn_if_inputs_changed(previous, inputs, manifest_path)
            if changed:
                raise ValueError(
                    "cached AudioShake stems came from changed input(s); rerun with --force"
                )
            recorded = previous.get("inputs")
            if isinstance(recorded, dict):
                return recorded
    return input_fingerprints(inputs)


def _write_manifest(
    run: RunPaths,
    *,
    source: str,
    stems: list[Path],
    variant: str,
    fmt: str,
    skipped: bool,
    inputs: dict[str, Any] | None = None,
    status: str = "completed",
    message: str = "",
) -> None:
    write_json(
        run.audio_separation_json,
        {
            "version": "1.0",
            "backend": "audioshake",
            "source": source,
            "variant": variant,
            "format": fmt,
            "output_dir": str(run.unmapped_speakers_dir),
            "skipped": skipped,
            "status": status,
            "message": message,
            "inputs": inputs or {},
            "stems": [str(path) for path in stems],
        },
    )


def _write_source_manifest(
    run: RunPaths,
    *,
    source: str,
    stems: dict[str, Path],
    fmt: str,
    backend: str,
    skipped: bool,
    input_audio: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "version": "1.0",
        "backend": backend,
        "source": source,
        "format": fmt,
        "output_dir": str(run.source_stems_dir),
        "skipped": skipped,
        "inputs": inputs or {},
        "dialogue_stem": str(stems["dialogue"]) if stems.get("dialogue") else "",
        "stems": {role: str(path) for role, path in stems.items()},
    }
    if input_audio:
        payload["input_audio"] = input_audio
    if metadata:
        payload.update(metadata)
    write_json(
        run.source_separation_json,
        payload,
    )
