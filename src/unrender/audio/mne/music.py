"""Music (MX) processing: de-bleed, silence-gated cue detection, and splits.

Pipeline:

1. *De-bleed* the MX stem through the AudioShake ``instrumental`` model to
   strip residual dialogue/vocal bleed, reusing the speech-island + ASR audit
   from :mod:`unrender.audio.mne.stem_audit` to report before/after bleed.
2. *Cue detection* by RMS-gating the cleaned MX into contiguous music regions,
   further splitting butt-joined cues on log-mel spectral-change boundaries.
3. *Optional instrument splitting* per cue via one AudioShake task with
   configurable targets.

AudioShake reuses the existing ``requests`` dependency; nothing heavy is
imported at module load.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from unrender.audio.mne.stem_audit import audit_stems
from unrender.audio.separation.audioshake_client import (
    DEFAULT_POLL_INTERVAL,
    DEFAULT_TIMEOUT,
    AudioShakeClient,
)
from unrender.lib.audio import read_wav_float, write_wav_float
from unrender.lib.audio_features import (
    Segment,
    activity_segments,
    as_float,
    spectral_change_points,
    to_mono,
)
from unrender.lib.fingerprints import input_fingerprints, warn_if_inputs_changed
from unrender.manifests import read_json, write_json
from unrender.project import RunPaths

DEFAULT_INSTRUMENT_MODELS: tuple[str, ...] = (
    "drums",
    "bass",
    "guitar",
    "keys",
    "strings",
    "wind",
    "vocals",
    "other",
)


@dataclass(frozen=True)
class MusicSettings:
    denoise: bool = True
    # Verify flagged bleed islands with ASR by default; ``audit_stems`` downgrades
    # to an RMS-only report on its own when whisperx is not installed.
    audit_verify_asr: bool = True
    audit_threshold_db: float = -45.0
    silence_db: float = -50.0
    min_cue_sec: float = 2.0
    min_silence_sec: float = 1.0
    pad_sec: float = 0.5
    spectral_split: bool = True
    spectral_window_sec: float = 1.0
    spectral_threshold: float = 0.35
    split_instruments: bool = False
    instrument_models: tuple[str, ...] = DEFAULT_INSTRUMENT_MODELS
    fmt: str = "wav"
    timeout: int = DEFAULT_TIMEOUT
    poll_interval: int = DEFAULT_POLL_INTERVAL


@dataclass
class MusicCue:
    index: int
    start_sec: float
    end_sec: float
    pad_start_sec: float
    pad_end_sec: float
    path: Path
    instruments_dir: Path | None = None
    instruments: list[str] = field(default_factory=list)


def process_music(
    *,
    run: RunPaths,
    mx_stem: str | Path,
    settings: MusicSettings | None = None,
    prefix: str,
    client: AudioShakeClient | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """De-bleed the MX stem, detect cues, and optionally split instruments."""
    run.ensure()
    active = settings or MusicSettings()
    source = Path(mx_stem).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"MX stem not found: {source}")
    output_json = run.mne_music_cues_json
    inputs = {"mx_stem": source}

    if output_json.exists() and not force:
        data = read_json(output_json)
        warn_if_inputs_changed(data, inputs, output_json)
        if active.split_instruments and _needs_instrument_split(data):
            print(
                "Music cues already exist; splitting instruments on the cached cues.",
                flush=True,
            )
            return _split_cached_instruments(
                run=run,
                data=data,
                settings=active,
                client=client,
                prefix=prefix,
                output_json=output_json,
            )
        print(f"Music cues already exist, skipping: {output_json}", flush=True)
        return data

    if dry_run:
        print(f"DRY RUN: would de-bleed and cue-detect MX stem: {source}", flush=True)
        print(f"DRY RUN: would write cues to {run.mne_music_dir / 'cues'}", flush=True)
        return {}

    cleaned, denoise_meta = _denoise(
        run=run, source=source, settings=active, client=client, prefix=prefix
    )

    audio, sample_rate, sample_width = read_wav_float(cleaned)
    mono = to_mono(audio)
    total_sec = mono.size / float(sample_rate) if sample_rate else 0.0

    regions = activity_segments(
        mono,
        sample_rate=sample_rate,
        threshold_db=active.silence_db,
        hop_sec=0.05,
        merge_gap_sec=active.min_silence_sec,
        min_duration_sec=active.min_cue_sec,
    )
    if active.spectral_split:
        regions = _split_on_spectral_change(mono, sample_rate, regions, active)

    cues_dir = run.mne_music_dir / "cues"
    cues = _write_cues(
        audio,
        sample_rate=sample_rate,
        sample_width=sample_width,
        regions=regions,
        total_sec=total_sec,
        pad_sec=active.pad_sec,
        output_dir=cues_dir,
        prefix=prefix,
    )

    if active.split_instruments:
        _split_instruments(run=run, cues=cues, settings=active, client=client, prefix=prefix)

    payload: dict[str, Any] = {
        "version": "1.0",
        "backend": "audioshake",
        "source": str(source),
        "cleaned_source": str(cleaned),
        "format": active.fmt,
        "output_dir": str(cues_dir),
        "duration_sec": round(total_sec, 3),
        "denoise": denoise_meta,
        "settings": {
            "split_instruments": active.split_instruments,
            "instrument_models": list(active.instrument_models),
            "silence_db": active.silence_db,
            "min_cue_sec": active.min_cue_sec,
            "spectral_split": active.spectral_split,
        },
        "inputs": input_fingerprints(inputs),
        "cues": [
            {
                "cue_id": f"cue_{cue.index:02d}",
                "start_sec": round(cue.start_sec, 3),
                "end_sec": round(cue.end_sec, 3),
                "pad_start_sec": round(cue.pad_start_sec, 3),
                "pad_end_sec": round(cue.pad_end_sec, 3),
                "path": str(cue.path),
                "instruments_dir": str(cue.instruments_dir) if cue.instruments_dir else "",
                "instruments": cue.instruments,
            }
            for cue in cues
        ],
    }
    write_json(output_json, payload)
    print(f"Music cues saved: {output_json} ({len(cues)} cue(s))", flush=True)
    return payload


def _denoise(
    *,
    run: RunPaths,
    source: Path,
    settings: MusicSettings,
    client: AudioShakeClient | None,
    prefix: str,
) -> tuple[Path, dict[str, Any]]:
    if not settings.denoise:
        return source, {"applied": False}

    raw_dir = run.mne_music_dir / "audioshake_raw"
    active_client = client or AudioShakeClient()
    downloaded = active_client.separate_targets(
        source,
        raw_dir,
        ["instrumental"],
        fmt=settings.fmt,
        prefix=f"{prefix}_denoise",
        timeout=settings.timeout,
        poll_interval=settings.poll_interval,
    )
    cleaned = _pick_output(downloaded, "instrumental")
    meta: dict[str, Any] = {"applied": True, "model": "instrumental", "cleaned": str(cleaned)}
    try:
        audit = audit_stems(
            stems={"mx_before": source, "mx_after": cleaned},
            output_json=run.mne_music_dir / "music_denoise_audit.json",
            threshold_db=settings.audit_threshold_db,
            verify_asr=settings.audit_verify_asr,
            force=True,
        )
        meta["bleed_before_sec"] = _stem_speech_sec(audit, "mx_before")
        meta["bleed_after_sec"] = _stem_speech_sec(audit, "mx_after")
    except (OSError, ValueError, RuntimeError) as exc:
        meta["audit_error"] = str(exc)
    return cleaned, meta


def _stem_speech_sec(audit: dict[str, Any], role: str) -> float:
    for row in audit.get("stems") or []:
        if row.get("role") == role:
            return float(row.get("speech_sec") or 0.0)
    return 0.0


def _pick_output(downloaded: list[Path], keyword: str) -> Path:
    if not downloaded:
        raise RuntimeError("AudioShake returned no de-bleed output")
    for path in downloaded:
        if keyword in path.name.lower():
            return path
    return downloaded[0]


def _split_on_spectral_change(
    mono: np.ndarray,
    sample_rate: int,
    regions: list[Segment],
    settings: MusicSettings,
) -> list[Segment]:
    out: list[Segment] = []
    for region in regions:
        start = int(region.start_sec * sample_rate)
        end = int(region.end_sec * sample_rate)
        clip = mono[start:end]
        points = spectral_change_points(
            clip,
            sample_rate=sample_rate,
            window_sec=settings.spectral_window_sec,
            threshold=settings.spectral_threshold,
        )
        bounds = [region.start_sec, *[region.start_sec + p for p in points], region.end_sec]
        pieces = [Segment(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
        # Pieces are butt-joined, so merging (any zero-gap join) would undo the
        # split; instead of dropping sub-cue fragments (which would lose audio),
        # fold each short fragment into its neighbour so coverage is preserved.
        out.extend(_merge_short_pieces(pieces, settings.min_cue_sec))
    return out


def _merge_short_pieces(pieces: list[Segment], min_sec: float) -> list[Segment]:
    """Absorb sub-``min_sec`` fragments into adjacent pieces without losing span."""
    if not pieces:
        return []
    merged: list[Segment] = []
    current = pieces[0]
    for nxt in pieces[1:]:
        if current.duration_sec < min_sec:
            current = Segment(current.start_sec, nxt.end_sec)
        else:
            merged.append(current)
            current = nxt
    if current.duration_sec < min_sec and merged:
        last = merged.pop()
        merged.append(Segment(last.start_sec, current.end_sec))
    else:
        merged.append(current)
    return merged


def _write_cues(
    audio: np.ndarray,
    *,
    sample_rate: int,
    sample_width: int,
    regions: list[Segment],
    total_sec: float,
    pad_sec: float,
    output_dir: Path,
    prefix: str,
) -> list[MusicCue]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cues: list[MusicCue] = []
    for index, region in enumerate(regions, 1):
        pad_start = max(0.0, region.start_sec - pad_sec)
        pad_end = min(total_sec, region.end_sec + pad_sec)
        start_sample = int(pad_start * sample_rate)
        end_sample = max(start_sample + 1, int(pad_end * sample_rate))
        clip = audio[start_sample:end_sample]
        target = output_dir / f"{prefix}_cue_{index:02d}.wav"
        write_wav_float(target, clip, sample_rate, sample_width=sample_width)
        cues.append(
            MusicCue(
                index=index,
                start_sec=region.start_sec,
                end_sec=region.end_sec,
                pad_start_sec=pad_start,
                pad_end_sec=pad_end,
                path=target,
            )
        )
    return cues


def _split_instruments(
    *,
    run: RunPaths,
    cues: list[MusicCue],
    settings: MusicSettings,
    client: AudioShakeClient | None,
    prefix: str,
) -> None:
    if not cues:
        return
    _warn_instrument_credits(cues, settings.instrument_models)
    active_client = client or AudioShakeClient()
    instruments_root = run.mne_music_dir / "instruments"
    for cue in cues:
        cue_dir = instruments_root / f"cue_{cue.index:02d}"
        downloaded = active_client.separate_targets(
            cue.path,
            cue_dir,
            list(settings.instrument_models),
            fmt=settings.fmt,
            prefix=f"{prefix}_cue_{cue.index:02d}",
            timeout=settings.timeout,
            poll_interval=settings.poll_interval,
        )
        cue.instruments_dir = cue_dir
        cue.instruments = [path.name for path in downloaded]


def _warn_instrument_credits(cues: list[MusicCue], models: tuple[str, ...]) -> None:
    """AudioShake bills per target per minute rounded up, so per-cue splitting
    can be surprisingly expensive; surface an estimate before uploading."""
    minutes = sum(max(1, math.ceil((cue.pad_end_sec - cue.pad_start_sec) / 60.0)) for cue in cues)
    credit_minutes = minutes * len(models)
    print(
        f"  Instrument split: {len(cues)} cue(s) x {len(models)} target(s) "
        f"~= {credit_minutes} AudioShake credit-minute(s) (rounded up per cue).",
        flush=True,
    )


def _needs_instrument_split(data: dict[str, Any]) -> bool:
    cues = data.get("cues") if isinstance(data, dict) else None
    if not isinstance(cues, list) or not cues:
        return False
    return any(not (isinstance(cue, dict) and cue.get("instruments")) for cue in cues)


def _cues_from_manifest(data: dict[str, Any]) -> list[MusicCue]:
    cues: list[MusicCue] = []
    for index, entry in enumerate(data.get("cues") or [], 1):
        if not isinstance(entry, dict):
            continue
        cues.append(
            MusicCue(
                index=index,
                start_sec=float(entry.get("start_sec") or 0.0),
                end_sec=float(entry.get("end_sec") or 0.0),
                pad_start_sec=float(entry.get("pad_start_sec") or entry.get("start_sec") or 0.0),
                pad_end_sec=float(entry.get("pad_end_sec") or entry.get("end_sec") or 0.0),
                path=Path(str(entry.get("path"))).expanduser(),
            )
        )
    return cues


def _split_cached_instruments(
    *,
    run: RunPaths,
    data: dict[str, Any],
    settings: MusicSettings,
    client: AudioShakeClient | None,
    prefix: str,
    output_json: Path,
) -> dict[str, Any]:
    cues = _cues_from_manifest(data)
    _split_instruments(run=run, cues=cues, settings=settings, client=client, prefix=prefix)
    by_index = {cue.index: cue for cue in cues}
    for index, entry in enumerate(data.get("cues") or [], 1):
        cue = by_index.get(index)
        if isinstance(entry, dict) and cue is not None:
            entry["instruments_dir"] = str(cue.instruments_dir) if cue.instruments_dir else ""
            entry["instruments"] = cue.instruments
    settings_block = data.get("settings")
    if isinstance(settings_block, dict):
        settings_block["split_instruments"] = True
        settings_block["instrument_models"] = list(settings.instrument_models)
    write_json(output_json, data)
    print(f"Music cues updated with instrument splits: {output_json}", flush=True)
    return data


def settings_from_values(
    *,
    denoise: bool | None = None,
    audit_verify_asr: bool | None = None,
    silence_db: float | str | None = None,
    min_cue_sec: float | str | None = None,
    min_silence_sec: float | str | None = None,
    pad_sec: float | str | None = None,
    spectral_split: bool | None = None,
    spectral_window_sec: float | str | None = None,
    spectral_threshold: float | str | None = None,
    split_instruments: bool | None = None,
    instrument_models: Any = None,
    timeout: int | str | None = None,
    poll_interval: int | str | None = None,
) -> MusicSettings:
    base = MusicSettings()
    return MusicSettings(
        denoise=base.denoise if denoise is None else bool(denoise),
        audit_verify_asr=(
            base.audit_verify_asr if audit_verify_asr is None else bool(audit_verify_asr)
        ),
        silence_db=as_float(silence_db, base.silence_db),
        min_cue_sec=as_float(min_cue_sec, base.min_cue_sec),
        min_silence_sec=as_float(min_silence_sec, base.min_silence_sec),
        pad_sec=as_float(pad_sec, base.pad_sec),
        spectral_split=base.spectral_split if spectral_split is None else bool(spectral_split),
        spectral_window_sec=as_float(spectral_window_sec, base.spectral_window_sec),
        spectral_threshold=as_float(spectral_threshold, base.spectral_threshold),
        split_instruments=(
            base.split_instruments if split_instruments is None else bool(split_instruments)
        ),
        instrument_models=_as_models(instrument_models, base.instrument_models),
        timeout=_as_int(timeout, base.timeout),
        poll_interval=_as_int(poll_interval, base.poll_interval),
    )


def _as_int(value: int | str | None, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _as_models(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",") if item.strip()]
        return tuple(parts) or default
    if isinstance(value, (list, tuple)):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return tuple(parts) or default
    raise ValueError("instrument_models must be a comma-separated string or a list")
