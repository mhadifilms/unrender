from __future__ import annotations

import argparse
from pathlib import Path

from unrender.audio.mne.fx_classify import classify_fx
from unrender.audio.mne.fx_classify import settings_from_values as _fx_settings
from unrender.audio.mne.music import process_music
from unrender.audio.mne.music import settings_from_values as _music_settings
from unrender.audio.mne.room_tone import extract_room_tone
from unrender.audio.mne.room_tone import settings_from_values as _room_tone_settings
from unrender.cli.commands.context import (
    _audio_output_prefix,
    _audioshake_client,
    _dx_stem_arg,
    _mne_arg,
    _run_paths,
    _scenes_manifest_arg,
)
from unrender.project import RunPaths


def _source_stem(
    args: argparse.Namespace, key: str, explicit: Path | None, run: RunPaths, *, suffix: str
) -> Path:
    """Resolve an original-language FX/MX stem."""
    if explicit is not None:
        return explicit.expanduser()
    project = getattr(args, "project_config", None)
    if project is not None:
        configured = project.audio_asset_path(f"dme.{suffix.lower()}")
        if configured is not None:
            return configured
    matches = sorted(run.source_stems_dir.glob(f"*_{suffix}_stem.*"))
    if matches:
        return matches[0]
    raise ValueError(
        f"missing --{key.replace('_', '-')} (or audio_assets.original.dme."
        f"{suffix.lower()} in project config, or a generated *_{suffix}_stem.* in "
        f"{run.source_stems_dir}). Run `unrender audio separate` first."
    )


def _mne_classify_fx(args: argparse.Namespace) -> int:
    run = _run_paths(args, explicit=args.run_dir)
    run.ensure()
    prefix = _audio_output_prefix(args, run)
    fx_stem = _source_stem(args, "fx_stem", args.fx_stem, run, suffix="FX")
    settings = _fx_settings(
        clap_model=_mne_arg(args, "clap_model", args.clap_model, None),
        device=_mne_arg(args, "device", args.device, None),
        activity_threshold_db=_mne_arg(
            args, "fx_activity_threshold_db", args.activity_threshold_db, None
        ),
        window_sec=_mne_arg(args, "fx_window_sec", args.window_sec, None),
        min_event_sec=_mne_arg(args, "fx_min_event_sec", args.min_event_sec, None),
        merge_gap_sec=_mne_arg(args, "fx_merge_gap_sec", args.merge_gap_sec, None),
        crossfade_ms=_mne_arg(args, "fx_crossfade_ms", args.crossfade_ms, None),
    )
    classify_fx(
        run=run,
        fx_stem=fx_stem,
        settings=settings,
        prefix=prefix,
        force=args.force,
        dry_run=args.dry_run,
    )
    return 0


def _mne_music(args: argparse.Namespace) -> int:
    run = _run_paths(args, explicit=args.run_dir)
    run.ensure()
    prefix = _audio_output_prefix(args, run)
    mx_stem = _source_stem(args, "mx_stem", args.mx_stem, run, suffix="MX")
    settings = _music_settings(
        denoise=_mne_arg(args, "music_denoise", args.denoise, None),
        audit_verify_asr=_mne_arg(args, "music_verify_asr", args.verify_asr, None),
        silence_db=_mne_arg(args, "music_silence_db", args.silence_db, None),
        min_cue_sec=_mne_arg(args, "music_min_cue_sec", args.min_cue_sec, None),
        min_silence_sec=_mne_arg(args, "music_min_silence_sec", args.min_silence_sec, None),
        pad_sec=_mne_arg(args, "music_pad_sec", args.pad_sec, None),
        spectral_split=_mne_arg(args, "music_spectral_split", args.spectral_split, None),
        spectral_window_sec=_mne_arg(
            args, "music_spectral_window_sec", args.spectral_window_sec, None
        ),
        spectral_threshold=_mne_arg(
            args, "music_spectral_threshold", args.spectral_threshold, None
        ),
        split_instruments=_mne_arg(args, "split_instruments", args.split_instruments, None),
        instrument_models=_mne_arg(args, "instrument_models", args.instruments, None),
        timeout=_mne_arg(args, "audioshake_timeout", args.timeout, None),
        poll_interval=_mne_arg(args, "audioshake_poll_interval", args.poll_interval, None),
    )
    process_music(
        run=run,
        mx_stem=mx_stem,
        settings=settings,
        prefix=prefix,
        client=None if args.api_key is None else _audioshake_client(args.api_key),
        force=args.force,
        dry_run=args.dry_run,
    )
    return 0


def _mne_room_tone(args: argparse.Namespace) -> int:
    run = _run_paths(args, explicit=args.run_dir)
    run.ensure()
    prefix = _audio_output_prefix(args, run)
    dx_stem = Path(_dx_stem_arg(args, run, explicit=args.dx_stem, required=True))
    scenes_path = _scenes_manifest_arg(args, run, explicit=args.scenes)
    settings = _room_tone_settings(
        min_gap_sec=_mne_arg(args, "room_tone_min_gap_sec", args.min_gap_sec, None),
        threshold_db=_mne_arg(args, "room_tone_threshold_db", args.threshold_db, None),
        digital_silence_db=_mne_arg(
            args, "room_tone_digital_silence_db", args.digital_silence_db, None
        ),
        transient_ratio=_mne_arg(args, "room_tone_transient_ratio", args.transient_ratio, None),
        edge_handle_sec=_mne_arg(args, "room_tone_edge_handle_sec", args.edge_handle_sec, None),
        grain_ms=_mne_arg(args, "room_tone_grain_ms", args.grain_ms, None),
        crossfade_ms=_mne_arg(args, "room_tone_crossfade_ms", args.crossfade_ms, None),
        loop_sec=_mne_arg(args, "room_tone_loop_sec", args.loop_sec, None),
        cluster_distance=_mne_arg(args, "room_tone_cluster_distance", args.cluster_distance, None),
        seed=_mne_arg(args, "room_tone_seed", args.seed, None),
    )
    extract_room_tone(
        run=run,
        dx_stem=dx_stem,
        settings=settings,
        scenes_path=scenes_path if scenes_path.exists() else None,
        prefix=prefix,
        force=args.force,
        dry_run=args.dry_run,
    )
    return 0
