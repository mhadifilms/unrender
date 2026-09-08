from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from unrender.audio.separation import separate_full_audio_source, separate_global_dx_stem
from unrender.cli.commands.context import (
    _audio_arg,
    _audio_output_prefix,
    _audioshake_client,
    _default_voice_clip_spec,
    _dx_stem_arg,
    _full_audio_arg,
    _path_arg,
    _run_paths,
    _shot_speakers_arg,
    _shots_manifest_arg,
    _speaker_db_path,
    _stem_sources_arg,
)
from unrender.cli.helpers import positive_float, threshold
from unrender.editorial.dialogue.clusters import resolve_voice_clips_clustered
from unrender.editorial.dialogue.lines import (
    build_voice_clips,
    import_transcript_lines,
    map_dialogue_to_shots,
    materialize_dialogue_line_stems,
    resolve_voice_clips,
    resolve_voice_clips_from_voice_db,
    transcribe_dialogue_lines,
)
from unrender.editorial.dialogue.resolvers import select_clip_resolver
from unrender.editorial.shots.dx import build_shot_dx, resolve_dx_stem
from unrender.manifests import load_dialogue_lines, load_shot_manifest, load_voice_inputs, read_json
from unrender.workflows import MANUAL_WORKFLOW, WorkflowProvenance, artifact_workflow


def _audio_language(args: argparse.Namespace) -> int:
    """Report what a file is spoken in, so the mixup is visible without a run."""
    from unrender.audio.language import probe_audio_language

    probe = probe_audio_language(
        args.source.expanduser(),
        expected=args.expect,
        windows=args.windows,
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        ffmpeg=args.ffmpeg or "ffmpeg",
    )
    for start_sec, detected in probe.windows:
        print(f"  {start_sec:9.1f}s  {detected or '--'}")
    print(f"{probe.path.name}: {probe.detected or 'undetermined'} ({probe.confidence:.0%})")
    if probe.issue is None:
        return 0
    print(f"ERROR: {probe.issue}")
    return 1


def _audio_separate(args: argparse.Namespace) -> int:
    run = _run_paths(args, explicit=args.run_dir)
    prefix = _audio_output_prefix(args, run)
    pair = tuple(args.full_audio_pair) if args.full_audio_pair else None
    explicit_sources = [value for value in (args.full_audio, pair, args.dx_stem) if value]
    if len(explicit_sources) > 1:
        raise ValueError("--full-audio, --full-audio-pair and --dx-stem are mutually exclusive")
    for key in ("source_backend", "speaker_strategy"):
        value = _audio_arg(args, key, getattr(args, key, None), "audioshake")
        if value != "audioshake":
            raise ValueError(f"audio.{key} must be audioshake")
    options = {
        "run": run,
        "prefix": prefix,
        "fmt": _audio_arg(args, "audioshake_format", args.format, "wav"),
        "timeout": int(
            positive_float(_audio_arg(args, "audioshake_timeout", args.timeout, 1800), "timeout")
        ),
        "poll_interval": int(
            positive_float(
                _audio_arg(args, "audioshake_poll_interval", args.poll_interval, 5), "poll-interval"
            )
        ),
        "force": args.force,
        "dry_run": args.dry_run,
        "client": (
            None if args.api_key is None or args.dry_run else _audioshake_client(args.api_key)
        ),
    }
    full_audio = pair or (None if args.dx_stem else _full_audio_arg(args, explicit=args.full_audio))
    dx_stem: str | Path
    if full_audio:
        separated = separate_full_audio_source(full_audio=full_audio, **options)
        if args.dry_run:
            print("DRY RUN: would separate the resulting dialogue into speaker stems")
            return 0
        if separated.dialogue_stem is None:
            raise RuntimeError("source separation did not produce a dialogue stem")
        dx_stem = separated.dialogue_stem
    else:
        dx_stem = _dx_stem_arg(args, run, explicit=args.dx_stem, required=True)
    separate_global_dx_stem(
        dx_stem=dx_stem,
        variant=_audio_arg(args, "audioshake_variant", args.variant, "n_speaker"),
        **options,
    )
    return 0


def _audio_transcribe_lines(args: argparse.Namespace) -> int:
    run = _run_paths(args, explicit=args.run_dir)
    run.ensure()
    if args.transcript is not None and args.dx_stem:
        raise ValueError("--transcript and --dx-stem are mutually exclusive for transcribe-lines")
    transcript = _path_arg(args, "transcript", args.transcript, required=False)
    if transcript is not None:
        import_transcript_lines(
            transcript_path=transcript,
            output_json=run.dialogue_lines_json,
            output_csv=run.dialogue_lines_csv,
            fps=float(_audio_arg(args, "transcript_fps", args.transcript_fps, 24.0)),
            default_duration_sec=float(
                _audio_arg(args, "default_duration_sec", args.default_duration_sec, 3.0)
            ),
            min_duration_sec=float(
                _audio_arg(args, "min_duration_sec", args.min_duration_sec, 0.3)
            ),
            handle_sec=float(_audio_arg(args, "handle_sec", args.handle_sec, 0.18)),
            force=args.force,
        )
        return 0
    shots_path = _shots_manifest_arg(args, run, explicit=args.shots, required=False)
    shots = load_shot_manifest(shots_path) if shots_path else []
    transcribe_dialogue_lines(
        audio_path=Path(
            _dx_stem_arg(
                args,
                run,
                explicit=args.dx_stem,
                required=True,
            )
        ),
        output_json=run.dialogue_lines_json,
        output_csv=run.dialogue_lines_csv,
        shots=shots,
        source_stems=_transcribe_source_stems(args, run),
        whisper_model=str(_audio_arg(args, "whisper_model", args.whisper_model, "small")),
        device=str(_audio_arg(args, "device", args.device, "cpu")),
        compute_type=str(_audio_arg(args, "compute_type", args.compute_type, "float32")),
        language=_audio_arg(args, "language", args.language, None),
        max_gap_sec=float(_audio_arg(args, "max_gap_sec", args.max_gap_sec, 0.5)),
        handle_sec=float(_audio_arg(args, "handle_sec", args.handle_sec, 0.18)),
        source_activity_threshold_db=float(
            _audio_arg(
                args,
                "source_activity_threshold_db",
                args.source_activity_threshold_db,
                -50.0,
            )
        ),
        source_margin_db=float(_audio_arg(args, "source_margin_db", args.source_margin_db, 1.0)),
        per_stem=bool(
            _audio_arg(
                args,
                "per_stem_transcription",
                args.per_stem,
                True,
            )
        ),
        force=args.force,
    )
    return 0


def _transcribe_source_stems(args: argparse.Namespace, run):
    if args.stems:
        return _stem_sources_arg(args, run)
    project = getattr(args, "project_config", None)
    if project is not None:
        configured = project.audio_asset_string("separated_stems") or project.data.get("stem_map")
        if configured:
            return _stem_sources_arg(args, run)
    if list(run.unmapped_speakers_dir.glob("*_speaker_*_stem.wav")):
        return _stem_sources_arg(args, run)
    return None


def _audio_build_clips(args: argparse.Namespace) -> int:
    run = _run_paths(args, explicit=args.run_dir)
    run.ensure()
    lines = load_dialogue_lines(
        _path_arg(args, "dialogue_lines", args.dialogue_lines, required=False)
        or run.dialogue_lines_json
    )
    build_voice_clips(
        lines=lines,
        stems=_stem_sources_arg(args, run),
        output_dir=run.voice_clips_dir,
        output_json=run.voice_clips_json,
        output_csv=run.voice_clips_csv,
        clip_type="dialogue_line",
        silence_threshold_db=float(
            _audio_arg(args, "silence_threshold_db", args.silence_threshold_db, -60.0)
        ),
        min_rms_dbfs=float(_audio_arg(args, "min_rms_dbfs", args.min_rms_dbfs, -55.0)),
        activity_threshold_db=float(
            _audio_arg(args, "activity_threshold_db", args.activity_threshold_db, -50.0)
        ),
        min_voiced_ratio=float(_audio_arg(args, "min_voiced_ratio", args.min_voiced_ratio, 0.15)),
        force=args.force,
        ffmpeg=str(_audio_arg(args, "ffmpeg", args.ffmpeg, "ffmpeg")),
    )
    return 0


def _audio_shot_dx(args: argparse.Namespace) -> int:
    run = _run_paths(args, explicit=args.run_dir)
    run.ensure()
    shots_path = _shots_manifest_arg(args, run, explicit=args.shots, required=True)
    assert shots_path is not None
    shots = load_shot_manifest(shots_path)
    explicit = _dx_stem_arg(
        args,
        run,
        explicit=args.dx_stem,
        required=False,
    )
    merge_sources = None
    if not explicit:
        # No DX anywhere: fall back to merging the separated speaker stems.
        merge_sources = _stem_sources_arg(args, run)
    dx_stem = resolve_dx_stem(
        run,
        explicit=Path(explicit) if explicit else None,
        merge_sources=merge_sources,
        force=args.force,
    )
    build_shot_dx(
        run=run,
        shots=shots,
        dx_stem=dx_stem,
        force=args.force,
        ffmpeg=str(_audio_arg(args, "ffmpeg", args.ffmpeg, "ffmpeg")),
    )
    return 0


def _audio_resolve_clips(args: argparse.Namespace) -> int:
    run = _run_paths(args, explicit=args.run_dir)
    run.ensure()
    clips = load_voice_inputs(args.clips or _default_voice_clip_spec(run))
    resolver = str(_audio_arg(args, "clip_resolver", args.resolver, "voice-db"))
    backend = str(_audio_arg(args, "voice_backend", args.backend, "pyannote"))
    if resolver == "auto":
        resolver = _auto_clip_resolver(args, run, clips_present=bool(clips))
    if resolver == "voice-db":
        resolve_voice_clips_from_voice_db(
            clips=clips,
            voice_db_path=(
                _path_arg(args, "voice_db", args.voice_db, required=False) or run.voice_db
            ),
            output_json=run.clip_stem_plan_json,
            output_csv=run.clip_stem_plan_csv,
        )
    elif resolver == "cluster":
        clusters = _audio_arg(args, "clusters", args.clusters, None)
        resolve_voice_clips_clustered(
            clips=clips,
            speaker_db_path=_speaker_db_path(args, args.speaker_db, run),
            output_json=run.clip_stem_plan_json,
            output_csv=run.clip_stem_plan_csv,
            backend=backend,
            n_clusters=None if clusters in (None, "") else int(clusters),
            override_min_sec=positive_float(
                _audio_arg(args, "override_sec", args.override_sec, 1.5), "override-sec"
            ),
            override_margin=float(_audio_arg(args, "override_margin", args.override_margin, 0.15)),
        )
    elif resolver == "clip":
        resolve_voice_clips(
            clips=clips,
            speaker_db_path=_speaker_db_path(args, args.speaker_db, run),
            voice_matches_json=run.voice_matches_json,
            output_json=run.clip_stem_plan_json,
            output_csv=run.clip_stem_plan_csv,
            backend=backend,
            sim_threshold=threshold(
                _audio_arg(args, "sim_threshold", args.sim_threshold, 0.65), "sim-threshold"
            ),
            min_margin=float(_audio_arg(args, "min_margin", args.min_margin, 0.05)),
        )
    else:
        raise ValueError(f"unsupported clip resolver: {resolver}")
    return 0


def _auto_clip_resolver(args: argparse.Namespace, run, *, clips_present: bool) -> str:
    speaker_db = _speaker_db_path(args, args.speaker_db, run)
    workflows: set[WorkflowProvenance] = set()
    provenance_paths = (speaker_db,)
    for path in dict.fromkeys(provenance_paths):
        if not path.is_file():
            continue
        payload = read_json(path)
        if not isinstance(payload, dict):
            continue
        discovered_workflow = artifact_workflow(payload)
        if discovered_workflow is not None:
            workflows.add(discovered_workflow)
    if len(workflows) > 1:
        raise ValueError(
            "auto clip resolver found mixed workflow provenance: " + ", ".join(sorted(workflows))
        )
    selected_workflow: WorkflowProvenance = MANUAL_WORKFLOW
    if workflows:
        selected_workflow = next(iter(workflows))
    voice_db = _path_arg(args, "voice_db", args.voice_db, required=False) or run.voice_db
    available = set()
    if clips_present:
        available.add("clips")
    if speaker_db.is_file():
        available.add("speaker_db")
    if voice_db.is_file():
        available.add("voice_db")
    return select_clip_resolver(selected_workflow, available_artifacts=available).name


def _audio_map_dialogue(args: argparse.Namespace) -> int:
    run = _run_paths(args, explicit=args.run_dir)
    run.ensure()
    lines = load_dialogue_lines(
        _path_arg(args, "dialogue_lines", args.dialogue_lines, required=False)
        or run.dialogue_lines_json
    )
    map_shots_path = _shots_manifest_arg(args, run, explicit=args.shots, required=True)
    assert map_shots_path is not None
    shots = load_shot_manifest(map_shots_path)
    clip_plan_path = _path_arg(args, "clip_plan", args.clip_plan, required=False) or (
        run.dialogue_stem_plan_json
        if run.dialogue_stem_plan_json.exists()
        else run.clip_stem_plan_json
    )
    clip_plan = read_json(clip_plan_path).get("clips") or []
    shot_speakers = _shot_speakers_arg(args, run)
    dialogue_plan = materialize_dialogue_line_stems(
        clip_plan=clip_plan,
        output_dir=run.dialogue_mapped_dir,
        output_json=run.dialogue_stem_plan_json,
        output_csv=run.dialogue_stem_plan_csv,
        force=args.force,
    )
    cut_shot_stems = bool(_audio_arg(args, "cut_shot_stems", args.cut_shot_stems or None, False))
    map_dialogue_to_shots(
        lines=lines,
        shots=shots,
        clip_plan=dialogue_plan,
        output_json=run.shot_dialogue_map_json,
        output_csv=run.shot_dialogue_map_csv,
        shot_plan_json=run.shot_stem_plan_json,
        shot_plan_csv=run.shot_stem_plan_csv,
        mapped_dir=run.shot_mapped_dir if cut_shot_stems else None,
        shot_speakers_by_id=shot_speakers,
        min_overlap_ratio=float(
            _audio_arg(args, "min_overlap_ratio", args.min_overlap_ratio, 0.01)
        ),
        force=args.force,
        ffmpeg=str(_audio_arg(args, "ffmpeg", args.ffmpeg, "ffmpeg")),
    )
    return 0


def _audio_analyze(args: argparse.Namespace) -> int:
    """Build the canonical, display-independent audio analysis bundle."""
    from unrender.audio.analysis import AnalysisSettings, analyze_audio_run

    run = _run_paths(args, explicit=args.run_dir)
    run.ensure()
    profile = str(_audio_arg(args, "analysis.profile", args.profile, "core"))
    analysis_dir = args.analysis_dir or _audio_arg(args, "analysis.output_dir", None, None)
    settings = AnalysisSettings.for_profile(
        profile,  # type: ignore[arg-type]
        analysis_dir=analysis_dir,
    )
    settings = replace(
        settings,
        frame_hop_sec=float(
            _audio_arg(args, "analysis.frame_hop_sec", args.hop_seconds, settings.frame_hop_sec)
        ),
        max_similarity_frames=int(
            _audio_arg(
                args,
                "analysis.max_similarity_frames",
                args.max_similarity_frames,
                settings.max_similarity_frames,
            )
        ),
        ffmpeg=str(_audio_arg(args, "analysis.ffmpeg", args.ffmpeg, "ffmpeg")),
        providers=tuple(args.provider or ()),
        permissive_only=not args.allow_unverified_licenses,
        allow_gated=bool(args.allow_gated),
        provider_opt_ins=tuple(args.provider_opt_in or ()),
        device=str(_audio_arg(args, "analysis.device", args.device, "cpu")),
        language=_audio_arg(args, "analysis.language", args.language, None),
        whisper_model=str(
            _audio_arg(args, "analysis.whisper_model", args.whisper_model, "large-v3")
        ),
        compute_type=str(_audio_arg(args, "analysis.compute_type", args.compute_type, "int8")),
        clap_window_sec=float(
            _audio_arg(
                args,
                "analysis.clap_window_sec",
                args.clap_window_seconds,
                settings.clap_window_sec,
            )
        ),
    )
    bundle = analyze_audio_run(
        run,
        inputs=args.input or None,
        settings=settings,
        force=args.force,
    )
    destination = (
        Path(settings.analysis_dir) / "audio_analysis.json"
        if settings.analysis_dir is not None
        else run.audio_analysis_json
    )
    print(
        f"Audio analysis: {len(bundle.artifacts)} artifact(s), "
        f"{len(bundle.features)} feature set(s) -> {destination}",
        flush=True,
    )
    failed = [status for status in bundle.analyzers if status.status == "failed"]
    if failed:
        print(f"  partial: {len(failed)} analyzer(s) failed", flush=True)
    return 0


def _audio_visualize(args: argparse.Namespace) -> int:
    """Render the canonical bundle in one or more standalone visual forms."""
    from unrender.audio.visualization import render_audio_analysis

    run = _run_paths(args, explicit=args.run_dir)
    analysis = args.analysis or run.audio_analysis_json
    if not Path(analysis).is_file():
        raise FileNotFoundError(
            f"audio analysis not found: {analysis}; run `unrender audio analyze`"
        )
    output_dir = (
        args.output_dir
        or _audio_arg(args, "visualization.output_dir", None, None)
        or run.audio_visualization_dir
    )
    renderer_names = args.renderer or _audio_arg(args, "visualization.renderers", None, None)
    outputs = render_audio_analysis(
        analysis,
        renderer_names=renderer_names,
        output_dir=output_dir,
        mapping=args.mapping or _audio_arg(args, "visualization.mapping", None, None),
        source_paths=args.source,
        alternate_paths=args.alternate,
        ffmpeg=_audio_arg(args, "visualization.ffmpeg", args.ffmpeg, None),
    )
    for name, path in outputs.items():
        print(f"Audio visualization [{name}]: {path}", flush=True)
    return 0


def _audio_transform(args: argparse.Namespace) -> int:
    """Render a non-destructive corrective or creative audio recipe."""
    from unrender.audio.transforms import RecipeSettings, run_recipe
    from unrender.lib.names import safe_name

    run = _run_paths(args, explicit=args.run_dir)
    run.ensure()
    configured_roles = _audio_arg(args, "transforms.roles", None, {})
    roles = (
        {str(key).lower(): Path(str(value)).expanduser() for key, value in configured_roles.items()}
        if isinstance(configured_roles, dict)
        else {}
    )
    roles.update(_key_path_values(args.role, label="--role", required=False))
    if not roles:
        raise ValueError("at least one --role ROLE=PATH value is required")
    configured_parameters = _audio_arg(args, "transforms.parameters", None, {})
    parameters = dict(configured_parameters) if isinstance(configured_parameters, dict) else {}
    parameters.update(_key_values(args.param, label="--param"))
    destination = (
        args.output_dir
        or _audio_arg(args, "transforms.output_dir", None, None)
        or run.derived_audio_dir / safe_name(args.recipe, fallback="recipe")
    )
    analysis = args.analysis or (
        run.audio_analysis_json if run.audio_analysis_json.is_file() else None
    )
    result = run_recipe(
        args.recipe,
        roles,
        destination,
        settings=RecipeSettings(
            dry_wet=float(_audio_arg(args, "transforms.dry_wet", args.dry_wet, 1.0)),
            seed=int(_audio_arg(args, "transforms.seed", args.seed, 0)),
            peak_ceiling_dbfs=float(
                _audio_arg(
                    args,
                    "transforms.peak_ceiling_dbfs",
                    args.peak_ceiling_dbfs,
                    -1.0,
                )
            ),
            sample_subtype=str(
                _audio_arg(args, "transforms.sample_subtype", args.sample_subtype, "FLOAT")
            ),
        ),
        parameters=parameters,
        analysis_bundle=analysis,
    )
    print(f"Audio transform [{result.name}]: {result.manifest}", flush=True)
    for role, path in result.outputs.items():
        print(f"  {role}: {path}", flush=True)
    return 0


def _key_path_values(
    values: list[str] | None, *, label: str, required: bool = True
) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for raw in values or []:
        key, separator, value = raw.partition("=")
        if not separator or not key.strip() or not value.strip():
            raise ValueError(f"{label} values must use ROLE=PATH")
        parsed[key.strip().lower()] = Path(value.strip()).expanduser()
    if required and not parsed:
        raise ValueError(f"at least one {label} ROLE=PATH value is required")
    return parsed


def _key_values(values: list[str] | None, *, label: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for raw in values or []:
        key, separator, value = raw.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"{label} values must use KEY=VALUE")
        try:
            parsed[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            parsed[key.strip()] = value
    return parsed
