from __future__ import annotations

import argparse
from pathlib import Path

from unrender import __version__
from unrender.cli.commands import (
    _audio_analyze,
    _audio_build_clips,
    _audio_language,
    _audio_map_dialogue,
    _audio_resolve_clips,
    _audio_separate,
    _audio_shot_dx,
    _audio_transcribe_lines,
    _audio_transform,
    _audio_visualize,
    _doctor,
    _export_artifacts,
    _face_build,
    _labels_apply,
    _labels_interactive,
    _labels_template,
    _mne_classify_fx,
    _mne_music,
    _mne_room_tone,
    _register_spectral,
    _scenes_group,
    _shots_cut,
    _shots_detect,
    _shots_match,
    _shots_proxy,
    _status,
    _timeline_build,
    _voice_build,
    _voice_match,
)

_SPEAKER_STRATEGY_HELP = "AudioShake separates concurrent voices into anonymous speaker stems"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="unrender")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="count", default=0)
    verbosity.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Show tracebacks for handled errors")
    sub = parser.add_subparsers(dest="command")

    audio = sub.add_parser("audio", help="Create and split speaker stems")
    audio_sub = audio.add_subparsers(dest="audio_command")

    audio_separate = audio_sub.add_parser(
        "separate",
        help="Separate source and speaker stems into the run directory",
    )
    _add_project_arg(audio_separate)
    audio_separate.add_argument(
        "--full-audio", help="Full mix/audio to split into dialogue/music/effects first"
    )
    audio_separate.add_argument(
        "--full-audio-pair",
        nargs=2,
        metavar=("LEFT", "RIGHT"),
        help="Separate mono L/R files merged as one stereo full mix",
    )
    audio_separate.add_argument(
        "--source-backend",
        choices=("audioshake",),
        help="Backend for --full-audio source separation (default: audioshake)",
    )
    audio_separate.add_argument(
        "--speaker-strategy",
        "--speaker-backend",
        dest="speaker_strategy",
        choices=("audioshake",),
        help=_SPEAKER_STRATEGY_HELP,
    )
    audio_separate.add_argument("--dx-stem", help="Local path or HTTPS URL to global DX stem")
    audio_separate.add_argument("--run-dir", "--out", dest="run_dir", type=Path)
    audio_separate.add_argument("--variant", choices=("two_speaker", "n_speaker"))
    audio_separate.add_argument("--format")
    audio_separate.add_argument("--prefix")
    audio_separate.add_argument("--timeout", type=int)
    audio_separate.add_argument("--poll-interval", type=int)
    audio_separate.add_argument(
        "--api-key", help="AudioShake API key (default: AUDIOSHAKE_API_KEY)"
    )
    audio_separate.add_argument("--force", "-f", action="store_true")
    audio_separate.add_argument("--dry-run", "-n", action="store_true")
    audio_separate.set_defaults(handler=_audio_separate)

    audio_transcribe = audio_sub.add_parser(
        "transcribe-lines",
        help="Transcribe DX into diarized dialogue lines",
    )
    _add_project_arg(audio_transcribe)
    audio_transcribe.add_argument("--dx-stem", help="Local path to global DX stem")
    audio_transcribe.add_argument(
        "--transcript", type=Path, help="Existing source transcript CSV/TSV/JSON/XLSX/DOCX"
    )
    audio_transcribe.add_argument("--shots", type=Path)
    audio_transcribe.add_argument("--run-dir", "--out", dest="run_dir", type=Path)
    audio_transcribe.add_argument(
        "--stems",
        help=(
            "Separated speaker stems transcribed independently with word timestamps "
            "(default: run audio/unmapped_speakers)"
        ),
    )
    per_stem = audio_transcribe.add_mutually_exclusive_group()
    per_stem.add_argument(
        "--per-stem",
        dest="per_stem",
        action="store_true",
        help="Transcribe each speaker stem independently (default when stems exist)",
    )
    per_stem.add_argument(
        "--no-per-stem",
        dest="per_stem",
        action="store_false",
        help="Transcribe only the DX stem and infer speaker channels from activity",
    )
    audio_transcribe.set_defaults(per_stem=None)
    audio_transcribe.add_argument("--whisper-model")
    audio_transcribe.add_argument("--device")
    audio_transcribe.add_argument("--compute-type")
    audio_transcribe.add_argument("--language")
    audio_transcribe.add_argument("--max-gap-sec", type=float)
    audio_transcribe.add_argument("--handle-sec", type=float)
    audio_transcribe.add_argument("--source-activity-threshold-db", type=float)
    audio_transcribe.add_argument("--source-margin-db", type=float)
    audio_transcribe.add_argument("--transcript-fps", type=float)
    audio_transcribe.add_argument("--default-duration-sec", type=float)
    audio_transcribe.add_argument("--min-duration-sec", type=float)
    audio_transcribe.add_argument("--force", "-f", action="store_true")
    audio_transcribe.set_defaults(handler=_audio_transcribe_lines)

    audio_language = audio_sub.add_parser(
        "language",
        help="Report the language a media file is actually spoken in",
    )
    audio_language.add_argument("source", type=Path, help="Audio or video file to sample")
    audio_language.add_argument("--expect", help="ISO code to compare against, e.g. de")
    audio_language.add_argument(
        "--windows", type=int, default=3, help="How many windows to sample (default: 3)"
    )
    audio_language.add_argument("--model", default="small", help="WhisperX model (default: small)")
    audio_language.add_argument("--device", default="cpu")
    audio_language.add_argument("--compute-type", default="float32")
    audio_language.add_argument("--ffmpeg")
    audio_language.set_defaults(handler=_audio_language)

    audio_build_clips = audio_sub.add_parser(
        "build-clips",
        help="Cut per-line voice clips from each full-length speaker stem",
    )
    _add_project_arg(audio_build_clips)
    audio_build_clips.add_argument("--dialogue-lines", type=Path)
    audio_build_clips.add_argument("--run-dir", "--out", dest="run_dir", type=Path)
    audio_build_clips.add_argument("--stems")
    audio_build_clips.add_argument("--silence-threshold-db", type=float)
    audio_build_clips.add_argument("--min-rms-dbfs", type=float)
    audio_build_clips.add_argument("--activity-threshold-db", type=float)
    audio_build_clips.add_argument("--min-voiced-ratio", type=float)
    audio_build_clips.add_argument("--ffmpeg")
    audio_build_clips.add_argument("--force", "-f", action="store_true")
    audio_build_clips.set_defaults(handler=_audio_build_clips)

    audio_shot_dx = audio_sub.add_parser(
        "shot-dx",
        help=(
            "Cut a per-shot DX stem from the full dialogue stem (given, from "
            "source separation, or merged from separated speaker stems)"
        ),
    )
    _add_project_arg(audio_shot_dx)
    audio_shot_dx.add_argument("--shots", type=Path)
    audio_shot_dx.add_argument("--dx-stem", help="Full DX stem (default: source separation output)")
    audio_shot_dx.add_argument(
        "--stems", help="Separated speaker stems to merge when no DX stem exists"
    )
    audio_shot_dx.add_argument("--run-dir", "--out", dest="run_dir", type=Path)
    audio_shot_dx.add_argument("--ffmpeg")
    audio_shot_dx.add_argument("--force", "-f", action="store_true")
    audio_shot_dx.set_defaults(handler=_audio_shot_dx)

    audio_resolve_clips = audio_sub.add_parser(
        "resolve-clips",
        help="Resolve native voice clips against labeled voice clusters",
    )
    _add_project_arg(audio_resolve_clips)
    audio_resolve_clips.add_argument("--clips", help="Glob, CSV, or JSON clip manifest")
    audio_resolve_clips.add_argument("--speaker-db", type=Path)
    audio_resolve_clips.add_argument("--voice-db", type=Path)
    audio_resolve_clips.add_argument("--run-dir", "--out", dest="run_dir", type=Path)
    audio_resolve_clips.add_argument("--backend", choices=("pyannote",))
    audio_resolve_clips.add_argument(
        "--resolver",
        choices=("voice-db", "cluster", "clip", "auto"),
        help=(
            "voice-db (default): preserve labeled voice_db membership; "
            "cluster: experimental reclustering fallback; clip: independent per-clip matching; "
            "auto: select from workflow provenance"
        ),
    )
    audio_resolve_clips.add_argument(
        "--clusters",
        type=int,
        help="Number of voice clusters (default: number of labeled speakers)",
    )
    audio_resolve_clips.add_argument(
        "--sim-threshold",
        type=float,
        help="Per-clip acceptance threshold (clip resolver only; default: 0.65)",
    )
    audio_resolve_clips.add_argument(
        "--min-margin",
        type=float,
        help="Required gap between best and second-best speaker (clip resolver; default: 0.05)",
    )
    audio_resolve_clips.add_argument(
        "--override-sec",
        type=float,
        help="Minimum clip duration for a per-clip cluster override (default: 1.5)",
    )
    audio_resolve_clips.add_argument(
        "--override-margin",
        type=float,
        help="Required best-vs-second margin for a per-clip override (default: 0.15)",
    )
    audio_resolve_clips.set_defaults(handler=_audio_resolve_clips)

    audio_map_dialogue = audio_sub.add_parser(
        "map-dialogue",
        help="Map resolved full dialogue lines onto shots",
    )
    _add_project_arg(audio_map_dialogue)
    audio_map_dialogue.add_argument("--dialogue-lines", type=Path)
    audio_map_dialogue.add_argument("--shots", type=Path)
    audio_map_dialogue.add_argument("--shot-matches", type=Path)
    audio_map_dialogue.add_argument("--clip-plan", type=Path)
    audio_map_dialogue.add_argument("--run-dir", "--out", dest="run_dir", type=Path)
    audio_map_dialogue.add_argument("--min-overlap-ratio", type=float)
    audio_map_dialogue.add_argument(
        "--cut-shot-stems",
        action="store_true",
        help=("Also build one exact-shot-length, silence-padded stem per active speaker"),
    )
    audio_map_dialogue.add_argument("--ffmpeg")
    audio_map_dialogue.add_argument("--force", "-f", action="store_true")
    audio_map_dialogue.set_defaults(handler=_audio_map_dialogue)

    audio_analyze = audio_sub.add_parser(
        "analyze",
        help="Build the canonical audio-analysis bundle from a mix and its stems",
    )
    _add_project_arg(audio_analyze)
    audio_analyze.add_argument("--run-dir", "--out", dest="run_dir", type=Path)
    audio_analyze.add_argument(
        "--input",
        action="append",
        help="Explicit ROLE=PATH or PATH input; may be repeated",
    )
    audio_analyze.add_argument(
        "--profile",
        choices=("core", "auto", "full", "quick", "standard", "deep"),
    )
    audio_analyze.add_argument("--analysis-dir", type=Path)
    audio_analyze.add_argument("--hop-seconds", type=float)
    audio_analyze.add_argument("--max-similarity-frames", type=int)
    audio_analyze.add_argument(
        "--provider",
        action="append",
        help="Optional provider id to preflight; may be repeated",
    )
    audio_analyze.add_argument(
        "--provider-opt-in",
        action="append",
        help="Explicitly accept a gated provider id; may be repeated",
    )
    audio_analyze.add_argument("--allow-gated", action="store_true")
    audio_analyze.add_argument("--allow-unverified-licenses", action="store_true")
    audio_analyze.add_argument("--device")
    audio_analyze.add_argument("--language")
    audio_analyze.add_argument("--whisper-model")
    audio_analyze.add_argument("--compute-type")
    audio_analyze.add_argument("--clap-window-seconds", type=float)
    audio_analyze.add_argument("--ffmpeg")
    audio_analyze.add_argument("--force", "-f", action="store_true")
    audio_analyze.set_defaults(handler=_audio_analyze)

    audio_visualize = audio_sub.add_parser(
        "visualize",
        help="Render standalone visual forms from audio_analysis.json",
    )
    _add_project_arg(audio_visualize)
    audio_visualize.add_argument("--run-dir", "--out", dest="run_dir", type=Path)
    audio_visualize.add_argument("--analysis", type=Path)
    audio_visualize.add_argument("--output-dir", type=Path)
    audio_visualize.add_argument(
        "--renderer",
        action="append",
        choices=(
            "timeline",
            "timeline_atlas",
            "structure",
            "structure_map",
            "territory",
            "mix_territory",
            "poster",
            "html",
            "overlay",
            "spectral",
        ),
        help="Renderer to build; may be repeated (default: core visual set)",
    )
    audio_visualize.add_argument("--mapping", type=Path)
    audio_visualize.add_argument("--source", type=Path, action="append")
    audio_visualize.add_argument("--alternate", type=Path, action="append")
    audio_visualize.add_argument("--ffmpeg")
    audio_visualize.set_defaults(handler=_audio_visualize)

    audio_transform = audio_sub.add_parser(
        "transform",
        help="Render a non-destructive corrective or creative audio recipe",
    )
    _add_project_arg(audio_transform)
    audio_transform.add_argument("recipe")
    audio_transform.add_argument("--run-dir", "--out", dest="run_dir", type=Path)
    audio_transform.add_argument(
        "--role",
        action="append",
        help="Recipe input as ROLE=PATH; may be repeated",
    )
    audio_transform.add_argument(
        "--param",
        action="append",
        help="Recipe parameter as KEY=JSON_VALUE; may be repeated",
    )
    audio_transform.add_argument("--analysis", type=Path)
    audio_transform.add_argument("--output-dir", type=Path)
    audio_transform.add_argument("--dry-wet", type=float)
    audio_transform.add_argument("--seed", type=int)
    audio_transform.add_argument("--peak-ceiling-dbfs", type=float)
    audio_transform.add_argument(
        "--sample-subtype",
        choices=("FLOAT", "PCM_16"),
    )
    audio_transform.set_defaults(handler=_audio_transform)

    face = sub.add_parser("face", help="Build face clusters and match faces to shots")
    face_sub = face.add_subparsers(dest="face_command")
    face_build = face_sub.add_parser("build", help="Build face_db.json and review grids")
    _add_project_arg(face_build)
    face_build.add_argument("--video", type=Path)
    face_build.add_argument("--run-dir", "--out", dest="run_dir", type=Path)
    face_build.add_argument(
        "--interval",
        type=float,
        help="Face-sampling interval in seconds (default: 1.0; high recall)",
    )
    face_build.add_argument("--max-frames", type=int, default=0)
    face_build.add_argument("--threshold", type=float)
    face_build.add_argument(
        "--min-cluster-size",
        type=int,
        help="Minimum detections for a face cluster (default: 2; preserves rare identities)",
    )
    face_build.add_argument(
        "--face-count",
        type=int,
        help="Target number of face identity clusters when known",
    )
    face_build.add_argument(
        "--min-confidence", type=float, help="Detection score floor (default: 0.5)"
    )
    face_build.add_argument(
        "--min-face-px",
        type=int,
        help="Skip faces smaller than this many pixels on their short side (default: 36)",
    )
    face_build.add_argument("--force", "-f", action="store_true")
    face_build.set_defaults(handler=_face_build)

    voice = sub.add_parser("voice", help="Build and match voice clusters")
    voice_sub = voice.add_subparsers(dest="voice_command")
    voice_build = voice_sub.add_parser("build", help="Build voice_db.json and review samples")
    _add_project_arg(voice_build)
    voice_build.add_argument(
        "--clips",
        help="Glob, CSV, or JSON voice clip manifest (default: run audio/_unmapped)",
    )
    voice_build.add_argument("--run-dir", "--out", dest="run_dir", type=Path)
    voice_build.add_argument("--backend", choices=("pyannote",), default="pyannote")
    voice_build.add_argument("--cluster-threshold", type=float)
    voice_build.add_argument("--min-cluster-size", type=int)
    voice_build.add_argument("--voice-count", type=int, help="Target number of voice clusters")
    voice_build.add_argument("--samples-per-cluster", type=int)
    voice_build.add_argument("--continuity-max-gap-sec", type=float)
    voice_build.add_argument("--continuity-min-similarity", type=float)
    voice_build.add_argument("--continuity-min-margin", type=float)
    voice_build.add_argument("--continuity-short-max-duration-sec", type=float)
    voice_build.add_argument("--continuity-short-max-gap-sec", type=float)
    voice_build.add_argument("--continuity-interjection-max-duration-sec", type=float)
    voice_build.add_argument("--continuity-interjection-max-gap-sec", type=float)
    voice_build.add_argument("--force", "-f", action="store_true")
    voice_build.set_defaults(handler=_voice_build)

    voice_match = voice_sub.add_parser("match", help="Match clips to labeled voice clusters")
    _add_project_arg(voice_match)
    voice_match.add_argument("--clips", help="Glob, CSV, or JSON voice clip manifest")
    voice_match.add_argument("--speaker-db", type=Path)
    voice_match.add_argument("--run-dir", "--out", dest="run_dir", type=Path)
    voice_match.add_argument("--backend", choices=("pyannote",), default="pyannote")
    voice_match.add_argument("--sim-threshold", type=float, default=0.65)
    voice_match.set_defaults(handler=_voice_match)

    labels = sub.add_parser("labels", help="Create and apply manual speaker labels")
    labels_sub = labels.add_subparsers(dest="labels_command")
    labels_template = labels_sub.add_parser("template", help="Write labels.csv for manual review")
    _add_project_arg(labels_template)
    labels_template.add_argument("--run-dir", "--run", dest="run_dir", type=Path)
    labels_template.add_argument("--out", type=Path)
    labels_template.set_defaults(handler=_labels_template)

    labels_apply = labels_sub.add_parser("apply", help="Apply labels.csv to speaker_db.json")
    _add_project_arg(labels_apply)
    labels_apply.add_argument("--run-dir", "--run", dest="run_dir", type=Path)
    labels_apply.add_argument("--labels", type=Path)
    labels_apply.add_argument("--config", type=Path)
    labels_apply.set_defaults(handler=_labels_apply)

    labels_interactive = labels_sub.add_parser(
        "interactive",
        help="Interactively map face/voice clusters to configured speakers",
    )
    _add_project_arg(labels_interactive)
    labels_interactive.add_argument("--run-dir", "--run", dest="run_dir", type=Path)
    labels_interactive.add_argument("--config", type=Path)
    labels_interactive.add_argument(
        "--type",
        choices=("all", "face", "voice"),
        default="all",
        help="Cluster type to label (default all)",
    )
    labels_interactive.add_argument(
        "--relabel-all",
        action="store_true",
        help="Prompt for already-labeled clusters too",
    )
    labels_interactive.set_defaults(handler=_labels_interactive)

    shots = sub.add_parser("shots", help="Detect, cut, and match shots from a master")
    shots_sub = shots.add_subparsers(dest="shots_command")

    shots_proxy = shots_sub.add_parser(
        "proxy",
        help="Create a full-program proxy video from the master (video or EXR sequence)",
    )
    _add_project_arg(shots_proxy)
    shots_proxy.add_argument("--master", type=Path, help="Master video file or sequence dir")
    shots_proxy.add_argument("--output", type=Path, help="Proxy output path")
    shots_proxy.add_argument("--fps", type=float, help="Frame rate (required for sequences)")
    shots_proxy.add_argument("--width", type=int, help="Proxy width (default: 1280)")
    shots_proxy.add_argument("--crf", type=int, help="H.264 CRF quality (default: 20)")
    shots_proxy.add_argument("--no-audio", action="store_true", help="Skip the audio track")
    shots_proxy.add_argument(
        "--exr-transfer", help="EXR linear-to-display transfer (default: bt709)"
    )
    shots_proxy.add_argument(
        "--frame-range",
        help=(
            "Program span inside a sequence master as START-END source frame numbers "
            "(default: every frame in the directory)"
        ),
    )
    shots_proxy.add_argument("--timecode-start", help="Timecode label of master frame 0")
    shots_proxy.add_argument("--run-dir", "--out", dest="run_dir", type=Path)
    shots_proxy.add_argument("--ffmpeg", help="ffmpeg binary (default: ffmpeg)")
    shots_proxy.add_argument("--ffprobe", help="ffprobe binary (default: ffprobe)")
    shots_proxy.add_argument("--force", "-f", action="store_true")
    shots_proxy.set_defaults(handler=_shots_proxy)

    shots_detect = shots_sub.add_parser(
        "detect",
        help="Detect scene cuts on the proxy and write shots.json (cuts media by default)",
    )
    _add_project_arg(shots_detect)
    shots_detect.add_argument("--master", type=Path, help="Master video file or sequence dir")
    shots_detect.add_argument("--proxy", type=Path, help="Detection proxy video")
    shots_detect.add_argument(
        "--detector",
        choices=("content", "adaptive", "ensemble"),
        help="Detection mode (default: content)",
    )
    shots_detect.add_argument(
        "--threshold", type=float, help="ContentDetector threshold (default: 27.0)"
    )
    shots_detect.add_argument(
        "--min-shot-frames", type=int, help="Minimum shot length in frames (default: 15)"
    )
    shots_detect.add_argument(
        "--min-gap-sec", type=float, help="Merge cuts closer than this (default: 0.33)"
    )
    shots_detect.add_argument(
        "--no-cut", action="store_true", help="Only write shots.json; do not cut media"
    )
    shots_detect.add_argument(
        "--no-proxy-create",
        action="store_true",
        help="Error instead of auto-creating a missing proxy",
    )
    shots_detect.add_argument(
        "--force-proxy",
        action="store_true",
        help="Never detect on the master directly, even when it is already compressed",
    )
    shots_detect.add_argument("--shots-out", type=Path, help="Shot manifest output path")
    shots_detect.add_argument(
        "--csv",
        nargs="?",
        const=True,
        default=None,
        help="Also write a spotting CSV (optionally to a custom path)",
    )
    _add_cut_args(shots_detect)
    shots_detect.set_defaults(handler=_shots_detect)

    shots_cut = shots_sub.add_parser(
        "cut",
        help="Cut the master into per-shot media from a shot list (json/csv/txt/edl)",
    )
    _add_project_arg(shots_cut)
    shots_cut.add_argument(
        "--shots", type=Path, help="Shot list: .json/.csv/.txt/.edl (default: paths.shots)"
    )
    shots_cut.add_argument("--master", type=Path, help="Master video file or sequence dir")
    shots_cut.add_argument("--proxy", type=Path, help="Full-program proxy for per-shot proxies")
    shots_cut.add_argument(
        "--edl-tc",
        choices=("record", "source"),
        default="record",
        help="Which EDL timecodes to cut with (default: record)",
    )
    shots_cut.add_argument("--shots-out", type=Path, help="Shot manifest output path")
    shots_cut.add_argument(
        "--no-cut", action="store_true", help="Only write shots.json; do not cut media"
    )
    shots_cut.add_argument(
        "--csv",
        nargs="?",
        const=True,
        default=None,
        help="Also write a spotting CSV (optionally to a custom path)",
    )
    _add_cut_args(shots_cut)
    shots_cut.set_defaults(handler=_shots_cut)

    shots_match = shots_sub.add_parser("match", help="Write shot_speaker_matches.json/csv")
    _add_project_arg(shots_match)
    shots_match.add_argument("--shots", type=Path)
    shots_match.add_argument("--speaker-db", type=Path)
    shots_match.add_argument("--run-dir", "--out", dest="run_dir", type=Path)
    shots_match.add_argument("--samples-per-shot", type=int)
    shots_match.add_argument("--sim-threshold", type=float)
    shots_match.add_argument(
        "--min-confidence", type=float, help="Detection score floor (default: 0.5)"
    )
    shots_match.add_argument("--min-votes", type=int)
    shots_match.add_argument(
        "--min-margin",
        type=float,
        help="Required best-vs-second similarity gap per face vote (default: 0.05)",
    )
    shots_match.set_defaults(handler=_shots_match)

    scenes = sub.add_parser("scenes", help="Group detected shots into narrative scenes")
    scenes_sub = scenes.add_subparsers(dest="scenes_command")
    scenes_group = scenes_sub.add_parser(
        "group",
        help="Score every cut and write scenes.json (optional stage after shots detect/cut)",
    )
    _add_project_arg(scenes_group)
    scenes_group.add_argument("--shots", type=Path, help="Shot manifest (default: paths.shots)")
    scenes_group.add_argument(
        "--video", type=Path, help="Keyframe source video (default: the detection proxy)"
    )
    scenes_group.add_argument(
        "--engine",
        choices=("fusion",),
        help="Boundary engine: pretrained feature fusion (default) or local Qwen2.5-VL",
    )
    scenes_group.add_argument(
        "--threshold", type=float, help="Boundary score threshold (default: 0.6)"
    )
    scenes_group.add_argument(
        "--dialogue-lines",
        type=Path,
        help="dialogue_lines.json for continuity evidence (default: the run's, when present)",
    )
    scenes_group.add_argument(
        "--no-audio", action="store_true", help="Skip audio features (fusion engine)"
    )
    scenes_group.add_argument("--scenes-out", type=Path, help="Scene manifest output path")
    scenes_group.add_argument(
        "--csv",
        nargs="?",
        const=True,
        default=None,
        help="Also write a scene spotting CSV (optionally to a custom path)",
    )
    scenes_group.add_argument("--run-dir", "--out", dest="run_dir", type=Path)
    scenes_group.add_argument("--ffmpeg", help="ffmpeg binary (default: ffmpeg)")
    scenes_group.add_argument("--force", "-f", action="store_true")
    scenes_group.set_defaults(handler=_scenes_group)

    mne = sub.add_parser(
        "mne", help="Process M&E stems into ambience/foley/FX, music cues, and room tone"
    )
    mne_sub = mne.add_subparsers(dest="mne_command")

    mne_classify_fx = mne_sub.add_parser(
        "classify-fx",
        help="Split the FX stem into AMB/FOLEY/HFX/DSGN tracks with CLAP zero-shot tagging",
    )
    _add_project_arg(mne_classify_fx)
    mne_classify_fx.add_argument("--fx-stem", type=Path, help="FX source stem (default: generated)")
    mne_classify_fx.add_argument("--run-dir", "--out", dest="run_dir", type=Path)
    mne_classify_fx.add_argument("--clap-model", help="CLAP checkpoint id")
    mne_classify_fx.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), help="Torch device for CLAP"
    )
    mne_classify_fx.add_argument("--activity-threshold-db", type=float)
    mne_classify_fx.add_argument("--window-sec", type=float)
    mne_classify_fx.add_argument("--min-event-sec", type=float)
    mne_classify_fx.add_argument("--merge-gap-sec", type=float)
    mne_classify_fx.add_argument("--crossfade-ms", type=float)
    mne_classify_fx.add_argument("--force", "-f", action="store_true")
    mne_classify_fx.add_argument("--dry-run", "-n", action="store_true")
    mne_classify_fx.set_defaults(handler=_mne_classify_fx)

    mne_music = mne_sub.add_parser(
        "music",
        help="De-bleed the MX stem, detect music cues, and optionally split instruments",
    )
    _add_project_arg(mne_music)
    mne_music.add_argument("--mx-stem", type=Path, help="MX source stem (default: generated)")
    mne_music.add_argument("--run-dir", "--out", dest="run_dir", type=Path)
    denoise = mne_music.add_mutually_exclusive_group()
    denoise.add_argument("--denoise", dest="denoise", action="store_true")
    denoise.add_argument("--no-denoise", dest="denoise", action="store_false")
    mne_music.set_defaults(denoise=None)
    verify = mne_music.add_mutually_exclusive_group()
    verify.add_argument("--verify-asr", dest="verify_asr", action="store_true")
    verify.add_argument("--no-verify-asr", dest="verify_asr", action="store_false")
    mne_music.set_defaults(verify_asr=None)
    spectral = mne_music.add_mutually_exclusive_group()
    spectral.add_argument("--spectral-split", dest="spectral_split", action="store_true")
    spectral.add_argument("--no-spectral-split", dest="spectral_split", action="store_false")
    mne_music.set_defaults(spectral_split=None)
    mne_music.add_argument("--silence-db", type=float)
    mne_music.add_argument("--min-cue-sec", type=float)
    mne_music.add_argument("--min-silence-sec", type=float)
    mne_music.add_argument("--pad-sec", type=float)
    mne_music.add_argument("--spectral-window-sec", type=float)
    mne_music.add_argument("--spectral-threshold", type=float)
    mne_music.add_argument(
        "--split-instruments", dest="split_instruments", action="store_true", default=None
    )
    mne_music.add_argument("--instruments", help="Comma-separated instrument targets")
    mne_music.add_argument("--api-key", help="AudioShake API key (default: AUDIOSHAKE_API_KEY)")
    mne_music.add_argument("--timeout", type=int)
    mne_music.add_argument("--poll-interval", type=int)
    mne_music.add_argument("--force", "-f", action="store_true")
    mne_music.add_argument("--dry-run", "-n", action="store_true")
    mne_music.set_defaults(handler=_mne_music)

    mne_room_tone = mne_sub.add_parser(
        "room-tone",
        help="Harvest speech-free DX gaps into per-scene room-tone loops and a full-length bed",
    )
    _add_project_arg(mne_room_tone)
    mne_room_tone.add_argument("--dx-stem", help="Full DX stem (default: source separation output)")
    mne_room_tone.add_argument(
        "--scenes", type=Path, help="Scene manifest for per-space grouping (default: scenes.json)"
    )
    mne_room_tone.add_argument("--run-dir", "--out", dest="run_dir", type=Path)
    mne_room_tone.add_argument("--min-gap-sec", type=float)
    mne_room_tone.add_argument("--threshold-db", type=float)
    mne_room_tone.add_argument("--digital-silence-db", type=float)
    mne_room_tone.add_argument("--transient-ratio", type=float)
    mne_room_tone.add_argument("--edge-handle-sec", type=float)
    mne_room_tone.add_argument("--grain-ms", type=float)
    mne_room_tone.add_argument("--crossfade-ms", type=float)
    mne_room_tone.add_argument("--loop-sec", type=float)
    mne_room_tone.add_argument("--cluster-distance", type=float)
    mne_room_tone.add_argument("--seed", type=int)
    mne_room_tone.add_argument("--force", "-f", action="store_true")
    mne_room_tone.add_argument("--dry-run", "-n", action="store_true")
    mne_room_tone.set_defaults(handler=_mne_room_tone)

    timeline = sub.add_parser("timeline", help="Reconstruct an editor timeline from run artifacts")
    timeline_sub = timeline.add_subparsers(dest="timeline_command")
    timeline_build = timeline_sub.add_parser(
        "build",
        help="Assemble an OpenTimelineIO timeline (.otio) from shots and stems",
    )
    _add_project_arg(timeline_build)
    timeline_build.add_argument("--shots", type=Path, help="Shot manifest for the video track")
    timeline_build.add_argument("--run-dir", "--run", dest="run_dir", type=Path)
    timeline_build.add_argument(
        "--out", type=Path, help="Output timeline path (default: <run>/timeline.otio)"
    )
    timeline_build.add_argument("--fps", type=float, help="Timeline frame rate (default: 24)")
    timeline_build.add_argument(
        "--start-timecode",
        help="Program start timecode (HH:MM:SS:FF) or seconds for the timeline origin",
    )
    timeline_build.add_argument(
        "--dialogue",
        choices=("auto", "lines", "shots"),
        help="Dialogue source granularity (default: auto)",
    )
    timeline_build.add_argument("--name", help="Timeline name (default: project/run name)")
    timeline_build.add_argument(
        "--scenes",
        type=Path,
        help="Scene manifest for scene markers/metadata (default: run scenes.json)",
    )
    timeline_build.add_argument(
        "--no-scenes",
        action="store_true",
        help="Ignore the scene manifest even if one exists",
    )
    timeline_build.add_argument(
        "--scene-stacks",
        action="store_true",
        default=None,
        help="Nest video clips into one OTIO stack per scene (default: flat + scene markers)",
    )
    timeline_build.add_argument(
        "--dialogue-bed",
        action="store_true",
        default=None,
        help="Add the full-length dialogue source stem as its own track",
    )
    timeline_build.add_argument(
        "--shot-dx",
        action="store_true",
        default=None,
        help="Add the per-shot DX stems (from `audio shot-dx`) as a single track",
    )
    timeline_build.add_argument(
        "--no-source-stems",
        action="store_true",
        default=None,
        help="Omit the music and effects source-stem tracks",
    )
    include_mne = timeline_build.add_mutually_exclusive_group()
    include_mne.add_argument(
        "--include-mne",
        dest="include_mne",
        action="store_true",
        help="Add mne AMB/FOLEY/HFX/DSGN, room-tone, and music-cue tracks (default: on if present)",
    )
    include_mne.add_argument(
        "--no-include-mne", dest="include_mne", action="store_false", help="Omit mne tracks"
    )
    timeline_build.set_defaults(include_mne=None)
    timeline_build.add_argument(
        "--no-markers",
        action="store_true",
        default=None,
        help="Do not add dialogue/speaker markers to clips",
    )
    timeline_build.add_argument(
        "--bundle",
        action="store_true",
        default=None,
        help="Also write a self-contained .otiod media bundle next to the timeline",
    )
    timeline_build.add_argument("--ffprobe", help="ffprobe executable for media probing")
    timeline_build.add_argument(
        "--dry-run", "-n", action="store_true", help="Report the plan without writing files"
    )
    timeline_build.set_defaults(handler=_timeline_build)

    export = sub.add_parser("export", help="Export run artifacts")
    export_sub = export.add_subparsers(dest="export_command")
    artifacts = export_sub.add_parser("artifacts", help="Export standalone JSON artifacts")
    _add_project_arg(artifacts)
    artifacts.add_argument("--run-dir", "--run", dest="run_dir", type=Path)
    artifacts.add_argument("--out", type=Path)
    artifacts.set_defaults(handler=_export_artifacts)

    status = sub.add_parser("status", help="Show run artifact status")
    _add_project_arg(status)
    status.add_argument("--run-dir", "--run", dest="run_dir", type=Path)
    status.set_defaults(handler=_status)

    doctor = sub.add_parser("doctor", help="Validate project inputs and local prerequisites")
    _add_project_arg(doctor)
    doctor.add_argument("--run-dir", "--run", dest="run_dir", type=Path)
    doctor.set_defaults(handler=_doctor)

    _register_spectral(sub)

    return parser


def _add_project_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-p",
        "--project",
        dest="projects",
        nargs="+",
        help="Project config name, path, or glob under config/",
    )
    # Lives here because the check it disables only fires for a project that
    # declares audio_assets.original.language, which only a project config can do.
    parser.add_argument(
        "--skip-language-check",
        action="store_true",
        help="Accept a DX stem whose spoken language contradicts the project config",
    )


def _add_cut_args(parser: argparse.ArgumentParser) -> None:
    """Flags shared by `shots detect` (default cut phase) and `shots cut`."""
    parser.add_argument("--fps", type=float, help="Frame rate (required for sequences)")
    parser.add_argument(
        "--tc-origin",
        help="Timecode label of master frame 0: HH:MM:SS:FF, 'auto', or 'zero'",
    )
    parser.add_argument("--timecode-start", help="Alias config for shots.timecode_start")
    parser.add_argument("--shots-dir", type=Path, help="Per-shot media root (default: run/shots)")
    parser.add_argument("--shot-prefix", help="Shot folder/file prefix (default: project name)")
    parser.add_argument("--padding", type=int, help="Shot id zero-padding (default: 3)")
    parser.add_argument(
        "--renumber",
        nargs="?",
        const="default",
        help="Renumber sequence frames from START (default 1001) instead of preserving",
    )
    parser.add_argument(
        "--link",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How to materialize sequence frames (default: copy)",
    )
    parser.add_argument(
        "--allow-gaps",
        action="store_true",
        help="Skip missing sequence frames instead of failing the shot",
    )
    parser.add_argument(
        "--per-shot-proxy",
        action="store_true",
        help="Also cut a per-shot proxy from the full-program proxy",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Skip per-shot proxies (sequence masters get them by default)",
    )
    parser.add_argument("--width", type=int, help="Proxy width when auto-creating (default: 1280)")
    parser.add_argument("--crf", type=int, help="Proxy H.264 CRF (default: 20)")
    parser.add_argument("--exr-transfer", help="EXR linear-to-display transfer (default: bt709)")
    parser.add_argument(
        "--frame-range",
        help=(
            "Program span inside a sequence master as START-END source frame numbers. "
            "Frame 0 of the timeline is START, so head/tail frames outside the program "
            "do not shift the cuts (default: every frame in the directory)"
        ),
    )
    parser.add_argument("--workers", type=int, help="Parallel cut workers")
    parser.add_argument("--timeout", type=int, help="Per-shot ffmpeg timeout in seconds")
    parser.add_argument("--run-dir", "--out", dest="run_dir", type=Path)
    parser.add_argument("--ffmpeg", help="ffmpeg binary (default: ffmpeg)")
    parser.add_argument("--ffprobe", help="ffprobe binary (default: ffprobe)")
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--dry-run", "-n", action="store_true")
