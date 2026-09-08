"""Full-program proxy creation from a master video or image sequence.

Scene detection (and ``face build``) must never chew through a full-res
ProRes master or an EXR folder; they read a compressed constant-frame-rate
proxy instead. This module builds that proxy: H.264 at a review width, exact
master frame rate, HDR tonemapped to Rec.709, EXR decoded through a display
transfer. Frame indices in the proxy map 1:1 to the master by construction.

For a sequence, "the master" is not always every file in the directory. A
delivered folder can carry head or tail frames outside the program, and the
host pipeline usually knows the real span. Passing ``frame_range`` makes proxy
frame 0 mean exactly that first program frame, which keeps the 1:1 mapping
honest: without it, a directory starting one file early silently shifts every
downstream cut by a frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from unrender.lib.fingerprints import input_fingerprints, warn_if_inputs_changed
from unrender.lib.names import safe_name
from unrender.lib.sequence import SequenceInfo, classify_master, detect_sequence
from unrender.lib.video import (
    DETECT_SAFE_CODECS,
    is_vfr,
    probe_video,
    run_video_tool,
    videotoolbox_available,
)
from unrender.manifests import read_json, write_json
from unrender.project import RunPaths

DEFAULT_PROXY_WIDTH = 1280
DEFAULT_PROXY_CRF = 20
DEFAULT_PROXY_TIMEOUT_SEC = 7200

_HDR_TONEMAP_FILTER = (
    "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
    "tonemap=hable,zscale=t=bt709:m=bt709:r=tv"
)


@dataclass(frozen=True)
class ProxyResult:
    path: Path
    created: bool
    source_kind: str


def default_proxy_path(run: RunPaths, master: Path) -> Path:
    name = safe_name(master.stem if master.is_file() else master.name, fallback="master")
    return run.media_dir / f"{name}_proxy.mov"


def proxy_inputs(
    master: Path, master_kind: str, frame_range: tuple[int, int] | None = None
) -> dict[str, object]:
    """Fingerprint spec for the master (first+last program frame for sequences)."""
    if master_kind == "sequence":
        sequence = detect_sequence(master)
        start, end = resolve_program_range(sequence, frame_range)
        return {"master": [sequence.frame_path(start), sequence.frame_path(end)]}
    return {"master": master}


def resolve_program_range(
    sequence: SequenceInfo, frame_range: tuple[int, int] | None
) -> tuple[int, int]:
    """Clamp a requested program span to the frames that exist, or raise.

    Returns the sequence's own bounds when no range is given, so callers that
    do not know the program span keep the previous whole-directory behavior.
    """
    if frame_range is None:
        return sequence.start_frame, sequence.end_frame
    start, end = int(frame_range[0]), int(frame_range[1])
    if start > end:
        raise ValueError(f"frame range start {start} is after end {end}")
    if start < sequence.start_frame or end > sequence.end_frame:
        raise ValueError(
            f"frame range {start}-{end} is outside the sequence in {sequence.directory}, "
            f"which holds {sequence.start_frame}-{sequence.end_frame}"
        )
    return start, end


def create_master_proxy(
    *,
    master: Path,
    output: Path,
    run: RunPaths,
    fps: float | None = None,
    width: int = DEFAULT_PROXY_WIDTH,
    crf: int = DEFAULT_PROXY_CRF,
    include_audio: bool = True,
    exr_transfer: str = "bt709",
    timecode_start: str | None = None,
    frame_range: tuple[int, int] | None = None,
    force: bool = False,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    timeout: int = DEFAULT_PROXY_TIMEOUT_SEC,
) -> ProxyResult:
    master_kind, master = classify_master(master)
    if frame_range is not None and master_kind != "sequence":
        raise ValueError(
            f"frame_range only applies to an image sequence; {master} is a video, "
            "whose program span is the whole file"
        )
    inputs = proxy_inputs(master, master_kind, frame_range)

    manifest_path = run.proxy_manifest_json
    if not force and output.exists() and output.stat().st_size > 1_000:
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if str(manifest.get("output")) == str(output):
                warn_if_inputs_changed(manifest, inputs, output)
        print(f"Proxy already exists, skipping: {output}", flush=True)
        return ProxyResult(path=output, created=False, source_kind=master_kind)

    output.parent.mkdir(parents=True, exist_ok=True)
    if master_kind == "sequence":
        _encode_sequence_proxy(
            master,
            output,
            fps=fps,
            width=width,
            crf=crf,
            exr_transfer=exr_transfer,
            timecode_start=timecode_start,
            frame_range=frame_range,
            ffmpeg=ffmpeg,
            timeout=timeout,
        )
    else:
        _encode_video_proxy(
            master,
            output,
            width=width,
            crf=crf,
            include_audio=include_audio,
            timecode_start=timecode_start,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            timeout=timeout,
        )

    write_json(
        manifest_path,
        {
            "version": "1.0",
            "master": str(master),
            "output": str(output),
            "inputs": input_fingerprints(inputs),
        },
    )
    print(f"Proxy saved: {output}", flush=True)
    return ProxyResult(path=output, created=True, source_kind=master_kind)


def master_usable_as_proxy(master: Path, *, ffprobe: str = "ffprobe") -> bool:
    """True when the master itself is a detection-safe compressed video."""
    if not master.is_file():
        return False
    try:
        probe = probe_video(master, ffprobe=ffprobe)
    except (RuntimeError, ValueError):
        return False
    return probe.codec in DETECT_SAFE_CODECS and not probe.is_hdr


def _encode_video_proxy(
    master: Path,
    output: Path,
    *,
    width: int,
    crf: int,
    include_audio: bool,
    timecode_start: str | None,
    ffmpeg: str,
    ffprobe: str,
    timeout: int,
) -> None:
    probe = probe_video(master, ffprobe=ffprobe)
    if is_vfr(probe):
        print(
            f"  WARNING: {master.name} looks variable-frame-rate "
            f"(r={probe.r_frame_rate}, avg={probe.avg_frame_rate}); the proxy is "
            "forced to constant frame rate and cut math assumes it.",
            flush=True,
        )
    filters = [f"scale={width}:-2"]
    if probe.is_hdr:
        print(f"  HDR master ({probe.color_transfer}); tonemapping proxy to Rec.709", flush=True)
        filters.append(_HDR_TONEMAP_FILTER)

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    if videotoolbox_available(ffmpeg=ffmpeg):
        cmd += ["-hwaccel", "videotoolbox"]
    cmd += ["-i", str(master)]
    cmd += ["-vf", ",".join(filters)]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p"]
    cmd += ["-fps_mode", "cfr", "-r", probe.r_frame_rate or f"{probe.fps:.6f}"]
    if include_audio and probe.has_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    timecode = probe.start_timecode or timecode_start
    if timecode:
        cmd += ["-timecode", timecode]
    cmd += ["-movflags", "+faststart", str(output)]
    run_video_tool(cmd, timeout=timeout)


def _encode_sequence_proxy(
    directory: Path,
    output: Path,
    *,
    fps: float | None,
    width: int,
    crf: int,
    exr_transfer: str,
    timecode_start: str | None,
    frame_range: tuple[int, int] | None,
    ffmpeg: str,
    timeout: int,
) -> None:
    if not fps or fps <= 0:
        raise ValueError(
            "an image-sequence master needs an explicit frame rate: pass --fps or set "
            "top-level `fps` in the project config"
        )
    sequence = detect_sequence(directory)
    start, end = resolve_program_range(sequence, frame_range)
    gap_frames = [frame for frame in sequence.missing_frames if start <= frame <= end]
    if gap_frames:
        raise ValueError(
            f"sequence {directory} has {len(gap_frames)} missing frame(s) "
            f"({_summarize_gaps(gap_frames)}) inside {start}-{end}; ffmpeg would "
            "misalign frames. Fill the gaps or provide a proxy via paths.proxy_master."
        )
    _preflight_decode(sequence, start, exr_transfer=exr_transfer, ffmpeg=ffmpeg)
    if frame_range is not None:
        print(f"  Program span: frames {start}-{end} of {directory.name}", flush=True)

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    if sequence.ext == ".exr":
        cmd += ["-apply_trc", exr_transfer]
    cmd += ["-framerate", f"{fps:.6f}", "-start_number", str(start)]
    cmd += ["-i", sequence.ffmpeg_pattern]
    cmd += ["-frames:v", str(end - start + 1)]
    cmd += ["-vf", f"scale={width}:-2"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p"]
    cmd += ["-an", "-timecode", timecode_start or "00:00:00:00"]
    cmd += ["-movflags", "+faststart", str(output)]
    run_video_tool(cmd, timeout=timeout)


def _preflight_decode(
    sequence: SequenceInfo, start_frame: int, *, exr_transfer: str, ffmpeg: str
) -> None:
    """Decode a single frame before committing to a long transcode."""
    first = sequence.frame_path(start_frame)
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    if sequence.ext == ".exr":
        cmd += ["-apply_trc", exr_transfer]
    cmd += ["-i", str(first), "-frames:v", "1", "-f", "null", "-"]
    try:
        run_video_tool(cmd, timeout=120)
    except RuntimeError as exc:
        raise RuntimeError(
            f"ffmpeg cannot decode this sequence ({first.name}): {exc}\n"
            "Provide a proxy via paths.proxy_master, or recompress the frames to a "
            "format ffmpeg can read (e.g. ZIP/PIZ EXR)."
        ) from exc


def _summarize_gaps(missing: list[int]) -> str:
    listed = ", ".join(str(frame) for frame in missing[:8])
    if len(missing) > 8:
        listed += ", ..."
    return listed
