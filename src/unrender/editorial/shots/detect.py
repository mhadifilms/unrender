"""Scene-cut detection on a compressed proxy, never on the master itself.

Detection runs PySceneDetect over the full-program proxy. The proxy is
resolved (or auto-created) by :func:`resolve_detection_input`, which is the
single gate guaranteeing an intra-frame master or an EXR folder never reaches
the detector. Cut frames are post-processed into frame-exact, end-exclusive
shot ranges shared with the cutting stage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from unrender.editorial.shots.proxy import (
    ProxyResult,
    create_master_proxy,
    default_proxy_path,
    master_usable_as_proxy,
)
from unrender.editorial.shots.sources import CutRange
from unrender.lib.timecode import seconds_to_frames
from unrender.lib.video import INTRA_FRAME_CODECS, probe_video
from unrender.project import RunPaths

# Detector defaults.
#
# The default is `content` (per-frame HSV+edge delta) because for a shot-cutting
# pipeline, RECALL of hard cuts matters most: a missed cut means a downstream
# plate spans a real edit, which breaks later timeline edits. `adaptive` (rolling-
# average ratio) was previously the default because it suppresses false cuts
# from flashes / fast motion, but it under-detects genuine hard cuts between
# consecutive *visually-similar* shots — e.g. a fast VFX/action montage where
# every shot is the same palette. Benchmarked on such content it merged 12
# distinct shots into one 24s "shot" (2/12 cuts found), which is the worst
# failure mode here. `adaptive` and `ensemble` remain available per-project.
#
# `content`'s only downside is brief flash frames producing sub-shot slivers.
# PySceneDetect's ContentDetector has a built-in FlashFilter (MERGE) that folds
# scenes shorter than `min_scene_len` into their neighbour, so the fix is simply
# to use scenedetect's own default min length (15 frames / ~0.6s at 24fps) --
# the earlier 8-frame value was below the flash-merge window, so ~10-frame
# flashes survived as false cuts.
DEFAULT_CONTENT_THRESHOLD = 27.0
DEFAULT_ADAPTIVE_THRESHOLD = 3.0
DEFAULT_MIN_CONTENT_VAL = 15.0
DEFAULT_ENSEMBLE_CONTENT_THRESHOLD = 20.0
DEFAULT_EDGE_WEIGHT = 0.5
DEFAULT_FADE_THRESHOLD = 12
DEFAULT_MIN_SHOT_FRAMES = 15
DEFAULT_MIN_GAP_SEC = 0.33


@dataclass(frozen=True)
class DetectOptions:
    detector: str = "content"  # "content" | "adaptive" | "ensemble"
    content_threshold: float = DEFAULT_CONTENT_THRESHOLD
    adaptive_threshold: float = DEFAULT_ADAPTIVE_THRESHOLD
    min_content_val: float = DEFAULT_MIN_CONTENT_VAL
    ensemble_content_threshold: float = DEFAULT_ENSEMBLE_CONTENT_THRESHOLD
    edge_weight: float = DEFAULT_EDGE_WEIGHT
    detect_fades: bool = True
    fade_threshold: int = DEFAULT_FADE_THRESHOLD
    min_shot_frames: int = DEFAULT_MIN_SHOT_FRAMES
    min_gap_sec: float = DEFAULT_MIN_GAP_SEC

    def summary(self) -> dict[str, object]:
        return {"mode": self.detector, **asdict(self)}


def resolve_detection_input(
    *,
    master: Path | None,
    proxy: Path | None,
    run: RunPaths,
    auto_create: bool = True,
    force_proxy: bool = False,
    fps: float | None = None,
    width: int,
    crf: int,
    exr_transfer: str,
    timecode_start: str | None,
    frame_range: tuple[int, int] | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> tuple[Path, ProxyResult | None]:
    """Return the video scene detection is allowed to read.

    This is the only path into the detector, and it never yields the master
    unless the master is itself a detection-safe compressed video.

    ``frame_range`` names the program span inside a sequence master, so cut
    frames come back numbered from that first program frame rather than from
    whatever file happens to sort first in the directory. It does not apply to
    an already-compressed input: a supplied proxy, or a video master detected
    on directly, defines its own span.
    """
    if proxy is not None:
        if not proxy.exists():
            raise ValueError(
                f"configured proxy does not exist: {proxy}. Run `unrender shots proxy` "
                "or fix paths.proxy_master."
            )
        _ensure_detection_safe(proxy, ffprobe=ffprobe)
        return proxy, None

    if master is None:
        raise ValueError(
            "nothing to detect on: set paths.master (or --master), or provide a proxy "
            "via paths.proxy_master (or --proxy)"
        )
    if not force_proxy and master_usable_as_proxy(master, ffprobe=ffprobe):
        print(f"Master is already a compressed video; detecting on it: {master}", flush=True)
        return master, None
    if not auto_create:
        raise ValueError(
            f"detection needs a proxy for {master}. Run `unrender shots proxy`, set "
            "paths.proxy_master, or drop --no-proxy-create."
        )

    output = default_proxy_path(run, master)
    print(f"No proxy configured; creating one at {output}", flush=True)
    result = create_master_proxy(
        master=master,
        output=output,
        run=run,
        fps=fps,
        width=width,
        crf=crf,
        include_audio=False,
        exr_transfer=exr_transfer,
        timecode_start=timecode_start,
        frame_range=frame_range,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
    )
    return result.path, result


def detect_cut_frames(
    video: Path, *, options: DetectOptions, ffprobe: str = "ffprobe"
) -> list[int]:
    """Frame numbers where new shots begin (frame 0 excluded)."""
    _ensure_detection_safe(video, ffprobe=ffprobe)
    scenedetect = _import_scenedetect()
    manager = scenedetect.SceneManager()
    min_len = max(1, options.min_shot_frames)

    if options.detector == "ensemble":
        manager.add_detector(
            scenedetect.detectors.AdaptiveDetector(
                adaptive_threshold=options.adaptive_threshold,
                min_content_val=options.min_content_val,
                min_scene_len=min_len,
            )
        )
        content_detector = scenedetect.detectors.ContentDetector
        manager.add_detector(
            content_detector(
                threshold=options.ensemble_content_threshold,
                min_scene_len=min_len,
                luma_only=False,
                weights=content_detector.Components(
                    delta_hue=1.0,
                    delta_sat=1.0,
                    delta_lum=1.0,
                    delta_edges=options.edge_weight,
                ),
            )
        )
        if options.detect_fades:
            manager.add_detector(
                scenedetect.detectors.ThresholdDetector(
                    threshold=options.fade_threshold, min_scene_len=min_len
                )
            )
    elif options.detector == "adaptive":
        manager.add_detector(
            scenedetect.detectors.AdaptiveDetector(
                adaptive_threshold=options.adaptive_threshold,
                min_content_val=options.min_content_val,
                min_scene_len=min_len,
            )
        )
        if options.detect_fades:
            manager.add_detector(
                scenedetect.detectors.ThresholdDetector(
                    threshold=options.fade_threshold, min_scene_len=min_len
                )
            )
    elif options.detector == "content":
        manager.add_detector(
            scenedetect.detectors.ContentDetector(
                threshold=options.content_threshold, min_scene_len=min_len, luma_only=False
            )
        )
    else:
        raise ValueError(f"unknown detector {options.detector!r} (adaptive/content/ensemble)")

    stream = scenedetect.open_video(str(video), backend="opencv")
    manager.detect_scenes(video=stream, show_progress=True)
    scenes = manager.get_scene_list()
    return sorted({_frame_num(start) for start, _ in scenes[1:]})


def _frame_num(timecode) -> int:
    # scenedetect 0.7 exposes `frame_num`; 0.6 only has get_frames().
    value = getattr(timecode, "frame_num", None)
    return int(value) if value is not None else int(timecode.get_frames())


def build_shots(
    cut_frames: list[int],
    *,
    total_frames: int,
    fps: float,
    min_gap_sec: float = DEFAULT_MIN_GAP_SEC,
    padding: int = 3,
) -> list[CutRange]:
    """Turn cut frames into contiguous, end-exclusive shot ranges."""
    if total_frames <= 0:
        raise ValueError(f"master has no frames ({total_frames})")
    min_gap = max(1, seconds_to_frames(min_gap_sec, fps))
    kept: list[int] = []
    for frame in sorted(set(cut_frames)):
        if frame <= 0 or frame >= total_frames:
            continue
        anchor = kept[-1] if kept else 0
        if frame - anchor >= min_gap:
            kept.append(frame)
    if kept and total_frames - kept[-1] < min_gap:
        kept.pop()
    edges = [0, *kept, total_frames]
    return [
        CutRange(
            shot_id=f"{index:0{padding}d}",
            start_frame=edges[index - 1],
            end_frame=edges[index],
        )
        for index in range(1, len(edges))
    ]


def _ensure_detection_safe(path: Path, *, ffprobe: str) -> None:
    if path.is_dir():
        raise ValueError(
            f"scene detection cannot run on an image-sequence directory: {path}. "
            "Create a proxy first (`unrender shots proxy`)."
        )
    probe = probe_video(path, ffprobe=ffprobe)
    if probe.codec in INTRA_FRAME_CODECS:
        raise ValueError(
            f"scene detection must not run on a full-resolution {probe.codec} master "
            f"({path}). Create a proxy first (`unrender shots proxy`) or set "
            "paths.proxy_master."
        )


def _import_scenedetect():
    try:
        import scenedetect
    except ImportError as exc:  # pragma: no cover - exercised via message test
        raise ImportError(
            "PySceneDetect is required for shot detection. Install it with: "
            'pip install "unrender[detect]"'
        ) from exc
    return scenedetect
