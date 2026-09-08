"""Per-shot DX stems cut from the full dialogue stem.

The full DX stem is resolved in priority order: an explicitly given path, the
dialogue stem produced by source separation (AudioShake or BandIt), or — when
neither exists because the pipeline started from already-separated speaker
stems — a stem merged by summing the separated stems. Each shot is then cut
to its exact timecode window, producing one DX clip per shot that can be laid
onto a single timeline track.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from unrender.editorial.shots.stems import StemSource
from unrender.lib.audio import mix_wav_files
from unrender.lib.ffmpeg import run_ffmpeg_slice
from unrender.lib.fingerprints import input_fingerprints, warn_if_inputs_changed
from unrender.lib.names import safe_name
from unrender.manifests import ShotRecord, read_json, write_json
from unrender.project import RunPaths


def resolve_dx_stem(
    run: RunPaths,
    *,
    explicit: Path | None = None,
    merge_sources: list[StemSource] | None = None,
    force: bool = False,
) -> Path:
    """Locate the full DX stem, merging separated stems as a last resort."""
    if explicit is not None:
        path = explicit.expanduser()
        if not path.exists():
            raise FileNotFoundError(f"DX stem not found: {path}")
        return path
    if run.source_separation_json.exists():
        data = read_json(run.source_separation_json)
        raw = str(data.get("dialogue_stem") or "").strip()
        if raw:
            path = Path(raw).expanduser()
            if path.exists():
                return path
    if merge_sources:
        merged = run.merged_dx_stem
        if merged.exists() and not force:
            print(f"Merged DX stem already exists, skipping: {merged}", flush=True)
            return merged
        print(
            f"No DX stem found; merging {len(merge_sources)} separated speaker stem(s)...",
            flush=True,
        )
        mix_wav_files([source.path for source in merge_sources], merged)
        print(f"Merged DX stem saved: {merged}", flush=True)
        return merged
    raise ValueError(
        "no DX stem available: pass --dx-stem, run `unrender audio separate` first, "
        "or provide separated speaker stems to merge"
    )


def build_shot_dx(
    *,
    run: RunPaths,
    shots: list[ShotRecord],
    dx_stem: Path,
    output_json: Path | None = None,
    force: bool = False,
    ffmpeg: str = "ffmpeg",
) -> list[dict[str, Any]]:
    """Cut the full DX stem into one clip per shot, exact to shot timecode."""
    if not shots:
        raise ValueError("no shots provided")
    plan_json = output_json or run.shot_dx_plan_json
    inputs = {"dx_stem": dx_stem}
    if plan_json.exists() and not force:
        print(f"Per-shot DX plan already exists, skipping: {plan_json}", flush=True)
        data = read_json(plan_json)
        warn_if_inputs_changed(data, inputs, plan_json)
        return list(data.get("shots") or [])

    run.shot_dx_dir.mkdir(parents=True, exist_ok=True)
    plan: list[dict[str, Any]] = []
    for shot in shots:
        if shot.start_sec is None or shot.end_sec is None:
            raise ValueError(f"shot {shot.shot_id} is missing start_sec/end_sec")
        if shot.end_sec <= shot.start_sec:
            raise ValueError(f"shot {shot.shot_id} has non-positive duration")
        target = run.shot_dx_dir / f"{safe_name(shot.shot_id)}_DX_stem.wav"
        print(
            f"  shot DX {shot.shot_id}: {shot.start_sec:.3f}-{shot.end_sec:.3f}",
            flush=True,
        )
        if force or not target.exists():
            run_ffmpeg_slice(
                ffmpeg=ffmpeg,
                source=dx_stem,
                target=target,
                start_sec=shot.start_sec,
                end_sec=shot.end_sec,
            )
        plan.append(
            {
                "shot_id": shot.shot_id,
                "stem_path": str(target),
                "start_sec": shot.start_sec,
                "end_sec": shot.end_sec,
            }
        )

    write_json(
        plan_json,
        {
            "version": "1.0",
            "dx_stem": str(dx_stem),
            "output_dir": str(run.shot_dx_dir),
            "inputs": input_fingerprints(inputs),
            "shots": plan,
        },
    )
    print(f"Per-shot DX plan saved: {plan_json} ({len(plan)} shot(s))", flush=True)
    return plan
