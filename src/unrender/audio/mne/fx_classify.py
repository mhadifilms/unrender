"""FX sub-classification: split the FX stem into AMB/FOLEY/HFX/DSGN tracks.

AudioShake has no ambience/foley/design model, so events cut from the FX stem
are labelled with local CLAP zero-shot tagging
(``laion/clap-htsat-unfused`` via ``transformers``) against per-category text
prompts. The CLAP/transformers/torch imports are deferred into
:func:`classify_events_clap` so this module (and the base package) imports with
no heavy dependencies installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from unrender.lib.audio import read_wav_float, write_wav_float
from unrender.lib.audio_features import (
    Segment,
    activity_segments,
    as_float,
    equal_power_mask,
    resample_mono,
    split_windows,
    to_mono,
)
from unrender.lib.fingerprints import input_fingerprints, warn_if_inputs_changed
from unrender.manifests import read_json, write_json
from unrender.project import RunPaths

# CLAP's feature extractor is trained at a fixed input rate; clips are
# resampled to this before tagging regardless of the stem's native rate.
CLAP_SAMPLE_RATE = 48_000
# Fixed hop for the activity gate that carves the stem into events; the
# classification window length is a separate, larger setting.
_SEGMENT_HOP_SEC = 0.1

# key, stem suffix, and the zero-shot text prompts describing the category.
FX_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "ambience",
        "AMB",
        (
            "ambient background noise",
            "wind blowing outdoors",
            "distant city traffic",
            "crowd murmur and walla",
        ),
    ),
    (
        "foley",
        "FOLEY",
        (
            "footsteps on a floor",
            "rustling cloth and clothing",
            "handling a small prop object",
        ),
    ),
    (
        "hard_fx",
        "HFX",
        (
            "a door slamming",
            "a gunshot",
            "a car passing by",
            "a loud impact sound effect",
        ),
    ),
    (
        "design_fx",
        "DSGN",
        (
            "a synthesized riser",
            "a whoosh transition",
            "a sci-fi drone",
            "a designed sound effect sting",
        ),
    ),
)
_DEFAULT_CATEGORY = "ambience"
_SUFFIX_BY_KEY = {key: suffix for key, suffix, _ in FX_CATEGORIES}


@dataclass(frozen=True)
class FxClassifySettings:
    clap_model: str = "laion/clap-htsat-unfused"
    device: str = "auto"
    activity_threshold_db: float = -50.0
    # Length of each CLAP classification window (long events are chunked into
    # windows of this size and their scores averaged).
    window_sec: float = 5.0
    min_event_sec: float = 0.5
    merge_gap_sec: float = 0.4
    crossfade_ms: float = 20.0


@dataclass
class FxEvent:
    start_sec: float
    end_sec: float
    label: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)


def classify_fx(
    *,
    run: RunPaths,
    fx_stem: str | Path,
    settings: FxClassifySettings | None = None,
    prefix: str,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Segment, CLAP-classify, and render per-category FX tracks."""
    run.ensure()
    active = settings or FxClassifySettings()
    source = Path(fx_stem).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"FX stem not found: {source}")
    output_json = run.mne_fx_classification_json
    inputs = {"fx_stem": source, "clap_model": active.clap_model}

    if output_json.exists() and not force:
        data = read_json(output_json)
        warn_if_inputs_changed(data, inputs, output_json)
        cached_model = data.get("clap_model") if isinstance(data, dict) else None
        if cached_model and cached_model != active.clap_model:
            print(
                f"  WARNING: clap_model changed ({cached_model} -> {active.clap_model}) since "
                f"{output_json.name} was written. Rerun with --force to reclassify.",
                flush=True,
            )
        print(f"FX classification already exists, skipping: {output_json}", flush=True)
        return data

    if dry_run:
        print(f"DRY RUN: would classify FX stem into AMB/FOLEY/HFX/DSGN: {source}", flush=True)
        print(f"DRY RUN: would write category stems to {run.mne_fx_dir}", flush=True)
        return {}

    audio, sample_rate, sample_width = read_wav_float(source)
    mono = to_mono(audio)
    total_sec = mono.size / float(sample_rate) if sample_rate else 0.0

    raw_events = activity_segments(
        mono,
        sample_rate=sample_rate,
        threshold_db=active.activity_threshold_db,
        hop_sec=_SEGMENT_HOP_SEC,
        merge_gap_sec=active.merge_gap_sec,
        min_duration_sec=active.min_event_sec,
    )
    events = _classify_segments(mono, sample_rate, raw_events, active)
    events = _smooth_labels(events)

    stems = _render_category_stems(
        audio,
        sample_rate=sample_rate,
        sample_width=sample_width,
        events=events,
        crossfade_ms=active.crossfade_ms,
        output_dir=run.mne_fx_dir,
        prefix=prefix,
    )

    payload: dict[str, Any] = {
        "version": "1.0",
        "backend": "clap",
        "source": str(source),
        "clap_model": active.clap_model,
        "format": "wav",
        "output_dir": str(run.mne_fx_dir),
        "duration_sec": round(total_sec, 3),
        "default_category": _DEFAULT_CATEGORY,
        "inputs": input_fingerprints(inputs),
        "categories": {key: str(path) for key, path in stems.items()},
        "events": [
            {
                "start_sec": round(event.start_sec, 3),
                "end_sec": round(event.end_sec, 3),
                "label": event.label,
                "confidence": round(event.confidence, 4),
                "scores": {k: round(v, 4) for k, v in event.scores.items()},
            }
            for event in events
        ],
    }
    write_json(output_json, payload)
    print(
        f"FX classification saved: {output_json} ({len(events)} event(s), "
        f"{len(stems)} track(s))",
        flush=True,
    )
    return payload


def classify_events_clap(
    clips: list[np.ndarray],
    *,
    sample_rate: int,
    model_name: str,
    device: str,
) -> list[dict[str, float]]:
    """Zero-shot CLAP category scores for each mono clip.

    Returns one ``{category_key: probability}`` dict per clip. Imports of
    ``torch``/``transformers`` happen here so the module stays importable
    without the ``mne`` extra; tests monkeypatch this function.
    """
    if not clips:
        return []
    try:
        import torch
        from transformers import ClapModel, ClapProcessor
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError("FX classification requires the 'mne' extra: pip install .[mne]") from exc

    resolved = _resolve_device(torch, device)
    model = ClapModel.from_pretrained(model_name).to(resolved)
    model.eval()
    processor = ClapProcessor.from_pretrained(model_name)
    # Apply CLAP's learned logit scale so softmax probabilities are calibrated;
    # raw cosine similarities (~+/-0.3) otherwise collapse to a near-uniform
    # distribution and make the confidences meaningless.
    logit_scale = (
        float(model.logit_scale_a.exp().item()) if hasattr(model, "logit_scale_a") else 1.0
    )

    prompts: list[str] = []
    prompt_owner: list[int] = []
    for index, (_key, _suffix, texts) in enumerate(FX_CATEGORIES):
        for text in texts:
            prompts.append(text)
            prompt_owner.append(index)

    results: list[dict[str, float]] = []
    with torch.no_grad():
        text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(resolved)
        text_embeds = model.get_text_features(**text_inputs)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        for clip in clips:
            resampled = resample_mono(
                np.asarray(clip, dtype=np.float32),
                orig_sr=sample_rate,
                target_sr=CLAP_SAMPLE_RATE,
            )
            audio_inputs = processor(
                audios=[resampled],
                sampling_rate=CLAP_SAMPLE_RATE,
                return_tensors="pt",
            ).to(resolved)
            audio_embed = model.get_audio_features(**audio_inputs)
            audio_embed = audio_embed / audio_embed.norm(dim=-1, keepdim=True)
            sims = (audio_embed @ text_embeds.T).reshape(-1) * logit_scale
            category_scores = torch.full((len(FX_CATEGORIES),), float("-inf"), device=resolved)
            for prompt_index, owner in enumerate(prompt_owner):
                category_scores[owner] = torch.maximum(category_scores[owner], sims[prompt_index])
            probs = torch.softmax(category_scores, dim=0).cpu().numpy()
            results.append(
                {FX_CATEGORIES[i][0]: float(probs[i]) for i in range(len(FX_CATEGORIES))}
            )
    return results


def _classify_segments(
    mono: np.ndarray,
    sample_rate: int,
    segments: list[Segment],
    settings: FxClassifySettings,
) -> list[FxEvent]:
    if not segments:
        return []
    # Window each (potentially long) event and remember which event owns each
    # window so per-window scores can be averaged back per event.
    windows: list[np.ndarray] = []
    owners: list[int] = []
    for index, seg in enumerate(segments):
        start = int(seg.start_sec * sample_rate)
        end = max(start + 1, int(seg.end_sec * sample_rate))
        for window in split_windows(
            mono[start:end], sample_rate=sample_rate, window_sec=settings.window_sec
        ):
            windows.append(window)
            owners.append(index)
    scored = classify_events_clap(
        windows,
        sample_rate=sample_rate,
        model_name=settings.clap_model,
        device=settings.device,
    )
    sums: list[dict[str, float]] = [{} for _ in segments]
    counts = [0 for _ in segments]
    for owner, window_scores in zip(owners, scored, strict=True):
        counts[owner] += 1
        for key, value in window_scores.items():
            sums[owner][key] = sums[owner].get(key, 0.0) + value

    events: list[FxEvent] = []
    for index, segment in enumerate(segments):
        if counts[index] > 0 and sums[index]:
            scores = {key: value / counts[index] for key, value in sums[index].items()}
            label = max(scores, key=lambda key: scores[key])
            confidence = float(scores[label])
        else:
            scores, label, confidence = {}, _DEFAULT_CATEGORY, 0.0
        events.append(
            FxEvent(
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                label=label,
                confidence=confidence,
                scores=scores,
            )
        )
    return events


def _smooth_labels(events: list[FxEvent]) -> list[FxEvent]:
    """Reassign a lone event whose both neighbours share a different label."""
    if len(events) < 3:
        return events
    labels = [event.label for event in events]
    for index in range(1, len(events) - 1):
        prev_label = labels[index - 1]
        next_label = labels[index + 1]
        if prev_label == next_label and events[index].label != prev_label:
            events[index].label = prev_label
    return events


def _render_category_stems(
    audio: np.ndarray,
    *,
    sample_rate: int,
    sample_width: int,
    events: list[FxEvent],
    crossfade_ms: float,
    output_dir: Path,
    prefix: str,
) -> dict[str, Path]:
    sample_count = audio.shape[0]
    segments_by_key: dict[str, list[Segment]] = {key: [] for key, _, _ in FX_CATEGORIES}
    covered: list[Segment] = []
    for event in events:
        segments_by_key.setdefault(event.label, []).append(Segment(event.start_sec, event.end_sec))
        if event.label != _DEFAULT_CATEGORY:
            covered.append(Segment(event.start_sec, event.end_sec))
    # Continuous low-level residual (everything not claimed by another
    # category) defaults to ambience.
    total_sec = sample_count / float(sample_rate) if sample_rate else 0.0
    segments_by_key[_DEFAULT_CATEGORY] = _residual_ambience(
        segments_by_key.get(_DEFAULT_CATEGORY, []), covered, total_sec
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stems: dict[str, Path] = {}
    for key, suffix, _ in FX_CATEGORIES:
        mask = equal_power_mask(
            sample_count,
            sample_rate=sample_rate,
            segments=segments_by_key.get(key, []),
            crossfade_ms=crossfade_ms,
        )
        track = audio * mask[:, None]
        target = output_dir / f"{prefix}_{suffix}_stem.wav"
        write_wav_float(target, track, sample_rate, sample_width=sample_width)
        stems[key] = target
    return stems


def _residual_ambience(
    ambience_events: list[Segment], covered: list[Segment], total_sec: float
) -> list[Segment]:
    """Ambience keeps its own events plus every uncovered residual region."""
    from unrender.lib.audio_features import gap_segments, merge_segments

    residual = gap_segments(sorted(covered, key=lambda s: s.start_sec), total_sec=total_sec)
    return merge_segments([*ambience_events, *residual])


def _resolve_device(torch: Any, value: str) -> Any:
    requested = value.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def settings_from_values(
    *,
    clap_model: str | None = None,
    device: str | None = None,
    activity_threshold_db: float | str | None = None,
    window_sec: float | str | None = None,
    min_event_sec: float | str | None = None,
    merge_gap_sec: float | str | None = None,
    crossfade_ms: float | str | None = None,
) -> FxClassifySettings:
    base = FxClassifySettings()
    return FxClassifySettings(
        clap_model=clap_model or base.clap_model,
        device=device or base.device,
        activity_threshold_db=as_float(activity_threshold_db, base.activity_threshold_db),
        window_sec=as_float(window_sec, base.window_sec),
        min_event_sec=as_float(min_event_sec, base.min_event_sec),
        merge_gap_sec=as_float(merge_gap_sec, base.merge_gap_sec),
        crossfade_ms=as_float(crossfade_ms, base.crossfade_ms),
    )
