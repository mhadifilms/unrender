"""Pillow-based static renderers for canonical audio-analysis features."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from unrender.audio.visualization.bundle import (
    AnalysisBundle,
    first_array,
    load_analysis_bundle,
)
from unrender.audio.visualization.mapping import VisualMapping, load_visual_mapping

BundleInput = AnalysisBundle | Mapping[str, Any] | str | Path
MappingInput = VisualMapping | Mapping[str, Any] | str | Path | None

_ROLE_ALIASES = {
    "dialogue": ("dialogue", "dialog", "dx", "speech", "voice"),
    "music": ("music", "mx"),
    "fx": ("fx", "effects", "sfx"),
}


def _rgb(color: str) -> tuple[int, int, int]:
    return ImageColor.getrgb(color)[:3]


def _rgba(color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    return (*_rgb(color), max(0, min(255, int(alpha))))


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _finite(array: np.ndarray | None) -> np.ndarray:
    if array is None:
        return np.zeros(1, dtype=np.float64)
    result = np.asarray(array, dtype=np.float64)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize(array: np.ndarray | None, *, symmetric: bool = False) -> np.ndarray:
    values = _finite(array)
    if not values.size:
        return np.zeros_like(values, dtype=np.float64)
    if symmetric:
        peak = float(np.max(np.abs(values)))
        return np.clip(values / peak, -1.0, 1.0) if peak > 0 else np.zeros_like(values)
    low = float(np.min(values))
    high = float(np.max(values))
    if low >= 0.0 and high <= 1.0:
        return np.clip(values, 0.0, 1.0)
    if math.isclose(low, high):
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _series(array: np.ndarray | None, width: int) -> np.ndarray:
    values = _finite(array).squeeze()
    if values.ndim == 0:
        values = values[None]
    if values.ndim > 1:
        values = np.mean(values, axis=tuple(range(1, values.ndim)))
    if values.size == width:
        return values
    if values.size <= 1:
        return np.full(width, float(values[0]) if values.size else 0.0)
    source = np.linspace(0.0, 1.0, values.size)
    target = np.linspace(0.0, 1.0, width)
    return np.interp(target, source, values)


def _matrix_image(
    matrix: np.ndarray,
    size: tuple[int, int],
    palette: Sequence[str],
    *,
    flip_y: bool = False,
) -> Image.Image:
    values = _normalize(matrix)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim > 2:
        values = np.mean(values, axis=tuple(range(2, values.ndim)))
    stops = [_rgb(color) for color in palette] or [(0, 0, 0), (255, 255, 255)]
    position = values * max(1, len(stops) - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, len(stops) - 1)
    mix = (position - lower)[..., None]
    colors = np.asarray(stops, dtype=np.float64)
    pixels = colors[lower] * (1.0 - mix) + colors[upper] * mix
    if flip_y:
        pixels = pixels[::-1]
    image = Image.fromarray(np.rint(pixels).astype(np.uint8), mode="RGB")
    return image.resize(size, Image.Resampling.BILINEAR)


def _role_activity(bundle: AnalysisBundle) -> tuple[np.ndarray, list[str]]:
    raw = bundle.get(
        "features.role_activity",
        "role_activity",
        "timeline.role_activity",
        default=None,
    )
    reference_keys = {"values", "data", "array", "path", "npy", "uri"}
    if isinstance(raw, Mapping) and not reference_keys.intersection(raw):
        arrays: list[np.ndarray] = []
        names: list[str] = []
        for canonical, aliases in _ROLE_ALIASES.items():
            match = next((raw[key] for key in aliases if key in raw), None)
            if match is not None:
                resolved = AnalysisBundle({"value": match}, bundle.base_dir).array("value")
                if resolved is not None:
                    arrays.append(_finite(resolved).reshape(-1))
                    names.append(canonical)
        for name, value in raw.items():
            if any(name in aliases for aliases in _ROLE_ALIASES.values()):
                continue
            resolved = AnalysisBundle({"value": value}, bundle.base_dir).array("value")
            if resolved is not None:
                arrays.append(_finite(resolved).reshape(-1))
                names.append(str(name))
        if arrays:
            length = max(array.size for array in arrays)
            columns = [_series(array, length) for array in arrays]
            return np.column_stack(columns), names
    array = first_array(
        bundle,
        ("features.role_activity", "role_activity", "timeline.role_activity"),
    )
    if array is None:
        return np.zeros((bundle.frame_count, 3)), ["dialogue", "music", "fx"]
    array = _finite(array)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim > 2:
        array = np.mean(array, axis=tuple(range(2, array.ndim)))
    defaults = ["dialogue", "music", "fx"]
    names = [
        defaults[index] if index < 3 else f"role {index + 1}" for index in range(array.shape[1])
    ]
    return array, names


def _lane_label(
    draw: ImageDraw.ImageDraw,
    label: str,
    box: tuple[int, int, int, int],
    mapping: VisualMapping,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=_rgba(mapping.colors["panel"], 230))
    draw.line((x0, y1, x1, y1), fill=_rgba(mapping.colors["grid"], 190))
    draw.text((10, y0 + 7), label, fill=mapping.colors["text"], font=_font())


def _timeline_image(
    bundle: AnalysisBundle,
    mapping: VisualMapping,
    width: int,
    height: int,
) -> Image.Image:
    image = Image.new("RGBA", (width, height), _rgba(mapping.colors["background"]))
    draw = ImageDraw.Draw(image, "RGBA")
    title_height = 42
    label_width = min(150, max(94, width // 7))
    bottom = 24
    lane_names = [
        "Role activity",
        "Timbral color",
        "Stereo space",
        "Masking",
        "Loudness / dynamics",
        "Dialogue / music / FX events",
    ]
    lane_height = max(1, (height - title_height - bottom) // len(lane_names))
    chart_x0 = label_width
    chart_width = max(1, width - chart_x0 - 1)
    duration = bundle.duration_sec
    draw.text((12, 13), "TIMELINE ATLAS", fill=mapping.colors["text"], font=_font())
    draw.text(
        (width - 125, 13),
        f"{duration:.2f} s",
        fill=mapping.colors["muted_text"],
        font=_font(),
    )

    for index, label in enumerate(lane_names):
        y0 = title_height + index * lane_height
        y1 = height - bottom if index == len(lane_names) - 1 else y0 + lane_height
        _lane_label(draw, label, (0, y0, width - 1, y1), mapping)
        for tick in range(11):
            x = chart_x0 + round(tick * (chart_width - 1) / 10)
            draw.line((x, y0, x, y1), fill=_rgba(mapping.colors["grid"], 90))

    # Role activity: stacked translucent areas.
    if mapping.layers.get("role_activity", True):
        activity, roles = _role_activity(bundle)
        activity = _normalize(activity)
        y0 = title_height
        y1 = y0 + lane_height
        palette = [
            mapping.colors.get(role, mapping.rules["role_activity"].palette[index % 3])
            for index, role in enumerate(roles)
        ]
        sampled = np.column_stack(
            [_series(activity[:, index], chart_width) for index in range(activity.shape[1])]
        )
        totals = np.maximum(1.0, np.sum(sampled, axis=1))
        sampled = sampled / totals[:, None]
        baseline = np.full(chart_width, y1 - 2.0)
        for index, color in enumerate(palette):
            top = baseline - sampled[:, index] * max(1, lane_height - 5)
            polygon = [(chart_x0 + x, float(baseline[x])) for x in range(chart_width)]
            polygon.extend((chart_x0 + x, float(top[x])) for x in range(chart_width - 1, -1, -1))
            draw.polygon(polygon, fill=_rgba(color, 190))
            baseline = top

    # Timbral color: RGB vectors are used directly; other shapes use a palette.
    if mapping.layers.get("timbral_color", True):
        timbre = first_array(
            bundle,
            ("features.timbral_color", "timbral_color", "features.timbre", "timbre"),
        )
        y0 = title_height + lane_height
        y1 = y0 + lane_height
        if timbre is not None and timbre.ndim >= 2 and timbre.shape[-1] >= 3:
            channels = np.column_stack(
                [_series(_normalize(timbre[..., index]), chart_width) for index in range(3)]
            )
            strip = np.rint(channels[None, :, :] * 255.0).astype(np.uint8)
            strip_image = Image.fromarray(strip, mode="RGB").resize(
                (chart_width, max(1, y1 - y0 - 3)), Image.Resampling.NEAREST
            )
        else:
            values = _series(_normalize(timbre), chart_width)[None, :]
            strip_image = _matrix_image(
                values,
                (chart_width, max(1, y1 - y0 - 3)),
                mapping.rules["timbral_color"].palette,
            )
        image.alpha_composite(strip_image.convert("RGBA"), (chart_x0, y0 + 2))

    # Stereo position and width.
    if mapping.layers.get("stereo_space", True):
        stereo = first_array(
            bundle,
            ("features.stereo_space", "stereo_space", "features.stereo", "stereo"),
        )
        y0 = title_height + lane_height * 2
        y1 = y0 + lane_height
        mid = (y0 + y1) / 2.0
        draw.line((chart_x0, mid, width, mid), fill=_rgba(mapping.colors["grid"], 180))
        if stereo is not None:
            stereo = _finite(stereo)
            pan = stereo[:, 0] if stereo.ndim > 1 else stereo
            spread = stereo[:, 1] if stereo.ndim > 1 and stereo.shape[1] > 1 else np.zeros_like(pan)
            pan = _series(_normalize(pan, symmetric=True), chart_width)
            spread = _series(_normalize(spread), chart_width)
            center = mid + pan * lane_height * 0.35
            radius = spread * lane_height * 0.28 + 1
            upper = [(chart_x0 + x, float(center[x] - radius[x])) for x in range(chart_width)]
            lower = [
                (chart_x0 + x, float(center[x] + radius[x])) for x in range(chart_width - 1, -1, -1)
            ]
            draw.polygon(
                upper + lower,
                fill=_rgba(mapping.rules["stereo_space"].palette[0], 120),
            )
            draw.line(
                [(chart_x0 + x, float(center[x])) for x in range(chart_width)],
                fill=_rgba(mapping.rules["stereo_space"].palette[0], 255),
                width=2,
            )

    # Masking heat strip.
    if mapping.layers.get("masking", True):
        masking = first_array(
            bundle,
            ("features.masking", "masking", "features.masking_score", "masking_score"),
        )
        y0 = title_height + lane_height * 3
        y1 = y0 + lane_height
        values = _series(_normalize(masking), chart_width)[None, :]
        strip = _matrix_image(
            values,
            (chart_width, max(1, y1 - y0 - 3)),
            mapping.rules["masking"].palette,
        )
        image.alpha_composite(strip.convert("RGBA"), (chart_x0, y0 + 2))

    # Loudness and dynamics line pair.
    if mapping.layers.get("loudness_dynamics", True):
        loudness = first_array(
            bundle,
            ("features.loudness", "loudness", "features.lufs", "lufs"),
        )
        dynamics = first_array(
            bundle,
            ("features.dynamics", "dynamics", "features.dynamic_range", "dynamic_range"),
        )
        y0 = title_height + lane_height * 4
        y1 = y0 + lane_height
        for values, color in (
            (loudness, mapping.colors["loudness"]),
            (dynamics, mapping.colors["dynamics"]),
        ):
            normalized = _series(_normalize(values), chart_width)
            points = [
                (chart_x0 + x, y1 - 3 - float(value) * max(1, lane_height - 7))
                for x, value in enumerate(normalized)
            ]
            draw.line(points, fill=_rgba(color), width=2)

    # Typed events.
    if mapping.layers.get("events", True):
        y0 = title_height + lane_height * 5
        y1 = height - bottom
        role_rows = {"dialogue": 0, "music": 1, "fx": 2}
        row_height = max(6, (y1 - y0 - 4) // 3)
        for event in bundle.events():
            role = str(event["role"]).lower()
            canonical = next(
                (
                    name
                    for name, aliases in _ROLE_ALIASES.items()
                    if role == name or role in aliases
                ),
                "fx",
            )
            x0 = chart_x0 + round(
                np.clip(float(event["start_sec"]) / duration, 0.0, 1.0) * chart_width
            )
            x1 = chart_x0 + round(
                np.clip(float(event["end_sec"]) / duration, 0.0, 1.0) * chart_width
            )
            ey0 = y0 + 2 + role_rows[canonical] * row_height
            color = mapping.colors[canonical]
            draw.rectangle((x0, ey0, max(x0 + 2, x1), ey0 + row_height - 2), fill=_rgba(color, 190))
            if x1 - x0 > 35:
                draw.text(
                    (x0 + 3, ey0 + 1),
                    str(event["label"])[:24],
                    fill=mapping.colors["text"],
                    font=_font(),
                )

    for tick in range(6):
        x = chart_x0 + round(tick * (chart_width - 1) / 5)
        draw.text(
            (x, height - 18),
            f"{duration * tick / 5:.1f}s",
            fill=mapping.colors["muted_text"],
            font=_font(),
        )
    return image


def render_timeline_atlas(
    bundle: BundleInput,
    output_path: str | Path,
    mapping: MappingInput = None,
    *,
    width: int | None = None,
    height: int | None = None,
) -> Path:
    """Render the semantic, whole-program timeline atlas."""

    loaded = load_analysis_bundle(bundle)
    visual = load_visual_mapping(mapping)
    width = int(width or visual.geometry["timeline_width"])
    height = int(height or visual.geometry["timeline_height"])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _timeline_image(loaded, visual, width, height).save(output, format="PNG")
    return output


def _structure_matrix(bundle: AnalysisBundle) -> np.ndarray:
    matrix = first_array(
        bundle,
        (
            "structure.self_similarity",
            "features.self_similarity",
            "self_similarity",
            "structure.similarity",
        ),
    )
    if matrix is not None:
        matrix = _finite(matrix).squeeze()
        if matrix.ndim == 2:
            return matrix
    embedding = first_array(
        bundle,
        ("structure.embedding", "features.embedding", "embedding", "features.timbre"),
    )
    if embedding is None:
        return np.eye(max(2, bundle.frame_count), dtype=np.float64)
    embedding = _finite(embedding)
    if embedding.ndim == 1:
        embedding = embedding[:, None]
    embedding = embedding.reshape(embedding.shape[0], -1)
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    normalized = embedding / np.maximum(norms, 1e-12)
    return normalized @ normalized.T


def _annotation_fraction(
    value: Any,
    bundle: AnalysisBundle,
    length: int,
    *,
    endpoint: str = "start",
) -> float | None:
    value_kind = "auto"
    if isinstance(value, Mapping):
        fraction_keys = (f"{endpoint}_fraction", "fraction", "progress")
        time_keys = (f"{endpoint}_sec", "time_sec", endpoint)
        frame_keys = (f"{endpoint}_frame", "frame", "index")
        raw = next((value[key] for key in fraction_keys if key in value), None)
        if raw is not None:
            value_kind = "fraction"
        else:
            raw = next((value[key] for key in time_keys if key in value), None)
            if raw is not None:
                value_kind = "time"
            else:
                raw = next((value[key] for key in frame_keys if key in value), None)
                value_kind = "frame"
    else:
        raw = value
    if raw is None:
        return None
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return None
    if value_kind == "fraction" and 0.0 <= number <= 1.0:
        return number
    if value_kind in {"auto", "time"} and 0.0 <= number <= bundle.duration_sec:
        return number / bundle.duration_sec
    if value_kind in {"auto", "frame"} and 0.0 <= number < length:
        return number / max(1, length - 1)
    return None


def _structure_image(
    bundle: AnalysisBundle,
    mapping: VisualMapping,
    width: int,
    height: int,
) -> Image.Image:
    image = Image.new("RGBA", (width, height), _rgba(mapping.colors["background"]))
    draw = ImageDraw.Draw(image, "RGBA")
    margin = max(24, min(70, width // 10))
    top = 42
    side = max(1, min(width - margin - 16, height - top - margin))
    matrix = _structure_matrix(bundle)
    heatmap = _matrix_image(
        matrix,
        (side, side),
        ("#020617", "#1d4ed8", "#a855f7", "#fef08a"),
        flip_y=True,
    )
    image.alpha_composite(heatmap.convert("RGBA"), (margin, top))
    draw.rectangle((margin, top, margin + side, top + side), outline=mapping.colors["grid"])
    draw.text((12, 13), "STRUCTURE / SELF-SIMILARITY", fill=mapping.colors["text"], font=_font())

    boundaries = bundle.get(
        "structure.boundaries",
        "boundaries",
        "annotations.boundaries",
        default=[],
    )
    if mapping.layers.get("boundaries", True) and isinstance(boundaries, Sequence):
        for boundary in boundaries:
            fraction = _annotation_fraction(boundary, bundle, matrix.shape[0])
            if fraction is None:
                continue
            x = margin + round(fraction * side)
            y = top + side - round(fraction * side)
            draw.line((x, top, x, top + side), fill=_rgba(mapping.colors["boundary"], 180))
            draw.line((margin, y, margin + side, y), fill=_rgba(mapping.colors["boundary"], 180))

    motifs = bundle.get("structure.motifs", "motifs", "annotations.motifs", default=[])
    if mapping.layers.get("motifs", True) and isinstance(motifs, Sequence):
        for index, motif in enumerate(motifs):
            if isinstance(motif, Mapping):
                start = _annotation_fraction(motif, bundle, matrix.shape[0])
                end = _annotation_fraction(
                    motif,
                    bundle,
                    matrix.shape[0],
                    endpoint="end",
                )
                label = str(motif.get("label") or motif.get("id") or f"M{index + 1}")
            else:
                start = _annotation_fraction(motif, bundle, matrix.shape[0])
                end = start
                label = f"M{index + 1}"
            if start is None:
                continue
            end = max(start, end if end is not None else start)
            x0 = margin + round(start * side)
            x1 = margin + round(end * side)
            draw.rectangle(
                (x0, top, max(x0 + 2, x1), top + side),
                outline=_rgba(mapping.colors["motif"], 210),
                width=2,
            )
            draw.text((x0 + 2, top + 2), label[:12], fill=mapping.colors["motif"], font=_font())
    draw.text(
        (margin, top + side + 8),
        "time →",
        fill=mapping.colors["muted_text"],
        font=_font(),
    )
    return image


def render_structure_map(
    bundle: BundleInput,
    output_path: str | Path,
    mapping: MappingInput = None,
    *,
    size: int | tuple[int, int] | None = None,
) -> Path:
    """Render a self-similarity matrix with structural annotations."""

    loaded = load_analysis_bundle(bundle)
    visual = load_visual_mapping(mapping)
    requested = size or int(visual.geometry["structure_size"])
    dimensions = (requested, requested) if isinstance(requested, int) else requested
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _structure_image(loaded, visual, int(dimensions[0]), int(dimensions[1])).save(
        output, format="PNG"
    )
    return output


def _dominance_matrix(bundle: AnalysisBundle) -> np.ndarray:
    dominance = first_array(
        bundle,
        (
            "mix.source_dominance",
            "features.source_dominance",
            "source_dominance",
            "mix.dominance",
        ),
    )
    bands = first_array(
        bundle,
        ("mix.band_power", "features.band_power", "band_power", "mix.band_powers"),
    )
    if dominance is None and bands is not None and bands.ndim >= 3:
        dominance = np.sum(_finite(bands), axis=-1)
    if dominance is None:
        return np.ones((bundle.frame_count, 1), dtype=np.float64)
    dominance = _finite(dominance).squeeze()
    if dominance.ndim == 1:
        rounded = np.rint(dominance).astype(int)
        if np.allclose(dominance, rounded) and np.min(rounded) >= 0 and np.max(rounded) < 32:
            result = np.zeros((dominance.size, int(np.max(rounded)) + 1))
            result[np.arange(dominance.size), rounded] = 1.0
            return result
        normalized = _normalize(dominance)
        return np.column_stack((1.0 - normalized, normalized))
    if dominance.ndim > 2:
        dominance = np.sum(dominance, axis=tuple(range(2, dominance.ndim)))
    dominance = np.maximum(0.0, dominance)
    totals = np.sum(dominance, axis=1, keepdims=True)
    return dominance / np.maximum(totals, 1e-12)


def _territory_image(
    bundle: AnalysisBundle,
    mapping: VisualMapping,
    width: int,
    height: int,
) -> Image.Image:
    image = Image.new("RGBA", (width, height), _rgba(mapping.colors["background"]))
    draw = ImageDraw.Draw(image, "RGBA")
    margin_x = max(80, width // 9)
    top = 42
    bottom = 26
    chart_width = max(1, width - margin_x - 12)
    chart_height = max(1, height - top - bottom)
    dominance_height = max(1, round(chart_height * 0.63))
    band_top = top + dominance_height + 8
    band_height = max(1, height - bottom - band_top)
    dominance = _dominance_matrix(bundle)
    sources = bundle.source_names(dominance.shape[1])
    palette = [
        mapping.colors["dialogue"],
        mapping.colors["music"],
        mapping.colors["fx"],
        "#22c55e",
        "#f472b6",
        "#f97316",
    ]
    sampled = np.column_stack(
        [_series(dominance[:, index], chart_width) for index in range(dominance.shape[1])]
    )
    sampled /= np.maximum(np.sum(sampled, axis=1, keepdims=True), 1e-12)
    baseline = np.full(chart_width, top + dominance_height, dtype=np.float64)
    for index, source in enumerate(sources):
        top_line = baseline - sampled[:, index] * dominance_height
        polygon = [(margin_x + x, float(baseline[x])) for x in range(chart_width)]
        polygon.extend((margin_x + x, float(top_line[x])) for x in range(chart_width - 1, -1, -1))
        draw.polygon(polygon, fill=_rgba(palette[index % len(palette)], 215))
        draw.text(
            (8, top + index * 14),
            source[:14],
            fill=palette[index % len(palette)],
            font=_font(),
        )
        baseline = top_line

    band_dominance = first_array(
        bundle,
        (
            "mix.source_dominance_by_band",
            "features.source_dominance_by_band",
            "source_dominance_by_band",
        ),
    )
    band_power = first_array(
        bundle,
        ("mix.band_power", "features.band_power", "band_power", "mix.band_powers"),
    )
    if band_dominance is not None and band_dominance.ndim == 3:
        territory = np.maximum(_finite(band_dominance), 0.0)
        territory /= np.maximum(np.sum(territory, axis=2, keepdims=True), 1e-12)
        source_colors = np.asarray(
            [_rgb(palette[index % len(palette)]) for index in range(territory.shape[2])],
            dtype=np.float64,
        )
        pixels = np.tensordot(territory, source_colors, axes=([2], [0]))
        pixels = np.clip(pixels.transpose(1, 0, 2)[::-1], 0, 255).astype(np.uint8)
        heatmap = Image.fromarray(pixels, mode="RGB").resize(
            (chart_width, band_height),
            Image.Resampling.NEAREST,
        )
    else:
        if band_power is None:
            band_power = dominance.T
        else:
            band_power = _finite(band_power)
            if band_power.ndim == 3:
                band_power = np.sum(band_power, axis=1)
            if band_power.ndim == 1:
                band_power = band_power[:, None]
            if band_power.shape[0] >= band_power.shape[1]:
                band_power = band_power.T
        heatmap = _matrix_image(
            band_power,
            (chart_width, band_height),
            ("#020617", "#0e7490", "#84cc16", "#facc15"),
            flip_y=True,
        )
    image.alpha_composite(heatmap.convert("RGBA"), (margin_x, band_top))
    draw.rectangle(
        (margin_x, top, margin_x + chart_width, top + dominance_height),
        outline=mapping.colors["grid"],
    )
    draw.rectangle(
        (margin_x, band_top, margin_x + chart_width, band_top + band_height),
        outline=mapping.colors["grid"],
    )
    draw.text((12, 13), "MIX TERRITORY", fill=mapping.colors["text"], font=_font())
    draw.text(
        (8, band_top + 4),
        "source\nterritory" if band_dominance is not None else "band\npower",
        fill=mapping.colors["muted_text"],
        font=_font(),
    )
    return image


def render_mix_territory(
    bundle: BundleInput,
    output_path: str | Path,
    mapping: MappingInput = None,
    *,
    width: int | None = None,
    height: int | None = None,
) -> Path:
    """Render source dominance and frequency-band power over time."""

    loaded = load_analysis_bundle(bundle)
    visual = load_visual_mapping(mapping)
    width = int(width or visual.geometry["territory_width"])
    height = int(height or visual.geometry["territory_height"])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _territory_image(loaded, visual, width, height).save(output, format="PNG")
    return output


def render_program_poster(
    bundle: BundleInput,
    output_path: str | Path,
    mapping: MappingInput = None,
    *,
    width: int | None = None,
    height: int | None = None,
) -> Path:
    """Combine timeline, structure, and mix territory into a summary poster."""

    loaded = load_analysis_bundle(bundle)
    visual = load_visual_mapping(mapping)
    width = int(width or visual.geometry["poster_width"])
    height = int(height or visual.geometry["poster_height"])
    image = Image.new("RGBA", (width, height), _rgba(visual.colors["background"]))
    draw = ImageDraw.Draw(image)
    margin = max(8, int(visual.geometry.get("margin", 24)))
    title_height = 50
    timeline_height = max(160, round((height - title_height - margin * 3) * 0.46))
    lower_height = max(120, height - title_height - timeline_height - margin * 3)
    lower_width = max(120, (width - margin * 3) // 2)
    timeline = _timeline_image(loaded, visual, width - margin * 2, timeline_height)
    structure = _structure_image(loaded, visual, lower_width, lower_height)
    territory = _territory_image(
        loaded,
        visual,
        width - margin * 3 - lower_width,
        lower_height,
    )
    image.alpha_composite(timeline, (margin, title_height))
    lower_y = title_height + timeline_height + margin
    image.alpha_composite(structure, (margin, lower_y))
    image.alpha_composite(territory, (margin * 2 + lower_width, lower_y))
    title = str(
        loaded.get("title", "metadata.title", "program.title", default="AUDIO ANALYSIS POSTER")
    )
    draw.text((margin, 18), title[:100], fill=visual.colors["text"], font=_font())
    draw.text(
        (width - 150, 18),
        f"{loaded.duration_sec:.2f} seconds",
        fill=visual.colors["muted_text"],
        font=_font(),
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")
    return output
