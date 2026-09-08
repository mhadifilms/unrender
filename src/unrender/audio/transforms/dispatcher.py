from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ._core import (
    LoadedRole,
    RecipeProducts,
    coerce_settings,
    load_analysis_bundle,
    load_roles,
    materialize_products,
    normalize_role_paths,
)
from .contracts import RecipeContract, RecipeResult, RecipeSettings, RoleInput
from .corrective import CORRECTIVE_CONTRACTS, CORRECTIVE_RENDERERS
from .creative import CREATIVE_CONTRACTS, CREATIVE_RENDERERS

Renderer = Callable[
    [Mapping[str, LoadedRole], RecipeSettings, Mapping[str, Any], Mapping[str, Any]],
    RecipeProducts,
]

_CONTRACTS = {**CORRECTIVE_CONTRACTS, **CREATIVE_CONTRACTS}
_RENDERERS: dict[str, Renderer] = {**CORRECTIVE_RENDERERS, **CREATIVE_RENDERERS}

RECIPE_REGISTRY: Mapping[str, RecipeContract] = MappingProxyType(_CONTRACTS)
RECIPES = RECIPE_REGISTRY

_ALIASES = {
    "ducking": "dialogue_aware_ducking",
    "dialogue_ducking": "dialogue_aware_ducking",
    "dialog_aware_ducking": "dialogue_aware_ducking",
    "unmasking": "spectral_unmasking",
    "bounded_spectral_unmasking": "spectral_unmasking",
    "rebalance": "dme_rebalance",
    "d_m_e_rebalance": "dme_rebalance",
    "perspective": "perspective_matching",
    "perspective_match": "perspective_matching",
    "review_regions": "review_region_variants",
    "review_region_extraction": "review_region_variants",
    "timbre_filter": "timbre_filter_modulation",
    "granular": "granular_resynthesis",
    "seeded_granular_resynthesis": "granular_resynthesis",
    "spatial_animation": "stereo_spatial_animation",
    "stereo_animation": "stereo_spatial_animation",
    "event_modulation": "event_beat_modulation",
    "beat_modulation": "event_beat_modulation",
    "cross_stem": "cross_stem_mapping",
}

_PARAMETER_ALIASES = {
    "dme_rebalance": {
        "d_gain_db": "dialogue_gain_db",
        "dx_gain_db": "dialogue_gain_db",
        "m_gain_db": "music_gain_db",
        "mx_gain_db": "music_gain_db",
        "e_gain_db": "effects_gain_db",
        "fx_gain_db": "effects_gain_db",
        "target_lufs": "target_loudness_lufs",
    },
    "dialogue_aware_ducking": {
        "max_reduction_db": "depth_db",
        "ducking_depth_db": "depth_db",
    },
    "spectral_unmasking": {
        "reduction_db": "max_reduction_db",
    },
    "perspective_matching": {
        "room_blend": "room_tone_blend",
    },
}


def list_recipes(*, category: str | None = None) -> tuple[str, ...]:
    """Return stable recipe names, optionally filtered by category."""
    names = (
        name
        for name, contract in RECIPE_REGISTRY.items()
        if category is None or contract.category == category
    )
    return tuple(sorted(names))


def get_recipe_contract(name: str) -> RecipeContract:
    canonical = _canonical_name(name)
    return RECIPE_REGISTRY[canonical]


def run_recipe(
    name: str,
    role_paths: Mapping[str, str | Path] | Sequence[RoleInput],
    output_dir: str | Path,
    settings: RecipeSettings | Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | None = None,
    analysis_bundle: Mapping[str, Any] | str | Path | None = None,
    *,
    parameters: Mapping[str, Any] | None = None,
) -> RecipeResult:
    """Render a named recipe without mutating any source.

    ``settings`` controls dry/wet, seed, peak ceiling, and WAV subtype. Recipe
    controls belong in ``params`` (or the equivalent ``parameters`` keyword).
    Unknown controls and out-of-range values are rejected before any output is
    written.
    """
    if params is not None and parameters is not None:
        raise ValueError("pass either params or parameters, not both")
    supplied_params = params if parameters is None else parameters
    canonical = _canonical_name(name)
    contract = RECIPE_REGISTRY[canonical]
    resolved_settings, recipe_params = coerce_settings(settings, supplied_params)
    recipe_params = _normalize_parameter_aliases(canonical, recipe_params)
    resolved_params = contract.resolve_parameters(recipe_params)
    normalized_paths = normalize_role_paths(role_paths)
    loaded_roles = load_roles(normalized_paths, contract)
    analysis = load_analysis_bundle(analysis_bundle)
    products = _RENDERERS[canonical](
        loaded_roles,
        resolved_settings,
        resolved_params,
        analysis,
    )
    outputs, automation, manifest, metrics, provenance = materialize_products(
        products=products,
        contract=contract,
        settings=resolved_settings,
        parameters=resolved_params,
        roles=loaded_roles,
        output_dir=Path(output_dir),
        analysis_bundle=analysis,
    )
    return RecipeResult(
        name=canonical,
        contract=contract,
        output_dir=Path(output_dir).expanduser().resolve(),
        outputs=MappingProxyType(outputs),
        automation=MappingProxyType(automation),
        manifest=manifest,
        metrics=metrics,
        provenance=provenance,
        report=products.report,
        latency_samples=products.latency_samples,
        tail_samples=products.tail_samples,
    )


def _canonical_name(name: str) -> str:
    normalized = name.strip().lower().replace("/", "_").replace("-", "_").replace(" ", "_")
    normalized = "_".join(part for part in normalized.split("_") if part)
    canonical = _ALIASES.get(normalized, normalized)
    if canonical not in RECIPE_REGISTRY:
        available = ", ".join(list_recipes())
        raise KeyError(f"unknown audio transform recipe '{name}'; available: {available}")
    return canonical


def _normalize_parameter_aliases(name: str, params: Mapping[str, Any]) -> dict[str, Any]:
    aliases = _PARAMETER_ALIASES.get(name, {})
    normalized: dict[str, Any] = {}
    for key, value in params.items():
        canonical = aliases.get(key, key)
        if canonical in normalized:
            raise ValueError(f"parameter supplied more than once: {canonical}")
        normalized[canonical] = value
    return normalized
