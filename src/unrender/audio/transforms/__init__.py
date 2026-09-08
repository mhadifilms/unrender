"""Non-destructive corrective and creative audio transformation recipes."""

from .contracts import (
    ParameterSpec,
    RecipeContract,
    RecipeResult,
    RecipeSettings,
    RoleInput,
    RoleInputSpec,
    TransformSettings,
)
from .corrective import (
    DIALOGUE_AWARE_DUCKING,
    DME_REBALANCE,
    PERSPECTIVE_MATCHING,
    REVIEW_REGION_VARIANTS,
    SPECTRAL_UNMASKING,
)
from .creative import (
    CROSS_STEM_MAPPING,
    EVENT_BEAT_MODULATION,
    GRANULAR_RESYNTHESIS,
    STEREO_SPATIAL_ANIMATION,
    TIMBRE_FILTER_MODULATION,
)
from .dispatcher import (
    RECIPE_REGISTRY,
    RECIPES,
    get_recipe_contract,
    list_recipes,
    run_recipe,
)

__all__ = [
    "CROSS_STEM_MAPPING",
    "DIALOGUE_AWARE_DUCKING",
    "DME_REBALANCE",
    "EVENT_BEAT_MODULATION",
    "GRANULAR_RESYNTHESIS",
    "PERSPECTIVE_MATCHING",
    "RECIPES",
    "RECIPE_REGISTRY",
    "REVIEW_REGION_VARIANTS",
    "SPECTRAL_UNMASKING",
    "STEREO_SPATIAL_ANIMATION",
    "TIMBRE_FILTER_MODULATION",
    "ParameterSpec",
    "RecipeContract",
    "RecipeResult",
    "RecipeSettings",
    "RoleInput",
    "RoleInputSpec",
    "TransformSettings",
    "get_recipe_contract",
    "list_recipes",
    "run_recipe",
]
