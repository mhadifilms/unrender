"""Narrative scene grouping: cluster detected shots into script-level scenes.

Runs after ``unrender shots detect``/``shots cut`` as an optional stage. The
default engine scores every cut with cheap local features (CLIP + DINOv2
keyframe embeddings, audio band energies, dialogue continuity) fused by
pretrained logistic weights.
"""

from unrender.editorial.scenes.group import (
    FUSION_WEIGHTS,
    BoundaryScores,
    assign_scenes,
    dp_persistence_scores,
    fuse_features,
    novelty_scores,
)
from unrender.editorial.scenes.manifest import (
    load_scene_manifest,
    scene_for_time,
    scene_manifest_entries,
    write_scenes_manifest,
)

__all__ = [
    "FUSION_WEIGHTS",
    "BoundaryScores",
    "assign_scenes",
    "dp_persistence_scores",
    "fuse_features",
    "load_scene_manifest",
    "novelty_scores",
    "scene_for_time",
    "scene_manifest_entries",
    "write_scenes_manifest",
]
