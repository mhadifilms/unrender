"""Boundary scoring and scene assignment over per-shot features.

Everything here is plain numpy on already-extracted features so it stays
unit-testable without torch. A boundary score near 1.0 after shot ``i``
means "shot i is the last shot of a scene" (MovieNet convention).

The fusion weights are logistic-regression coefficients fit on hand-labeled
scene boundaries of two open films (Tears of Steel, Sintel) segmented by
*this repo's own* `shots detect` at default settings (adaptive detector;
131 + 168 shots, 26 boundaries — every hand-labeled boundary exists as a
detected cut), over exactly the features computed by
:mod:`unrender.editorial.scenes.features`. Biases are recentered so DEFAULT_THRESHOLD
sits at the calibrated joint-F1 optimum (raise the threshold for coarser
grouping). Cross-film transfer of this feature set measures F1 ~= 0.5 /
mIoU ~= 0.55 against an unseen film's hand labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Weight vectors (bias last) keyed by available-feature variant. Feature
# order matches FUSION_FEATURES with unavailable columns dropped.
FUSION_FEATURES = ("clip_dp", "clip_nov", "dino_nov", "audio_nov", "dlg_span")
FUSION_WEIGHTS: dict[str, list[float]] = {
    "full": [3.8306, -0.1764, 4.4968, 2.6071, -0.177, -3.9381],
    "no_audio": [3.6853, -0.4299, 4.1534, -0.288, -3.0746],
    "no_dialogue": [3.8298, -0.1709, 4.4887, 2.6242, -3.962],
    "visual_only": [3.6778, -0.425, 4.1488, -3.1971],
}
DEFAULT_THRESHOLD = 0.6
NOVELTY_WINDOW = 4


@dataclass(frozen=True)
class BoundaryScores:
    """Per-cut boundary scores plus the variant used to produce them."""

    scores: np.ndarray  # [n_shots]; scores[i] = P(scene ends after shot i)
    variant: str
    engine: str
    reasons: list[str] = field(default_factory=list)


def _norm01(values: np.ndarray) -> np.ndarray:
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def novelty_scores(embeddings: np.ndarray, window: int = NOVELTY_WINDOW) -> np.ndarray:
    """TextTiling-style novelty: distance between mean embeddings of the
    ``window`` shots before and after each cut. Robust to shot/reverse-shot
    coverage because both sides of an intra-scene cut average over the same
    location."""
    count = len(embeddings)
    scores = np.zeros(count, dtype=np.float32)
    for index in range(count - 1):
        before = embeddings[max(0, index - window + 1) : index + 1].mean(0)
        after = embeddings[index + 1 : min(count, index + 1 + window)].mean(0)
        before = before / (np.linalg.norm(before) + 1e-9)
        after = after / (np.linalg.norm(after) + 1e-9)
        scores[index] = 1.0 - float(before @ after)
    return _norm01(scores)


def _dp_partition(embeddings: np.ndarray, penalty: float, max_len: int = 60) -> list[int]:
    """Optimal partition minimizing within-segment scatter + penalty/segment."""
    count = len(embeddings)
    csum = np.vstack([np.zeros(embeddings.shape[1]), np.cumsum(embeddings, axis=0)])

    def scatter(a: int, b: int) -> float:  # segment [a, b)
        centroid = csum[b] - csum[a]
        norm = np.linalg.norm(centroid) + 1e-9
        # unit-norm rows: sum(1 - cos) = (b - a) - rows . centroid/|centroid|
        return (b - a) - float(centroid @ (centroid / norm))

    best = np.full(count + 1, np.inf)
    prev = np.zeros(count + 1, dtype=int)
    best[0] = 0.0
    for end in range(1, count + 1):
        for start in range(max(0, end - max_len), end):
            cost = best[start] + scatter(start, end) + penalty
            if cost < best[end]:
                best[end], prev[end] = cost, start
    bounds = []
    end = count
    while end > 0:
        start = prev[end]
        if end < count:
            bounds.append(end - 1)
        end = start
    return sorted(bounds)


def dp_persistence_scores(
    embeddings: np.ndarray, penalties: np.ndarray | None = None
) -> np.ndarray:
    """Fraction of segmentation penalties whose optimal partition cuts after
    each shot — a well-calibrated continuous boundary score (LGSS-style
    global grouping, swept over the scene-count prior)."""
    count = len(embeddings)
    if penalties is None:
        penalties = np.geomspace(0.05, 3.0, 25)
    votes = np.zeros(count, dtype=np.float32)
    for penalty in penalties:
        for bound in _dp_partition(embeddings, float(penalty)):
            votes[bound] += 1.0
    return votes / len(penalties)


def fuse_features(
    *,
    clip_embeddings: np.ndarray,
    dino_embeddings: np.ndarray,
    audio_features: np.ndarray | None = None,
    dialogue_spans: np.ndarray | None = None,
    novelty_window: int = NOVELTY_WINDOW,
) -> BoundaryScores:
    """Score every cut by fusing the available features with the pretrained
    weights for that availability variant."""
    columns = {
        "clip_dp": dp_persistence_scores(clip_embeddings),
        "clip_nov": novelty_scores(clip_embeddings, window=novelty_window),
        "dino_nov": novelty_scores(dino_embeddings, window=novelty_window),
    }
    if audio_features is not None:
        normed = audio_features / (np.linalg.norm(audio_features, axis=1, keepdims=True) + 1e-9)
        columns["audio_nov"] = novelty_scores(normed, window=novelty_window)
    if dialogue_spans is not None:
        columns["dlg_span"] = dialogue_spans.astype(np.float32)

    if audio_features is None and dialogue_spans is None:
        variant = "visual_only"
    elif audio_features is None:
        variant = "no_audio"
    elif dialogue_spans is None:
        variant = "no_dialogue"
    else:
        variant = "full"
    names = [name for name in FUSION_FEATURES if name in columns]
    matrix = np.column_stack([columns[name] for name in names])
    weights = np.asarray(FUSION_WEIGHTS[variant])
    logits = matrix @ weights[:-1] + weights[-1]
    return BoundaryScores(
        scores=1.0 / (1.0 + np.exp(-logits)),
        variant=variant,
        engine="fusion",
    )


def assign_scenes(
    shot_ids: list[str],
    scores: np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, int]:
    """Map shot_id -> 1-based scene number. The final shot always ends a
    scene; no minimum-length post-processing (real scenes can be one shot —
    a title card, a single-take flashback)."""
    if len(shot_ids) != len(scores):
        raise ValueError(f"{len(shot_ids)} shots but {len(scores)} scores")
    assignment: dict[str, int] = {}
    scene = 1
    for index, shot_id in enumerate(shot_ids):
        assignment[shot_id] = scene
        if scores[index] >= threshold and index < len(shot_ids) - 1:
            scene += 1
    return assignment
