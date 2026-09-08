"""Face identity: build a face DB from a proxy master and match shots.

Design notes:

* Detection and embedding are a single InsightFace pass per frame
  (:class:`FaceEngine`). The previous MTCNN-detect-then-InsightFace-re-detect
  pipeline ran two detectors per face and could embed a *different* face than
  the one MTCNN found inside the padded crop.
* Frames stream through sequentially (``grab``/``retrieve``) instead of
  seeking per sample, and only small JPEG-encoded crops are retained, so a
  feature-length master fits in bounded memory.
* Clustering is agglomerative with a cosine-distance threshold; clusters
  smaller than ``min_cluster_size`` are reabsorbed into the nearest large
  cluster when close enough instead of being discarded as noise.
* Shot matching votes per detected face with a best-vs-second margin gate, so
  a face that looks almost equally like two enrolled people abstains rather
  than guessing.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from unrender.lib.fingerprints import input_fingerprints, warn_if_inputs_changed
from unrender.manifests import (
    ShotRecord,
    read_json,
    write_json,
    write_shot_matches_csv,
)
from unrender.speakers import load_labeled_face_centroids, load_speaker_db

logger = logging.getLogger("unrender.analysis.identity.face")

DEFAULT_MIN_CONFIDENCE = 0.5  # InsightFace SCRFD det_score scale (~0-1)
DEFAULT_MIN_FACE_PX = 36
DEFAULT_MIN_MARGIN = 0.05
CROP_MAX_PX = 200
GRID_EXEMPLARS = 36


@dataclass(frozen=True)
class FaceObservation:
    """One detected face: box (x, y, w, h), score, and aligned embedding."""

    box: tuple[int, int, int, int]
    confidence: float
    embedding: np.ndarray


class FaceEngine:
    """InsightFace detection plus aligned 512-d embeddings in one pass."""

    def __init__(self, det_size: int = 640) -> None:
        try:
            from insightface import app as face_app
        except ImportError as exc:
            raise ImportError(
                "face analysis requires the 'face' extra: pip install .[face]"
            ) from exc
        # Only detection + recognition: the landmark and gender/age models
        # FaceAnalysis loads by default run per face and are never used here.
        self._app = face_app.FaceAnalysis(
            allowed_modules=["detection", "recognition"],
            providers=_onnx_providers(),
        )
        self._app.prepare(ctx_id=-1, det_size=(det_size, det_size))

    def analyze(self, frame: np.ndarray) -> list[FaceObservation]:
        """Detect and embed every face in a BGR frame."""
        observations: list[FaceObservation] = []
        for face in self._app.get(frame):
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                continue
            x1, y1, x2, y2 = (int(value) for value in face.bbox)
            observations.append(
                FaceObservation(
                    box=(x1, y1, max(1, x2 - x1), max(1, y2 - y1)),
                    confidence=float(face.det_score),
                    embedding=np.asarray(embedding, dtype=np.float32),
                )
            )
        return observations


def _onnx_providers() -> list[str]:
    """Prefer CUDA when available; CoreML is opt-in via UNRENDER_FACE_COREML=1.

    CPU always stays in the list so onnxruntime can fall back per-node.
    """
    import os

    try:
        import onnxruntime

        available = set(onnxruntime.get_available_providers())
    except ImportError:
        return ["CPUExecutionProvider"]
    preferred = ["CUDAExecutionProvider"]
    if os.environ.get("UNRENDER_FACE_COREML") == "1":
        preferred.append("CoreMLExecutionProvider")
    providers = [name for name in preferred if name in available]
    providers.append("CPUExecutionProvider")
    return providers


def iter_sampled_frames(
    video_path: Path,
    interval_sec: float,
    max_frames: int | None = None,
):
    """Stream ``(timestamp, frame)`` samples without per-frame seeking."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, round(fps * interval_sec))
        print(
            f"Video: {total_frames / fps:.1f}s @ {fps:.2f}fps; sampling every {interval_sec}s",
            flush=True,
        )
        index = 0
        yielded = 0
        while cap.grab():
            if index % step == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    yield index / fps, frame
                    yielded += 1
                    if max_frames and yielded >= max_frames:
                        return
            index += 1
    finally:
        cap.release()


def sample_video_frames(video_path: Path, n_samples: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = np.linspace(0, max(0, total - 1), num=min(n_samples, total)).astype(int)
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    cap.release()
    return frames


def cluster_embeddings(
    embeddings: list[np.ndarray],
    *,
    threshold: float = 0.5,
    min_cluster_size: int = 5,
    target_clusters: int | None = None,
) -> list[int]:
    """Cluster identity embeddings; reabsorb small clusters instead of dropping.

    Agglomerative clustering with average cosine linkage forms identity
    groups; groups below ``min_cluster_size`` are merged into the nearest
    large cluster when its centroid is within ``threshold`` cosine distance,
    and only truly isolated faces become noise (label ``-1``). Labels are
    renumbered by descending cluster size.
    """
    if not embeddings:
        return []
    matrix = _normalize(np.asarray(embeddings, dtype=np.float32))
    if len(matrix) == 1:
        raw = np.zeros(1, dtype=int)
    else:
        try:
            from sklearn.cluster import AgglomerativeClustering
        except ImportError as exc:
            raise ImportError("face clustering requires scikit-learn: pip install .[face]") from exc

        if target_clusters is not None and target_clusters > 0:
            raw = AgglomerativeClustering(
                n_clusters=min(target_clusters, len(matrix)),
                metric="cosine",
                linkage="average",
            ).fit_predict(matrix)
        else:
            raw = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=threshold,
                metric="cosine",
                linkage="average",
            ).fit_predict(matrix)

    counts = Counter(int(label) for label in raw)
    kept = [label for label, count in counts.items() if count >= min_cluster_size]
    if not kept:
        return [-1] * len(matrix)

    centroids = {
        label: _normalize(matrix[raw == label].mean(axis=0, keepdims=True))[0] for label in kept
    }
    # Rank kept clusters by size so cluster 0 is always the most frequent face.
    ranked = sorted(kept, key=lambda label: (-counts[label], label))
    renumber = {label: index for index, label in enumerate(ranked)}

    labels: list[int] = []
    for row, label in zip(matrix, (int(v) for v in raw), strict=True):
        if label in renumber:
            labels.append(renumber[label])
            continue
        sims = {kept_label: float(row @ centroids[kept_label]) for kept_label in kept}
        best = max(sims, key=lambda key: sims[key])
        labels.append(renumber[best] if 1.0 - sims[best] <= threshold else -1)
    return labels


def build_face_db(
    *,
    video_path: Path,
    face_db_path: Path,
    review_dir: Path,
    interval_sec: float = 2.0,
    max_frames: int | None = None,
    threshold: float = 0.5,
    min_cluster_size: int = 5,
    target_clusters: int | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_face_px: int = DEFAULT_MIN_FACE_PX,
    force: bool = False,
) -> dict[str, Any]:
    inputs = {"video": video_path}
    if face_db_path.exists() and not force:
        print(f"Face DB already exists, skipping: {face_db_path}", flush=True)
        data = read_json(face_db_path)
        warn_if_inputs_changed(data, inputs, face_db_path)
        return data
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    review_dir.mkdir(parents=True, exist_ok=True)
    engine = FaceEngine()

    detections: list[dict[str, Any]] = []
    frames_seen = 0
    for ts, frame in iter_sampled_frames(video_path, interval_sec, max_frames):
        frames_seen += 1
        for obs in engine.analyze(frame):
            if obs.confidence < min_confidence:
                continue
            _, _, width, height = obs.box
            if min(width, height) < min_face_px:
                continue
            detections.append(
                {
                    "ts": float(ts),
                    "frame_idx": frames_seen - 1,
                    "box": [int(v) for v in obs.box],
                    "confidence": obs.confidence,
                    "quality": obs.confidence * float(min(width, height)),
                    "crop_jpeg": _encode_crop(frame, obs.box),
                    "embedding": obs.embedding,
                }
            )
        if frames_seen % 50 == 0:
            print(
                f"  sampled {frames_seen} frames; detections={len(detections)}",
                flush=True,
            )
    if not detections:
        raise RuntimeError("no faces detected")
    print(f"  sampled {frames_seen} frames; detections={len(detections)}", flush=True)

    print("Clustering face embeddings...", flush=True)
    labels = cluster_embeddings(
        [det["embedding"] for det in detections],
        threshold=threshold,
        min_cluster_size=min_cluster_size,
        target_clusters=target_clusters,
    )
    by_cluster: dict[int, list[dict[str, Any]]] = {}
    noise = 0
    for det, label in zip(detections, labels, strict=True):
        if label >= 0:
            by_cluster.setdefault(label, []).append(det)
        else:
            noise += 1

    clusters: list[dict[str, Any]] = []
    for cluster_id in sorted(by_cluster):
        members = by_cluster[cluster_id]
        exemplars = _select_exemplars(members, GRID_EXEMPLARS)
        grid_path = review_dir / f"cluster_{cluster_id:03d}.jpg"
        crops = [_decode_crop(det["crop_jpeg"]) for det in exemplars]
        make_grid([crop for crop in crops if crop is not None]).save(grid_path, quality=88)
        centroid = _weighted_centroid(members)
        member_matrix = _normalize(
            np.asarray([det["embedding"] for det in members], dtype=np.float32)
        )
        mean_similarity = float(np.mean(member_matrix @ centroid))
        clusters.append(
            {
                "cluster_id": int(cluster_id),
                "name": "",
                "face_count": len(members),
                "centroid": [float(v) for v in centroid.tolist()],
                "mean_similarity": mean_similarity,
                "grid_path": str(grid_path),
                "faces": [
                    {
                        "ts": det["ts"],
                        "frame_idx": det["frame_idx"],
                        "box": det["box"],
                        "confidence": det["confidence"],
                    }
                    for det in members
                ],
            }
        )
        print(
            f"  cluster {cluster_id}: {len(members)} faces "
            f"(cohesion {mean_similarity:.2f}) -> {grid_path}",
            flush=True,
        )
    if noise:
        print(f"  {noise} isolated face(s) left unclustered", flush=True)

    data = {
        "version": "1.0",
        "video": str(video_path),
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z",
        "settings": {
            "interval_sec": interval_sec,
            "max_frames": max_frames,
            "threshold": threshold,
            "min_cluster_size": min_cluster_size,
            "target_clusters": target_clusters,
            "min_confidence": min_confidence,
            "min_face_px": min_face_px,
        },
        "inputs": input_fingerprints(inputs),
        "clusters": clusters,
    }
    write_json(face_db_path, data)
    print(f"Face DB saved: {face_db_path}", flush=True)
    return data


def match_shots(
    *,
    shots: list[ShotRecord],
    speaker_db_path: Path,
    output_json: Path,
    output_csv: Path,
    samples_per_shot: int = 8,
    sim_threshold: float = 0.45,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_votes: int = 2,
    min_margin: float = DEFAULT_MIN_MARGIN,
) -> list[dict[str, Any]]:
    identity_db = load_speaker_db(speaker_db_path)
    cluster_names, raw_centroids = load_labeled_face_centroids(identity_db)
    if not cluster_names:
        raise ValueError(f"speaker DB has no labeled face clusters: {speaker_db_path}")
    centroids = _normalize(np.asarray(raw_centroids, dtype=np.float32))
    engine = FaceEngine()

    matches: list[dict[str, Any]] = []
    for index, shot in enumerate(shots, 1):
        votes, faces = detect_speakers_for_shot(
            shot.video_path,
            engine,
            cluster_names,
            centroids,
            samples_per_shot,
            sim_threshold,
            min_confidence,
            target_box=shot.target_face_box,
            min_margin=min_margin,
        )
        accepted = [name for name, count in votes.most_common() if count >= min_votes]
        entry: dict[str, Any] = {
            "shot_id": shot.shot_id,
            "video_path": str(shot.video_path),
            "faces_detected": faces,
            "votes": dict(votes),
            "accepted": accepted,
        }
        if accepted:
            entry["proposed_value"] = ", ".join(accepted)
        matches.append(entry)
        top = ", ".join(f"{name}x{count}" for name, count in votes.most_common(5))
        print(
            f"  [{index}/{len(shots)}] {shot.shot_id} faces={faces} votes: {top or '(none)'}",
            flush=True,
        )

    write_json(
        output_json,
        {
            "version": "1.0",
            "speaker_db": str(speaker_db_path),
            "min_votes": min_votes,
            "sim_threshold": sim_threshold,
            "min_margin": min_margin,
            "shots": matches,
        },
    )
    write_shot_matches_csv(output_csv, matches)
    print(f"Shot matches saved: {output_json}", flush=True)
    return matches


def detect_speakers_for_shot(
    shot_video: Path,
    engine: FaceEngine,
    cluster_names: list[str],
    centroids: np.ndarray,
    n_samples: int,
    sim_threshold: float,
    min_confidence: float,
    target_box: tuple[int, int, int, int] | None = None,
    min_margin: float = DEFAULT_MIN_MARGIN,
) -> tuple[Counter[str], int]:
    """Vote per detected face; ambiguous faces (small margin) abstain."""
    frames = sample_video_frames(shot_video, n_samples)
    votes: Counter[str] = Counter()
    faces = 0
    for frame in frames:
        observations = [obs for obs in engine.analyze(frame) if obs.confidence >= min_confidence]
        if target_box is not None:
            observations = _best_target_face(observations, target_box)
        for obs in observations:
            faces += 1
            emb = _normalize(np.asarray([obs.embedding], dtype=np.float32))[0]
            sims = centroids @ emb
            order = np.argsort(sims)[::-1]
            best = float(sims[int(order[0])])
            second = float(sims[int(order[1])]) if len(order) > 1 else -1.0
            if best >= sim_threshold and best - second >= min_margin:
                votes[cluster_names[int(order[0])]] += 1
    return votes, faces


def _best_target_face(
    observations: list[FaceObservation],
    target_box: tuple[int, int, int, int],
    min_iou: float = 0.1,
) -> list[FaceObservation]:
    if not observations:
        return []
    best = max(observations, key=lambda obs: _box_iou(obs.box, target_box))
    return [best] if _box_iou(best.box, target_box) >= min_iou else []


def _box_iou(a: list[int] | tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = (float(value) for value in a)
    bx, by, bw, bh = (float(value) for value in b)
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_w = max(0.0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0.0, min(ay2, by2) - max(ay, by))
    intersection = inter_w * inter_h
    union = aw * ah + bw * bh - intersection
    return 0.0 if union <= 0 else intersection / union


def make_grid(
    images: list[np.ndarray], cell: int = 200, label_h: int = 22, cols: int = 6
) -> Image.Image:
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cell * cols, (cell + label_h) * max(1, rows)), color=(0, 0, 0))
    draw = ImageDraw.Draw(grid)
    try:
        font: Any = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except OSError:
        font = ImageFont.load_default()
    for index, image in enumerate(images):
        if image is None or image.size == 0:
            continue
        h, w = image.shape[:2]
        scale = cell / max(h, w)
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        resized = cv2.resize(image, (nw, nh))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        x = (index % cols) * cell + (cell - nw) // 2
        y_cell_top = (index // cols) * (cell + label_h)
        y = y_cell_top + (cell - nh) // 2
        grid.paste(Image.fromarray(rgb), (x, y))
        label = f"{index + 1}"
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text(
            ((index % cols) * cell + (cell - (bbox[2] - bbox[0])) // 2, y_cell_top + cell + 2),
            label,
            fill=(220, 220, 220),
            font=font,
        )
    return grid


def _select_exemplars(members: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Highest-quality faces, spread across the timeline for scene diversity.

    Taking the top-N by confidence tends to return near-duplicates from a
    single scene; spreading a quality-sorted shortlist over time keeps the
    review grid representative.
    """
    if len(members) <= limit:
        return sorted(members, key=lambda det: det["ts"])
    shortlist = sorted(members, key=lambda det: -det.get("quality", det["confidence"]))
    shortlist = sorted(shortlist[: limit * 2], key=lambda det: det["ts"])
    indices = np.linspace(0, len(shortlist) - 1, num=limit).astype(int)
    return [shortlist[int(i)] for i in indices]


def _encode_crop(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    padding: float = 0.15,
    max_px: int = CROP_MAX_PX,
) -> bytes:
    x, y, width, height = box
    img_h, img_w = frame.shape[:2]
    pad_w = int(width * padding)
    pad_h = int(height * padding)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + width + pad_w)
    y2 = min(img_h, y + height + pad_h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((8, 8, 3), dtype=np.uint8)
    h, w = crop.shape[:2]
    scale = max_px / max(h, w)
    if scale < 1.0:
        crop = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))))
    ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return encoded.tobytes() if ok else b""


def _decode_crop(data: bytes) -> np.ndarray | None:
    if not data:
        return None
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return decoded if decoded is not None and decoded.size else None


def _weighted_centroid(members: list[dict[str, Any]]) -> np.ndarray:
    """Confidence-weighted centroid so marginal detections drag it less."""
    matrix = _normalize(np.asarray([det["embedding"] for det in members], dtype=np.float32))
    weights = np.asarray([max(1e-3, det["confidence"]) for det in members], dtype=np.float32)
    center = (matrix * weights[:, None]).sum(axis=0) / weights.sum()
    norm = float(np.linalg.norm(center))
    return center / norm if norm > 0 else center


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms
