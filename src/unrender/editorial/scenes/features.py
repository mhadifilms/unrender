"""Per-shot feature extraction for scene grouping.

Keyframes are decoded from the detection proxy (never a raw master — the
same rule the detector follows), embedded shot-by-shot so memory stays flat
regardless of movie length. torch/transformers load lazily behind the
``scenes`` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from unrender.lib.ffmpeg import decode_audio_f32le

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DINO_MODEL_ID = "facebook/dinov2-small"
KEYFRAMES_PER_SHOT = 3
KEYFRAME_MAX_WIDTH = 480
EDGE_SKIP_FRAMES = 2  # avoid frames adjacent to the cut (transition blur)
AUDIO_SAMPLE_RATE = 22_050
AUDIO_BANDS = 40


def _import_torch_stack() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as exc:  # pragma: no cover - exercised via message test
        raise ImportError(
            "scene grouping needs torch + transformers. Install them with: "
            'pip install "unrender[scenes]"'
        ) from exc
    return torch, transformers


def keyframe_indices(
    start_frame: int, end_frame: int, count: int = KEYFRAMES_PER_SHOT
) -> list[int]:
    """Evenly spaced frame indices inside [start, end), skipping cut-adjacent
    frames when the shot is long enough."""
    low, high = start_frame, end_frame - 1
    if high - low >= 2 * EDGE_SKIP_FRAMES + count:
        low, high = low + EDGE_SKIP_FRAMES, high - EDGE_SKIP_FRAMES
    if count == 1 or high <= low:
        return [(low + high) // 2]
    return sorted({low + round(i * (high - low) / (count - 1)) for i in range(count)})


def shot_embeddings(
    video: Path,
    shots: list[dict[str, Any]],
    *,
    model: str = "clip",
    keyframes: int = KEYFRAMES_PER_SHOT,
) -> np.ndarray:
    """[n_shots, dim] L2-normalized embeddings, mean-pooled over keyframes.

    ``shots`` are shot-manifest entries carrying ``start_frame``/``end_frame``
    (end-exclusive). Frames are decoded and embedded one shot at a time.
    """
    torch, _ = _import_torch_stack()
    from PIL import Image

    if model == "clip":
        from transformers import CLIPModel, CLIPProcessor

        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        net = CLIPModel.from_pretrained(CLIP_MODEL_ID)

        def embed(images: list[Image.Image]) -> Any:
            features = net.get_image_features(**processor(images=images, return_tensors="pt"))
            if not torch.is_tensor(features):  # transformers >= 5 returns an output object
                features = features.pooler_output
            return features

    elif model == "dino":
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
        net = AutoModel.from_pretrained(DINO_MODEL_ID)

        def embed(images: list[Image.Image]) -> Any:
            output = net(**processor(images=images, return_tensors="pt"))
            pooled = getattr(output, "pooler_output", None)
            return pooled if pooled is not None else output.last_hidden_state[:, 0]

    else:
        raise ValueError(f"unknown embedding model {model!r} (clip/dino)")

    net.eval()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open video for keyframe extraction: {video}")
    vectors: list[np.ndarray] = []
    dim = 0
    with torch.no_grad():
        for index, entry in enumerate(shots):
            frames = _read_keyframes(capture, entry, keyframes)
            if not frames:
                vectors.append(np.zeros(dim, dtype=np.float32))
                continue
            images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
            features = torch.nn.functional.normalize(embed(images), dim=-1)
            pooled = torch.nn.functional.normalize(features.mean(0), dim=-1)
            vector = pooled.float().numpy().astype(np.float32)
            dim = vector.shape[0]
            vectors.append(vector)
            if (index + 1) % 100 == 0:
                print(f"  {model} embeddings: {index + 1}/{len(shots)} shots", flush=True)
    capture.release()
    # backfill any leading zero vectors created before dim was known
    return np.stack(
        [vec if vec.shape[0] == dim else np.zeros(dim, dtype=np.float32) for vec in vectors]
    )


def _read_keyframes(
    capture: cv2.VideoCapture, entry: dict[str, Any], count: int
) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for frame_num in keyframe_indices(int(entry["start_frame"]), int(entry["end_frame"]), count):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ok, frame = capture.read()
        if not ok:
            continue
        height, width = frame.shape[:2]
        if width > KEYFRAME_MAX_WIDTH:
            frame = cv2.resize(
                frame,
                (KEYFRAME_MAX_WIDTH, round(height * KEYFRAME_MAX_WIDTH / width)),
                interpolation=cv2.INTER_AREA,
            )
        frames.append(frame)
    return frames


def shot_audio_features(
    audio_source: Path,
    shots: list[dict[str, Any]],
    *,
    ffmpeg: str = "ffmpeg",
    sample_rate: int = AUDIO_SAMPLE_RATE,
    bands: int = AUDIO_BANDS,
) -> np.ndarray | None:
    """[n_shots, 2*bands] mean+std of log band energies per shot, or ``None``
    when the source has no decodable audio. Scene changes usually change the
    soundscape; intra-scene cuts don't."""
    wave = decode_audio_f32le(audio_source, ffmpeg=ffmpeg, sample_rate=sample_rate)
    if wave is None or wave.size < sample_rate:
        return None

    n_fft, hop = 1024, 512
    window = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    edges = np.geomspace(50, sample_rate / 2, bands + 2)
    filterbank = np.zeros((bands, len(freqs)), dtype=np.float32)
    for band in range(bands):
        low, mid, high = edges[band], edges[band + 1], edges[band + 2]
        rising = (freqs >= low) & (freqs <= mid)
        falling = (freqs > mid) & (freqs <= high)
        filterbank[band, rising] = (freqs[rising] - low) / (mid - low)
        filterbank[band, falling] = (high - freqs[falling]) / (high - mid)

    features = []
    for entry in shots:
        start = int(float(entry["start_sec"]) * sample_rate)
        end = int(float(entry["end_sec"]) * sample_rate)
        segment = wave[start:end]
        if segment.size < n_fft:
            features.append(np.zeros(2 * bands, dtype=np.float32))
            continue
        frame_count = 1 + (segment.size - n_fft) // hop
        idx = np.arange(n_fft)[None, :] + hop * np.arange(frame_count)[:, None]
        spectrum = np.abs(np.fft.rfft(segment[idx] * window, axis=1)) ** 2
        band_energy = np.log10(spectrum @ filterbank.T + 1e-9)
        features.append(
            np.concatenate([band_energy.mean(0), band_energy.std(0)]).astype(np.float32)
        )
    return np.stack(features)


def dialogue_spans(lines: list[dict[str, Any]], shots: list[dict[str, Any]]) -> np.ndarray | None:
    """spans[i] = 1 when a dialogue line temporally crosses the cut after shot
    ``i`` — strong evidence both shots belong to the same scene. ``None``
    when there are no usable lines."""
    usable = [
        (float(line.get("start_sec") or 0.0), float(line.get("end_sec") or 0.0))
        for line in lines
        if line.get("start_sec") is not None
    ]
    if not usable:
        return None
    spans = np.zeros(len(shots), dtype=np.float32)
    for index, entry in enumerate(shots[:-1]):
        cut_sec = float(entry["end_sec"])
        for start, end in usable:
            if start < cut_sec - 0.04 and end > cut_sec + 0.04:
                spans[index] = 1.0
                break
    return spans
