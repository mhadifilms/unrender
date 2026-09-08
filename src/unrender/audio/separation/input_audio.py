from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unrender.lib.ffmpeg import DEFAULT_FFMPEG_TIMEOUT_SEC
from unrender.lib.fingerprints import changed_inputs, input_fingerprints
from unrender.lib.names import safe_name
from unrender.project import RunPaths

AudioInputPair = tuple[str | Path, str | Path]
AudioInputSource = str | Path | AudioInputPair


@dataclass(frozen=True)
class AudioStream:
    index: int
    channels: int
    channel_layout: str
    codec_name: str
    sample_rate: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "channels": self.channels,
            "channel_layout": self.channel_layout,
            "codec_name": self.codec_name,
            "sample_rate": self.sample_rate,
        }


@dataclass(frozen=True)
class NormalizedAudioInput:
    source: str | tuple[str, str]
    prepared_source: str | Path
    normalized: bool
    selection: str
    selected_streams: tuple[int, ...]
    streams: tuple[AudioStream, ...]

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "prepared_source": str(self.prepared_source),
            "normalized": self.normalized,
            "selection": self.selection,
            "selected_streams": list(self.selected_streams),
            "streams": [stream.as_dict() for stream in self.streams],
        }


def probe_audio_streams(
    source: Path,
    *,
    ffprobe: str = "ffprobe",
) -> tuple[AudioStream, ...]:
    """Probe every audio stream in a local media file."""

    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_streams",
        "-of",
        "json",
        str(source),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"ffprobe executable not found: {ffprobe!r}. Install ffmpeg and ensure it is on PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffprobe timed out while probing audio streams: {source}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"ffprobe could not inspect audio streams in {source}{suffix}") from exc

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned invalid JSON for {source}") from exc
    raw_streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(raw_streams, list):
        raise RuntimeError(f"ffprobe returned no stream list for {source}")

    streams: list[AudioStream] = []
    for raw in raw_streams:
        if not isinstance(raw, dict):
            continue
        sample_rate_raw = raw.get("sample_rate")
        try:
            sample_rate = int(str(sample_rate_raw)) if sample_rate_raw not in (None, "") else None
        except (TypeError, ValueError):
            sample_rate = None
        streams.append(
            AudioStream(
                index=int(raw["index"]) if raw.get("index") is not None else len(streams),
                channels=int(raw.get("channels") or 0),
                channel_layout=str(raw.get("channel_layout") or ""),
                codec_name=str(raw.get("codec_name") or ""),
                sample_rate=sample_rate,
            )
        )
    if not streams:
        raise ValueError(f"media contains no audio streams: {source}")
    return tuple(streams)


def normalize_full_audio_input(
    *,
    run: RunPaths,
    source: AudioInputSource,
    prefix: str = "audio",
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    force: bool = False,
    dry_run: bool = False,
) -> NormalizedAudioInput:
    """Select a stereo program and stage it as PCM WAV when necessary."""

    source_paths: Path | tuple[Path, Path]
    source_value: str | tuple[str, str]
    streams: tuple[AudioStream, ...]
    selected: tuple[int, ...]
    if isinstance(source, tuple):
        if len(source) != 2:
            raise ValueError("full audio pair must contain exactly left and right paths")
        source_paths = (Path(source[0]).expanduser(), Path(source[1]).expanduser())
        for side, path in zip(("left", "right"), source_paths, strict=True):
            if not path.is_file():
                raise FileNotFoundError(f"full audio {side} source not found: {path}")
        pair_streams = tuple(probe_audio_streams(path, ffprobe=ffprobe) for path in source_paths)
        if any(
            len(side_streams) != 1 or side_streams[0].channels != 1 for side_streams in pair_streams
        ):
            raise ValueError(
                "full audio pair requires one mono stream per side; "
                f"left={len(pair_streams[0])} stream(s)/"
                f"{pair_streams[0][0].channels if pair_streams[0] else 0} channel(s), "
                f"right={len(pair_streams[1])} stream(s)/"
                f"{pair_streams[1][0].channels if pair_streams[1] else 0} channel(s)"
            )
        sample_rates = {side_streams[0].sample_rate for side_streams in pair_streams}
        sample_rates.discard(None)
        if len(sample_rates) > 1:
            raise ValueError(
                "full audio pair sample rates do not match: "
                f"left={pair_streams[0][0].sample_rate}, right={pair_streams[1][0].sample_rate}"
            )
        streams = (pair_streams[0][0], pair_streams[1][0])
        selection = "external_mono_pair_2_0_program"
        selected = (streams[0].index, streams[1].index)
        source_value = (str(source_paths[0]), str(source_paths[1]))
    else:
        source_text = str(source)
        if source_text.startswith(("http://", "https://")):
            return NormalizedAudioInput(
                source=source_text,
                prepared_source=source_text,
                normalized=False,
                selection="remote_passthrough",
                selected_streams=(),
                streams=(),
            )

        source_path = Path(source).expanduser()
        if not source_path.is_file():
            raise FileNotFoundError(f"full audio source not found: {source_path}")
        source_paths = source_path
        source_value = str(source_path)
        streams = probe_audio_streams(source_path, ffprobe=ffprobe)
        selection, selected = _select_stereo_program(streams)

    if (
        selection == "stereo_stream"
        and len(streams) == 1
        and isinstance(source_paths, Path)
        and source_paths.suffix.lower() in {".wav", ".wave"}
    ):
        return NormalizedAudioInput(
            source=source_value,
            prepared_source=source_paths,
            normalized=False,
            selection="direct_stereo_wav",
            selected_streams=selected,
            streams=streams,
        )

    target = run.normalized_audio_dir / f"{safe_name(prefix, fallback='audio')}_full_audio.wav"
    result = NormalizedAudioInput(
        source=source_value,
        prepared_source=target,
        normalized=True,
        selection=selection,
        selected_streams=selected,
        streams=streams,
    )
    if dry_run:
        print(
            f"DRY RUN: would normalize {selection} audio stream(s) {list(selected)} "
            f"to stereo PCM: {target}",
            flush=True,
        )
        return result
    normalization_manifest = target.with_suffix(".json")
    inputs = {
        "source": source_paths,
        "selection": selection,
        "selected_streams": list(selected),
    }
    if target.exists() and normalization_manifest.exists() and not force:
        try:
            previous = json.loads(normalization_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
        changed = changed_inputs(previous, inputs) if isinstance(previous, dict) else ["source"]
        if changed:
            raise ValueError(
                "normalized full-audio input changed "
                f"({', '.join(sorted(changed))}); rerun with --force"
            )
        print(f"Normalized full audio already exists, skipping: {target}", flush=True)
        return result

    target.parent.mkdir(parents=True, exist_ok=True)
    command = _normalization_command(
        ffmpeg=ffmpeg,
        source=source_paths,
        target=target,
        selection=selection,
        selected_streams=selected,
    )
    try:
        subprocess.run(command, check=True, timeout=DEFAULT_FFMPEG_TIMEOUT_SEC)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"ffmpeg executable not found: {ffmpeg!r}. Install ffmpeg and ensure it is on PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ffmpeg timed out while normalizing full audio: {source_value}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg exited with status {exc.returncode} while normalizing {source_value}"
        ) from exc
    normalization_manifest.write_text(
        json.dumps(
            {
                "version": "1.0",
                "input_audio": result.provenance,
                "inputs": input_fingerprints(inputs),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def _select_stereo_program(streams: tuple[AudioStream, ...]) -> tuple[str, tuple[int, ...]]:
    first_eight = streams[:8]
    if len(first_eight) == 8 and all(stream.channels == 1 for stream in first_eight):
        return "first_mono_pair_2_0_program", (first_eight[0].index, first_eight[1].index)

    stereo = next(
        (
            stream
            for stream in streams
            if stream.channels == 2 or stream.channel_layout.lower() == "stereo"
        ),
        None,
    )
    if stereo is not None:
        return "stereo_stream", (stereo.index,)

    surround = next(
        (
            stream
            for stream in streams
            if stream.channels >= 6 or "5.1" in stream.channel_layout.lower()
        ),
        None,
    )
    if surround is not None:
        return "downmix_5_1_stream", (surround.index,)

    first_six = streams[:6]
    if len(first_six) == 6 and all(stream.channels == 1 for stream in first_six):
        return "downmix_six_mono_5_1", tuple(stream.index for stream in first_six)

    first = streams[0]
    return "first_audio_stream_to_stereo", (first.index,)


def _normalization_command(
    *,
    ffmpeg: str,
    source: Path | tuple[Path, Path],
    target: Path,
    selection: str,
    selected_streams: tuple[int, ...],
) -> list[str]:
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    if selection == "external_mono_pair_2_0_program":
        assert isinstance(source, tuple)
        command.extend(
            [
                "-i",
                str(source[0]),
                "-i",
                str(source[1]),
                "-vn",
                "-filter_complex",
                "[0:a:0][1:a:0]amerge=inputs=2,pan=stereo|c0=c0|c1=c1[out]",
                "-map",
                "[out]",
            ]
        )
    elif selection == "first_mono_pair_2_0_program":
        assert isinstance(source, Path)
        command.extend(["-i", str(source), "-vn"])
        left, right = selected_streams
        command.extend(
            [
                "-filter_complex",
                f"[0:{left}][0:{right}]amerge=inputs=2," "pan=stereo|c0=c0|c1=c1[out]",
                "-map",
                "[out]",
            ]
        )
    elif selection == "downmix_six_mono_5_1":
        assert isinstance(source, Path)
        command.extend(["-i", str(source), "-vn"])
        labels = "".join(f"[0:{index}]" for index in selected_streams)
        command.extend(
            [
                "-filter_complex",
                f"{labels}amerge=inputs=6,"
                "pan=stereo|c0=c0+0.707*c2+0.5*c3+0.707*c4|"
                "c1=c1+0.707*c2+0.5*c3+0.707*c5[out]",
                "-map",
                "[out]",
            ]
        )
    else:
        assert isinstance(source, Path)
        command.extend(["-i", str(source), "-vn"])
        command.extend(["-map", f"0:{selected_streams[0]}", "-ac", "2"])
    command.extend(["-c:a", "pcm_s24le", str(target)])
    return command
