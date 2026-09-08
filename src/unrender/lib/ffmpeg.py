from __future__ import annotations

import struct
import subprocess
import tempfile
import wave
from collections.abc import Sequence
from pathlib import Path

import numpy as np

# Upper bound for a single ffmpeg invocation. Long enough for large files,
# short enough that a genuinely hung process does not stall the CLI forever.
DEFAULT_FFMPEG_TIMEOUT_SEC = 3600

DEFAULT_SLICE_CODEC = "pcm_s16le"

_WAVE_FORMAT_PCM = 0x0001
_WAVE_FORMAT_IEEE_FLOAT = 0x0003
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE

_PCM_CODEC_BY_FORMAT: dict[tuple[int, int], str] = {
    (_WAVE_FORMAT_PCM, 8): "pcm_u8",
    (_WAVE_FORMAT_PCM, 16): "pcm_s16le",
    (_WAVE_FORMAT_PCM, 24): "pcm_s24le",
    (_WAVE_FORMAT_PCM, 32): "pcm_s32le",
    (_WAVE_FORMAT_IEEE_FLOAT, 32): "pcm_f32le",
    (_WAVE_FORMAT_IEEE_FLOAT, 64): "pcm_f64le",
}

# PCM codec keyed by sample width in bytes (for re-encoding float buffers).
_PCM_CODEC_BY_WIDTH: dict[int, str] = {1: "pcm_u8", 2: "pcm_s16le", 3: "pcm_s24le", 4: "pcm_s32le"}


def pcm_codec_for_width(sample_width: int) -> str:
    """PCM codec name for a sample width in bytes (defaults to 24-bit)."""
    return _PCM_CODEC_BY_WIDTH.get(sample_width, "pcm_s24le")


def decode_audio_f32le(
    source: Path,
    *,
    ffmpeg: str = "ffmpeg",
    sample_rate: int,
    channels: int = 1,
) -> np.ndarray | None:
    """Decode ``source`` to raw float32 PCM samples, or ``None`` if undecodable.

    Returns a flat mono (or interleaved) float32 array. Used by feature
    extractors that need the whole waveform in memory rather than a file.
    """
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    completed = subprocess.run(cmd, capture_output=True, check=False)
    if completed.returncode != 0:
        return None
    return np.frombuffer(completed.stdout, dtype=np.float32)


def probe_media_duration_sec(source: Path, *, ffprobe: str = "ffprobe") -> float:
    """Container duration in seconds, or ``0.0`` when the file does not report one."""
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(source),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"ffprobe executable not found: {ffprobe!r}. "
            "Install ffmpeg and ensure it is on your PATH."
        ) from exc
    try:
        return max(0.0, float(completed.stdout.strip()))
    except ValueError:
        return 0.0


def extract_audio_window(
    source: Path,
    target: Path,
    *,
    ffmpeg: str = "ffmpeg",
    start_sec: float,
    duration_sec: float,
    sample_rate: int,
    audio_streams: int = 1,
) -> None:
    """Write one mono window of ``source`` as a 16-bit WAV.

    Pass ``audio_streams`` above 1 to merge that many streams. ffmpeg otherwise
    decodes its default stream alone, which on a multi-track master is a single
    discrete track and may hold no dialogue at all.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-v", "error", "-ss", f"{start_sec:.3f}", "-i", str(source)]
    cmd += ["-t", f"{duration_sec:.3f}", "-vn"]
    if audio_streams > 1:
        cmd += [
            "-filter_complex",
            f"[0:a]amix=inputs={audio_streams}:normalize=0[merged]",
            "-map",
            "[merged]",
        ]
    cmd += ["-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", "-y", str(target)]
    _run_ffmpeg(cmd)


def _run_ffmpeg(cmd: list[str], *, timeout: int = DEFAULT_FFMPEG_TIMEOUT_SEC) -> None:
    try:
        subprocess.run(cmd, check=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"ffmpeg executable not found: {cmd[0]!r}. "
            "Install ffmpeg and ensure it is on your PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg timed out after {timeout}s: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg exited with status {exc.returncode} while writing {cmd[-1]}"
        ) from exc


def run_ffmpeg_slice(
    *,
    ffmpeg: str,
    source: Path,
    target: Path,
    start_sec: float,
    end_sec: float,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    # PCM WAV -> WAV slices are a sample-accurate byte copy; skipping the
    # ffmpeg process spawn makes per-line clip cutting orders of magnitude
    # faster. Anything the wave module cannot read (float, compressed,
    # non-WAV) falls through to ffmpeg below.
    if _slice_wav_native(source=source, target=target, start_sec=start_sec, end_sec=end_sec):
        return
    duration = end_sec - start_sec
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_sec:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(source),
        "-vn",
        "-acodec",
        pcm_codec_for_source(source),
        str(target),
    ]
    _run_ffmpeg(cmd)


def run_ffmpeg_exact_slice(
    *,
    ffmpeg: str,
    source: Path,
    target: Path,
    start_sec: float,
    end_sec: float,
) -> None:
    """Cut and silence-pad a source to the exact requested duration."""

    duration = end_sec - start_sec
    if duration <= 0:
        raise ValueError(f"audio slice duration must be positive, got {duration}")
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_sec:.6f}",
        "-i",
        str(source),
        "-vn",
        "-af",
        (f"apad=whole_dur={duration:.6f},atrim=start=0:end={duration:.6f},asetpts=PTS-STARTPTS"),
        "-t",
        f"{duration:.6f}",
        "-acodec",
        pcm_codec_for_source(source),
        str(target),
    ]
    _run_ffmpeg(command)


def run_ffmpeg_timeline_mix(
    *,
    ffmpeg: str,
    segments: Sequence[tuple[Path, float, float, float]],
    target: Path,
    duration_sec: float,
) -> None:
    """Place source slices on a silent timeline with an exact output duration.

    Each segment is ``(source, source_start_sec, source_end_sec,
    output_offset_sec)``. This is used for shot-length speaker stems: only the
    dialogue intervals assigned to that speaker are audible, while silence
    preserves their original timing within the shot.
    """
    if duration_sec <= 0:
        raise ValueError(f"timeline duration must be positive, got {duration_sec}")
    if not segments:
        raise ValueError("timeline mix requires at least one audio segment")

    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    filters: list[str] = []
    labels: list[str] = []
    for index, (source, start_sec, end_sec, offset_sec) in enumerate(segments):
        segment_duration = end_sec - start_sec
        if segment_duration <= 0:
            raise ValueError(f"audio segment must have positive duration: {source}")
        if offset_sec < 0 or offset_sec + segment_duration > duration_sec + 1e-6:
            raise ValueError(
                f"audio segment lies outside the {duration_sec:.6f}s timeline: "
                f"{source} at {offset_sec:.6f}s for {segment_duration:.6f}s"
            )
        cmd.extend(
            [
                "-ss",
                f"{start_sec:.6f}",
                "-t",
                f"{segment_duration:.6f}",
                "-i",
                str(source),
            ]
        )
        label = f"segment{index}"
        labels.append(f"[{label}]")
        delay_ms = round(offset_sec * 1000.0)
        filters.append(f"[{index}:a]asetpts=PTS-STARTPTS,adelay={delay_ms}:all=1[{label}]")

    labels.append("[silence]")
    filters.append(f"anullsrc=r=48000:cl=mono:d={duration_sec:.6f}[silence]")
    filters.append(
        f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:normalize=0,"
        f"apad=whole_dur={duration_sec:.6f},"
        f"atrim=start=0:end={duration_sec:.6f},asetpts=PTS-STARTPTS[out]"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-acodec",
            pcm_codec_for_source(segments[0][0]),
            "-t",
            f"{duration_sec:.6f}",
            str(target),
        ]
    )
    _run_ffmpeg(cmd)


def _slice_wav_native(
    *,
    source: Path,
    target: Path,
    start_sec: float,
    end_sec: float,
) -> bool:
    """Copy a PCM WAV slice directly; returns False when ffmpeg is needed."""
    if source.suffix.lower() != ".wav" or target.suffix.lower() != ".wav":
        return False
    try:
        with wave.open(str(source), "rb") as reader:
            rate = reader.getframerate()
            channels = reader.getnchannels()
            sample_width = reader.getsampwidth()
            total = reader.getnframes()
            if rate <= 0:
                return False
            start_frame = min(total, max(0, round(start_sec * rate)))
            count = max(0, min(total - start_frame, round((end_sec - start_sec) * rate)))
            reader.setpos(start_frame)
            data = reader.readframes(count)
    except (wave.Error, EOFError, OSError):
        return False
    with wave.open(str(target), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(rate)
        writer.writeframes(data)
    return True


def pcm_codec_for_source(source: Path) -> str:
    """Pick the slice codec that preserves the source WAV sample format.

    Slices always re-encoded to 16-bit PCM before, silently quantizing 24-bit
    and float stems (BandIt writes 32-bit float by default). Non-WAV and
    unreadable sources keep the 16-bit default.
    """
    if source.suffix.lower() != ".wav":
        return DEFAULT_SLICE_CODEC
    fmt = _wav_sample_format(source)
    if fmt is None:
        return DEFAULT_SLICE_CODEC
    return _PCM_CODEC_BY_FORMAT.get(fmt, DEFAULT_SLICE_CODEC)


def _wav_sample_format(path: Path) -> tuple[int, int] | None:
    """Return ``(format_tag, bits_per_sample)`` from a WAV ``fmt`` chunk."""
    try:
        with path.open("rb") as handle:
            riff = handle.read(12)
            if len(riff) < 12 or riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
                return None
            while True:
                header = handle.read(8)
                if len(header) < 8:
                    return None
                chunk_id = header[:4]
                size = struct.unpack("<I", header[4:])[0]
                if chunk_id != b"fmt ":
                    handle.seek(size + (size % 2), 1)
                    continue
                data = handle.read(size)
                if len(data) < 16:
                    return None
                format_tag = struct.unpack_from("<H", data, 0)[0]
                bits = struct.unpack_from("<H", data, 14)[0]
                if format_tag == _WAVE_FORMAT_EXTENSIBLE and len(data) >= 26:
                    # WAVE_FORMAT_EXTENSIBLE stores the real tag in the first
                    # two bytes of the SubFormat GUID at offset 24.
                    format_tag = struct.unpack_from("<H", data, 24)[0]
                return format_tag, bits
    except OSError:
        return None


def cut_audio_with_fades(
    *,
    ffmpeg: str,
    source: Path,
    target: Path,
    duration_sec: float,
    sample_rate: int,
    codec: str,
    fade_ms: int,
    channels: int = 1,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fade_out_start = max(0.0, duration_sec - (fade_ms / 1000.0))
    filters = (
        f"afade=t=in:st=0:d={fade_ms / 1000.0:.3f},"
        f"afade=t=out:st={fade_out_start:.6f}:d={fade_ms / 1000.0:.3f}"
    )
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-t",
        f"{duration_sec:.6f}",
        "-vn",
        "-af",
        filters,
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-acodec",
        codec,
        str(target),
    ]
    _run_ffmpeg(cmd)


def concat_audio(*, ffmpeg: str, parts: list[Path], output_path: Path, codec: str) -> None:
    if not parts:
        raise ValueError("cannot build a reference from zero clips")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        concat_list = Path(handle.name)
        for part in parts:
            handle.write(f"file '{ffmpeg_concat_path(part)}'\n")
    try:
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-acodec",
            codec,
            str(output_path),
        ]
        _run_ffmpeg(cmd)
    finally:
        concat_list.unlink(missing_ok=True)


def ffmpeg_concat_path(path: Path) -> str:
    return str(path).replace("'", r"'\''")
