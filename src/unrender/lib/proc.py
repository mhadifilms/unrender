"""Single subprocess runner for external media tools (ffmpeg / ffprobe).

Capture-style callers (ffprobe, one-shot ffmpeg encodes) use ``capture=True``
to get stdout back with a stderr tail on failure. Streaming callers pass
``capture=False`` to let output flow to the console and rely on a non-zero
exit raising. The audio-slice helper in :mod:`unrender.lib.ffmpeg` keeps its
own runner because its fast path and error text are exercised directly by
tests.
"""

from __future__ import annotations

import subprocess

DEFAULT_TOOL_TIMEOUT_SEC = 7200


def run_tool(
    cmd: list[str], *, timeout: int = DEFAULT_TOOL_TIMEOUT_SEC, capture: bool = False
) -> str:
    """Run ``cmd``; return stdout when ``capture`` is set, else an empty string.

    Failures raise ``RuntimeError`` with an actionable message (missing binary,
    timeout, or non-zero exit with a stderr tail when captured).
    """
    try:
        completed = subprocess.run(
            cmd,
            capture_output=capture,
            text=True if capture else None,
            check=not capture,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{cmd[0]!r} was not found. Install ffmpeg and ensure it is on your PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{cmd[0]} timed out after {timeout}s") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"{cmd[0]} exited with status {exc.returncode} while writing {cmd[-1]}"
        ) from exc
    if capture and completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        tail = "\n".join(stderr.splitlines()[-12:])
        raise RuntimeError(f"{cmd[0]} exited with status {completed.returncode}:\n{tail}")
    return (completed.stdout or "") if capture else ""
