from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from unrender.audio.separation.models import SpeakerVariant

AUDIOSHAKE_API_BASE = "https://api.audioshake.ai"
HTTP_TIMEOUT_JSON = 120
HTTP_TIMEOUT_UPLOAD = 600
HTTP_TIMEOUT_DOWNLOAD = 300
DEFAULT_POLL_INTERVAL = 5
DEFAULT_TIMEOUT = 1800
DEFAULT_MAX_ATTEMPTS = 4


class AudioShakeError(RuntimeError):
    pass


class AudioShakeNoSpeakersError(AudioShakeError):
    pass


class AudioShakeClient:
    """Small AudioShake Tasks API client for multi-speaker separation."""

    def __init__(self, api_key: str | None = None) -> None:
        resolved = api_key or os.environ.get("AUDIOSHAKE_API_KEY")
        if not resolved:
            raise ValueError(
                "AudioShake API key not found. Set AUDIOSHAKE_API_KEY or pass --api-key."
            )
        self.api_key = resolved
        self.session = requests.Session()
        self.session.headers.update({"x-api-key": self.api_key, "Content-Type": "application/json"})

    def separate_speakers(
        self,
        audio_source: str | Path,
        output_dir: Path,
        *,
        variant: SpeakerVariant = "n_speaker",
        fmt: str = "wav",
        prefix: str = "",
        timeout: int = DEFAULT_TIMEOUT,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> list[Path]:
        source = str(audio_source)
        if source.startswith(("http://", "https://")):
            url = source
            asset_id = None
            print(f"  Using URL source: {url}", flush=True)
        else:
            path = Path(audio_source).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"DX stem not found: {path}")
            url = None
            asset_id = self.upload_asset(path)

        print(f"  Creating AudioShake multi-speaker task (variant={variant})...", flush=True)
        task_id = self.create_task(
            model="multi_voice",
            url=url,
            asset_id=asset_id,
            variant=variant,
            formats=[fmt],
        )
        print(f"  Processing AudioShake task: {task_id}", flush=True)
        task_data = self.wait_for_task(task_id, timeout=timeout, poll_interval=poll_interval)
        print("  Downloading speaker stems...", flush=True)
        return self.download_outputs(task_data, output_dir, prefix=prefix)

    def separate_dme(
        self,
        audio_source: str | Path,
        output_dir: Path,
        *,
        fmt: str = "wav",
        prefix: str = "",
        timeout: int = DEFAULT_TIMEOUT,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> list[Path]:
        source = str(audio_source)
        if source.startswith(("http://", "https://")):
            url = source
            asset_id = None
            print(f"  Using URL source: {url}", flush=True)
        else:
            path = Path(audio_source).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"full audio source not found: {path}")
            url = None
            asset_id = self.upload_asset(path)

        print("  Creating AudioShake DME task (dialogue, music_fx, effects)...", flush=True)
        task_id = self.create_task(
            url=url,
            asset_id=asset_id,
            formats=[fmt],
            targets=[
                {"model": "dialogue", "formats": [fmt]},
                {"model": "music_fx", "formats": [fmt]},
                {"model": "effects", "formats": [fmt]},
            ],
        )
        print(f"  Processing AudioShake DME task: {task_id}", flush=True)
        task_data = self.wait_for_task(task_id, timeout=timeout, poll_interval=poll_interval)
        print("  Downloading DME stems...", flush=True)
        return self.download_outputs(task_data, output_dir, prefix=prefix)

    def separate_targets(
        self,
        audio_source: str | Path,
        output_dir: Path,
        models: list[str],
        *,
        fmt: str = "wav",
        prefix: str = "",
        timeout: int = DEFAULT_TIMEOUT,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> list[Path]:
        """Run one AudioShake task with arbitrary target models.

        Powers both the ``instrumental`` music de-bleed pass and per-cue
        instrument stem splitting; the existing multi-target ``create_task``
        already accepts a ``targets`` list, so this only wires the standard
        upload -> task -> download flow around it.
        """
        if not models:
            raise ValueError("separate_targets requires at least one model")
        source = str(audio_source)
        if source.startswith(("http://", "https://")):
            url: str | None = source
            asset_id: str | None = None
            print(f"  Using URL source: {url}", flush=True)
        else:
            path = Path(audio_source).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"audio source not found: {path}")
            url = None
            asset_id = self.upload_asset(path)

        print(f"  Creating AudioShake task (models={', '.join(models)})...", flush=True)
        task_id = self.create_task(
            url=url,
            asset_id=asset_id,
            formats=[fmt],
            targets=[{"model": model, "formats": [fmt]} for model in models],
        )
        print(f"  Processing AudioShake task: {task_id}", flush=True)
        task_data = self.wait_for_task(task_id, timeout=timeout, poll_interval=poll_interval)
        print("  Downloading target stems...", flush=True)
        return self.download_outputs(task_data, output_dir, prefix=prefix)

    def upload_asset(self, file_path: Path) -> str:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"  Uploading {file_path.name} ({size_mb:.1f} MB)...", flush=True)
        headers = {"x-api-key": self.api_key}
        response: requests.Response | None = None
        for attempt in range(1, DEFAULT_MAX_ATTEMPTS + 1):
            try:
                with file_path.open("rb") as handle:
                    response = requests.post(
                        f"{AUDIOSHAKE_API_BASE}/assets",
                        files={"file": (file_path.name, handle)},
                        headers=headers,
                        timeout=HTTP_TIMEOUT_UPLOAD,
                    )
                if response.status_code >= 500 and attempt < DEFAULT_MAX_ATTEMPTS:
                    _sleep_before_retry("upload", attempt, f"HTTP {response.status_code}")
                    continue
                break
            except requests.exceptions.RequestException as exc:
                if attempt >= DEFAULT_MAX_ATTEMPTS:
                    raise AudioShakeError(
                        f"AudioShake upload failed after {attempt} attempts: {exc}"
                    ) from exc
                _sleep_before_retry("upload", attempt, f"{type(exc).__name__}: {exc}")
        if response is None:
            raise AudioShakeError("AudioShake upload failed before receiving a response")
        if response.status_code >= 400:
            raise AudioShakeError(_http_error("AudioShake upload failed", response))
        asset_id = response.json().get("id")
        if not asset_id:
            raise AudioShakeError(
                f"AudioShake upload response did not include an asset id: {response.text}"
            )
        print("  Upload complete.", flush=True)
        return str(asset_id)

    def create_task(
        self,
        *,
        model: str | None = None,
        url: str | None,
        asset_id: str | None,
        variant: SpeakerVariant | None = None,
        formats: list[str],
        targets: list[dict[str, Any]] | None = None,
    ) -> str:
        if bool(url) == bool(asset_id):
            raise ValueError("provide exactly one of url or asset_id")
        if targets is not None:
            body_targets = []
            for target in targets:
                normalized = dict(target)
                normalized.setdefault("formats", formats)
                body_targets.append(normalized)
        else:
            if model is None:
                raise ValueError("provide model or targets")
            single_target: dict[str, Any] = {"model": model, "formats": formats}
            if variant:
                single_target["variant"] = variant
            body_targets = [single_target]
        body: dict[str, Any] = {"targets": body_targets}
        if url:
            body["url"] = url
        else:
            body["assetId"] = asset_id
        data = self._request("POST", "/tasks", json=body)
        task_id = data.get("id")
        if not task_id:
            raise AudioShakeError(f"AudioShake task response did not include an id: {data}")
        return str(task_id)

    def wait_for_task(self, task_id: str, *, timeout: int, poll_interval: int) -> dict[str, Any]:
        start = time.time()
        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                raise AudioShakeError(f"AudioShake task {task_id} timed out after {timeout}s")
            data = self._request("GET", f"/tasks/{task_id}")
            status = str(data.get("status") or "").lower()
            targets = data.get("targets") or []
            failed_targets = [
                target
                for target in targets
                if str(target.get("status") or "").lower() in {"error", "failed"}
            ]
            if status in {"error", "failed"} or failed_targets:
                if any(_target_error_code(target) == 11 for target in failed_targets):
                    raise AudioShakeNoSpeakersError(
                        f"AudioShake task {task_id} found no speakers: {_task_error(data)}"
                    )
                raise AudioShakeError(f"AudioShake task {task_id} failed: {_task_error(data)}")
            if status == "completed" or (
                targets and all(t.get("status") == "completed" for t in targets)
            ):
                print("", flush=True)
                return data
            completed = sum(1 for target in targets if target.get("status") == "completed")
            total = len(targets) or 1
            print(
                f"\r  [AudioShake] {int(elapsed)}s - {completed}/{total} targets completed",
                end="",
                flush=True,
            )
            time.sleep(poll_interval)

    def download_outputs(
        self, task_data: dict[str, Any], output_dir: Path, *, prefix: str
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []
        for target in task_data.get("targets") or []:
            for output in target.get("output") or []:
                link = output.get("link")
                if not link:
                    continue
                fallback = Path(urlparse(link).path).name or "output.wav"
                filename = Path(str(output.get("stem") or fallback)).name
                if prefix:
                    filename = f"{prefix}_{filename}"
                if not Path(filename).suffix:
                    filename += ".wav"
                destination = output_dir / filename
                print(f"    -> {destination.name}", flush=True)
                response = _download_with_retries(link)
                with destination.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            handle.write(chunk)
                downloaded.append(destination)
        if not downloaded:
            raise AudioShakeError(
                "AudioShake task completed but no downloadable speaker stems were found"
            )
        return downloaded

    def _request(self, method: str, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("timeout", HTTP_TIMEOUT_JSON)
        url = f"{AUDIOSHAKE_API_BASE}{endpoint}"
        response: requests.Response | None = None
        for attempt in range(1, DEFAULT_MAX_ATTEMPTS + 1):
            try:
                response = self.session.request(method, url, **kwargs)
                if response.status_code >= 500 and attempt < DEFAULT_MAX_ATTEMPTS:
                    _sleep_before_retry(
                        f"{method} {endpoint}", attempt, f"HTTP {response.status_code}"
                    )
                    continue
                break
            except requests.exceptions.RequestException as exc:
                if attempt >= DEFAULT_MAX_ATTEMPTS:
                    raise AudioShakeError(
                        f"AudioShake API request failed after {attempt} attempts: {exc}"
                    ) from exc
                _sleep_before_retry(f"{method} {endpoint}", attempt, f"{type(exc).__name__}: {exc}")
        if response is None:
            raise AudioShakeError("AudioShake API request failed before receiving a response")
        if response.status_code >= 400:
            raise AudioShakeError(_http_error("AudioShake API request failed", response))
        data = response.json()
        if not isinstance(data, dict):
            raise AudioShakeError(f"AudioShake API returned unexpected response: {data!r}")
        return data


def _task_error(data: dict[str, Any]) -> str:
    if data.get("error") or data.get("message"):
        return _error_message(data.get("error") or data.get("message"))
    for target in data.get("targets") or []:
        if target.get("error") or target.get("message"):
            return _error_message(target.get("error") or target.get("message"))
    return "unknown error"


def _error_message(value: Any) -> str:
    if isinstance(value, dict) and value.get("message"):
        return str(value["message"])
    return str(value)


def _target_error_code(target: dict[str, Any]) -> int | None:
    error = target.get("error")
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    if code is None:
        return None
    try:
        return int(code)
    except (TypeError, ValueError):
        return None


def _download_with_retries(link: str) -> requests.Response:
    for attempt in range(1, DEFAULT_MAX_ATTEMPTS + 1):
        try:
            response = requests.get(link, stream=True, timeout=HTTP_TIMEOUT_DOWNLOAD)
            if response.status_code >= 500 and attempt < DEFAULT_MAX_ATTEMPTS:
                _sleep_before_retry("download", attempt, f"HTTP {response.status_code}")
                continue
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as exc:
            if attempt >= DEFAULT_MAX_ATTEMPTS:
                raise AudioShakeError(
                    f"AudioShake download failed after {attempt} attempts: {exc}"
                ) from exc
            _sleep_before_retry("download", attempt, f"{type(exc).__name__}: {exc}")
    raise AudioShakeError("AudioShake download failed before receiving a response")


def _sleep_before_retry(label: str, attempt: int, reason: str) -> None:
    backoff = 5 * (2 ** (attempt - 1))
    print(
        f"  {label} attempt {attempt} failed ({reason}). Retrying in {backoff}s...",
        flush=True,
    )
    time.sleep(backoff)


def _http_error(prefix: str, response: requests.Response) -> str:
    return f"{prefix}: HTTP {response.status_code}. {response.text[:1000]}"
