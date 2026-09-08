"""Montreal Forced Aligner external-process adapter and TextGrid parser."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .contract import (
    CapabilityState,
    LicenseMetadata,
    ModelProvenance,
    PackageProvenance,
    ProviderPolicy,
    ProviderProvenance,
    ProviderResult,
    statuses_for_capabilities,
)

_ITEM_RE = re.compile(
    r"(?ms)^[ \t]*item[ \t]*\[\d+\][ \t]*:[ \t]*\n" r"(.*?)(?=^[ \t]*item[ \t]*\[\d+\][ \t]*:|\Z)"
)
_INTERVAL_RE = re.compile(
    r"(?ms)^[ \t]*intervals[ \t]*\[\d+\][ \t]*:[^\n]*\n"
    r"[ \t]*xmin[ \t]*=[ \t]*([-+0-9.eE]+)[ \t]*\n"
    r"[ \t]*xmax[ \t]*=[ \t]*([-+0-9.eE]+)[ \t]*\n"
    r'[ \t]*text[ \t]*=[ \t]*"((?:""|[^"])*)"'
)
_TIER_NAME_RE = re.compile(r'(?m)^[ \t]*name[ \t]*=[ \t]*"((?:""|[^"])*)"')
_OOV_LABELS = {"<unk>", "<oov>", "oov", "spn", "<spn>"}


class MfaProvider:
    provider_id = "mfa"
    capabilities = ("word_alignment", "phone_alignment")
    license_metadata = LicenseMetadata(
        license_id="MIT (runtime); model-dependent",
        source="https://montreal-forced-aligner.readthedocs.io/",
        permissive=None,
    )

    def __init__(
        self,
        *,
        executable: str = "mfa",
        dictionary: str | Path | None = None,
        acoustic_model: str | Path | None = None,
        dictionary_revision: str | None = None,
        acoustic_model_revision: str | None = None,
    ) -> None:
        self.executable = executable
        self.dictionary = str(dictionary) if dictionary is not None else None
        self.acoustic_model = str(acoustic_model) if acoustic_model is not None else None
        self.dictionary_revision = dictionary_revision
        self.acoustic_model_revision = acoustic_model_revision

    @property
    def provenance(self) -> ProviderProvenance:
        models: list[ModelProvenance] = []
        if self.dictionary:
            models.append(
                ModelProvenance(
                    model_id=self.dictionary,
                    revision=self.dictionary_revision,
                    source="mfa_dictionary",
                )
            )
        if self.acoustic_model:
            models.append(
                ModelProvenance(
                    model_id=self.acoustic_model,
                    revision=self.acoustic_model_revision,
                    source="mfa_acoustic",
                )
            )
        version = _mfa_version(self.executable)
        return ProviderProvenance(
            packages=(PackageProvenance(name="montreal-forced-aligner", version=version),),
            models=tuple(models),
        )

    def preflight(self, policy: ProviderPolicy | None = None):
        active_policy = policy or ProviderPolicy()
        blocked = active_policy.block_reason(self.provider_id, self.license_metadata)
        executable = _resolve_executable(self.executable)
        if blocked:
            state, reason = CapabilityState.BLOCKED_BY_POLICY, blocked
        elif executable is None:
            state = CapabilityState.UNAVAILABLE
            reason = f"MFA executable is not available: {self.executable}"
        elif not self.dictionary or not self.acoustic_model:
            state = CapabilityState.UNAVAILABLE
            reason = "MFA requires explicit dictionary and acoustic-model identifiers"
        else:
            missing = [
                f"{kind}={reference}"
                for kind, reference in (
                    ("dictionary", self.dictionary),
                    ("acoustic", self.acoustic_model),
                )
                if not _mfa_model_available(executable, kind, reference)
            ]
            if missing:
                state = CapabilityState.UNAVAILABLE
                reason = "MFA model unavailable: " + ", ".join(missing)
            else:
                state = CapabilityState.AVAILABLE
                reason = "MFA executable, dictionary, and acoustic model are available"
        return statuses_for_capabilities(
            provider_id=self.provider_id,
            capabilities=self.capabilities,
            state=state,
            reason=reason,
            provenance=self.provenance,
            license_metadata=self.license_metadata,
            backend="mfa_external_process",
        )

    def align(
        self,
        audio_path: str | Path,
        *,
        transcript: str,
        language: str,
        dictionary: str | Path | None = None,
        acoustic_model: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> ProviderResult:
        dictionary_ref = str(dictionary) if dictionary is not None else self.dictionary
        acoustic_ref = str(acoustic_model) if acoustic_model is not None else self.acoustic_model
        unsupported = _unsupported_request(
            audio_path=audio_path,
            transcript=transcript,
            language=language,
            dictionary=dictionary_ref,
            acoustic_model=acoustic_ref,
        )
        if unsupported:
            return self._failure(unsupported)

        executable = _resolve_executable(self.executable)
        if executable is None:
            return ProviderResult(
                provider_id=self.provider_id,
                capability="word_alignment",
                state=CapabilityState.UNAVAILABLE,
                provenance=self.provenance,
                error=f"MFA executable is not available: {self.executable}",
            )
        for kind, reference in (("dictionary", dictionary_ref), ("acoustic", acoustic_ref)):
            assert reference is not None
            if not _mfa_model_available(executable, kind, reference):
                return self._failure(f"MFA {kind} model is unavailable: {reference}")

        source = Path(audio_path)
        try:
            with tempfile.TemporaryDirectory(prefix="unrender-mfa-") as temporary:
                workspace = Path(temporary)
                corpus = workspace / "corpus"
                aligned = Path(output_dir) if output_dir is not None else workspace / "aligned"
                corpus.mkdir(parents=True)
                aligned.mkdir(parents=True, exist_ok=True)
                local_audio = corpus / source.name
                shutil.copy2(source, local_audio)
                local_audio.with_suffix(".lab").write_text(transcript.strip(), encoding="utf-8")
                command = [
                    executable,
                    "align",
                    str(corpus),
                    str(dictionary_ref),
                    str(acoustic_ref),
                    str(aligned),
                    "--clean",
                    "--single_speaker",
                ]
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=3600,
                )
                if completed.returncode != 0:
                    detail = (completed.stderr or completed.stdout or "unknown MFA error").strip()
                    category = "oov" if "oov" in detail.lower() else "external_process"
                    return self._failure(f"MFA {category} failure: {detail}")
                textgrids = sorted(aligned.rglob("*.TextGrid"))
                if not textgrids:
                    return self._failure("MFA completed but produced no TextGrid")
                parsed = parse_textgrid(textgrids[0])
        except (OSError, subprocess.SubprocessError) as exc:
            return self._failure(f"MFA process failed: {exc}")

        oov = _find_oov(parsed)
        failures = [{"kind": "oov", "label": label} for label in oov]
        data: dict[str, Any] = {
            **parsed,
            "language": language,
            "dictionary": dictionary_ref,
            "acoustic_model": acoustic_ref,
            "failures": failures,
        }
        if oov:
            return ProviderResult(
                provider_id=self.provider_id,
                capability="word_alignment",
                state=CapabilityState.FAILED,
                data=data,
                provenance=self.provenance,
                error="MFA emitted out-of-vocabulary intervals: " + ", ".join(oov),
            )
        return ProviderResult(
            provider_id=self.provider_id,
            capability="word_alignment",
            state=CapabilityState.COMPLETE,
            data=data,
            metadata={"backend": "mfa_external_process"},
            provenance=self.provenance,
        )

    def _failure(self, error: str) -> ProviderResult:
        return ProviderResult(
            provider_id=self.provider_id,
            capability="word_alignment",
            state=CapabilityState.FAILED,
            provenance=self.provenance,
            error=error,
        )


def parse_textgrid(path_or_text: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Parse long TextGrid interval tiers into honest word and phone records."""

    if isinstance(path_or_text, Path):
        text = path_or_text.read_text(encoding="utf-8")
    else:
        candidate = Path(path_or_text)
        try:
            text = candidate.read_text(encoding="utf-8") if candidate.exists() else path_or_text
        except (OSError, ValueError):
            text = path_or_text

    words: list[dict[str, Any]] = []
    phones: list[dict[str, Any]] = []
    for item in _ITEM_RE.findall(text):
        name_match = _TIER_NAME_RE.search(item)
        if not name_match:
            continue
        tier_name = _unquote(name_match.group(1)).strip().lower()
        if tier_name in {"word", "words", "transcript"}:
            destination, label_key = words, "word"
        elif tier_name in {"phone", "phones", "phoneme", "phonemes"}:
            destination, label_key = phones, "phone"
        else:
            continue
        for start, end, label in _INTERVAL_RE.findall(item):
            value = _unquote(label).strip()
            if value:
                destination.append(
                    {
                        label_key: value,
                        "start": float(start),
                        "end": float(end),
                        "tier": tier_name,
                    }
                )
    if not words and not phones:
        raise ValueError("unsupported or empty TextGrid interval format")
    return {"words": words, "phones": phones}


def _resolve_executable(executable: str) -> str | None:
    candidate = Path(executable).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        return str(candidate) if candidate.is_file() else None
    return shutil.which(executable)


def _mfa_model_available(executable: str, kind: str, reference: str) -> bool:
    path = Path(reference).expanduser()
    if path.exists():
        return path.is_file() or path.is_dir()
    try:
        result = subprocess.run(
            [executable, "model", "inspect", kind, reference],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _mfa_version(executable: str) -> str | None:
    resolved = _resolve_executable(executable)
    if resolved is None:
        return None
    try:
        result = subprocess.run(
            [resolved, "version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    match = re.search(r"\d+(?:\.\d+)+(?:[-+.\w]*)?", result.stdout)
    return match.group(0) if match else None


def _unsupported_request(
    *,
    audio_path: str | Path,
    transcript: str,
    language: str,
    dictionary: str | None,
    acoustic_model: str | None,
) -> str | None:
    if not Path(audio_path).is_file():
        return f"audio file does not exist: {audio_path}"
    if not transcript.strip():
        return "MFA requires a non-empty transcript"
    if not language.strip():
        return "MFA requires an explicit language"
    if not dictionary:
        return "MFA requires an explicit pronunciation dictionary"
    if not acoustic_model:
        return "MFA requires an explicit acoustic model"
    return None


def _find_oov(parsed: dict[str, list[dict[str, Any]]]) -> list[str]:
    labels: list[str] = []
    for interval in (*parsed.get("words", []), *parsed.get("phones", [])):
        value = str(interval.get("word") or interval.get("phone") or "").strip()
        if value.lower() in _OOV_LABELS and value not in labels:
            labels.append(value)
    return labels


def _unquote(value: str) -> str:
    return value.replace('""', '"')
