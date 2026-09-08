from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unrender.audio.analysis import AnalysisSettings, AudioArtifact, AudioMetadata
from unrender.audio.analysis.providers import (
    BrouhahaProvider,
    CapabilityState,
    CapabilityStatus,
    ClapProvider,
    MfaProvider,
    Profile,
    ProfilePreflightError,
    ProviderPolicy,
    ProviderProvenance,
    ProviderRegistry,
    ProviderResult,
    WhisperXProvider,
    default_registry,
    parse_textgrid,
)
from unrender.audio.analysis.providers.execution import run_installed_providers


def test_missing_optional_dependency_is_reported_without_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "unrender.audio.analysis.providers.whisperx.module_available", lambda _module: False
    )
    provider = WhisperXProvider(model_revision="model-commit")

    status = provider.preflight()[0]
    result = provider.align("missing.wav")

    assert status.state is CapabilityState.UNAVAILABLE
    assert result.state is CapabilityState.UNAVAILABLE
    assert status.provenance.models[0].revision == "model-commit"


def test_brouhaha_requires_gated_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "unrender.audio.analysis.providers.brouhaha.module_available", lambda _module: True
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")
    provider = BrouhahaProvider(
        enabled=True,
        hf_token="fake-token",
        model_revision="model-commit",
    )

    blocked = provider.preflight(ProviderPolicy())[0]
    opted_in = provider.preflight(
        ProviderPolicy(allow_gated=True, explicit_opt_ins=frozenset({"brouhaha"}))
    )[0]
    permissive_only = provider.preflight(
        ProviderPolicy(
            permissive_only=True,
            allow_gated=True,
            explicit_opt_ins=frozenset({"brouhaha"}),
        )
    )[0]

    assert blocked.state is CapabilityState.BLOCKED_BY_POLICY
    assert opted_in.state is CapabilityState.AVAILABLE
    assert permissive_only.state is CapabilityState.BLOCKED_BY_POLICY


def test_default_registry_enables_explicit_gated_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "unrender.audio.analysis.providers.brouhaha.module_available", lambda _module: True
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")
    registry = default_registry(enabled_gated=("brouhaha",))
    report = registry.preflight(
        Profile.AUTO,
        configured_providers=("brouhaha",),
        policy=ProviderPolicy(
            allow_gated=True,
            explicit_opt_ins=frozenset({"brouhaha"}),
        ),
    )

    assert report.statuses[0].state is CapabilityState.AVAILABLE


def test_provider_execution_materializes_canonical_word_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeWhisper:
        def __init__(self, **_kwargs) -> None:
            pass

        def preflight(self, _policy):
            return [
                CapabilityStatus(
                    provider_id="whisperx",
                    capability="word_alignment",
                    state=CapabilityState.AVAILABLE,
                )
            ]

        def align(self, _path, *, language=None):
            return ProviderResult(
                provider_id="whisperx",
                capability="word_alignment",
                state=CapabilityState.COMPLETE,
                data={
                    "words": [{"word": "hello", "start": 0.1, "end": 0.4}],
                    "characters": [],
                },
                metadata={"language": language or "en"},
                provenance=ProviderProvenance(),
            )

    monkeypatch.setattr(
        "unrender.audio.analysis.providers.execution.WhisperXProvider",
        FakeWhisper,
    )
    artifact = AudioArtifact(
        id="dx-123",
        role="dialogue",
        path=str(tmp_path / "dx.wav"),
        hash="abc",
        audio=AudioMetadata(16_000, 1, 16_000, 1.0, "pcm", "s16"),
    )

    features, results = run_installed_providers(
        artifacts=[artifact],
        settings=AnalysisSettings(
            profile="auto",
            providers=("whisperx",),
            language="en",
        ),
        analysis_dir=tmp_path,
        policy=ProviderPolicy(permissive_only=False),
    )

    assert results[0]["state"] == "complete"
    assert features[0].name == "model_words"
    assert features[0].value[0]["word"] == "hello"


def test_brouhaha_emits_named_frame_series(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "unrender.audio.analysis.providers.brouhaha.module_available", lambda _module: True
    )
    provider = BrouhahaProvider(
        enabled=True,
        hf_token="fake-token",
        model_revision="model-commit",
    )
    output = SimpleNamespace(
        data=np.asarray([[0.75, 18.0, 4.5], [0.25, 6.0, -1.0]], dtype=np.float32),
        sliding_window=SimpleNamespace(start=0.1, step=0.02),
    )
    monkeypatch.setattr(provider, "_load_inference", lambda: lambda _path: output)
    policy = ProviderPolicy(
        allow_gated=True,
        explicit_opt_ins=frozenset({"brouhaha"}),
    )

    result = provider.analyze("fake.wav", policy=policy)

    assert result.state is CapabilityState.COMPLETE
    assert result.data["series"]["vad"][0] == {"time_sec": 0.1, "probability": 0.75}
    assert result.data["series"]["snr"][0]["snr_db"] == 18.0
    assert result.data["series"]["c50_estimate_db"][1]["c50_estimate_db"] == -1.0


def test_provenance_records_exact_package_and_model_revisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "unrender.audio.analysis.providers.contract.importlib.metadata.version",
        lambda distribution: {"transformers": "9.1.2", "torch": "3.4.5"}[distribution],
    )
    provider = ClapProvider(model_revision="0123456789abcdef")

    provenance = provider.provenance

    assert provenance.exact
    assert [(package.name, package.version) for package in provenance.packages] == [
        ("transformers", "9.1.2"),
        ("torch", "3.4.5"),
    ]
    assert provenance.models[0].model_id == "laion/clap-htsat-unfused"
    assert provenance.models[0].revision == "0123456789abcdef"


class _FakeProcessor:
    def __call__(self, **kwargs):
        return kwargs


class _FakeClapModel:
    def get_audio_features(self, **_kwargs):
        return np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)

    def get_text_features(self, **_kwargs):
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


def test_clap_embeddings_are_float16_normalized_and_cosine_searchable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ClapProvider(model_revision="model-commit")
    monkeypatch.setattr(
        provider,
        "_load_backend",
        lambda: (_FakeClapModel(), _FakeProcessor(), SimpleNamespace(no_grad=nullcontext)),
    )

    result = provider.embed(
        [np.ones(8, dtype=np.float32), np.zeros(8, dtype=np.float32)],
        sample_rate=48_000,
        times=[(0.0, 1.0), (1.0, 2.0)],
    )
    vectors = [record["embedding"] for record in result.data["embeddings"]]
    matches = provider.search(np.asarray([0.0, 1.0]), result)
    prompt_scores = provider.score_prompts(np.asarray([1.0, 0.0]), ["speech", "music"])

    assert result.state is CapabilityState.COMPLETE
    assert all(vector.dtype == np.float16 for vector in vectors)
    vector_norms = [float(np.linalg.norm(vector.astype(np.float32))) for vector in vectors]
    assert vector_norms == pytest.approx([1.0, 1.0], abs=1e-3)
    assert matches[0]["index"] == 1
    assert "cosine_score" in matches[0]
    assert prompt_scores["calibrated"] is False
    assert sum(row["prompt_set_probability"] for row in prompt_scores["scores"]) == pytest.approx(
        1.0
    )
    assert all("cosine_score" in row for row in prompt_scores["scores"])


def test_textgrid_parser_keeps_words_and_phones_distinct() -> None:
    textgrid = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 1
            text = "hello"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 0.4
            text = "HH"
        intervals [2]:
            xmin = 0.4
            xmax = 1
            text = "AH"
"""

    parsed = parse_textgrid(textgrid)

    assert parsed["words"] == [{"word": "hello", "start": 0.0, "end": 1.0, "tier": "words"}]
    assert [interval["phone"] for interval in parsed["phones"]] == ["HH", "AH"]
    assert all("grapheme" not in interval for interval in parsed["phones"])


def test_mfa_missing_models_fail_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "unrender.audio.analysis.providers.mfa._resolve_executable", lambda _executable: "/fake/mfa"
    )
    monkeypatch.setattr(
        "unrender.audio.analysis.providers.mfa._mfa_version", lambda _executable: "3.3.7"
    )
    monkeypatch.setattr(
        "unrender.audio.analysis.providers.mfa._mfa_model_available",
        lambda _executable, kind, _reference: kind == "dictionary",
    )
    provider = MfaProvider(dictionary="english", acoustic_model="english_us")

    statuses = provider.preflight()

    assert {status.state for status in statuses} == {CapabilityState.UNAVAILABLE}
    assert "acoustic=english_us" in statuses[0].reason
    assert statuses[0].provenance.packages[0].version == "3.3.7"


def test_whisperx_persists_alignment_degradation_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "unrender.audio.analysis.providers.whisperx.module_available", lambda _module: True
    )

    def fake_helper(_path: Path, **_kwargs):
        return (
            [{"word": "Hi", "start": 1.0, "end": 1.4, "speaker": ""}],
            {"language": "en", "aligned": True},
        )

    monkeypatch.setattr("unrender.audio.asr.transcribe_word_aligned", fake_helper)
    result = WhisperXProvider(model_revision="model-commit").align("audio.wav")

    assert result.state is CapabilityState.COMPLETE
    assert result.metadata["timing_quality"]["word"] == "model_aligned"
    assert result.metadata["timing_quality"]["character"] == "interpolated_from_words"
    assert result.metadata["timing_quality"]["degraded"] is True
    assert [item["character"] for item in result.data["characters"]] == ["H", "i"]


class _FakeProvider:
    capabilities = ("example",)

    def __init__(self, provider_id: str, state: CapabilityState) -> None:
        self.provider_id = provider_id
        self.state = state

    def preflight(self, _policy=None):
        return [
            CapabilityStatus(
                provider_id=self.provider_id,
                capability="example",
                state=self.state,
                reason="fake status",
            )
        ]


def test_profile_preflight_auto_records_and_full_fails() -> None:
    registry = ProviderRegistry(
        [
            _FakeProvider("ready", CapabilityState.AVAILABLE),
            _FakeProvider("missing", CapabilityState.UNAVAILABLE),
        ]
    )

    automatic = registry.preflight(Profile.AUTO)

    assert automatic.ok
    assert [status.state for status in automatic.statuses] == [
        CapabilityState.AVAILABLE,
        CapabilityState.UNAVAILABLE,
    ]
    with pytest.raises(ProfilePreflightError) as error:
        registry.preflight(Profile.FULL, configured_providers=["ready", "missing"])
    assert error.value.report.ok is False


def test_core_profile_has_no_optional_provider_requirements() -> None:
    registry = ProviderRegistry([_FakeProvider("missing", CapabilityState.UNAVAILABLE)])

    report = registry.preflight(Profile.CORE)

    assert report.ok
    assert report.statuses == ()
