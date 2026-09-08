from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from unrender.audio.separation.audioshake import (
    normalize_source_outputs,
    normalize_speaker_outputs,
    separate_full_audio_source,
    separate_global_dx_stem,
    speech_denoise_file,
)
from unrender.audio.separation.audioshake_client import (
    AudioShakeClient,
    AudioShakeNoSpeakersError,
)
from unrender.audio.separation.input_audio import AudioStream
from unrender.project import RunPaths


class FakeAudioShakeClient:
    def __init__(
        self,
        downloaded: list[Path] | None = None,
        dme_downloaded: list[Path] | None = None,
        speaker_downloaded: list[Path] | None = None,
    ) -> None:
        self.downloaded = downloaded or []
        self.dme_downloaded = dme_downloaded or []
        self.speaker_downloaded = speaker_downloaded or []
        self.calls = 0
        self.dme_calls = 0
        self.speaker_calls = 0
        self.targets_calls = 0
        self.last_speaker_source = None
        self.last_targets_models: list[str] | None = None
        self.last_targets_source = None

    def separate_speakers(self, *args, **kwargs) -> list[Path]:
        self.calls += 1
        self.speaker_calls += 1
        self.last_speaker_source = args[0] if args else None
        return self.speaker_downloaded or self.downloaded

    def separate_dme(self, *args, **kwargs) -> list[Path]:
        self.dme_calls += 1
        return self.dme_downloaded or self.downloaded

    def separate_targets(
        self, audio_source, output_dir: Path, models: list[str], **kwargs
    ) -> list[Path]:
        self.targets_calls += 1
        self.last_targets_source = audio_source
        self.last_targets_models = list(models)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        for model in models:
            path = output_dir / f"clip_{model}.wav"
            path.write_bytes(b"clean")
            outputs.append(path)
        return outputs


def test_wait_for_task_treats_error_status_as_terminal_no_speakers(monkeypatch) -> None:
    client = object.__new__(AudioShakeClient)
    monkeypatch.setattr(
        client,
        "_request",
        lambda *args, **kwargs: {
            "targets": [
                {
                    "status": "error",
                    "error": {
                        "code": 11,
                        "message": "No speakers detected in vocal diarization",
                    },
                }
            ]
        },
    )

    with pytest.raises(AudioShakeNoSpeakersError, match="No speakers detected"):
        client.wait_for_task("task-id", timeout=30, poll_interval=0)


def test_speaker_separation_records_valid_empty_result_when_no_speakers(tmp_path: Path) -> None:
    class NoSpeakersClient(FakeAudioShakeClient):
        def separate_speakers(self, *args, **kwargs) -> list[Path]:
            self.calls += 1
            raise AudioShakeNoSpeakersError("No speakers detected in vocal diarization")

    run = RunPaths.from_path(tmp_path / "run")
    dialogue = tmp_path / "dialogue.wav"
    dialogue.write_bytes(b"silent")
    client = NoSpeakersClient()

    result = separate_global_dx_stem(
        run=run,
        dx_stem=dialogue,
        client=client,
        prefix="show-a",
    )

    assert result.stems == []
    manifest = json.loads(run.audio_separation_json.read_text(encoding="utf-8"))
    assert manifest["status"] == "no_speakers_detected"
    assert manifest["stems"] == []

    skipped = separate_global_dx_stem(
        run=run,
        dx_stem=dialogue,
        client=client,
        prefix="show-a",
    )

    assert skipped.skipped is True
    assert skipped.stems == []
    assert client.calls == 1


def test_normalize_speaker_outputs_uses_stable_unmapped_names(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    raw.mkdir()
    first = raw / "voice_1_speaker_03.wav"
    second = raw / "voice_2.wav"
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    stems = normalize_speaker_outputs([second, first], out, prefix="show-a", force=False)

    assert [path.name for path in stems] == [
        "show-a_speaker_02_stem.wav",
        "show-a_speaker_03_stem.wav",
    ]
    assert all(path.exists() for path in stems)


def test_normalize_source_outputs_uses_stable_dme_names(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    raw.mkdir()
    dialogue = raw / "task_dialogue.wav"
    music = raw / "task_music_fx.wav"
    effects = raw / "task_effects.wav"
    for path in [dialogue, music, effects]:
        path.write_bytes(b"audio")

    stems = normalize_source_outputs([effects, dialogue, music], out, prefix="show-a", force=False)

    assert {role: path.name for role, path in stems.items()} == {
        "dialogue": "show-a_DX_stem.wav",
        "effects": "show-a_FX_stem.wav",
        "music_fx": "show-a_MX_stem.wav",
    }
    assert all(path.exists() for path in stems.values())


def test_normalize_source_outputs_ignores_role_tokens_in_prefix(tmp_path: Path) -> None:
    # A prefix like "redux" contains "dx"; roles must come from the stem name
    # AudioShake produced, never from the user-supplied prefix.
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    raw.mkdir()
    names = ["redux_dialogue.wav", "redux_music_fx.wav", "redux_effects.wav"]
    for name in names:
        (raw / name).write_bytes(b"audio")

    stems = normalize_source_outputs(
        [raw / name for name in names], out, prefix="redux", force=False
    )

    assert {role: path.name for role, path in stems.items()} == {
        "dialogue": "redux_DX_stem.wav",
        "effects": "redux_FX_stem.wav",
        "music_fx": "redux_MX_stem.wav",
    }


def test_normalize_speaker_outputs_ignores_digits_in_prefix(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    raw.mkdir()
    first = raw / "speaker_9_project_voice_1.wav"
    second = raw / "speaker_9_project_voice_2.wav"
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    stems = normalize_speaker_outputs([first, second], out, prefix="speaker_9_project", force=False)

    # Without prefix stripping both files would claim speaker 09 and collide.
    assert [path.name for path in stems] == [
        "speaker_9_project_speaker_01_stem.wav",
        "speaker_9_project_speaker_02_stem.wav",
    ]


def test_speech_denoise_file_runs_speech_denoise_target(tmp_path: Path) -> None:
    client = FakeAudioShakeClient()
    source = tmp_path / "prompt.wav"
    source.write_bytes(b"noisy")

    cleaned = speech_denoise_file(source, tmp_path / "denoise", client=client)

    assert client.targets_calls == 1
    assert client.last_targets_models == ["speech_denoise"]
    assert client.last_targets_source == source
    assert cleaned.name == "clip_speech_denoise.wav"
    assert cleaned.exists()


def test_separate_full_audio_source_writes_manifest_and_skips_existing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    full_audio = tmp_path / "full.wav"
    full_audio.write_bytes(b"audio")
    monkeypatch.setattr(
        "unrender.audio.separation.input_audio.probe_audio_streams",
        lambda source, **kwargs: (AudioStream(0, 2, "stereo", "pcm_s24le", 48_000),),
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    downloaded = [raw / "dialogue.wav", raw / "music_fx.wav", raw / "effects.wav"]
    for path in downloaded:
        path.write_bytes(b"audio")
    client = FakeAudioShakeClient(dme_downloaded=downloaded)

    result = separate_full_audio_source(
        run=run,
        full_audio=full_audio,
        client=client,
        prefix="show-a",
        force=False,
    )

    assert client.dme_calls == 1
    assert result.dialogue_stem == run.source_stems_dir / "show-a_DX_stem.wav"
    manifest = json.loads(run.source_separation_json.read_text(encoding="utf-8"))
    assert manifest["dialogue_stem"] == str(result.dialogue_stem)
    assert manifest["stems"]["music_fx"].endswith("show-a_MX_stem.wav")

    skipped = separate_full_audio_source(
        run=run,
        full_audio=full_audio,
        client=client,
        prefix="show-a",
        force=False,
    )

    assert client.dme_calls == 1
    assert skipped.skipped
    assert skipped.dialogue_stem == result.dialogue_stem


def test_source_manifest_records_embedded_audio_normalization(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    source = tmp_path / "delivery.mov"
    source.write_bytes(b"media")
    raw = tmp_path / "raw"
    raw.mkdir()
    downloaded = [raw / "dialogue.wav", raw / "music_fx.wav", raw / "effects.wav"]
    for path in downloaded:
        path.write_bytes(b"audio")

    streams = [
        {
            "index": index,
            "channels": 1,
            "channel_layout": "mono",
            "codec_name": "pcm_s24le",
            "sample_rate": "48000",
        }
        for index in range(1, 9)
    ]

    def fake_run(command, **kwargs):
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"streams": streams}),
                stderr="",
            )
        target = Path(command[-1])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"stereo pcm")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("unrender.audio.separation.input_audio.subprocess.run", fake_run)

    result = separate_full_audio_source(
        run=run,
        full_audio=source,
        client=FakeAudioShakeClient(dme_downloaded=downloaded),
        prefix="show-a",
    )

    manifest = json.loads(run.source_separation_json.read_text(encoding="utf-8"))
    provenance = manifest["input_audio"]
    assert provenance["source"] == str(source)
    assert provenance["prepared_source"] == str(run.normalized_audio_dir / "show-a_full_audio.wav")
    assert provenance["selection"] == "first_mono_pair_2_0_program"
    assert provenance["selected_streams"] == [1, 2]
    assert result.source == str(source)


def test_separate_global_dx_stem_writes_manifest_and_skips_existing(tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    raw = tmp_path / "raw"
    raw.mkdir()
    downloaded = [raw / "audioshake_speaker_01.wav", raw / "audioshake_speaker_02.wav"]
    for path in downloaded:
        path.write_bytes(b"audio")
    client = FakeAudioShakeClient(downloaded)

    result = separate_global_dx_stem(
        run=run,
        dx_stem=tmp_path / "dx.wav",
        client=client,
        prefix="show-a",
        force=False,
    )

    assert client.calls == 1
    assert not result.skipped
    assert [path.name for path in result.stems] == [
        "show-a_speaker_01_stem.wav",
        "show-a_speaker_02_stem.wav",
    ]
    manifest = json.loads(run.audio_separation_json.read_text(encoding="utf-8"))
    assert manifest["backend"] == "audioshake"
    assert manifest["skipped"] is False

    skipped = separate_global_dx_stem(
        run=run,
        dx_stem=tmp_path / "dx.wav",
        client=client,
        prefix="show-a",
        force=False,
    )

    assert client.calls == 1
    assert skipped.skipped
    assert len(skipped.stems) == 2


def test_separation_never_reuses_stems_after_source_changes(tmp_path: Path) -> None:
    run = RunPaths.from_path(tmp_path / "run")
    raw = tmp_path / "raw"
    raw.mkdir()
    downloaded = [raw / "audioshake_speaker_01.wav"]
    downloaded[0].write_bytes(b"audio")
    source = tmp_path / "dx.wav"
    source.write_bytes(b"first")
    client = FakeAudioShakeClient(downloaded)
    separate_global_dx_stem(
        run=run,
        dx_stem=source,
        client=client,
        prefix="show-a",
    )
    source.write_bytes(b"changed source")

    with pytest.raises(ValueError, match="changed input"):
        separate_global_dx_stem(
            run=run,
            dx_stem=source,
            client=client,
            prefix="show-a",
        )
