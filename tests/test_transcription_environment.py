from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from unrender.audio import asr, transcription_worker


def _fake_interpreter(tmp_path: Path) -> Path:
    interpreter = tmp_path / "transcription-python"
    interpreter.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    interpreter.chmod(0o755)
    return interpreter


def test_word_alignment_uses_configured_worker_and_ignores_stdout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    interpreter = _fake_interpreter(tmp_path)
    monkeypatch.setenv("UNRENDER_TRANSCRIPTION_PYTHON", str(interpreter))
    captured: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["env"] = kwargs["env"]
        request_path = Path(command[-2])
        response_path = Path(command[-1])
        captured["request"] = json.loads(request_path.read_text(encoding="utf-8"))
        response_path.write_text(
            json.dumps(
                {
                    "ok": True,
                    "result": {
                        "words": [
                            {
                                "word": "hello",
                                "start": 0.0,
                                "end": 0.5,
                                "speaker": "",
                            }
                        ],
                        "metadata": {"language": "en", "aligned": True},
                    },
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="model log\n{not json}", stderr="")

    monkeypatch.setattr(asr.subprocess, "run", fake_run)

    words, metadata = asr.transcribe_word_aligned(tmp_path / "audio.wav")

    assert words[0]["word"] == "hello"
    assert metadata == {"language": "en", "aligned": True}
    assert captured["command"][1:3] == [
        "-m",
        "unrender.audio.transcription_worker",
    ]
    assert captured["request"]["operation"] == "transcribe_word_aligned"
    assert captured["env"]["_UNRENDER_TRANSCRIPTION_WORKER"] == "1"


def test_same_interpreter_keeps_word_alignment_in_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("UNRENDER_TRANSCRIPTION_PYTHON", sys.executable)
    expected = ([{"word": "local"}], {"language": "en", "aligned": False})
    monkeypatch.setattr(
        asr, "_transcribe_word_aligned_in_process", lambda *args, **kwargs: expected
    )
    monkeypatch.setattr(
        asr.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("same interpreter must not launch a worker"),
    )

    assert asr.transcribe_word_aligned(tmp_path / "audio.wav") == expected


def test_distinct_virtualenv_launcher_is_not_collapsed_to_base_python(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    interpreter = tmp_path / "other-venv-python"
    interpreter.symlink_to(sys.executable)
    monkeypatch.setenv("UNRENDER_TRANSCRIPTION_PYTHON", str(interpreter))

    assert asr._external_transcription_python() == str(interpreter)


def test_worker_guard_prevents_recursive_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("UNRENDER_TRANSCRIPTION_PYTHON", str(_fake_interpreter(tmp_path)))
    monkeypatch.setenv("_UNRENDER_TRANSCRIPTION_WORKER", "1")

    assert asr._external_transcription_python() is None


def test_path_segments_use_worker_but_arrays_remain_in_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    interpreter = _fake_interpreter(tmp_path)
    monkeypatch.setenv("UNRENDER_TRANSCRIPTION_PYTHON", str(interpreter))
    operations: list[str] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        request = json.loads(Path(command[-2]).read_text(encoding="utf-8"))
        operations.append(request["operation"])
        Path(command[-1]).write_text(
            json.dumps(
                {
                    "ok": True,
                    "result": {"segments": [{"text": "remote"}]},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(asr.subprocess, "run", fake_run)
    monkeypatch.setattr(
        asr,
        "_transcribe_segments_in_process",
        lambda *args, **kwargs: [{"text": "local"}],
    )

    assert asr.transcribe_path(tmp_path / "audio.wav") == "remote"
    assert asr.transcribe_segments(np.zeros(16, dtype=np.float32)) == [{"text": "local"}]
    assert operations == ["transcribe_path_segments"]


def test_missing_or_non_executable_interpreter_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-python"
    monkeypatch.setenv("UNRENDER_TRANSCRIPTION_PYTHON", str(missing))
    with pytest.raises(RuntimeError, match=r"UNRENDER_TRANSCRIPTION_PYTHON.*missing"):
        asr.transcribe_word_aligned(tmp_path / "audio.wav")

    not_executable = tmp_path / "not-executable"
    not_executable.write_text("", encoding="utf-8")
    monkeypatch.setenv("UNRENDER_TRANSCRIPTION_PYTHON", str(not_executable))
    with pytest.raises(RuntimeError, match=r"UNRENDER_TRANSCRIPTION_PYTHON.*executable"):
        asr.transcribe_word_aligned(tmp_path / "audio.wav")


@pytest.mark.parametrize(
    ("response", "returncode", "match"),
    [
        ("not-json", 0, "UNRENDER_TRANSCRIPTION_PYTHON.*malformed JSON"),
        (
            json.dumps(
                {
                    "ok": False,
                    "error": {
                        "type": "ImportError",
                        "message": "word-level transcription requires whisperx",
                    },
                }
            ),
            1,
            "UNRENDER_TRANSCRIPTION_PYTHON.*requires whisperx",
        ),
    ],
)
def test_worker_failures_are_actionable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    response: str,
    returncode: int,
    match: str,
) -> None:
    interpreter = _fake_interpreter(tmp_path)
    monkeypatch.setenv("UNRENDER_TRANSCRIPTION_PYTHON", str(interpreter))

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        Path(command[-1]).write_text(response, encoding="utf-8")
        return subprocess.CompletedProcess(command, returncode, stdout="", stderr="")

    monkeypatch.setattr(asr.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=match):
        asr.transcribe_word_aligned(tmp_path / "audio.wav")


def test_worker_validates_operation_and_payload() -> None:
    with pytest.raises(ValueError, match="unsupported transcription operation"):
        transcription_worker.execute_request(
            {"version": 1, "operation": "transcribe", "payload": {}}
        )
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        transcription_worker.execute_request(
            {
                "version": 1,
                "operation": "transcribe_word_aligned",
                "payload": {
                    "path": "audio.wav",
                    "model_name": "small",
                    "device": "cpu",
                    "compute_type": "float32",
                    "language": None,
                    "batch_size": 0,
                },
            }
        )


def test_worker_writes_response_separately_from_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    request_path.write_text("{}", encoding="utf-8")

    def fake_execute(request: Any) -> dict[str, Any]:
        print("WhisperX emitted a log line")
        return {"segments": []}

    monkeypatch.setattr(transcription_worker, "execute_request", fake_execute)

    assert transcription_worker.main([str(request_path), str(response_path)]) == 0
    assert capsys.readouterr().out == "WhisperX emitted a log line\n"
    assert json.loads(response_path.read_text(encoding="utf-8")) == {
        "ok": True,
        "result": {"segments": []},
    }
