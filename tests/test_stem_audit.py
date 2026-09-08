from pathlib import Path

import numpy as np

from conftest import write_wav
from unrender.audio.mne.stem_audit import audit_stems


def test_audit_stems_reports_residual_speech(tmp_path: Path) -> None:
    fx = tmp_path / "fx.wav"
    mx = tmp_path / "mx.wav"
    write_wav(fx, np.full(16_000, 5000, dtype=np.int16))
    write_wav(mx, np.zeros(16_000, dtype=np.int16))

    report = audit_stems(
        stems={"fx": fx, "mx": mx},
        output_json=tmp_path / "audit.json",
        threshold_db=-45.0,
        max_allowed_speech_sec=0.1,
        verify_asr=False,
        force=True,
    )

    assert report["status"] == "fail"
    assert report["method"] == "rms"
    assert report["stems"][0]["speech_sec"] > 0.9
