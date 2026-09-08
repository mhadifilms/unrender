from __future__ import annotations

import unrender
from unrender import audio


def test_unrender_package_exports_public_api() -> None:
    assert unrender.__version__ == "0.0.0"
    assert hasattr(unrender, "RunPaths")
    assert callable(audio.normalize_full_audio_input)
    assert callable(audio.separate_global_dx_stem)
    assert callable(audio.separate_full_audio_source)
