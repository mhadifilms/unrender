from __future__ import annotations

import pytest

from unrender.lib.languages import (
    BY_CODE,
    is_spaceless,
    is_supported,
    language_name,
    resolve,
    supported_codes,
    supported_names,
    to_whisper_code,
    validate,
)


def test_thirty_one_languages() -> None:
    assert len(supported_codes()) == 31
    assert len(supported_names()) == 31
    # a spot-check of the display-name registry
    for code in ("zh", "yue", "en", "de", "fa", "th", "vi", "sw", "mk", "tl"):
        assert code in BY_CODE


def test_name_code_roundtrip_all() -> None:
    for code, lang in BY_CODE.items():
        assert to_whisper_code(lang.name) == code
        assert to_whisper_code(code) == code
        assert language_name(code) == lang.name
        assert language_name(lang.name) == lang.name


def test_aliases_resolve() -> None:
    cases = {
        "deutsch": "de",
        "Deutsch": "de",
        "farsi": "fa",
        "Persian (Farsi)": "fa",
        "mandarin": "zh",
        "Mandarin Chinese": "zh",
        "cantonese": "yue",
        "filipino": "tl",
        "castilian": "es",
        "brazilian portuguese": "pt",
    }
    for alias, code in cases.items():
        assert to_whisper_code(alias) == code, alias


def test_spaceless_flags() -> None:
    for code in ("zh", "yue", "ja", "th"):
        assert is_spaceless(code)
        assert is_spaceless(BY_CODE[code].name)
    for code in ("en", "de", "es", "ko", "vi", "ar"):
        assert not is_spaceless(code)


def test_validate_and_support() -> None:
    assert is_supported("German") and is_supported("de") and is_supported("deutsch")
    assert not is_supported("klingon")
    assert not is_supported(None)
    assert validate("French").code == "fr"
    with pytest.raises(ValueError):
        validate("klingon")


def test_unknown_passthrough_and_none() -> None:
    # unknown code passes through lowercased (a caller may hold a code Whisper
    # knows outside the display-name registry); resolve() still reports it unsupported.
    assert to_whisper_code("xx") == "xx"
    assert resolve("xx") is None
    assert to_whisper_code(None) is None
    assert language_name("xx") is None
