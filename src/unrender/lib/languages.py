"""Language names and ISO codes for transcription."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    """One supported language.

    ``name`` is the canonical display name;
    ``code`` is the ISO-639 code Whisper/WhisperX expects; ``spaceless`` marks
    scripts written without spaces between words (word-level text matching must
    fall back to character-level for these).
    """

    name: str
    code: str
    spaceless: bool = False


# Supported language names. ``spaceless`` = Chinese,
# Cantonese, Japanese, Thai (no inter-word spaces in normal orthography).
_LANGUAGES: tuple[Language, ...] = (
    Language("Chinese", "zh", spaceless=True),
    Language("Cantonese", "yue", spaceless=True),
    Language("English", "en"),
    Language("Arabic", "ar"),
    Language("Czech", "cs"),
    Language("Danish", "da"),
    Language("Dutch", "nl"),
    Language("Finnish", "fi"),
    Language("French", "fr"),
    Language("German", "de"),
    Language("Greek", "el"),
    Language("Hebrew", "he"),
    Language("Hindi", "hi"),
    Language("Hungarian", "hu"),
    Language("Italian", "it"),
    Language("Japanese", "ja", spaceless=True),
    Language("Korean", "ko"),
    Language("Macedonian", "mk"),
    Language("Malay", "ms"),
    Language("Persian", "fa"),
    Language("Polish", "pl"),
    Language("Portuguese", "pt"),
    Language("Romanian", "ro"),
    Language("Russian", "ru"),
    Language("Spanish", "es"),
    Language("Swahili", "sw"),
    Language("Swedish", "sv"),
    Language("Tagalog", "tl"),
    Language("Thai", "th", spaceless=True),
    Language("Turkish", "tr"),
    Language("Vietnamese", "vi"),
)

# code -> Language
BY_CODE: dict[str, Language] = {lang.code: lang for lang in _LANGUAGES}

# Common alternate names / endonyms that should resolve to a canonical entry.
_ALIASES: dict[str, str] = {
    "deutsch": "de",
    "farsi": "fa",
    "persian (farsi)": "fa",
    "mandarin": "zh",
    "mandarin chinese": "zh",
    "putonghua": "zh",
    "brazilian portuguese": "pt",
    "castilian": "es",
    "castellano": "es",
    "filipino": "tl",
    "bahasa melayu": "ms",
    "bahasa malaysia": "ms",
    "kiswahili": "sw",
}

# lookup key (lowercased name / code / alias) -> code
_BY_KEY: dict[str, str] = {}
for _lang in _LANGUAGES:
    _BY_KEY[_lang.name.lower()] = _lang.code
    _BY_KEY[_lang.code.lower()] = _lang.code
_BY_KEY.update(_ALIASES)


def resolve(language: str | None) -> Language | None:
    """Resolve a name, code, or alias (case-insensitive) to a ``Language``.

    Returns ``None`` for empty or unsupported input.
    """
    if not language:
        return None
    return BY_CODE.get(_BY_KEY.get(str(language).strip().lower(), ""))


def to_whisper_code(language: str | None) -> str | None:
    """ISO code for Whisper. Unknown values pass through lowercased (so a caller
    that already holds an exotic code Whisper knows is not blocked), matching the
    legacy ``whisper_language_code`` behaviour."""
    if not language:
        return None
    lang = resolve(language)
    if lang is not None:
        return lang.code
    return str(language).strip().lower() or None


def language_name(language: str | None) -> str | None:
    """Canonical speech language name, or ``None`` if unsupported."""
    lang = resolve(language)
    return lang.name if lang is not None else None


def is_spaceless(language: str | None) -> bool:
    """True for scripts written without inter-word spaces (zh/yue/ja/th)."""
    lang = resolve(language)
    return bool(lang and lang.spaceless)


def is_supported(language: str | None) -> bool:
    return resolve(language) is not None


def supported_names() -> list[str]:
    return sorted(lang.name for lang in _LANGUAGES)


def supported_codes() -> list[str]:
    return sorted(BY_CODE)


def validate(language: str | None) -> Language:
    """Resolve a known language name or raise with the display-name registry."""
    lang = resolve(language)
    if lang is None:
        raise ValueError(
            f"Unsupported language {language!r}. Supported speech languages: "
            f"{', '.join(supported_names())}"
        )
    return lang
