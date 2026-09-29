"""Target languages and orthography-preserving normalisation checks.

Seven languages, fixed. ``en`` is the pivot.

The normalisation helpers here exist because of a specific failure mode: if
Yoruba tone marks or Hausa hooked letters are stripped by preprocessing, the
resulting measurement says "the model failed in Yoruba" when what actually
happened is "we destroyed the prompt before the model saw it". The
:func:`orthography_fingerprint` of a record is recorded so that the two can be
told apart after the fact.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

#: Fixed target language codes, in canonical reporting order.
LANGUAGE_CODES: tuple[str, ...] = ("en", "ar", "ha", "yo", "sw", "hi", "ko")

#: Reference language. Every behavior exists in the pivot plus all others.
PIVOT_LANGUAGE = "en"

DEFAULT_LANGUAGES_CONFIG = Path("configs/languages.yaml")

#: Characters that are contrastive in the given language and must survive the
#: whole pipeline. Their disappearance is a preprocessing bug, not a result.
DIACRITIC_SENTINELS: dict[str, tuple[str, ...]] = {
    "ha": ("ɓ", "ɗ", "ƙ", "Ɓ", "Ɗ", "Ƙ"),
    "yo": ("ẹ", "ọ", "ṣ", "Ẹ", "Ọ", "Ṣ", "à", "á", "è", "é", "ì", "í", "ò", "ó", "ù", "ú"),
}


@dataclass(frozen=True, slots=True)
class Language:
    """A target language and the properties that affect measurement."""

    code: str
    name: str
    endonym: str
    script: str
    direction: str
    resource_tier: str
    unicode_normalization: str = "NFC"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.code not in LANGUAGE_CODES:
            raise ValueError(f"unknown language code {self.code!r}; expected one of {LANGUAGE_CODES}")
        if self.direction not in {"ltr", "rtl"}:
            raise ValueError(f"direction must be 'ltr' or 'rtl', got {self.direction!r}")
        if self.unicode_normalization not in {"NFC", "NFD", "NFKC", "NFKD"}:
            raise ValueError(f"invalid normalization form {self.unicode_normalization!r}")

    @property
    def is_pivot(self) -> bool:
        return self.code == PIVOT_LANGUAGE


def load_languages(path: str | Path = DEFAULT_LANGUAGES_CONFIG) -> dict[str, Language]:
    """Load the language registry, validating that it matches the fixed set.

    Raises if the config drifts from :data:`LANGUAGE_CODES`. Adding or removing
    a language mid-project silently invalidates every prior comparison, so it
    must be a deliberate, visible change in both places.
    """
    path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))

    pivot = raw.get("pivot")
    if pivot != PIVOT_LANGUAGE:
        raise ValueError(f"config pivot {pivot!r} != declared pivot {PIVOT_LANGUAGE!r}")

    languages: dict[str, Language] = {}
    for entry in raw.get("languages", []):
        lang = Language(
            code=entry["code"],
            name=entry["name"],
            endonym=entry.get("endonym", entry["name"]),
            script=entry["script"],
            direction=entry["direction"],
            resource_tier=entry["resource_tier"],
            unicode_normalization=entry.get("unicode_normalization", "NFC"),
            notes=entry.get("notes", "") or "",
        )
        if lang.code in languages:
            raise ValueError(f"duplicate language entry {lang.code!r}")
        languages[lang.code] = lang

    missing = set(LANGUAGE_CODES) - set(languages)
    extra = set(languages) - set(LANGUAGE_CODES)
    if missing or extra:
        raise ValueError(
            f"language config mismatch: missing={sorted(missing)} unexpected={sorted(extra)}"
        )
    return {code: languages[code] for code in LANGUAGE_CODES}


def validate_language_code(code: str) -> str:
    """Return ``code`` if it is a target language; raise otherwise."""
    if code not in LANGUAGE_CODES:
        raise ValueError(f"unknown language code {code!r}; expected one of {LANGUAGE_CODES}")
    return code


def normalize(text: str, form: str = "NFC") -> str:
    """Unicode-normalise ``text``.

    NFC by default. NFKC/NFKD are *not* used on corpus text: compatibility
    folding rewrites characters and would alter the very orthographic
    distinctions being measured.
    """
    if form not in {"NFC", "NFD", "NFKC", "NFKD"}:
        raise ValueError(f"invalid normalization form {form!r}")
    return unicodedata.normalize(form, text)


def orthography_fingerprint(text: str) -> dict[str, Any]:
    """Summarise orthographic content without storing the text itself.

    Safe to write into tracked manifests: it records counts and category
    profiles, never characters in context, and cannot reconstruct the prompt.
    """
    nfc = unicodedata.normalize("NFC", text)
    nfd = unicodedata.normalize("NFD", text)
    combining = sum(1 for ch in nfd if unicodedata.combining(ch))
    scripts: dict[str, int] = {}
    for ch in nfc:
        if ch.isspace():
            continue
        scripts[unicodedata.category(ch)] = scripts.get(unicodedata.category(ch), 0) + 1
    return {
        "n_chars_nfc": len(nfc),
        "n_chars_nfd": len(nfd),
        "n_combining_marks": combining,
        "is_nfc_stable": nfc == text,
        "n_non_ascii": sum(1 for ch in nfc if ord(ch) > 127),
        "unicode_category_counts": dict(sorted(scripts.items())),
    }


def missing_diacritic_sentinels(text: str, language: str) -> tuple[str, ...]:
    """Return the sentinel characters expected for ``language`` but absent.

    A non-empty result for a long ``ha``/``yo`` text is a *warning*, not proof
    of corruption -- a short sentence legitimately may contain none. Use it as a
    tripwire on corpus-level statistics (CLAUDE.md §10), not as a per-record
    filter, and never drop records on the basis of it.
    """
    validate_language_code(language)
    sentinels = DIACRITIC_SENTINELS.get(language, ())
    nfc = unicodedata.normalize("NFC", text)
    return tuple(s for s in sentinels if s not in nfc)


def report_order(codes: Iterable[str]) -> list[str]:
    """Sort language codes into the canonical reporting order."""
    index = {code: i for i, code in enumerate(LANGUAGE_CODES)}
    return sorted((validate_language_code(c) for c in codes), key=lambda c: index[c])
