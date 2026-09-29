"""Heuristic language checks for translated text.

**These are heuristics, not language identification, and never validation.**
They exist to catch the failures that are cheap to catch automatically -- empty
output, wrong script, untranslated English passed through, truncation,
duplicates -- so that a human reviewer's attention goes to the rows that need
it. A row that passes every check here is still `review_required=True` until an
actual native speaker looks at it (CLAUDE.md: no claim of native-speaker
validation without a native speaker).

Asymmetry worth stating: for `ar`, `hi`, and `ko` the expected script is a
strong signal, so wrong-script output is caught reliably. For `ha`, `yo`, and
`sw` the target shares the Latin script with English, so the only automatic
signals are function-word overlap and (for `yo`/`ha`) diacritic presence. Those
are weaker, which means **the lower-resource Latin-script languages are exactly
where automatic checking is least reliable** -- so uncertainty there resolves to
"send to a human", never to "accept".
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from multilingual_safety.languages import LANGUAGE_CODES, validate_language_code

# --- issue codes -----------------------------------------------------------
EMPTY_TEXT = "EMPTY_TEXT"
NOT_TRANSLATED = "NOT_TRANSLATED"
IDENTICAL_TO_SOURCE = "IDENTICAL_TO_SOURCE"
WRONG_SCRIPT = "WRONG_SCRIPT"
ENGLISH_LEAKAGE = "ENGLISH_LEAKAGE"
SUSPICIOUSLY_SHORT = "SUSPICIOUSLY_SHORT"
SUSPICIOUSLY_LONG = "SUSPICIOUSLY_LONG"
MISSING_EXPECTED_DIACRITICS = "MISSING_EXPECTED_DIACRITICS"
NO_TARGET_LANGUAGE_EVIDENCE = "NO_TARGET_LANGUAGE_EVIDENCE"
DUPLICATE_TRANSLATION = "DUPLICATE_TRANSLATION"
CONTAINS_PLACEHOLDER = "CONTAINS_PLACEHOLDER"

ALL_ISSUE_CODES: tuple[str, ...] = (
    CONTAINS_PLACEHOLDER,
    DUPLICATE_TRANSLATION,
    EMPTY_TEXT,
    ENGLISH_LEAKAGE,
    IDENTICAL_TO_SOURCE,
    MISSING_EXPECTED_DIACRITICS,
    NOT_TRANSLATED,
    NO_TARGET_LANGUAGE_EVIDENCE,
    SUSPICIOUSLY_LONG,
    SUSPICIOUSLY_SHORT,
    WRONG_SCRIPT,
)

# --- script ranges ---------------------------------------------------------
SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "Latn": ((0x0041, 0x024F), (0x1E00, 0x1EFF), (0x0250, 0x02AF)),
    "Arab": ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
    "Deva": ((0x0900, 0x097F), (0xA8E0, 0xA8FF)),
    "Hang": ((0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F), (0xAC00, 0xD7AF)),
}

EXPECTED_SCRIPT: dict[str, str] = {
    "en": "Latn", "ar": "Arab", "ha": "Latn",
    "yo": "Latn", "sw": "Latn", "hi": "Deva", "ko": "Hang",
}

#: Languages whose script alone distinguishes them from English.
SCRIPT_DISTINGUISHES_FROM_ENGLISH: frozenset[str] = frozenset({"ar", "hi", "ko"})

# --- function words --------------------------------------------------------
ENGLISH_FUNCTION_WORDS: frozenset[str] = frozenset(["a", "about", "an", "and", "are", "as", "at", "be", "been", "by", "can", "could", "do", "does", "for", "from", "give", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "make", "of", "on", "or", "provide", "should", "so", "some", "such", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "to", "use", "used", "using", "was", "were", "what", "when", "where", "which", "who", "will", "with", "would", "write", "you", "your", "explain", "describe", "detail", "detailed", "step", "steps", "instructions", "tutorial", "guide", "article", "post", "list"])

#: Positive evidence that the target language is actually present. Short and
#: high-frequency by design: these are function words, not content words, so
#: they survive paraphrase.
TARGET_FUNCTION_WORDS: dict[str, frozenset[str]] = {
    "sw": frozenset(["na", "ya", "wa", "ni", "kwa", "katika", "kuwa", "hii", "hiyo", "ambayo", "kwamba", "jinsi", "ili", "au", "za", "la", "kama", "yake", "wake", "zao", "huu", "hizo", "baada", "kabla", "ndani", "juu", "chini", "kutoka", "kwenda", "andika", "eleza", "tengeneza", "toa", "fanya", "orodha", "makala"]),
    "ha": frozenset(["da", "na", "ya", "a", "ba", "don", "wannan", "yadda", "sannan", "ka", "ku", "wani", "wata", "su", "ne", "ce", "ta", "mai", "domin", "amma", "sai", "ko", "yin", "rubuta", "bayyana", "yi", "jerin", "kuma", "cikin", "akan", "gaba"]),
    "yo": frozenset(["ni", "ti", "si", "ati", "fun", "awon", "àwọn", "bawo", "báwo", "kan", "tabi", "tàbí", "pe", "pé", "lati", "láti", "ko", "kọ", "won", "wọn", "ohun", "ipa", "lori", "lórí", "ninu", "nínú", "je", "jẹ", "ki", "kí", "se", "ṣe"]),
}

#: Contrastive characters. Their total absence from a long text is suspicious,
#: though a short sentence may legitimately contain none.
EXPECTED_DIACRITICS: dict[str, tuple[str, ...]] = {
    "ha": ("ɓ", "ɗ", "ƙ", "Ɓ", "Ɗ", "Ƙ", "'y"),
    "yo": ("ẹ", "ọ", "ṣ", "Ẹ", "Ọ", "Ṣ", "à", "á", "è", "é", "ì", "í", "ò", "ó", "ù", "ú", "ń"),
}

#: Text a translator or template might leave behind.
PLACEHOLDER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\{\{.*?\}\}"),
    re.compile(r"\[(?:TODO|TRANSLATE|PLACEHOLDER|INSERT)[^\]]*\]", re.IGNORECASE),
    re.compile(r"(?i)\b(?:lorem ipsum|translation (?:pending|needed|unavailable))\b"),
    re.compile(r"(?i)^\s*(?:n/?a|none|null|todo)\s*$"),
)

# --- thresholds (recorded in every validation report) ----------------------
MIN_CHARS = 10
SHORT_RATIO = 0.40          # < 40% of the source length is suspicious
LONG_RATIO = 3.00           # > 3x the source length is suspicious
ENGLISH_LEAKAGE_RATIO = 0.50  # >= half the tokens are English function words
DIACRITIC_MIN_CHARS = 40    # only check diacritics on reasonably long text

#: Coarse character-length buckets, as ``(exclusive_upper_bound, label)`` pairs
#: with ``None`` as the open-ended final bound.
#:
#: A tracked report may state how long a row is, but not how long it is
#: *exactly*: an exact character count is a fingerprint of the text, and for a
#: ``final_test`` row it is a detail about held-out data that no metadata file
#: needs. Buckets carry the order of magnitude and nothing else.
CHARACTER_LENGTH_BUCKETS: tuple[tuple[int | None, str], ...] = (
    (50, "0-49"),
    (100, "50-99"),
    (200, "100-199"),
    (400, "200-399"),
    (800, "400-799"),
    (None, "800+"),
)

_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens, Unicode-aware (keeps ẹ, ɗ, ā, etc.)."""
    return [t.lower() for t in _TOKEN_RE.findall(unicodedata.normalize("NFC", text))]


def character_length_bucket(n_chars: int) -> str:
    """Coarse bucket label for a character count (``CHARACTER_LENGTH_BUCKETS``).

    Takes an ``int``, never text, so a caller can measure a string in memory and
    put only the bucket into a tracked report without the text or its exact
    length passing through the report at all.
    """
    if n_chars < 0:
        raise ValueError(f"character count cannot be negative: {n_chars}")
    for upper, label in CHARACTER_LENGTH_BUCKETS:
        if upper is None or n_chars < upper:
            return label
    raise AssertionError("CHARACTER_LENGTH_BUCKETS must end in an open-ended bound")


def script_counts(text: str) -> dict[str, int]:
    """Count characters per script. Deterministic, sorted output."""
    counts = dict.fromkeys(SCRIPT_RANGES, 0)
    counts["Other"] = 0
    for ch in unicodedata.normalize("NFC", text):
        if not ch.isalpha():
            continue
        code = ord(ch)
        for script, ranges in SCRIPT_RANGES.items():
            if any(lo <= code <= hi for lo, hi in ranges):
                counts[script] += 1
                break
        else:
            counts["Other"] += 1
    return dict(sorted(counts.items()))


def dominant_script(text: str) -> str | None:
    """Most frequent script, or None when there are no alphabetic characters."""
    counts = {k: v for k, v in script_counts(text).items() if v}
    if not counts:
        return None
    # max() over a sorted list keeps ties deterministic.
    return max(sorted(counts.items()), key=lambda kv: kv[1])[0]


def english_function_word_ratio(text: str) -> float:
    """Fraction of tokens that are English function words."""
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in ENGLISH_FUNCTION_WORDS) / len(tokens)


def target_function_word_ratio(text: str, language: str) -> float | None:
    """Fraction of tokens that are known function words of ``language``.

    None when no word list exists for that language -- reported honestly as
    "no evidence available", never as "evidence of correctness".
    """
    words = TARGET_FUNCTION_WORDS.get(language)
    if words is None:
        return None
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in words) / len(tokens)


def missing_diacritics(text: str, language: str) -> tuple[str, ...]:
    """Expected contrastive characters absent from ``text``."""
    expected = EXPECTED_DIACRITICS.get(language, ())
    nfc = unicodedata.normalize("NFC", text)
    return tuple(ch for ch in expected if ch not in nfc)


def contains_placeholder(text: str) -> bool:
    return any(p.search(text) for p in PLACEHOLDER_PATTERNS)


@dataclass(frozen=True, slots=True)
class LanguageCheck:
    """Outcome of the heuristic checks for one row."""

    language: str
    issues: tuple[str, ...]
    metrics: dict[str, Any]

    @property
    def ok(self) -> bool:
        """No issues found. Note this is NOT the same as 'validated'."""
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "issues": sorted(self.issues),
            "n_issues": len(self.issues),
            "metrics": dict(sorted(self.metrics.items())),
            "heuristics_passed": self.ok,
            "note": "Heuristics only. Passing does not constitute validation and "
                    "never implies native-speaker review.",
        }


def check_translation(
    *,
    translated_text: str | None,
    source_text: str,
    language: str,
    duplicate_of: Sequence[str] = (),
) -> LanguageCheck:
    """Run every heuristic for one translated row.

    ``duplicate_of`` carries record IDs sharing this row's prompt hash within the
    same language; a non-empty value raises DUPLICATE_TRANSLATION.
    """
    validate_language_code(language)
    issues: list[str] = []
    metrics: dict[str, Any] = {}

    if translated_text is None:
        return LanguageCheck(language, (NOT_TRANSLATED,), {"n_chars": 0})

    text = unicodedata.normalize("NFC", translated_text)
    stripped = text.strip()
    metrics["n_chars"] = len(stripped)
    metrics["n_tokens"] = len(tokenize(stripped))
    metrics["source_n_chars"] = len(source_text.strip())

    if not stripped:
        return LanguageCheck(language, (EMPTY_TEXT,), metrics)

    if contains_placeholder(stripped):
        issues.append(CONTAINS_PLACEHOLDER)

    # --- length ----------------------------------------------------------
    source_len = max(len(source_text.strip()), 1)
    ratio = len(stripped) / source_len
    metrics["length_ratio"] = round(ratio, 4)
    if len(stripped) < MIN_CHARS or ratio < SHORT_RATIO:
        issues.append(SUSPICIOUSLY_SHORT)
    elif ratio > LONG_RATIO:
        # Over-long output can mean the translator elaborated -- which would add
        # operational detail the source did not contain.
        issues.append(SUSPICIOUSLY_LONG)

    if duplicate_of:
        issues.append(DUPLICATE_TRANSLATION)
        metrics["duplicate_of"] = sorted(duplicate_of)

    # --- English rows are canonical source text; nothing further applies --
    if language == "en":
        return LanguageCheck(language, tuple(sorted(set(issues))), metrics)

    if stripped == unicodedata.normalize("NFC", source_text.strip()):
        issues.append(IDENTICAL_TO_SOURCE)

    # --- script ----------------------------------------------------------
    counts = script_counts(stripped)
    expected = EXPECTED_SCRIPT[language]
    dominant = dominant_script(stripped)
    metrics["script_counts"] = counts
    metrics["dominant_script"] = dominant
    metrics["expected_script"] = expected
    if dominant is not None and dominant != expected:
        issues.append(WRONG_SCRIPT)

    # --- English leakage --------------------------------------------------
    en_ratio = english_function_word_ratio(stripped)
    metrics["english_function_word_ratio"] = round(en_ratio, 4)
    if language in SCRIPT_DISTINGUISHES_FROM_ENGLISH:
        # Any substantial Latin run in an Arabic/Devanagari/Hangul target is
        # untranslated English, modulo the odd loanword or proper noun.
        latin = counts.get("Latn", 0)
        total = sum(v for k, v in counts.items() if k != "Other") or 1
        metrics["latin_char_ratio"] = round(latin / total, 4)
        if latin / total > 0.30:
            issues.append(ENGLISH_LEAKAGE)
    elif en_ratio >= ENGLISH_LEAKAGE_RATIO:
        issues.append(ENGLISH_LEAKAGE)

    # --- positive evidence of the target language -------------------------
    target_ratio = target_function_word_ratio(stripped, language)
    metrics["target_function_word_ratio"] = (
        None if target_ratio is None else round(target_ratio, 4)
    )
    if (
        language not in SCRIPT_DISTINGUISHES_FROM_ENGLISH
        and target_ratio == 0.0
        and metrics["n_tokens"] >= 5
    ):
        issues.append(NO_TARGET_LANGUAGE_EVIDENCE)

    # --- diacritics -------------------------------------------------------
    if language in EXPECTED_DIACRITICS and len(stripped) >= DIACRITIC_MIN_CHARS:
        absent = missing_diacritics(stripped, language)
        metrics["n_missing_diacritic_sentinels"] = len(absent)
        if len(absent) == len(EXPECTED_DIACRITICS[language]):
            # Not one contrastive character in a long text: either ASCII-folded
            # in transit, or not really this language.
            issues.append(MISSING_EXPECTED_DIACRITICS)

    return LanguageCheck(language, tuple(sorted(set(issues))), metrics)


def thresholds() -> dict[str, Any]:
    """The thresholds in force, recorded in every validation report."""
    return {
        "min_chars": MIN_CHARS,
        "short_ratio": SHORT_RATIO,
        "long_ratio": LONG_RATIO,
        "english_leakage_ratio": ENGLISH_LEAKAGE_RATIO,
        "latin_char_ratio_for_non_latin_targets": 0.30,
        "diacritic_min_chars": DIACRITIC_MIN_CHARS,
        "languages": list(LANGUAGE_CODES),
        "caveat": "Heuristics, not language identification. Latin-script targets "
                  "(ha, yo, sw) are checked least reliably; uncertainty there must "
                  "resolve to human review, never to acceptance.",
    }
