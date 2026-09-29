"""Language registry and orthography-preservation invariants."""

from __future__ import annotations

import pytest

from multilingual_safety.languages import (
    LANGUAGE_CODES,
    PIVOT_LANGUAGE,
    load_languages,
    missing_diacritic_sentinels,
    normalize,
    orthography_fingerprint,
    report_order,
    validate_language_code,
)
from multilingual_safety.paths import LANGUAGES_CONFIG


def test_seven_target_languages_are_fixed():
    assert LANGUAGE_CODES == ("en", "ar", "ha", "yo", "sw", "hi", "ko")
    assert len(set(LANGUAGE_CODES)) == 7
    assert PIVOT_LANGUAGE == "en"


def test_config_matches_declared_language_set():
    languages = load_languages(LANGUAGES_CONFIG)
    assert tuple(languages) == LANGUAGE_CODES
    assert languages["ar"].direction == "rtl"
    assert languages["ko"].script == "Hang"
    assert languages["en"].is_pivot


def test_unknown_language_code_rejected():
    with pytest.raises(ValueError, match="unknown language code"):
        validate_language_code("fr")


def test_report_order_is_canonical():
    assert report_order(["ko", "en", "ha"]) == ["en", "ha", "ko"]


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("yo", "Ẹ ṣé gan-an, ọmọ mi"),   # sub-dots + tone
        ("ha", "Yaya kake? Ɓarawo ɗaya ƙarami"),  # hooked letters
        ("ar", "مرحبا بالعالم"),
        ("hi", "नमस्ते दुनिया"),
        ("ko", "안녕하세요 세계"),
    ],
)
def test_nfc_normalization_is_idempotent_and_lossless(language, text):
    once = normalize(text, "NFC")
    twice = normalize(once, "NFC")
    assert once == twice
    # Round-tripping through NFD and back must not lose characters.
    assert normalize(normalize(text, "NFD"), "NFC") == once


def test_orthography_fingerprint_is_text_free():
    """The fingerprint is committed to manifests, so it must not leak content."""
    text = "Ẹ ṣé gan-an"
    fp = orthography_fingerprint(text)
    serialized = str(fp)
    for word in text.split():
        assert word not in serialized
    assert fp["n_combining_marks"] > 0
    assert fp["n_non_ascii"] > 0


def test_diacritic_sentinels_detect_ascii_folding():
    """ASCII-folding Yoruba is a preprocessing bug, not a language property."""
    intact = "Ẹ ṣé ọmọ"
    folded = "E se omo"
    assert missing_diacritic_sentinels(intact, "yo") != missing_diacritic_sentinels(folded, "yo")
    assert len(missing_diacritic_sentinels(folded, "yo")) > len(
        missing_diacritic_sentinels(intact, "yo")
    )


def test_hausa_hooked_letters_are_sentinels():
    assert "ɓ" not in missing_diacritic_sentinels("ɓarawo ɗaya ƙarami", "ha")
    assert "ɓ" in missing_diacritic_sentinels("barawo daya karami", "ha")
