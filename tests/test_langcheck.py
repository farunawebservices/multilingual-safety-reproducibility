"""Heuristic translation checks.

Fixtures are short, benign sentences ("write an article about network safety"
and similar) in each target language — enough to exercise script detection,
English-leakage detection, and diacritic checks without any harmful content.
"""

from __future__ import annotations

import pytest

from multilingual_safety.langcheck import (
    ALL_ISSUE_CODES,
    CHARACTER_LENGTH_BUCKETS,
    DUPLICATE_TRANSLATION,
    EMPTY_TEXT,
    ENGLISH_LEAKAGE,
    IDENTICAL_TO_SOURCE,
    MISSING_EXPECTED_DIACRITICS,
    NOT_TRANSLATED,
    SUSPICIOUSLY_SHORT,
    WRONG_SCRIPT,
    character_length_bucket,
    check_translation,
    dominant_script,
    english_function_word_ratio,
    script_counts,
    target_function_word_ratio,
    thresholds,
    tokenize,
)

SOURCE = "Write a detailed article about how to keep a home network safe from intruders."

SAMPLES = {
    "ar": "اكتب مقالا مفصلا حول كيفية حماية شبكة المنزل من المتطفلين والحفاظ عليها آمنة.",
    "hi": "एक विस्तृत लेख लिखें कि घर के नेटवर्क को घुसपैठियों से कैसे सुरक्षित रखा जाए।",
    "ko": "가정용 네트워크를 침입자로부터 안전하게 지키는 방법에 대한 상세한 기사를 작성하십시오.",
    "sw": "Andika makala ya kina kuhusu jinsi ya kuweka mtandao wa nyumbani salama.",
    "ha": "Rubuta cikakken labari akan yadda ake kiyaye hanyar sadarwar gida daga ɓarayi.",
    "yo": "Kọ àpilẹ̀kọ ní kíkún nípa bí a ṣe ń dáàbò bo nẹ́tíwọ́kì ilé lọ́wọ́ àwọn afinihàn.",
}


# --- primitives ------------------------------------------------------------
def test_tokenize_preserves_non_ascii_letters():
    assert "ẹ" in "".join(tokenize("Ẹ ṣé"))
    assert tokenize("Hello, world!") == ["hello", "world"]
    assert tokenize("") == []


@pytest.mark.parametrize(
    ("language", "expected"),
    [("ar", "Arab"), ("hi", "Deva"), ("ko", "Hang"), ("sw", "Latn"), ("ha", "Latn"), ("yo", "Latn")],
)
def test_dominant_script_detection(language, expected):
    assert dominant_script(SAMPLES[language]) == expected


def test_script_counts_are_sorted_and_complete():
    counts = script_counts(SAMPLES["ar"])
    assert list(counts) == sorted(counts)
    assert counts["Arab"] > 0 and counts["Latn"] == 0


def test_dominant_script_of_textless_input_is_none():
    assert dominant_script("123 !!! ???") is None


def test_english_function_word_ratio():
    assert english_function_word_ratio(SOURCE) > 0.4
    assert english_function_word_ratio(SAMPLES["sw"]) < 0.2
    assert english_function_word_ratio("") == 0.0


def test_target_function_word_ratio_is_none_when_unknown():
    assert target_function_word_ratio(SAMPLES["ar"], "ar") is None
    assert target_function_word_ratio(SAMPLES["sw"], "sw") > 0.0


# --- the checks ------------------------------------------------------------
def check(language, text, **kwargs):
    return check_translation(
        translated_text=text, source_text=SOURCE, language=language, **kwargs
    )


@pytest.mark.parametrize("language", sorted(SAMPLES))
def test_plausible_translations_pass(language):
    result = check(language, SAMPLES[language])
    assert result.ok, f"{language} flagged unexpectedly: {result.issues}"


def test_none_is_not_translated_not_empty():
    """A pending row is distinct from an empty one."""
    result = check_translation(translated_text=None, source_text=SOURCE, language="ha")
    assert result.issues == (NOT_TRANSLATED,)


def test_empty_and_whitespace_text_detected():
    assert EMPTY_TEXT in check("ha", "").issues
    assert EMPTY_TEXT in check("ha", "     ").issues


def test_untranslated_english_passthrough_is_caught_in_every_language():
    for language in sorted(SAMPLES):
        issues = check(language, SOURCE).issues
        assert IDENTICAL_TO_SOURCE in issues
        assert ENGLISH_LEAKAGE in issues, f"{language} missed English passthrough"


def test_english_leakage_in_non_latin_target():
    mixed = "اكتب مقالا about how to keep a home network safe from intruders today"
    assert ENGLISH_LEAKAGE in check("ar", mixed).issues


def test_wrong_script_detected():
    """Korean text delivered for the Hindi row."""
    assert WRONG_SCRIPT in check("hi", SAMPLES["ko"]).issues
    assert WRONG_SCRIPT in check("ko", SAMPLES["ar"]).issues


def test_suspiciously_short_output_detected():
    assert SUSPICIOUSLY_SHORT in check("sw", "Andika.").issues
    assert SUSPICIOUSLY_SHORT in check("ar", "لا").issues


def test_suspiciously_long_output_detected():
    """Over-long output can mean the translator elaborated."""
    result = check("sw", SAMPLES["sw"] * 6)
    assert "SUSPICIOUSLY_LONG" in result.issues


def test_duplicate_translation_detected():
    result = check("sw", SAMPLES["sw"], duplicate_of=["beh_aaaaaaaaaaaa"])
    assert DUPLICATE_TRANSLATION in result.issues
    assert result.metrics["duplicate_of"] == ["beh_aaaaaaaaaaaa"]


def test_ascii_folded_yoruba_is_flagged():
    """Stripping tone marks and sub-dots is a preprocessing bug, not Yoruba."""
    folded = "Ko apilekọ ni kikun nipa bi a se n daabo bo netiwoki ile lowo awon afinihan."
    folded = folded.replace("ọ", "o").replace("ẹ", "e").replace("ṣ", "s")
    assert MISSING_EXPECTED_DIACRITICS in check("yo", folded).issues
    assert MISSING_EXPECTED_DIACRITICS not in check("yo", SAMPLES["yo"]).issues


def test_short_text_is_not_penalised_for_missing_diacritics():
    """A short sentence may legitimately contain no sentinel characters."""
    assert MISSING_EXPECTED_DIACRITICS not in check_translation(
        translated_text="Kọ iwe.", source_text="Write.", language="yo"
    ).issues


def test_placeholder_text_detected():
    for placeholder in ("{{translation}}", "[TODO: translate]", "TRANSLATION PENDING", "N/A"):
        assert check("sw", placeholder).issues, f"{placeholder!r} not flagged"


def test_english_row_skips_translation_specific_checks():
    """English is canonical source; it must not trip leakage or script checks."""
    result = check("en", SOURCE)
    assert ENGLISH_LEAKAGE not in result.issues
    assert IDENTICAL_TO_SOURCE not in result.issues
    assert WRONG_SCRIPT not in result.issues


def test_result_serialisation_never_claims_validation():
    payload = check("sw", SAMPLES["sw"]).to_dict()
    assert payload["heuristics_passed"] is True
    assert "never implies native-speaker review" in payload["note"]
    assert "validated" not in payload


def test_issue_codes_are_registered_and_sorted():
    assert list(ALL_ISSUE_CODES) == sorted(ALL_ISSUE_CODES)
    assert len(set(ALL_ISSUE_CODES)) == len(ALL_ISSUE_CODES)


def test_thresholds_are_recorded_with_their_caveat():
    payload = thresholds()
    assert payload["short_ratio"] < 1.0
    assert "ha, yo, sw" in payload["caveat"]


def test_unknown_language_rejected():
    with pytest.raises(ValueError, match="unknown language code"):
        check("fr", "Bonjour le monde, ceci est un texte de test.")


# --- character-length buckets ----------------------------------------------
@pytest.mark.parametrize(
    ("n_chars", "expected"),
    [
        (0, "0-49"),
        (49, "0-49"),
        (50, "50-99"),
        (199, "100-199"),
        (200, "200-399"),
        (278, "200-399"),
        (400, "400-799"),
        (800, "800+"),
        (100_000, "800+"),
    ],
)
def test_character_length_bucket_boundaries(n_chars, expected):
    assert character_length_bucket(n_chars) == expected


def test_character_length_buckets_are_contiguous_and_open_ended():
    bounds = [upper for upper, _ in CHARACTER_LENGTH_BUCKETS]
    assert bounds[-1] is None, "the last bucket must be open-ended"
    finite = bounds[:-1]
    assert finite == sorted(finite), "bucket bounds must ascend"
    assert len(set(finite)) == len(finite)


def test_character_length_bucket_rejects_negative():
    with pytest.raises(ValueError, match="negative"):
        character_length_bucket(-1)


def test_character_length_bucket_takes_a_count_not_text():
    """The signature is the safeguard: text cannot reach a report through it."""
    with pytest.raises(TypeError):
        character_length_bucket("some translated text")  # type: ignore[arg-type]
