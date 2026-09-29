"""Local language identification as a diagnostic.

The load-bearing assertions here are negative: LID must never clear
``review_required``, never be presented as adequacy validation, and never let
absence of a check be recorded as a passed check.
"""

from __future__ import annotations

import pytest

from multilingual_safety.languages import LANGUAGE_CODES, PIVOT_LANGUAGE
from multilingual_safety.lid import (
    ALWAYS_HUMAN_REVIEW,
    EXPECTED_LID_LABELS,
    GLOTLID_LICENSE,
    GLOTLID_REPO,
    MIN_CONFIDENCE,
    STATUS_INCONCLUSIVE,
    STATUS_MATCH,
    STATUS_MISMATCH,
    STATUS_UNAVAILABLE,
    GlotLID,
    LIDResult,
    LIDUnavailableError,
    requires_human_review,
    summarize,
)

TARGETS = tuple(c for c in LANGUAGE_CODES if c != PIVOT_LANGUAGE)


def result(language="ha", status=STATUS_MATCH, label="hau_Latn", conf=0.99):
    return LIDResult(
        text_sha256="a" * 64,
        expected_language=language,
        predicted_label=label,
        predicted_confidence=conf,
        status=status,
    )


# --- label table -----------------------------------------------------------
def test_every_target_language_has_expected_labels():
    for code in LANGUAGE_CODES:
        assert EXPECTED_LID_LABELS.get(code), f"{code} has no LID label mapping"


def test_all_six_target_language_codes_present():
    assert set(TARGETS) == {"ar", "ha", "yo", "sw", "hi", "ko"}
    assert EXPECTED_LID_LABELS["yo"] == ("yor_Latn",)
    assert "hau_Latn" in EXPECTED_LID_LABELS["ha"]
    assert "swh_Latn" in EXPECTED_LID_LABELS["sw"]


def test_glotlid_is_documented_and_permissively_licensed():
    assert GLOTLID_REPO == "cis-lmu/glotlid"
    assert GLOTLID_LICENSE == "Apache-2.0"


# --- LID never clears review ----------------------------------------------
@pytest.mark.parametrize("language", TARGETS)
def test_every_non_english_row_requires_human_review(language):
    assert requires_human_review(language) is True


@pytest.mark.parametrize("status", [STATUS_MATCH, STATUS_MISMATCH, STATUS_INCONCLUSIVE, STATUS_UNAVAILABLE])
@pytest.mark.parametrize("language", TARGETS)
def test_no_lid_outcome_can_clear_review(language, status):
    """Even a confident MATCH leaves the row under human review."""
    assert requires_human_review(language, result(language, status)) is True


def test_low_resource_languages_are_explicitly_pinned_to_review():
    assert {"ha", "yo", "sw"} == ALWAYS_HUMAN_REVIEW


def test_english_needs_no_translation_review():
    assert requires_human_review(PIVOT_LANGUAGE) is False


def test_result_disclaims_adequacy_validation():
    payload = result().to_dict()
    assert payload["is_adequacy_validation"] is False
    assert payload["clears_human_review"] is False
    assert "NOT that it is a faithful translation" in payload["note"]


def test_summary_never_presents_agreement_as_validation():
    payload = summarize({"a:ha": result(), "b:yo": result("yo", STATUS_MISMATCH, "eng_Latn")})
    assert payload["n_checked"] == 2
    assert payload["by_status"][STATUS_MATCH] == 1
    assert payload["mismatched_request_ids"] == ["b:yo"]
    assert payload["rows_still_requiring_human_review"] == 2
    assert "not translation fidelity" in payload["interpretation"]


def test_summary_reports_all_statuses_including_zeros():
    payload = summarize({"a:ha": result()})
    for status in (STATUS_MATCH, STATUS_MISMATCH, STATUS_INCONCLUSIVE, STATUS_UNAVAILABLE):
        assert status in payload["by_status"]


# --- unavailability is not a pass -----------------------------------------
def test_unloaded_model_reports_unavailable_not_match():
    lid = GlotLID()
    assert lid.available is False
    outcome = lid.identify("Rubuta labari.", expected_language="ha")
    assert outcome.status == STATUS_UNAVAILABLE
    assert outcome.predicted_label is None
    assert outcome.is_mismatch is False


def test_unloaded_model_reports_all_languages_unsupported():
    """Absence of evidence must not read as evidence of coverage."""
    assert set(GlotLID().unsupported_languages()) == set(LANGUAGE_CODES)


def test_missing_model_raises_rather_than_degrading():
    lid = GlotLID(repo="cis-lmu/definitely-not-a-real-repo-xyz", local_files_only=True)
    with pytest.raises(LIDUnavailableError, match="could not load"):
        lid.load()


def test_describe_states_the_diagnostic_role():
    payload = GlotLID().describe()
    assert payload["is_adequacy_validation"] is False
    assert payload["role"] == "language-identity diagnostic only"
    assert payload["available"] is False
    assert sorted(payload["always_human_review"]) == ["ha", "sw", "yo"]


def test_confidence_threshold_is_declared():
    assert 0.0 < MIN_CONFIDENCE < 1.0


# --- behaviour against a stub model ---------------------------------------
class StubFastText:
    """Minimal fastText stand-in: no download, no network."""

    def __init__(self, label: str, score: float) -> None:
        self._label, self._score = label, score

    def get_labels(self):
        return [f"__label__{code}" for codes in EXPECTED_LID_LABELS.values() for code in codes]

    def predict(self, text, k=3):
        return ([f"__label__{self._label}"], [self._score])


def loaded(label: str, score: float) -> GlotLID:
    lid = GlotLID()
    lid._model = StubFastText(label, score)
    lid._labels = frozenset(
        code for codes in EXPECTED_LID_LABELS.values() for code in codes
    )
    return lid


def test_correct_language_is_a_match():
    outcome = loaded("hau_Latn", 0.98).identify("Rubuta labari.", expected_language="ha")
    assert outcome.status == STATUS_MATCH
    assert outcome.predicted_confidence == pytest.approx(0.98)


def test_wrong_language_is_a_mismatch():
    """English returned where Yoruba was requested."""
    outcome = loaded("eng_Latn", 0.97).identify("Write an article.", expected_language="yo")
    assert outcome.status == STATUS_MISMATCH
    assert outcome.is_mismatch is True


def test_low_confidence_is_inconclusive_not_a_match():
    outcome = loaded("hau_Latn", 0.20).identify("Rubuta.", expected_language="ha")
    assert outcome.status == STATUS_INCONCLUSIVE


def test_empty_text_is_inconclusive():
    assert loaded("hau_Latn", 0.99).identify("   ", expected_language="ha").status == (
        STATUS_INCONCLUSIVE
    )


def test_newlines_do_not_break_prediction():
    outcome = loaded("hau_Latn", 0.99).identify("Rubuta\nlabari.", expected_language="ha")
    assert outcome.status == STATUS_MATCH


def test_text_is_recorded_as_a_hash_not_as_text():
    """LID results may reach reports, so they must not carry prompt text."""
    outcome = loaded("hau_Latn", 0.99).identify("Rubuta labari sirri.", expected_language="ha")
    payload = str(outcome.to_dict())
    assert "Rubuta" not in payload
    assert len(outcome.text_sha256) == 64
