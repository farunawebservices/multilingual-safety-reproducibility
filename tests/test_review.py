"""Review decisions and the reviewer-competence gate.

The load-bearing assertions are negative: a decision recorded by someone without
target-language competence must never clear ``review_required``, and must never
be reportable as native-speaker validation.
"""

from __future__ import annotations

import json

import pytest

from multilingual_safety.hashing import text_sha256
from multilingual_safety.languages import LANGUAGE_CODES, PIVOT_LANGUAGE
from multilingual_safety.paths import RESULTS_DIR
from multilingual_safety.review import (
    COMPETENCE_CAN_CLEAR_REVIEW,
    DECISION_CAN_CLEAR_REVIEW,
    ReviewDecision,
    ReviewDecisionType,
    ReviewerCompetence,
    ReviewError,
    can_clear_human_review,
    competence_rationale,
    summarize,
    validate_against_rows,
)

TARGETS = tuple(c for c in LANGUAGE_CODES if c != PIVOT_LANGUAGE)


def decision(**overrides) -> ReviewDecision:
    payload = {
        "review_id": "1",
        "request_id": "beh_0123456789ab:ha",
        "behavior_id": "beh_0123456789ab",
        "language": "ha",
        "decision": "ACCEPT",
        "reviewer_id": "Faruna_Godwin",
        "reviewer_language_competence": "machine_assisted",
        "source_text_sha256": text_sha256("Write an article."),
        "response_sha256": text_sha256("Rubuta labari."),
        "source_hash_verified": True,
        "response_hash_verified": True,
    }
    payload.update(overrides)
    return ReviewDecision(**payload)


# --- the competence gate ---------------------------------------------------
def test_machine_assisted_cannot_clear_review():
    """The decisive rule for this import."""
    assert can_clear_human_review("machine_assisted", "ACCEPT") is False
    assert ReviewerCompetence.MACHINE_ASSISTED not in COMPETENCE_CAN_CLEAR_REVIEW


@pytest.mark.parametrize(
    "competence", ["machine_assisted", "fluent_non_native", "none"]
)
def test_insufficient_competences_never_clear_review(competence):
    for verdict in ("ACCEPT", "REJECT", "REVISE", "PENDING"):
        assert can_clear_human_review(competence, verdict) is False


@pytest.mark.parametrize("competence", ["native_speaker", "professional_translator"])
def test_sufficient_competences_clear_only_on_accept(competence):
    assert can_clear_human_review(competence, "ACCEPT") is True
    for verdict in ("REJECT", "REVISE", "PENDING"):
        assert can_clear_human_review(competence, verdict) is False


def test_only_accept_can_ever_clear_review():
    assert {ReviewDecisionType.ACCEPT} == DECISION_CAN_CLEAR_REVIEW


def test_rationale_explains_the_refusal():
    text = competence_rationale("machine_assisted")
    assert "NOT sufficient" in text
    assert "meaning was preserved" in text
    assert "separate review layer" in text


def test_clears_review_is_computed_not_supplied():
    """A caller must not be able to assert its way past the gate."""
    assert decision().clears_review_required is False
    with pytest.raises(TypeError):
        ReviewDecision(
            review_id="1", request_id="beh_0123456789ab:ha", behavior_id="beh_0123456789ab",
            language="ha", decision="ACCEPT", reviewer_id="X",
            reviewer_language_competence="machine_assisted",
            clears_review_required=True,      # not an accepted argument
        )


def test_native_speaker_accept_does_clear():
    assert decision(reviewer_language_competence="native_speaker").clears_review_required is True


# --- schema ----------------------------------------------------------------
def test_decision_requires_a_named_reviewer():
    with pytest.raises(ReviewError, match="requires a named reviewer_id"):
        decision(reviewer_id="   ")


def test_unknown_decision_and_competence_rejected():
    with pytest.raises(ReviewError, match="unknown decision"):
        decision(decision="LOOKS_FINE")
    with pytest.raises(ReviewError, match="unknown competence"):
        decision(reviewer_language_competence="vibes")


def test_english_rows_are_not_reviewed_as_translations():
    with pytest.raises(ReviewError, match="canonical source text"):
        decision(language=PIVOT_LANGUAGE, request_id="beh_0123456789ab:en")


def test_serialised_decision_carries_no_text():
    payload = decision(notes_present=True, notes_sha256=text_sha256("looks ok")).to_dict()
    blob = json.dumps(payload, ensure_ascii=False)
    assert "Rubuta" not in blob and "Write an article" not in blob
    assert "looks ok" not in blob
    assert payload["contains_text"] is False
    assert payload["clears_review_required"] is False
    assert len(payload["response_sha256"]) == 64


def test_hashes_verified_requires_both():
    assert decision().hashes_verified is True
    assert decision(source_hash_verified=False).hashes_verified is False
    assert decision(response_hash_verified=False).hashes_verified is False


# --- aggregation -----------------------------------------------------------
def test_summary_counts_by_language_and_decision():
    decisions = [
        decision(review_id=str(i), language=lang, request_id=f"beh_0123456789ab:{lang}")
        for i, lang in enumerate(TARGETS, start=1)
    ]
    summary = summarize(decisions)
    assert summary["n_decisions"] == 6
    assert summary["counts_by_decision"]["ACCEPT"] == 6
    assert set(summary["counts_by_language"]) == set(TARGETS)
    assert summary["counts_by_competence"] == {"machine_assisted": 6}
    assert summary["n_clearing_review_required"] == 0
    assert summary["n_review_required_retained"] == 6
    assert summary["native_speaker_validation_claimed"] is False
    assert summary["contains_prompt_text"] is False


def test_summary_reports_all_decision_types_including_zeros():
    summary = summarize([decision()])
    for verdict in ("ACCEPT", "REJECT", "REVISE", "PENDING"):
        assert verdict in summary["counts_by_decision"]


def test_summary_flags_unverified_hashes():
    summary = summarize([decision(review_id="7", response_hash_verified=False)])
    assert summary["n_hash_unverified"] == 1
    assert summary["unverified_review_ids"] == ["7"]


# --- cross-checks against the translated rows ------------------------------
def test_decision_for_an_untranslated_row_is_reported():
    problems = validate_against_rows([decision()], {"beh_other000000:ha": {}})
    assert any("not a currently translated row" in p for p in problems)


def test_duplicate_decisions_are_reported():
    problems = validate_against_rows(
        [decision(review_id="1"), decision(review_id="2")],
        {"beh_0123456789ab:ha": {}},
    )
    assert any("2 decisions" in p for p in problems)


def test_missing_decisions_are_reported():
    problems = validate_against_rows(
        [decision()], {"beh_0123456789ab:ha": {}, "beh_0123456789ab:yo": {}}
    )
    assert any("have no decision" in p for p in problems)


def test_complete_matching_set_has_no_problems():
    assert validate_against_rows([decision()], {"beh_0123456789ab:ha": {}}) == []


# --- live check on the imported summary ------------------------------------
def test_imported_summary_claims_no_cleared_reviews():
    path = RESULTS_DIR / "translation_pilot_review_summary.json"
    if not path.exists():
        pytest.skip("review decisions have not been imported in this working tree")
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["competence_clears_review_required"] is False
    assert summary["n_clearing_review_required"] == 0
    assert summary["human_reviewed_rows_written"] == 0
    assert summary["review_required_cleared"] == 0
    assert summary["translation_statuses_modified"] == 0
    assert summary["native_speaker_validation_claimed"] is False
    assert summary["contains_prompt_text"] is False
    assert summary["n_hash_unverified"] == 0
