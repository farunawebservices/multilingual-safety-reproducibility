"""The six-category taxonomy: exhaustive, no defaults, no silent recoding."""

from __future__ import annotations

import pytest

from multilingual_safety.decisions import (
    DECISION_CATEGORIES,
    FAILURE_CATEGORIES,
    SEMANTIC_CATEGORIES,
    DecisionCategory,
    DecisionCoercionError,
    StageDecision,
    coerce_to_binary,
    is_failure_category,
    parse_category,
    rates,
    tally,
)


def test_exactly_six_categories():
    assert len(DECISION_CATEGORIES) == 6
    assert {c.value for c in DECISION_CATEGORIES} == {
        "SEMANTIC_ALLOW",
        "SEMANTIC_VETO",
        "MISSING_DECISION",
        "MALFORMED_DECISION",
        "TIMEOUT",
        "NON_TERMINATION",
    }


def test_semantic_and_failure_partition_the_space():
    assert set(DECISION_CATEGORIES) == SEMANTIC_CATEGORIES | FAILURE_CATEGORIES
    assert not (SEMANTIC_CATEGORIES & FAILURE_CATEGORIES)
    assert len(FAILURE_CATEGORIES) == 4


@pytest.mark.parametrize("category", sorted(FAILURE_CATEGORIES, key=lambda c: c.value))
def test_failure_categories_cannot_be_coerced_to_binary(category):
    """MISSING must not become a veto; MALFORMED must not become an allow."""
    assert is_failure_category(category)
    with pytest.raises(DecisionCoercionError):
        coerce_to_binary(category)


def test_semantic_categories_coerce():
    assert coerce_to_binary(DecisionCategory.SEMANTIC_ALLOW) is True
    assert coerce_to_binary(DecisionCategory.SEMANTIC_VETO) is False


def test_unknown_stored_category_raises_rather_than_mapping():
    with pytest.raises(ValueError, match="unrecognised decision category"):
        parse_category("PROBABLY_FINE")
    assert parse_category("TIMEOUT") is DecisionCategory.TIMEOUT


def test_tally_always_reports_all_six_categories():
    counts = tally([DecisionCategory.SEMANTIC_ALLOW, DecisionCategory.TIMEOUT])
    assert set(counts) == set(DECISION_CATEGORIES)
    assert counts[DecisionCategory.SEMANTIC_ALLOW] == 1
    assert counts[DecisionCategory.NON_TERMINATION] == 0


def test_rates_use_the_attempted_denominator():
    """Rates over parsed-only denominators are how a cross-lingual gap vanishes."""
    counts = tally([DecisionCategory.SEMANTIC_VETO] * 3 + [DecisionCategory.TIMEOUT] * 7)
    r = rates(counts, attempted=10)
    assert r[DecisionCategory.SEMANTIC_VETO] == pytest.approx(0.3)
    assert r[DecisionCategory.TIMEOUT] == pytest.approx(0.7)
    assert sum(r.values()) == pytest.approx(1.0)


def test_rates_reject_inconsistent_denominator():
    counts = tally([DecisionCategory.SEMANTIC_ALLOW] * 5)
    with pytest.raises(ValueError, match="counts and denominator disagree"):
        rates(counts, attempted=3)


def test_zero_attempted_is_a_stop_condition():
    with pytest.raises(ValueError, match="attempted must be positive"):
        rates(tally([]), attempted=0)


def test_stage_decision_validates_stage_name():
    ok = StageDecision(stage="critic", category=DecisionCategory.SEMANTIC_VETO)
    assert ok.is_failure is False
    with pytest.raises(ValueError, match="unknown stage"):
        StageDecision(stage="referee", category=DecisionCategory.SEMANTIC_ALLOW)


def test_stage_decision_serialises_without_model_text():
    decision = StageDecision(
        stage="planner",
        category=DecisionCategory.MALFORMED_DECISION,
        rationale_present=True,
        detector="schema_v1",
        evidence={"missing_key": "decision"},
    )
    payload = decision.to_dict()
    assert payload["category"] == "MALFORMED_DECISION"
    assert "text" not in payload
    assert payload["rationale_present"] is True
