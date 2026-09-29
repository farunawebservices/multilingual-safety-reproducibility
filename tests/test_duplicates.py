"""Duplicate detection, including the case that matters for HarmBench.

The load-bearing distinction: a repeated *primary text* column is not a
duplicated *item* when another text column distinguishes the rows. HarmBench
pairs one `Behavior` with several `ContextString`s; deduplicating on `Behavior`
alone would delete real behaviors.
"""

from __future__ import annotations

import pytest

from multilingual_safety.duplicates import (
    duplicate_summary,
    find_duplicate_composite,
    find_duplicate_ids,
    find_duplicate_text,
)

CLEAN = [
    {"BehaviorID": "a1", "Behavior": "alpha", "ContextString": ""},
    {"BehaviorID": "a2", "Behavior": "beta", "ContextString": ""},
    {"BehaviorID": "a3", "Behavior": "gamma", "ContextString": ""},
]

# The real HarmBench pattern: one instruction, three different contexts.
CONTEXTUAL = [
    {"BehaviorID": "sqli_1", "Behavior": "same instruction", "ContextString": "context one"},
    {"BehaviorID": "sqli_2", "Behavior": "same instruction", "ContextString": "context two"},
    {"BehaviorID": "sqli_3", "Behavior": "same instruction", "ContextString": "context three"},
]

TRUE_DUPES = [
    {"BehaviorID": "d1", "Behavior": "identical", "ContextString": "same"},
    {"BehaviorID": "d2", "Behavior": "identical", "ContextString": "same"},
]


def test_clean_table_has_no_duplicates():
    assert not find_duplicate_ids(CLEAN, "BehaviorID").has_duplicates
    assert not find_duplicate_text(CLEAN, "Behavior").has_duplicates
    assert not find_duplicate_composite(CLEAN, ["Behavior", "ContextString"]).has_duplicates


def test_duplicate_ids_are_detected_and_reported_verbatim():
    rows = [*CLEAN, {"BehaviorID": "a1", "Behavior": "delta", "ContextString": ""}]
    report = find_duplicate_ids(rows, "BehaviorID")
    assert report.has_duplicates
    assert report.n_duplicate_keys == 1
    assert report.n_rows_affected == 2
    assert report.duplicate_keys == ("a1",)  # IDs are metadata, safe to echo


def test_repeated_primary_text_is_not_a_duplicate_item():
    """The HarmBench case: same Behavior, different ContextString."""
    text = find_duplicate_text(CONTEXTUAL, "Behavior")
    composite = find_duplicate_composite(CONTEXTUAL, ["Behavior", "ContextString"])
    assert text.has_duplicates, "the primary column does repeat"
    assert text.n_rows_affected == 3
    assert not composite.has_duplicates, "but each row is a distinct item"

    summary = duplicate_summary(CONTEXTUAL, id_column="BehaviorID", text_columns=["Behavior", "ContextString"])
    assert summary["is_defective"] is False


def test_true_duplicates_are_flagged_as_defective():
    summary = duplicate_summary(TRUE_DUPES, id_column="BehaviorID", text_columns=["Behavior", "ContextString"])
    assert summary["composite"]["n_duplicate_keys"] == 1
    assert summary["is_defective"] is True


def test_duplicate_text_is_reported_as_hashes_not_text():
    """Findings go into a committed report, so they must not carry prompt text."""
    report = find_duplicate_text(CONTEXTUAL, "Behavior")
    for key in report.duplicate_keys:
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)
        assert "same instruction" not in key
    assert "Behavior" in report.key_name and report.key_name.startswith("sha256(")


def test_composite_separator_prevents_boundary_collisions():
    """'ab'+'c' must not collide with 'a'+'bc'."""
    a = [{"x": "ab", "y": "c"}]
    b = [{"x": "a", "y": "bc"}]
    assert (
        find_duplicate_composite([*a, *b], ["x", "y"]).n_duplicate_keys == 0
    ), "naive concatenation would have merged these"


def test_composite_requires_at_least_one_column():
    with pytest.raises(ValueError, match="at least one column"):
        find_duplicate_composite(CLEAN, [])


def test_missing_column_treated_as_empty_not_crash():
    report = find_duplicate_text(CLEAN, "NoSuchColumn")
    assert report.n_duplicate_keys == 1  # all three share the empty value


def test_summary_reports_all_three_levels():
    summary = duplicate_summary(CONTEXTUAL, id_column="BehaviorID", text_columns=["Behavior", "ContextString"])
    assert set(summary) >= {"ids", "primary_text", "composite", "is_defective", "note"}
    assert summary["composite_key_columns"] == ["Behavior", "ContextString"]


def test_report_is_deterministic_and_sorted():
    rows = [{"BehaviorID": f"id{i % 3}", "Behavior": f"t{i % 2}"} for i in range(12)]
    first = find_duplicate_ids(rows, "BehaviorID")
    second = find_duplicate_ids(list(reversed(rows)), "BehaviorID")
    assert first.duplicate_keys == second.duplicate_keys == tuple(sorted(first.duplicate_keys))
