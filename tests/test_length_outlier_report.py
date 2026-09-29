"""The metadata-only length-outlier review record stays honest about its limits.

``results/length_outlier_review.json`` records that one ``final_test`` Yoruba row
was inspected out of band and accepted. The check performed had no structured
reviewer identity, so the record must never read as native-speaker sign-off, and
must never carry the text or the exact length of a held-out row.

These tests fail if a later edit quietly upgrades the record's authority.
"""

from __future__ import annotations

import json

import pytest

from multilingual_safety.paths import RESULTS_DIR

REPORT_PATH = RESULTS_DIR / "length_outlier_review.json"

#: Fields whose presence would mean text or an exact length leaked into a
#: tracked file.
FORBIDDEN_REPORT_FIELDS = (
    "source_text",
    "translated_text",
    "prompt",
    "text",
    "character_length",
    "n_chars",
    "length_chars",
)


@pytest.fixture
def report() -> dict:
    if not REPORT_PATH.exists():
        pytest.skip("length_outlier_review.json has not been generated in this checkout")
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_records_the_requested_metadata(report):
    assert report["behavior_id"] == "beh_b117279f5262"
    assert report["language"] == "yo"
    assert report["split"] == "final_test"
    assert report["decision"] == "ACCEPT"
    assert report["outlier_label"] == "valid_translation_length_outlier"
    assert report["review_provenance"] == "external_native_check_unstructured"
    assert report["character_length_bucket"]


def test_does_not_claim_native_speaker_validation(report):
    """An unstructured external check is not sign-off of record."""
    assert report["structured_reviewer_identity_captured"] is False
    assert report["clears_review_required"] is False
    assert report["native_speaker_validation_of_record"] is False
    assert report["review_decision_schema_representable"] is False
    assert "no reviewer_id" in report["structured_reviewer_identity_note"]


def test_row_state_is_recorded_as_unchanged(report):
    """The ACCEPT does not promote the row's status in the frozen dataset."""
    state = report["row_state_unchanged"]
    assert state["translation_status"] == "MACHINE_TRANSLATED_UNCHECKED"
    assert state["review_required"] is True
    assert state["reviewer_id"] is None


def test_not_frozen_yet(report):
    assert report["frozen"] is False


def test_declares_the_inputs_untouched(report):
    for key in (
        "frozen_dataset_unmodified",
        "splits_unmodified",
        "split_assignment_unmodified",
        "source_hashes_unmodified",
        "translation_hashes_unmodified",
    ):
        assert report[key] is True, f"{key} must be asserted true"


def test_is_text_free(report):
    assert report["contains_prompt_text"] is False
    assert report["final_test_text_loaded_into_report"] is False
    blob = json.dumps(report, ensure_ascii=False)
    for field in FORBIDDEN_REPORT_FIELDS:
        assert f'"{field}":' not in blob, f"report carries a {field} field"


def test_is_traceable_to_a_run(report):
    assert report["run_id"]
    assert report["stage"] == "stage3.2_record_length_outlier_review"
    assert report["deterministic_digest"]
