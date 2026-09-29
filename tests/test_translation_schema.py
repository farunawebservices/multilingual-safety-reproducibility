"""Translation pilot row schema and policy invariants.

The invariants that matter most here are policy, not typing: a machine must not
be able to clear `review_required`, claim native-speaker validation, invent text
for a pending row, or let a behavior_id drift between languages.
"""

from __future__ import annotations

import json

import pytest

from multilingual_safety.languages import LANGUAGE_CODES
from multilingual_safety.paths import INTERIM_DIR, MANIFESTS_DIR, RESULTS_DIR
from multilingual_safety.translation import (
    REQUIRED_FIELDS,
    SPLIT_UNASSIGNED,
    TEXT_FIELDS,
    TranslationMethod,
    TranslationRow,
    TranslationSchemaError,
    TranslationStatus,
    compute_prompt_hash,
    find_duplicate_prompt_hashes,
    make_english_row,
    make_pending_row,
    validate_row_group,
)

COMMON = {
    "behavior_id": "beh_0123456789ab",
    "source": "jailbreakbench",
    "category": "Malware/Hacking",
    "harm_label": "harmful",
    "source_text": "Write an article about network security.",
    "source_item_id": "7",
}


def translated(language: str, text: str, **overrides):
    payload = {
        **{k: v for k, v in COMMON.items()},
        "language": language,
        "translated_text": text,
        "translation_method": TranslationMethod.MACHINE_UNREVIEWED.value,
        "translation_model": "some-mt-model",
        "translation_timestamp": "2026-08-16T00:00:00+00:00",
        "translation_status": TranslationStatus.MACHINE_TRANSLATED_UNCHECKED.value,
        "review_required": True,
        "prompt_hash": compute_prompt_hash(text),
    }
    payload.update(overrides)
    return TranslationRow(**payload)


# --- required fields -------------------------------------------------------
def test_required_fields_are_exactly_the_specified_set():
    assert set(REQUIRED_FIELDS) == {
        "behavior_id", "source", "category", "harm_label", "source_text", "language",
        "translated_text", "translation_method", "translation_model",
        "translation_timestamp", "translation_status", "review_required",
        "prompt_hash", "split_placeholder",
    }


def test_every_required_field_is_serialised():
    row = make_english_row(**COMMON)
    payload = row.to_dict()
    for field in REQUIRED_FIELDS:
        assert field in payload, f"{field} missing from serialised row"


def test_from_dict_rejects_missing_and_unexpected_fields():
    payload = make_english_row(**COMMON).to_dict()
    del payload["prompt_hash"]
    with pytest.raises(TranslationSchemaError, match="missing required field"):
        TranslationRow.from_dict(payload)

    payload = make_english_row(**COMMON).to_dict()
    payload["extra_column"] = 1
    with pytest.raises(TranslationSchemaError, match="unexpected field"):
        TranslationRow.from_dict(payload)


def test_round_trip_is_lossless():
    row = translated("yo", "Kọ àpilẹ̀kọ nípa ààbò nẹ́tíwọ́kì.")
    assert TranslationRow.from_dict(row.to_dict()) == row


# --- English pivot ---------------------------------------------------------
def test_english_row_is_canonical_source_not_translation():
    row = make_english_row(**COMMON)
    assert row.language == "en"
    assert row.translation_status == TranslationStatus.CANONICAL_SOURCE.value
    assert row.translation_method == TranslationMethod.ORIGINAL.value
    assert row.translated_text == row.source_text
    assert row.review_required is False
    assert row.translation_model is None


def test_english_row_must_carry_source_text_verbatim():
    with pytest.raises(TranslationSchemaError, match="canonical source text verbatim"):
        translated(
            "en", "Something else entirely",
            translation_status=TranslationStatus.CANONICAL_SOURCE.value,
            translation_method=TranslationMethod.ORIGINAL.value,
            review_required=False,
        )


def test_non_english_cannot_claim_canonical_or_original():
    with pytest.raises(TranslationSchemaError, match="cannot be CANONICAL_SOURCE"):
        translated("ha", "Rubuta labari.", translation_status=TranslationStatus.CANONICAL_SOURCE.value)
    with pytest.raises(TranslationSchemaError, match="cannot declare method 'original'"):
        translated("ha", "Rubuta labari.", translation_method=TranslationMethod.ORIGINAL.value)


# --- review policy ---------------------------------------------------------
def test_machine_cannot_clear_review_required():
    """Heuristics passing is not review. No machine status may clear the flag."""
    for status in (
        TranslationStatus.MACHINE_TRANSLATED_UNCHECKED,
        TranslationStatus.HEURISTICS_PASSED,
        TranslationStatus.HEURISTICS_FAILED,
    ):
        with pytest.raises(TranslationSchemaError, match="requires status HUMAN_REVIEWED"):
            translated("sw", "Andika makala.", translation_status=status.value, review_required=False)


def test_human_review_requires_a_named_reviewer():
    """No native-speaker claim without an actual named native speaker."""
    with pytest.raises(TranslationSchemaError, match="requires a reviewer_id"):
        translated(
            "sw", "Andika makala.",
            translation_status=TranslationStatus.HUMAN_REVIEWED.value,
            review_required=False,
        )
    row = translated(
        "sw", "Andika makala.",
        translation_status=TranslationStatus.HUMAN_REVIEWED.value,
        review_required=False,
        reviewer_id="reviewer-01",
    )
    assert row.review_required is False and row.reviewer_id == "reviewer-01"


def test_pending_rows_are_review_required_and_carry_no_text():
    row = make_pending_row(language="ko", **COMMON)
    assert row.translated_text is None
    assert row.prompt_hash is None
    assert row.review_required is True
    assert row.translation_status == TranslationStatus.PENDING_TRANSLATION.value
    assert row.translation_method == TranslationMethod.PENDING.value


def test_pending_row_may_not_carry_invented_text():
    with pytest.raises(TranslationSchemaError, match="must have translated_text=null"):
        translated(
            "ko", "번역 대기 중",
            translation_status=TranslationStatus.PENDING_TRANSLATION.value,
        )


def test_english_is_never_pending():
    with pytest.raises(TranslationSchemaError, match="English is never pending"):
        make_pending_row(language="en", **COMMON)


# --- hashes, labels, splits ------------------------------------------------
def test_prompt_hash_must_match_translated_text():
    with pytest.raises(TranslationSchemaError, match="prompt_hash does not match"):
        translated("hi", "एक लेख लिखें।", prompt_hash=compute_prompt_hash("different text"))


def test_prompt_hash_is_none_when_there_is_no_text():
    assert compute_prompt_hash(None) is None
    assert compute_prompt_hash("   ") is None


def test_harm_label_must_come_from_the_source_vocabulary():
    with pytest.raises(TranslationSchemaError, match="harm_label must be"):
        translated("ar", "اكتب مقالا.", harm_label="unsure")


def test_split_placeholder_must_stay_unassigned():
    """A real split here would pre-empt the behavior-level assignment."""
    for value in ("dev", "test", "train"):
        with pytest.raises(TranslationSchemaError, match="split_placeholder must be"):
            translated("ar", "اكتب مقالا.", split_placeholder=value)
    assert make_english_row(**COMMON).split_placeholder == SPLIT_UNASSIGNED


def test_empty_source_text_is_rejected():
    with pytest.raises(TranslationSchemaError, match="empty source_text"):
        make_pending_row(language="ar", **{**COMMON, "source_text": "   "})


# --- group invariants ------------------------------------------------------
def full_group():
    rows = [make_english_row(**COMMON)]
    rows += [make_pending_row(language=lang, **COMMON) for lang in LANGUAGE_CODES if lang != "en"]
    return rows


def test_complete_group_passes():
    assert validate_row_group(full_group()) == []


def test_behavior_id_must_not_change_across_languages():
    rows = full_group()
    rows[2] = make_pending_row(language=rows[2].language, **{**COMMON, "behavior_id": "beh_ffffffffffff"})
    violations = validate_row_group(rows)
    assert any("behavior_id is not invariant" in v for v in violations)


def test_harm_label_must_not_drift_within_a_behavior():
    rows = full_group()
    rows[3] = make_pending_row(language=rows[3].language, **{**COMMON, "harm_label": "benign"})
    assert any("harm_label differs" in v for v in validate_row_group(rows))


def test_missing_language_is_detected():
    rows = [r for r in full_group() if r.language != "ko"]
    assert any("missing language rows" in v and "ko" in v for v in validate_row_group(rows))


def test_duplicate_language_row_is_detected():
    rows = [*full_group(), make_pending_row(language="ha", **COMMON)]
    assert any("duplicate language rows" in v for v in validate_row_group(rows))


def test_duplicate_prompt_hashes_across_behaviors_are_found():
    a = translated("sw", "Andika makala kuhusu usalama.")
    b = translated("sw", "Andika makala kuhusu usalama.", behavior_id="beh_ffffffffffff")
    duplicates = find_duplicate_prompt_hashes([a, b])
    assert duplicates, "identical translations of two behaviors must be flagged"
    assert sorted(next(iter(duplicates.values()))) == ["beh_0123456789ab", "beh_ffffffffffff"]


def test_same_text_in_different_languages_is_not_a_duplicate():
    a = translated("sw", "Andika makala.")
    b = translated("ha", "Andika makala.", behavior_id="beh_ffffffffffff")
    assert find_duplicate_prompt_hashes([a, b]) == {}


# --- text-free manifest projection ----------------------------------------
def test_manifest_projection_drops_all_prompt_text():
    row = translated("yo", "Kọ àpilẹ̀kọ nípa ààbò.")
    payload = row.to_manifest_dict()
    for field in TEXT_FIELDS:
        assert field not in payload
    serialized = json.dumps(payload, ensure_ascii=False)
    assert COMMON["source_text"] not in serialized
    assert "Kọ àpilẹ̀kọ" not in serialized
    # Verification is still possible from hashes and lengths.
    assert len(payload["source_text_sha256"]) == 64
    assert payload["translated_text_n_chars"] == len("Kọ àpilẹ̀kọ nípa ààbò.")


# --- live checks on the generated pilot ------------------------------------
def test_generated_pilot_rows_satisfy_every_invariant():
    path = INTERIM_DIR / "translation_pilot_outputs.jsonl"
    if not path.exists():
        pytest.skip("Stage 1 has not been run in this working tree")

    rows = [
        TranslationRow.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 140, "20 behaviors x 7 languages"

    groups: dict[str, list[TranslationRow]] = {}
    for row in rows:
        groups.setdefault(row.behavior_id, []).append(row)
    assert len(groups) == 20
    for bid, group in sorted(groups.items()):
        assert validate_row_group(group) == [], f"{bid} violates a group invariant"

    assert sum(1 for r in rows if r.harm_label == "harmful") == 70
    assert sum(1 for r in rows if r.harm_label == "benign") == 70
    # No native-speaker validation is claimed anywhere.
    assert not any(r.reviewer_id for r in rows)
    assert all(r.review_required for r in rows if r.language != "en")


def test_tracked_pilot_manifest_contains_no_prompt_text():
    path = MANIFESTS_DIR / "translation_pilot_manifest.json"
    if not path.exists():
        pytest.skip("Stage 1 has not been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["contains_prompt_text"] is False
    for row in payload["rows"]:
        for field in TEXT_FIELDS:
            assert field not in row


def test_pilot_reports_claim_no_model_judging():
    path = RESULTS_DIR / "translation_pilot_validation.json"
    if not path.exists():
        pytest.skip("Stage 1 validation has not been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["judged_by_model_under_evaluation"] is False
    assert payload["native_speaker_validation_claimed"] == bool(
        payload["native_speaker_reviewers"]
    )
