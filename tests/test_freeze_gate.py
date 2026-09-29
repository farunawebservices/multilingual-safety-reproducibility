"""The freeze gate: `frozen: true` is only legitimate if the preconditions hold.

Two halves, and both are needed:

* the committed config passes the gate, and is frozen consistently;
* the gate REJECTS each precondition when it is violated. A gate that passes
  everything is indistinguishable from no gate, so every precondition is mutated
  on a deep copy and the failure is asserted.

Nothing here reads corpus text, and nothing writes.
"""

from __future__ import annotations

import copy
import json

import pytest
import yaml

from multilingual_safety.freeze import (
    EXPECTED_MODEL_REVISIONS,
    EXPECTED_OUTLIER_BEHAVIOR,
    FREEZE_PRECONDITIONS,
    check_freeze_preconditions,
    freeze_is_consistent,
)
from multilingual_safety.paths import CONFIGS_DIR, PROCESSED_DIR, RESULTS_DIR

PREREG_PATH = CONFIGS_DIR / "experiment_preregistration_v1.yaml"
SPLITS_PATH = PROCESSED_DIR / "splits_v1.json"
OUTLIER_PATH = RESULTS_DIR / "length_outlier_review.json"


@pytest.fixture(scope="module")
def prereg() -> dict:
    return yaml.safe_load(PREREG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def splits() -> dict | None:
    if not SPLITS_PATH.exists():
        return None
    return json.loads(SPLITS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def outlier() -> dict | None:
    if not OUTLIER_PATH.exists():
        return None
    return json.loads(OUTLIER_PATH.read_text(encoding="utf-8"))


def gate(prereg, splits=None, outlier=None) -> list[str]:
    return check_freeze_preconditions(prereg, splits, outlier)


# --- the committed state -----------------------------------------------------
def test_the_committed_config_passes_the_gate(prereg, splits, outlier):
    failures = gate(prereg, splits, outlier)
    assert failures == [], f"the frozen config violates its own preconditions: {failures}"


def test_the_config_is_frozen_consistently(prereg):
    assert prereg["frozen"] is True
    assert prereg["frozen_at"], "frozen requires a timestamp"
    assert freeze_is_consistent(prereg) == []


def test_frozen_at_is_an_utc_timestamp(prereg):
    from datetime import datetime, timezone

    stamp = datetime.fromisoformat(prereg["frozen_at"])
    assert stamp.tzinfo is not None, "frozen_at must be timezone-aware"
    assert stamp.utcoffset().total_seconds() == 0, "frozen_at must be UTC"
    assert stamp <= datetime.now(timezone.utc), "frozen_at must not be in the future"


def test_freezing_did_not_open_the_holdout(prereg):
    """Freezing makes final_test eligible to be read. It does not read it."""
    policy = prereg["holdout_policy"]
    assert policy["final_test_readable"] is False
    assert policy["final_test_reads"] == []
    assert prereg["conditions"]["intervention_policy"]["models_run"] is False


def test_freezing_did_not_manufacture_a_native_speaker_claim(prereg):
    claims = prereg["dataset"]["translation_validation"]["claims_permitted"]
    assert claims["native_speaker_signoff"] is False
    assert claims["row_level_native_speaker_validation"] is False
    assert prereg["dataset"]["review_state_at_preregistration"]["rows_human_reviewed"] == 0


def test_split_hash_is_the_verified_value_not_the_mistyped_one(prereg, splits):
    """The mistyped variants differ from the verified hash by one character."""
    declared = prereg["splits"]["split_hash"]
    provenance = prereg["splits"]["hash_provenance"]
    assert len(declared) == 64
    assert declared == provenance["verified_actual"]
    assert declared != provenance["supplied_in_request"]
    if splits is not None:
        assert declared == splits["split_hash"]


def test_every_precondition_has_a_stable_id():
    assert len(set(FREEZE_PRECONDITIONS)) == len(FREEZE_PRECONDITIONS)
    assert all(name.islower() for name in FREEZE_PRECONDITIONS)


# --- the gate rejects violations --------------------------------------------
def test_gate_rejects_a_drifted_model_revision(prereg, splits, outlier):
    broken = copy.deepcopy(prereg)
    broken["models"]["under_evaluation"][0]["revision"] = "main"
    failures = gate(broken, splits, outlier)
    assert any("model_revisions_pinned" in f for f in failures)


@pytest.mark.parametrize("model_id", sorted(EXPECTED_MODEL_REVISIONS))
def test_gate_rejects_a_missing_model(prereg, splits, outlier, model_id):
    broken = copy.deepcopy(prereg)
    broken["models"]["under_evaluation"] = [
        m for m in broken["models"]["under_evaluation"] if m["id"] != model_id
    ]
    failures = gate(broken, splits, outlier)
    assert any(model_id in f and "model_revisions_pinned" in f for f in failures)


def test_gate_rejects_the_mistyped_split_hash(prereg, splits, outlier):
    """The exact regression that hash_provenance exists to record."""
    broken = copy.deepcopy(prereg)
    mistyped = broken["splits"]["hash_provenance"]["supplied_in_request"]
    broken["splits"]["split_hash"] = mistyped
    failures = gate(broken, splits, outlier)
    assert any("split_hash_verified" in f for f in failures)


def test_gate_rejects_a_truncated_split_hash(prereg, splits, outlier):
    broken = copy.deepcopy(prereg)
    broken["splits"]["split_hash"] = prereg["splits"]["split_hash"][:63]
    failures = gate(broken, splits, outlier)
    assert any("64-char" in f for f in failures)


def test_gate_rejects_a_readable_holdout(prereg, splits, outlier):
    broken = copy.deepcopy(prereg)
    broken["holdout_policy"]["final_test_readable"] = True
    failures = gate(broken, splits, outlier)
    assert any("holdout_closed" in f for f in failures)


def test_gate_rejects_a_pre_existing_holdout_read(prereg, splits, outlier):
    broken = copy.deepcopy(prereg)
    broken["holdout_policy"]["final_test_reads"] = [{"run_id": "r", "date": "d", "reason": "oops"}]
    failures = gate(broken, splits, outlier)
    assert any("holdout_closed" in f for f in failures)


def test_gate_rejects_no_interventions(prereg, splits, outlier):
    broken = copy.deepcopy(prereg)
    broken["conditions"]["interventions"] = []
    failures = gate(broken, splits, outlier)
    assert any("interventions_have_matched_controls" in f for f in failures)


def test_gate_rejects_a_vector_arm_without_a_control(prereg, splits, outlier):
    broken = copy.deepcopy(prereg)
    arm = next(a for a in broken["conditions"]["interventions"] if a["applies_vector"])
    arm["control"] = "none"
    failures = gate(broken, splits, outlier)
    assert any("applies a vector with no control" in f for f in failures)


@pytest.mark.parametrize(
    "key", ["random_vector_matched_norm", "random_vector_matched_shape"]
)
def test_gate_rejects_an_unmatched_random_vector(prereg, splits, outlier, key):
    broken = copy.deepcopy(prereg)
    arm = next(a for a in broken["conditions"]["interventions"] if a["applies_vector"])
    arm["control_spec"][key] = False
    failures = gate(broken, splits, outlier)
    assert any(key in f for f in failures)


@pytest.mark.parametrize(
    "key",
    [
        "vector_construction_selected_on",
        "layer_selection_selected_on",
        "strength_grid_selected_on",
    ],
)
def test_gate_rejects_selection_on_final_test(prereg, splits, outlier, key):
    broken = copy.deepcopy(prereg)
    arm = next(a for a in broken["conditions"]["interventions"] if a["applies_vector"])
    arm["selection"][key] = "final_test"
    failures = gate(broken, splits, outlier)
    assert any(key in f for f in failures)


def test_gate_rejects_final_test_access_during_selection(prereg, splits, outlier):
    broken = copy.deepcopy(prereg)
    arm = next(a for a in broken["conditions"]["interventions"] if a["applies_vector"])
    arm["selection"]["final_test_accessible_during_selection"] = True
    failures = gate(broken, splits, outlier)
    assert any("allows final_test during selection" in f for f in failures)


def test_gate_rejects_the_wrong_reported_effect(prereg, splits, outlier):
    broken = copy.deepcopy(prereg)
    arm = next(a for a in broken["conditions"]["interventions"] if a["applies_vector"])
    arm["reporting"]["reported_effect"] = "intervention_minus_baseline"
    failures = gate(broken, splits, outlier)
    assert any("must be intervention_minus_matched_control" in f for f in failures)


@pytest.mark.parametrize("key", ["retain_negative_results", "retain_unstable_results"])
def test_gate_rejects_discarding_negative_or_unstable_results(prereg, splits, outlier, key):
    broken = copy.deepcopy(prereg)
    broken["conditions"]["interventions"][0]["reporting"][key] = False
    failures = gate(broken, splits, outlier)
    assert any(key in f for f in failures)


def test_gate_rejects_a_joint_arm_controlled_at_one_role(prereg, splits, outlier):
    broken = copy.deepcopy(prereg)
    joint = next(
        a for a in broken["conditions"]["interventions"] if "_and_" in str(a["target_role"])
    )
    joint["control_spec"]["matched_at_every_targeted_role"] = False
    failures = gate(broken, splits, outlier)
    assert any("control at every targeted role" in f for f in failures)


def test_gate_rejects_baseline_promoted_to_comparator(prereg, splits, outlier):
    broken = copy.deepcopy(prereg)
    baseline = next(a for a in broken["conditions"]["interventions"] if not a["applies_vector"])
    baseline["usable_as_effect_comparator"] = True
    failures = gate(broken, splits, outlier)
    assert any("effect comparator" in f for f in failures)


def test_gate_rejects_a_missing_translation_limitation(prereg, splits, outlier):
    broken = copy.deepcopy(prereg)
    broken["dataset"]["translation_validation"]["residual_limitation"][
        "translation_quality_remains_a_limitation"
    ] = False
    failures = gate(broken, splits, outlier)
    assert any("translation_limitations_explicit" in f for f in failures)


def test_gate_rejects_dropping_the_low_resource_languages(prereg, splits, outlier):
    broken = copy.deepcopy(prereg)
    broken["dataset"]["translation_validation"]["residual_limitation"]["weakest_languages"] = ["ha"]
    failures = gate(broken, splits, outlier)
    assert any("weakest_languages" in f for f in failures)


def test_gate_rejects_a_native_speaker_signoff_claim(prereg, splits, outlier):
    broken = copy.deepcopy(prereg)
    broken["dataset"]["translation_validation"]["claims_permitted"][
        "native_speaker_signoff"
    ] = True
    failures = gate(broken, splits, outlier)
    assert any("native_speaker_signoff must be false" in f for f in failures)


def test_gate_rejects_silent_limitations(prereg, splits, outlier):
    broken = copy.deepcopy(prereg)
    broken["declared_limitations"] = ["The tool surface is mocked."]
    failures = gate(broken, splits, outlier)
    assert any("declared_limitations never mentions" in f for f in failures)


def test_gate_rejects_missing_outlier_metadata(prereg, splits, outlier):
    broken = copy.deepcopy(prereg)
    broken["dataset"]["translation_validation"]["length_outliers"]["recorded"] = []
    failures = gate(broken, splits, outlier)
    assert any(EXPECTED_OUTLIER_BEHAVIOR in f for f in failures)


def test_gate_rejects_an_outlier_record_that_clears_review(prereg, splits, outlier):
    broken = copy.deepcopy(prereg)
    broken["dataset"]["translation_validation"]["length_outliers"]["recorded"][0][
        "clears_review_required"
    ] = True
    failures = gate(broken, splits, outlier)
    assert any("must not clear review_required" in f for f in failures)


def test_gate_rejects_an_outlier_record_that_contradicts_its_report(prereg, splits, outlier):
    if outlier is None:
        pytest.skip("outlier report not present")
    broken = copy.deepcopy(prereg)
    broken["dataset"]["translation_validation"]["length_outliers"]["recorded"][0][
        "decision"
    ] = "REJECT"
    failures = gate(broken, splits, outlier)
    assert any("disagrees with the cited report" in f for f in failures)


def test_gate_rejects_prompt_text_in_the_outlier_report(prereg, splits, outlier):
    if outlier is None:
        pytest.skip("outlier report not present")
    tainted = copy.deepcopy(outlier)
    tainted["contains_prompt_text"] = True
    failures = gate(prereg, splits, tainted)
    assert any("carries prompt text" in f for f in failures)


def test_gate_rejects_a_freeze_after_models_have_run(prereg, splits, outlier):
    broken = copy.deepcopy(prereg)
    broken["conditions"]["intervention_policy"]["models_run"] = True
    failures = gate(broken, splits, outlier)
    assert any("models_run is not false" in f for f in failures)


# --- the disclosure requirement ---------------------------------------------
def test_the_pre_freeze_deviation_is_declared(prereg):
    """One final_test row was inspected before freezing; it must be on the record."""
    deviations = prereg["holdout_policy"]["pre_freeze_deviations"]
    assert isinstance(deviations, list)
    entry = next(d for d in deviations if EXPECTED_OUTLIER_BEHAVIOR in d["behavior_ids"])
    assert entry["rule"]
    assert entry["scope"]
    assert entry["what_was_not_done"]
    assert entry["rationale"]
    assert entry["recorded_in"] == "results/length_outlier_review.json"


def test_gate_rejects_an_undisclosed_holdout_inspection(prereg, splits, outlier):
    """The gate must not let a known departure be quietly dropped."""
    if outlier is None:
        pytest.skip("outlier report not present")
    broken = copy.deepcopy(prereg)
    broken["holdout_policy"]["pre_freeze_deviations"] = []
    failures = gate(broken, splits, outlier)
    assert any("no deviation discloses it" in f for f in failures)


def test_gate_rejects_a_missing_deviations_key(prereg, splits, outlier):
    broken = copy.deepcopy(prereg)
    del broken["holdout_policy"]["pre_freeze_deviations"]
    failures = gate(broken, splits, outlier)
    assert any("pre_freeze_deviations is missing" in f for f in failures)


def test_gate_rejects_an_unexplained_deviation(prereg, splits, outlier):
    broken = copy.deepcopy(prereg)
    broken["holdout_policy"]["pre_freeze_deviations"][0]["rationale"] = ""
    failures = gate(broken, splits, outlier)
    assert any("has no rationale" in f for f in failures)


# --- freeze consistency ------------------------------------------------------
def test_frozen_without_a_timestamp_is_inconsistent():
    assert freeze_is_consistent({"frozen": True, "frozen_at": None}) != []


def test_timestamp_without_frozen_is_inconsistent():
    assert freeze_is_consistent({"frozen": False, "frozen_at": "2026-08-17T00:00:00+00:00"}) != []


def test_non_utc_timestamp_is_rejected():
    assert freeze_is_consistent({"frozen": True, "frozen_at": "2026-08-17T00:00:00+02:00"}) != []


def test_unfrozen_and_unstamped_is_consistent():
    assert freeze_is_consistent({"frozen": False, "frozen_at": None}) == []
