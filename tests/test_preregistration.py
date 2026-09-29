"""The pre-registration config, checked against the artifacts it claims to pin.

A pre-registration that drifts from the data it describes is worse than none: it
lends false authority to numbers it no longer covers. These tests tie every
claim in the file back to something verifiable — the frozen split file, the
decision taxonomy, the language registry, the agent schemas.
"""

from __future__ import annotations

import json

import pytest
import yaml

from multilingual_safety.agents.schemas import SCHEMA_VERSIONS
from multilingual_safety.decisions import DECISION_CATEGORIES, FAILURE_CATEGORIES
from multilingual_safety.freeze import freeze_is_consistent
from multilingual_safety.languages import LANGUAGE_CODES, PIVOT_LANGUAGE
from multilingual_safety.paths import CONFIGS_DIR, PROCESSED_DIR
from multilingual_safety.review import ReviewDecisionType  # noqa: F401  (vocabulary sanity)
from multilingual_safety.splits import FINAL_TEST, HOLDOUT_SPLIT_NAMES, load_splits

PREREG_PATH = CONFIGS_DIR / "experiment_preregistration_v1.yaml"
SPLITS_PATH = PROCESSED_DIR / "splits_v1.json"
TRANSLATION_CONFIG = CONFIGS_DIR / "translation.yaml"

MODELS = {"llama-3.3-70b-instruct", "qwen-2.5-72b-instruct"}


@pytest.fixture(scope="module")
def prereg() -> dict:
    assert PREREG_PATH.exists(), "pre-registration config is missing"
    return yaml.safe_load(PREREG_PATH.read_text(encoding="utf-8"))


# --- identity and freeze state ---------------------------------------------
def test_schema_and_status(prereg):
    assert prereg["schema_version"] == "preregistration.v1"
    # Frozen as of 2026-08-17. The preconditions that licence the freeze, and the
    # mutation tests proving the gate enforces them, live in tests/test_freeze_gate.py.
    assert prereg["frozen"] is True
    assert prereg["frozen_at"], "a freeze requires a timestamp"
    assert freeze_is_consistent(prereg) == []


def test_dataset_version_and_counts(prereg):
    dataset = prereg["dataset"]
    assert dataset["version"] == "v1"
    assert dataset["n_behaviors"] == 200
    assert dataset["n_behaviors_per_arm"] == {"harmful": 100, "benign": 100}
    assert dataset["n_rows"] == 1400
    assert dataset["n_translated_rows"] == 1200
    assert dataset["n_languages"] == len(LANGUAGE_CODES)
    assert dataset["harmbench_used"] is False


# --- the split hash is the load-bearing claim -------------------------------
def test_split_hash_matches_the_frozen_split_file(prereg):
    if not SPLITS_PATH.exists():
        pytest.skip("splits not frozen in this working tree")
    actual = load_splits(SPLITS_PATH)["split_hash"]
    assert prereg["splits"]["split_hash"] == actual, (
        "the pre-registered split_hash does not match the frozen split file"
    )


def test_hash_discrepancy_is_recorded_not_hidden(prereg):
    """A corrected hash must leave an audit trail."""
    provenance = prereg["splits"]["hash_provenance"]
    assert provenance["verified_actual"] == prereg["splits"]["split_hash"]
    assert provenance["supplied_in_request"] != provenance["verified_actual"]
    assert provenance["differing_characters"] == 1
    supplied, verified = provenance["supplied_in_request"], provenance["verified_actual"]
    assert sum(1 for a, b in zip(supplied, verified, strict=True) if a != b) == 1


def test_split_counts_match_the_frozen_file(prereg):
    if not SPLITS_PATH.exists():
        pytest.skip("splits not frozen")
    payload = load_splits(SPLITS_PATH)
    assert prereg["splits"]["dev"]["n_behaviors"] == payload["counts"]["dev"]
    assert prereg["splits"]["final_test"]["n_behaviors"] == payload["counts"][FINAL_TEST]
    assert prereg["splits"]["salt"] == payload["salt"]
    assert prereg["splits"]["stratified_by"] == payload["stratified_by"]
    assert prereg["splits"]["holdout_per_stratum"] == payload["holdout_per_stratum"]


def test_dev_and_holdout_arm_counts(prereg):
    assert prereg["splits"]["dev"] == {
        "n_behaviors": 140, "harmful": 70, "benign": 70, "per_category": 14
    }
    assert prereg["splits"]["final_test"] == {
        "n_behaviors": 60, "harmful": 30, "benign": 30, "per_category": 6
    }


# --- holdout policy ---------------------------------------------------------
def test_holdout_stays_closed_after_freezing(prereg):
    """Freezing satisfies one of two requirements; the read event is the other."""
    policy = prereg["holdout_policy"]
    assert policy["final_test_readable"] is False
    assert policy["final_test_reads"] == [], "no holdout read may have happened yet"
    assert "frozen == true" in policy["requires"]
    assert "recorded holdout access event" in policy["requires"]


def test_final_test_is_guarded_in_code(prereg):
    """The policy must be backed by the runtime guard, not just prose."""
    assert FINAL_TEST in HOLDOUT_SPLIT_NAMES
    assert "allow_test=True" in prereg["holdout_policy"]["enforced_by"]


def test_strength_selection_happens_on_dev(prereg):
    assert prereg["conditions"]["intervention_policy"]["strength_selected_on"] == "dev"


# --- languages --------------------------------------------------------------
def test_seven_languages_match_the_registry(prereg):
    languages = prereg["languages"]
    assert languages["targets"] == list(LANGUAGE_CODES)
    assert languages["pivot"] == PIVOT_LANGUAGE
    tiers = languages["resource_tiers"]
    assert {lang for group in tiers.values() for lang in group} == set(LANGUAGE_CODES)
    assert set(tiers["low"]) == {"ha", "yo", "sw"}


# --- models -----------------------------------------------------------------
def test_both_models_have_a_pinned_revision(prereg):
    """A moving ref would silently change the system under test between runs."""
    models = {m["id"]: m for m in prereg["models"]["under_evaluation"]}
    assert set(models) == MODELS
    for model_id, model in models.items():
        revision = model["revision"]
        assert isinstance(revision, str) and len(revision) == 40, (
            f"{model_id}: revision must be a full 40-char commit sha, got {revision!r}"
        )
        assert all(c in "0123456789abcdef" for c in revision), f"{model_id}: not hex"
        assert revision not in {"main", "master", "HEAD"}, f"{model_id}: moving ref"
        assert model["revision_type"] == "hf_model_commit_sha"
        # Credentials are still referenced by NAME only, and never a value.
        assert model["api_key_env"] is None
        assert model["provider"] is None, "serving stack is chosen at run time"


def test_model_and_tokenizer_ids_are_recorded(prereg):
    models = {m["id"]: m for m in prereg["models"]["under_evaluation"]}
    assert models["llama-3.3-70b-instruct"]["hf_model_id"] == "meta-llama/Llama-3.3-70B-Instruct"
    assert models["qwen-2.5-72b-instruct"]["hf_model_id"] == "Qwen/Qwen2.5-72B-Instruct"
    for model in models.values():
        assert model["hf_tokenizer_id"] == model["hf_model_id"]
        assert model["retrieved_at"], "retrieval timestamp is required"
        assert model["licence"]
        assert model["access_verified"] is True


def test_revisions_are_distinct_and_resolution_is_recorded(prereg):
    models = prereg["models"]["under_evaluation"]
    assert len({m["revision"] for m in models}) == len(models)
    resolution = prereg["models"]["revision_resolution"]
    assert resolution["method"] == "huggingface_hub.HfApi.model_info"
    assert resolution["access_verified"] is True
    assert resolution["retrieved_at"]


def test_gated_model_records_its_agreement_requirement(prereg):
    """Llama is gated; forgetting that surfaces as a failure mid-run."""
    models = {m["id"]: m for m in prereg["models"]["under_evaluation"]}
    llama = models["llama-3.3-70b-instruct"]
    assert llama["gated"] == "manual"
    assert llama["requires_agreement"] is True
    assert models["qwen-2.5-72b-instruct"]["requires_agreement"] is False


def test_models_under_evaluation_may_not_translate_or_judge(prereg):
    assert set(prereg["models"]["excluded_from_translation_and_judging"]) == MODELS
    assert prereg["models"]["judge_policy"]["llm_judge_allowed"] is False
    # And the exclusion is actually enforced in the translation config.
    translation = yaml.safe_load(TRANSLATION_CONFIG.read_text(encoding="utf-8"))
    excluded = set(translation["llm_backend"]["excluded_models"])
    assert excluded >= MODELS, "evaluation targets must be excluded as translators/judges"
    assert translation["llm_backend"]["must_not_be_under_evaluation"] is True


# --- roles and conditions ---------------------------------------------------
def test_three_roles_match_the_agent_schemas(prereg):
    roles = prereg["roles"]
    assert set(roles) == {"planner", "tool_caller", "critic"}
    for role, spec in roles.items():
        assert spec["schema"] == SCHEMA_VERSIONS[role]
    assert roles["tool_caller"]["allow_real_execution"] is False
    assert roles["tool_caller"]["tools"] == "mock_only"


def test_direct_single_agent_and_ablations_declared(prereg):
    conditions = prereg["conditions"]
    primary = {c["id"] for c in conditions["primary"]}
    assert {"multi_agent_full", "direct_single_agent"} <= primary
    ablations = {a["id"] for a in conditions["ablations"]}
    assert {"no_critic", "no_tool_caller", "critic_language_en"} <= ablations


def test_interventions_require_a_matched_control(prereg):
    policy = prereg["conditions"]["intervention_policy"]
    assert policy["require_matched_control"] is True
    assert policy["reported_effect"] == "intervention_minus_matched_control"


# --- intervention arms ------------------------------------------------------
INTERVENTION_ARMS = {
    "baseline": ("none", "none"),
    "planner_only": ("planner", "random_vector_matched_norm_planner"),
    "critic_only": ("critic", "random_vector_matched_norm_critic"),
    "planner_critic": ("planner_and_critic", "matched_random_vectors_at_both_roles"),
}


@pytest.fixture(scope="module")
def interventions(prereg) -> list[dict]:
    return prereg["conditions"]["interventions"]


def test_exactly_the_four_declared_arms(interventions):
    assert [arm["id"] for arm in interventions] == list(INTERVENTION_ARMS), (
        "the declared arms, in order, must be exactly these four"
    )
    for arm in interventions:
        target_role, control = INTERVENTION_ARMS[arm["id"]]
        assert arm["target_role"] == target_role, f"{arm['id']}: wrong target_role"
        assert arm["control"] == control, f"{arm['id']}: wrong control"


def test_target_roles_are_real_pipeline_roles(interventions):
    """A vector cannot be applied at a role the pipeline does not have."""
    from multilingual_safety.agents.schemas import ROLES

    for arm in interventions:
        role = arm["target_role"]
        if role == "none":
            assert arm["applies_vector"] is False
            continue
        for component in role.split("_and_"):
            assert component in ROLES, f"{arm['id']}: {component!r} is not a pipeline role"


def test_intervention_ids_are_namespaced_against_the_ablations(prereg, interventions):
    """`planner_only` is both an ablation and an intervention; they must not collide."""
    ablation_ids = {a["id"] for a in prereg["conditions"]["ablations"]}
    for arm in interventions:
        assert arm["qualified_id"] == f"intervention.{arm['id']}"
        if arm["id"] in ablation_ids:
            assert arm["qualified_id"] not in ablation_ids
    qualified = [arm["qualified_id"] for arm in interventions]
    assert len(set(qualified)) == len(qualified), "qualified ids must be unique"


def test_every_arm_selects_on_dev_only(interventions):
    """Vector, layer, and strength are hyperparameters -- dev only (CLAUDE.md §1)."""
    for arm in interventions:
        selection = arm["selection"]
        for key in (
            "vector_construction_selected_on",
            "layer_selection_selected_on",
            "strength_grid_selected_on",
        ):
            assert selection[key] == "dev", f"{arm['id']}: {key} must be selected on dev"
        assert selection["final_test_accessible_during_selection"] is False, (
            f"{arm['id']}: final_test must be inaccessible during selection"
        )


def test_vector_arms_have_a_matched_random_vector_control(interventions):
    for arm in interventions:
        if not arm["applies_vector"]:
            continue
        control = arm["control_spec"]
        assert control["random_vector_matched_norm"] is True, f"{arm['id']}: norm not matched"
        assert control["random_vector_matched_shape"] is True, f"{arm['id']}: shape not matched"
        assert control["applied_at_matched_layer"] is True
        assert control["applied_at_matched_token_positions"] is True
        assert control["same_items"] is True
        assert control["same_seeds"] is True
        assert control["same_session"] is True
        assert arm["control"] != "none", f"{arm['id']} applies a vector but declares no control"


def test_both_roles_are_controlled_in_the_joint_arm(interventions):
    """A control at one role would leave the second perturbation uncontrolled."""
    joint = next(a for a in interventions if a["id"] == "planner_critic")
    assert joint["control_spec"]["matched_at_every_targeted_role"] is True
    assert joint["control"] == "matched_random_vectors_at_both_roles"


def test_vector_arms_report_intervention_minus_matched_control(interventions):
    for arm in interventions:
        if not arm["applies_vector"]:
            continue
        assert arm["reporting"]["reported_effect"] == "intervention_minus_matched_control", (
            f"{arm['id']}: the effect must be measured against its matched control"
        )


def test_baseline_is_not_the_effect_comparator(interventions):
    """CLAUDE.md §4: never intervention minus no-intervention."""
    baseline = next(a for a in interventions if a["id"] == "baseline")
    assert baseline["applies_vector"] is False
    assert baseline["control"] == "none"
    assert baseline["usable_as_effect_comparator"] is False
    # An arm with no vector must not assert a matched random vector it lacks.
    assert baseline["control_spec"]["random_vector_matched_norm"] is None
    assert baseline["control_spec"]["random_vector_matched_shape"] is None
    assert baseline["control_spec"]["not_applicable_reason"]
    assert baseline["reporting"]["reported_effect"] is None
    # And no arm may be used as a comparator in place of a matched control.
    assert all(a["usable_as_effect_comparator"] is False for a in interventions)


def test_baseline_records_no_intervention_id(interventions):
    """Otherwise TrialResult raises: an intervention_id demands a control_id."""
    from multilingual_safety.agents.base import TrialResult

    baseline = next(a for a in interventions if a["id"] == "baseline")
    assert baseline["records_intervention_id"] is False
    # The config's promise is only meaningful because the runtime enforces it.
    assert "intervention_id" in TrialResult.__dataclass_fields__
    assert "control_id" in TrialResult.__dataclass_fields__


def test_declared_arms_are_recordable_with_their_declared_controls(interventions):
    """Each vector arm's (id, control) pair must satisfy the runtime guard.

    The guard's rejection path is covered in tests/test_agents_contract.py; what
    is checked here is that the pairs THIS config declares are accepted.
    """
    from multilingual_safety.agents.base import StageOutput, TrialResult
    from multilingual_safety.decisions import DecisionCategory, StageDecision

    stages = [
        StageOutput(decision=StageDecision(stage="critic", category=DecisionCategory.SEMANTIC_VETO))
    ]
    for arm in interventions:
        if not arm["applies_vector"]:
            continue
        trial = TrialResult(
            run_id="run_20260817T120000Z_abcdef12",
            record_id="rec_yo_0123456789ab",
            behavior_id="beh_0123456789ab",
            arm="harmful",
            split="dev",
            prompt_language="yo",
            trial_index=0,
            seed=20260817,
            stages=stages,
            intervention_id=arm["qualified_id"],
            control_id=arm["control"],
        )
        assert trial.control_id == arm["control"]
        assert trial.split == "dev", "selection runs on dev"


def test_every_arm_retains_negative_and_unstable_results(interventions):
    for arm in interventions:
        assert arm["reporting"]["retain_negative_results"] is True, f"{arm['id']}"
        assert arm["reporting"]["retain_unstable_results"] is True, f"{arm['id']}"


def test_policy_matches_the_per_arm_declarations(prereg, interventions):
    """The global policy and every arm must say the same thing."""
    policy = prereg["conditions"]["intervention_policy"]
    assert policy["vector_construction_selected_on"] == "dev"
    assert policy["layer_selection_selected_on"] == "dev"
    assert policy["strength_grid_selected_on"] == "dev"
    assert policy["strength_selected_on"] == "dev"
    assert policy["control_matched_norm"] is True
    assert policy["control_matched_shape"] is True
    assert policy["final_test_accessible_during_selection"] is False
    assert policy["retain_negative_results"] is True
    assert policy["retain_unstable_results"] is True
    for arm in interventions:
        assert arm["selection"]["final_test_accessible_during_selection"] == (
            policy["final_test_accessible_during_selection"]
        )


def test_no_model_has_been_run_for_these_arms(prereg):
    """Declaration only: these are commitments, not results."""
    policy = prereg["conditions"]["intervention_policy"]
    assert policy["models_run"] is False
    assert policy["arms_executed"] == 0
    assert policy["vectors_constructed"] == 0
    assert policy["layers_chosen"] == 0
    assert policy["strength_grids_swept"] == 0
    # Declaring arms -- and freezing them -- must not open the holdout.
    assert prereg["holdout_policy"]["final_test_readable"] is False
    assert prereg["holdout_policy"]["final_test_reads"] == []


# --- decoding and seeds -----------------------------------------------------
def test_decoding_is_deterministic_with_one_length_control(prereg):
    decoding = prereg["decoding"]
    assert decoding["temperature"] == 0.0
    assert decoding["do_sample"] is False
    assert decoding["length_control"] == "max_new_tokens"
    assert "max_length" not in decoding, "two length controls make the limit ambiguous"
    assert decoding["max_retries"] == 0


def test_seed_policy(prereg):
    policy = prereg["seed_policy"]
    assert len(policy["seeds"]) == policy["trials_per_item"] == 3
    assert len(set(policy["seeds"])) == 3
    assert policy["passed_to_generate"] is False


# --- metrics and failure labels ---------------------------------------------
def test_metrics_use_the_attempted_denominator_and_paired_arms(prereg):
    metrics = prereg["metrics"]
    assert metrics["denominator"] == "attempted"
    assert metrics["paired_arms_required"] is True
    primary = {m["id"] for m in metrics["primary"]}
    assert {"harmful_veto_rate", "benign_false_veto_rate"} <= primary


def test_failure_labels_are_the_six_categories(prereg):
    labels = prereg["failure_labels"]
    assert set(labels["decision_categories"]) == {c.value for c in DECISION_CATEGORIES}
    assert set(labels["apparatus_failure"]) == {c.value for c in FAILURE_CATEGORIES}
    assert set(labels["semantic"]) == {"SEMANTIC_ALLOW", "SEMANTIC_VETO"}


def test_no_silent_coercion_is_permitted(prereg):
    policy = prereg["failure_labels"]["policy"]
    assert policy["coerce_missing_to_veto"] is False
    assert policy["coerce_malformed_to_allow"] is False
    assert policy["drop_unparseable_rows"] is False


def test_critic_failure_vocabulary_matches_the_schema(prereg):
    from multilingual_safety.agents.schemas import CRITIC_SCHEMA

    declared = prereg["failure_labels"]["critic_failure_types"]
    schema_values = CRITIC_SCHEMA["properties"]["failure_type"]["enum"]
    assert set(declared) == set(schema_values)


# --- honesty about limitations ----------------------------------------------
def test_limitations_declared_in_advance(prereg):
    limitations = " ".join(prereg["declared_limitations"]).lower()
    assert "unvalidated" in limitations
    assert "mock" in limitations
    assert "outlier" in limitations
    # The tokenizer/context asymmetry between the two families is a confound.
    assert "tokenis" in limitations or "tokeniz" in limitations
    assert "context window" in limitations
    assert prereg["dataset"]["review_state_at_preregistration"]["rows_human_reviewed"] == 0


# --- the translation validation statement -----------------------------------
def test_machine_translation_revision_is_pinned_and_matches_the_translation_config(prereg):
    machine = prereg["dataset"]["translation_validation"]["machine_translation"]
    revision = machine["model_revision"]
    assert machine["revision_pinned"] is True
    assert machine["revision_type"] == "hf_model_commit_sha"
    assert len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)
    assert revision not in {"main", "master", "HEAD"}, "a moving ref is not a pin"
    # The pin must agree with the config that actually ran the translation.
    translation = yaml.safe_load(TRANSLATION_CONFIG.read_text(encoding="utf-8"))
    assert translation["nllb_local_settings"]["revision"] == revision
    assert prereg["dataset"]["translation"]["model_revision"] == revision


def test_manual_review_is_complete_but_not_captured_row_by_row(prereg):
    review = prereg["dataset"]["translation_validation"]["manual_review"]
    assert review["rows_reviewed_by_project_researcher"] == 1200
    assert review["rows_requiring_review"] == prereg["dataset"]["n_translated_rows"] == 1200
    assert review["complete"] is True
    # The assertion and the evidence are different things, and stay different.
    assert review["captured_row_by_row"] is False
    assert review["recorded_in_structured_audit"] is False


def test_external_native_speaker_checking_is_recorded_without_row_attribution(prereg):
    external = prereg["dataset"]["translation_validation"]["external_native_speaker_checking"]
    assert external["obtained"] is True
    assert external["row_level_reviewer_identities_recorded"] is False
    assert external["row_level_language_assignments_recorded"] is False
    assert external["represented_in_structured_repository_audit"] is False


def test_row_level_native_speaker_validation_is_never_claimed(prereg):
    """The load-bearing negative claim. Guarded recursively, not by one field."""
    claims = prereg["dataset"]["translation_validation"]["claims_permitted"]
    assert claims["native_speaker_signoff"] is False
    assert claims["row_level_native_speaker_validation"] is False
    assert claims["corpus_level_native_speaker_checking_performed"] is True
    assert "must NOT claim row-level native-speaker validation" in claims["rationale"]

    forbidden_keys = {"native_speaker_signoff", "row_level_native_speaker_validation"}

    def walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in forbidden_keys:
                    assert value is False, f"{path}.{key} claims native-speaker sign-off"
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]")

    walk(prereg)


def test_forbidden_phrasings_appear_only_as_the_forbidden_list(prereg):
    """A phrase banned in the write-up must not be asserted elsewhere in the file."""
    claims = prereg["dataset"]["translation_validation"]["claims_permitted"]
    forbidden = claims["forbidden_phrasing"]
    assert forbidden, "the ban list must not be empty"
    blob = PREREG_PATH.read_text(encoding="utf-8")
    for phrase in forbidden:
        # Once as the list entry itself; any second occurrence is a claim.
        assert blob.count(phrase) == 1, f"{phrase!r} is asserted outside the ban list"


def test_translation_quality_remains_a_declared_limitation(prereg):
    residual = prereg["dataset"]["translation_validation"]["residual_limitation"]
    assert residual["translation_quality_remains_a_limitation"] is True
    # The named languages must be exactly the low-resource tier.
    assert set(residual["weakest_languages"]) == set(prereg["languages"]["resource_tiers"]["low"])
    assert set(residual["weakest_languages"]) == {"ha", "yo", "sw"}


def test_yoruba_final_test_length_outlier_is_recorded_as_valid(prereg):
    outliers = prereg["dataset"]["translation_validation"]["length_outliers"]
    assert outliers["n_yoruba_length_outliers"] == 3
    recorded = outliers["recorded"]
    assert len(recorded) == 1
    entry = recorded[0]
    assert entry["behavior_id"] == "beh_b117279f5262"
    assert entry["language"] == "yo"
    assert entry["split"] == FINAL_TEST
    assert entry["decision"] == "ACCEPT"
    assert entry["outlier_label"] == "valid_translation_length_outlier"
    assert entry["review_provenance"] == "external_native_check_unstructured"
    assert entry["structured_reviewer_identity_captured"] is False
    assert entry["clears_review_required"] is False


def test_outlier_entry_agrees_with_its_report(prereg):
    """The pre-registration must not drift from the artifact it cites."""
    entry = prereg["dataset"]["translation_validation"]["length_outliers"]["recorded"][0]
    report_path = PROCESSED_DIR.parent.parent / entry["recorded_in"]
    if not report_path.exists():
        pytest.skip("length_outlier_review.json not generated in this working tree")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for field in (
        "behavior_id",
        "language",
        "split",
        "decision",
        "outlier_label",
        "character_length_bucket",
        "review_provenance",
        "clears_review_required",
        "structured_reviewer_identity_captured",
    ):
        assert entry[field] == report[field], f"{field} disagrees with {entry['recorded_in']}"


def test_structured_audit_counts_stay_separate_from_the_assertions(prereg):
    """Manual review did not silently promote any row's recorded status."""
    state = prereg["dataset"]["review_state_at_preregistration"]
    assert state["rows_human_reviewed"] == 0
    assert state["rows_with_a_structured_review_record"] == 18
    assert state["rows_without_a_structured_review_record"] == 1182
    assert (
        state["rows_with_a_structured_review_record"]
        + state["rows_without_a_structured_review_record"]
        == state["rows_requiring_review"]
        == 1200
    )
    # Freezing the study did not promote any row's recorded review status.
    claims = prereg["dataset"]["translation_validation"]["claims_permitted"]
    assert claims["native_speaker_signoff"] is False
    assert claims["row_level_native_speaker_validation"] is False


def test_config_contains_no_prompt_text():
    """A pre-registration is tracked; it must carry no corpus text."""
    blob = PREREG_PATH.read_text(encoding="utf-8")
    frozen = PROCESSED_DIR / "frozen_dataset_v1.jsonl"
    if not frozen.exists():
        pytest.skip("frozen dataset not present")
    texts = []
    for line in frozen.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            texts.append(row["source_text"])
            if row.get("translated_text"):
                texts.append(row["translated_text"])
    leaked = [t for t in texts if t and t in blob]
    assert not leaked, f"{len(leaked)} corpus string(s) leaked into the pre-registration"
