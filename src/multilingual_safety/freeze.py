"""The freeze gate: what must hold before a pre-registration may be frozen.

Freezing is the moment the study's commitments become binding and ``final_test``
becomes reachable (with a logged read event). A pre-registration frozen while any
of these conditions is false would lend authority to claims the artifacts do not
support, so the conditions are checked here rather than trusted to a checklist.

The gate is deliberately a pure function over already-loaded data: it opens no
files of its own, reads no corpus text, and returns a list of human-readable
failures rather than raising, so a caller can report all of them at once.

``check_freeze_preconditions`` is used in two places, and both matter:

* ``scripts/18_freeze_preregistration.py`` runs it BEFORE writing ``frozen: true``;
* ``tests/test_freeze_gate.py`` runs it against the committed config, and also
  mutates each precondition to prove the gate actually catches a violation.
"""

from __future__ import annotations

import re
from typing import Any

#: Model revisions the study is pinned to. A revision drifting between
#: pre-registration and run would change the system under test silently.
EXPECTED_MODEL_REVISIONS: dict[str, str] = {
    "llama-3.3-70b-instruct": "6f6073b423013f6a7d4d9f39144961bfbfbc386b",
    "qwen-2.5-72b-instruct": "495f39366efef23836d0cfae4fbe635880d2be31",
}

#: The Yoruba length outlier whose metadata must be recorded before freezing.
EXPECTED_OUTLIER_BEHAVIOR = "beh_b117279f5262"

#: Substrings that must appear in ``declared_limitations`` for the translation
#: limitation to count as explicit.
REQUIRED_LIMITATION_TERMS: tuple[str, ...] = ("translation quality", "hausa", "yoruba", "swahili")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: Stable ids for each precondition, so a report can name what failed.
FREEZE_PRECONDITIONS: tuple[str, ...] = (
    "model_revisions_pinned",
    "split_hash_verified",
    "holdout_closed",
    "interventions_have_matched_controls",
    "translation_limitations_explicit",
    "yoruba_outlier_metadata_recorded",
    "no_final_test_text_in_tracked_artifacts",
    "pre_freeze_deviations_disclosed",
)


def _check_model_revisions(prereg: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    models = {m["id"]: m for m in prereg["models"]["under_evaluation"]}
    for model_id, expected in EXPECTED_MODEL_REVISIONS.items():
        model = models.get(model_id)
        if model is None:
            failures.append(f"model_revisions_pinned: {model_id} is not under evaluation")
            continue
        actual = model.get("revision")
        if actual != expected:
            failures.append(
                f"model_revisions_pinned: {model_id} revision is {actual!r}, expected {expected!r}"
            )
        if model.get("revision_type") != "hf_model_commit_sha":
            failures.append(f"model_revisions_pinned: {model_id} revision is not a commit sha")
    return failures


def _check_split_hash(prereg: dict[str, Any], splits: dict[str, Any] | None) -> list[str]:
    failures: list[str] = []
    declared = prereg["splits"]["split_hash"]
    if not _SHA256_RE.match(declared or ""):
        failures.append(
            f"split_hash_verified: declared split_hash is not a 64-char lowercase sha256: "
            f"{declared!r} (length {len(declared or '')})"
        )
    if splits is not None and declared != splits.get("split_hash"):
        failures.append(
            "split_hash_verified: declared split_hash does not match the frozen split file"
        )
    provenance = prereg["splits"].get("hash_provenance") or {}
    if provenance.get("verified_actual") != declared:
        failures.append(
            "split_hash_verified: hash_provenance.verified_actual disagrees with split_hash"
        )
    # A known-mistyped value must never become the declared hash.
    if provenance.get("supplied_in_request") == declared:
        failures.append("split_hash_verified: the mistyped supplied hash is being used as verified")
    return failures


def _check_holdout_closed(prereg: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    policy = prereg["holdout_policy"]
    if policy.get("final_test_readable") is not False:
        failures.append("holdout_closed: final_test_readable must be false at freeze time")
    if policy.get("final_test_reads") != []:
        failures.append(
            f"holdout_closed: final_test_reads must be empty, got {policy.get('final_test_reads')!r}"
        )
    return failures


def _check_interventions(prereg: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    conditions = prereg["conditions"]
    arms = conditions.get("interventions") or []
    if not arms:
        failures.append("interventions_have_matched_controls: no intervention arm is declared")
    policy = conditions["intervention_policy"]
    if policy.get("require_matched_control") is not True:
        failures.append("interventions_have_matched_controls: require_matched_control is not true")

    for arm in arms:
        arm_id = arm.get("id", "<unnamed>")
        selection = arm.get("selection") or {}
        for key in (
            "vector_construction_selected_on",
            "layer_selection_selected_on",
            "strength_grid_selected_on",
        ):
            if selection.get(key) != "dev":
                failures.append(
                    f"interventions_have_matched_controls: {arm_id}.{key} is "
                    f"{selection.get(key)!r}, must be 'dev'"
                )
        if selection.get("final_test_accessible_during_selection") is not False:
            failures.append(
                f"interventions_have_matched_controls: {arm_id} allows final_test during selection"
            )
        reporting = arm.get("reporting") or {}
        for key in ("retain_negative_results", "retain_unstable_results"):
            if reporting.get(key) is not True:
                failures.append(f"interventions_have_matched_controls: {arm_id}.{key} is not true")

        if not arm.get("applies_vector"):
            # A no-vector arm must not be dressed up as a control or a comparator.
            if arm.get("usable_as_effect_comparator") is not False:
                failures.append(
                    f"interventions_have_matched_controls: {arm_id} applies no vector but is "
                    "marked usable as an effect comparator (CLAUDE.md §4)"
                )
            continue

        control_spec = arm.get("control_spec") or {}
        if arm.get("control") in (None, "none", ""):
            failures.append(
                f"interventions_have_matched_controls: {arm_id} applies a vector with no control"
            )
        for key in (
            "random_vector_matched_norm",
            "random_vector_matched_shape",
            "applied_at_matched_layer",
            "applied_at_matched_token_positions",
            "same_items",
            "same_seeds",
            "same_session",
        ):
            if control_spec.get(key) is not True:
                failures.append(
                    f"interventions_have_matched_controls: {arm_id}.control_spec.{key} is "
                    f"{control_spec.get(key)!r}, must be true"
                )
        if reporting.get("reported_effect") != "intervention_minus_matched_control":
            failures.append(
                f"interventions_have_matched_controls: {arm_id} reports "
                f"{reporting.get('reported_effect')!r}, must be intervention_minus_matched_control"
            )
        # A joint arm needs a control at every role it perturbs.
        if "_and_" in str(arm.get("target_role")) and (
            control_spec.get("matched_at_every_targeted_role") is not True
        ):
            failures.append(
                f"interventions_have_matched_controls: {arm_id} targets multiple roles without a "
                "control at every targeted role"
            )
    return failures


def _check_translation_limitations(prereg: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    validation = (prereg["dataset"].get("translation_validation") or {})
    if not validation:
        failures.append("translation_limitations_explicit: translation_validation block is missing")
        return failures

    residual = validation.get("residual_limitation") or {}
    if residual.get("translation_quality_remains_a_limitation") is not True:
        failures.append(
            "translation_limitations_explicit: translation quality is not declared a limitation"
        )
    if set(residual.get("weakest_languages") or []) != {"ha", "yo", "sw"}:
        failures.append(
            "translation_limitations_explicit: weakest_languages must be exactly ha, yo, sw"
        )

    claims = validation.get("claims_permitted") or {}
    for key in ("native_speaker_signoff", "row_level_native_speaker_validation"):
        if claims.get(key) is not False:
            failures.append(
                f"translation_limitations_explicit: claims_permitted.{key} must be false"
            )

    limitations = " ".join(prereg.get("declared_limitations") or []).lower()
    for term in REQUIRED_LIMITATION_TERMS:
        if term not in limitations:
            failures.append(
                f"translation_limitations_explicit: declared_limitations never mentions {term!r}"
            )
    return failures


def _check_outlier_metadata(
    prereg: dict[str, Any], outlier_report: dict[str, Any] | None
) -> list[str]:
    failures: list[str] = []
    validation = prereg["dataset"].get("translation_validation") or {}
    recorded = ((validation.get("length_outliers") or {}).get("recorded")) or []
    entry = next((e for e in recorded if e.get("behavior_id") == EXPECTED_OUTLIER_BEHAVIOR), None)
    if entry is None:
        failures.append(
            f"yoruba_outlier_metadata_recorded: {EXPECTED_OUTLIER_BEHAVIOR} is not recorded"
        )
        return failures

    if entry.get("language") != "yo":
        failures.append("yoruba_outlier_metadata_recorded: recorded language is not yo")
    if entry.get("outlier_label") != "valid_translation_length_outlier":
        failures.append("yoruba_outlier_metadata_recorded: outlier_label is wrong")
    if entry.get("clears_review_required") is not False:
        failures.append(
            "yoruba_outlier_metadata_recorded: the outlier record must not clear review_required"
        )
    if not entry.get("recorded_in"):
        failures.append("yoruba_outlier_metadata_recorded: no report path is cited")

    if outlier_report is not None:
        for field in ("behavior_id", "language", "split", "decision", "outlier_label"):
            if entry.get(field) != outlier_report.get(field):
                failures.append(
                    f"yoruba_outlier_metadata_recorded: {field} disagrees with the cited report"
                )
    return failures


def _check_no_final_test_text(
    prereg: dict[str, Any], outlier_report: dict[str, Any] | None
) -> list[str]:
    """No final_test text in tracked artifacts, and no model has seen the holdout.

    Note what this does NOT assert: that no human ever looked at a final_test
    row. That is what ``pre_freeze_deviations`` is for -- see
    :func:`_check_deviations_disclosed`.
    """
    failures: list[str] = []
    if prereg["conditions"]["intervention_policy"].get("models_run") is not False:
        failures.append("no_final_test_text_in_tracked_artifacts: models_run is not false")
    if outlier_report is not None:
        if outlier_report.get("contains_prompt_text") is not False:
            failures.append(
                "no_final_test_text_in_tracked_artifacts: the outlier report carries prompt text"
            )
        if outlier_report.get("final_test_text_loaded_into_report") is not False:
            failures.append(
                "no_final_test_text_in_tracked_artifacts: the outlier report loaded final_test text"
            )
    return failures


def _check_deviations_disclosed(
    prereg: dict[str, Any], outlier_report: dict[str, Any] | None
) -> list[str]:
    """Any pre-freeze departure from the holdout rules must be declared.

    The gate refuses a freeze that hides one. Concretely: a recorded outlier on a
    ``final_test`` row whose provenance is a human/external inspection means that
    row's text WAS inspected before freezing, which the holdout rules forbid. The
    freeze may still proceed -- but only with the departure written down.
    """
    failures: list[str] = []
    policy = prereg["holdout_policy"]
    deviations = policy.get("pre_freeze_deviations")
    if deviations is None:
        failures.append(
            "pre_freeze_deviations_disclosed: holdout_policy.pre_freeze_deviations is missing "
            "(use an empty list to assert there were none)"
        )
        return failures
    if not isinstance(deviations, list):
        failures.append("pre_freeze_deviations_disclosed: pre_freeze_deviations must be a list")
        return failures

    for i, deviation in enumerate(deviations):
        for key in ("rule", "behavior_ids", "scope", "what_was_not_done", "rationale"):
            if not deviation.get(key):
                failures.append(f"pre_freeze_deviations_disclosed: deviation[{i}] has no {key}")

    disclosed_ids = {
        behavior_id
        for deviation in deviations
        for behavior_id in (deviation.get("behavior_ids") or [])
    }

    # Cross-check against the artifact: an inspected final_test row must appear.
    if outlier_report is not None and outlier_report.get("split") == "final_test":
        inspected = (outlier_report.get("asserted_facts") or {}).get(
            "translation_manually_inspected"
        ) or (outlier_report.get("asserted_facts") or {}).get(
            "external_native_speaker_check_performed"
        )
        behavior_id = outlier_report.get("behavior_id")
        if inspected and behavior_id not in disclosed_ids:
            failures.append(
                f"pre_freeze_deviations_disclosed: {behavior_id} is a final_test row that was "
                "inspected before freezing, but no deviation discloses it"
            )
    return failures


def check_freeze_preconditions(
    prereg: dict[str, Any],
    splits: dict[str, Any] | None = None,
    outlier_report: dict[str, Any] | None = None,
) -> list[str]:
    """Every reason this pre-registration may not be frozen. Empty means clear.

    Parameters
    ----------
    prereg:
        The parsed pre-registration config.
    splits:
        The parsed frozen split file, when available, so the declared
        ``split_hash`` can be checked against it rather than against itself.
    outlier_report:
        The parsed ``results/length_outlier_review.json``, when available, so the
        config's outlier claim can be checked against the artifact it cites.
    """
    return [
        *_check_model_revisions(prereg),
        *_check_split_hash(prereg, splits),
        *_check_holdout_closed(prereg),
        *_check_interventions(prereg),
        *_check_translation_limitations(prereg),
        *_check_outlier_metadata(prereg, outlier_report),
        *_check_no_final_test_text(prereg, outlier_report),
        *_check_deviations_disclosed(prereg, outlier_report),
    ]


def freeze_is_consistent(prereg: dict[str, Any]) -> list[str]:
    """``frozen`` and its provenance fields must agree with each other."""
    failures: list[str] = []
    frozen = prereg.get("frozen")
    frozen_at = prereg.get("frozen_at")
    if frozen is True and not frozen_at:
        failures.append("frozen is true but frozen_at is not set")
    if frozen is not True and frozen_at:
        failures.append("frozen_at is set but frozen is not true")
    if frozen_at is not None and not isinstance(frozen_at, str):
        failures.append(f"frozen_at must be an ISO-8601 string, got {type(frozen_at).__name__}")
    if isinstance(frozen_at, str) and not frozen_at.endswith("+00:00"):
        failures.append(f"frozen_at must be UTC (+00:00), got {frozen_at!r}")
    return failures
