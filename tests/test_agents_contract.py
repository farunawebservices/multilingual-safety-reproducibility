"""Pipeline contracts: no inference yet, and no intervention without a control."""

from __future__ import annotations

import pytest

from multilingual_safety.agents import Critic, Planner, ToolCaller
from multilingual_safety.agents.base import CriticInput, PlannerInput, StageOutput, TrialResult
from multilingual_safety.decisions import DecisionCategory, StageDecision


def test_no_model_inference_is_wired_up():
    """Backends are contracts only at this stage of the project."""
    with pytest.raises(NotImplementedError):
        Planner().run(
            PlannerInput(
                record_id="rec_en_0123456789ab",
                behavior_id="beh_0123456789ab",
                prompt_language="en",
                stage_language="en",
            )
        )
    with pytest.raises(NotImplementedError):
        Critic().run(
            CriticInput(
                record_id="rec_yo_0123456789ab",
                behavior_id="beh_0123456789ab",
                prompt_language="yo",
                stage_language="en",
            )
        )


def test_tool_surface_is_mock_only():
    assert ToolCaller.allow_real_execution is False


def test_prompt_language_and_stage_language_are_independent():
    """RQ3 needs a Yoruba prompt with an English critic to be expressible."""
    stage_input = CriticInput(
        record_id="rec_yo_0123456789ab",
        behavior_id="beh_0123456789ab",
        prompt_language="yo",
        stage_language="en",
    )
    assert stage_input.prompt_language != stage_input.stage_language


def _output(stage: str, category: DecisionCategory) -> StageOutput:
    return StageOutput(decision=StageDecision(stage=stage, category=category))


def test_trial_records_a_per_stage_decision_vector():
    trial = TrialResult(
        run_id="run_20260816T120000Z_abcdef12",
        record_id="rec_ha_0123456789ab",
        behavior_id="beh_0123456789ab",
        arm="harmful",
        split="dev",
        prompt_language="ha",
        trial_index=0,
        seed=1,
        stages=[
            _output("planner", DecisionCategory.SEMANTIC_ALLOW),
            _output("tool_caller", DecisionCategory.MALFORMED_DECISION),
            _output("critic", DecisionCategory.SEMANTIC_VETO),
        ],
    )
    categories = trial.categories()
    assert categories["planner"] is DecisionCategory.SEMANTIC_ALLOW
    assert categories["tool_caller"] is DecisionCategory.MALFORMED_DECISION
    assert categories["critic"] is DecisionCategory.SEMANTIC_VETO


def test_intervention_without_matched_control_is_rejected():
    with pytest.raises(ValueError, match="matched control"):
        TrialResult(
            run_id="run_20260816T120000Z_abcdef12",
            record_id="rec_ha_0123456789ab",
            behavior_id="beh_0123456789ab",
            arm="harmful",
            split="dev",
            prompt_language="ha",
            trial_index=0,
            seed=1,
            stages=[_output("critic", DecisionCategory.SEMANTIC_VETO)],
            intervention_id="steer_refusal_v1",
            control_id=None,
        )


def test_intervention_with_matched_control_is_accepted():
    trial = TrialResult(
        run_id="run_20260816T120000Z_abcdef12",
        record_id="rec_ha_0123456789ab",
        behavior_id="beh_0123456789ab",
        arm="harmful",
        split="dev",
        prompt_language="ha",
        trial_index=0,
        seed=1,
        stages=[_output("critic", DecisionCategory.SEMANTIC_VETO)],
        intervention_id="steer_refusal_v1",
        control_id="random_vector_matched_norm_seed1",
    )
    assert trial.control_id


def test_trial_with_no_stages_is_degenerate():
    with pytest.raises(ValueError, match="degenerate"):
        TrialResult(
            run_id="run_20260816T120000Z_abcdef12",
            record_id="rec_en_0123456789ab",
            behavior_id="beh_0123456789ab",
            arm="benign",
            split="dev",
            prompt_language="en",
            trial_index=0,
            seed=1,
            stages=[],
        )
