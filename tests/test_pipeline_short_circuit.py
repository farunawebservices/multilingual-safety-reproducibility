"""Test stage outcomes; real short-circuit logic lives in pipeline.run_trial()."""

from __future__ import annotations

from typing import Any, Mapping

from multilingual_safety.agents.base import StageInput, StageOutput
from multilingual_safety.agents.stage_runner import StageOutcome
from tests.fake_backend import FakeBackend


class FakePrompt:
    def __init__(self, stage_name: str, row: Mapping[str, Any]) -> None:
        self.prompt_id = f"{stage_name}_prompt"
        self.messages = [{"role": "user", "content": row.get("translated_text", "")}]
        self.prompt_hash = "fake_hash"
        self.tool_results_sha256 = None
        self.template_id = "fake_template"
        self.template_sha256 = "fake_sha"
        self.orthography_fingerprint = "fake_ortho"

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "prompt_hash": self.prompt_hash,
            "template_id": self.template_id,
            "template_sha256": self.template_sha256,
            "orthography_fingerprint": self.orthography_fingerprint,
        }


class FakePromptSet:
    def build(self, stage_name: str, row: Mapping[str, Any], **kwargs: Any) -> FakePrompt:
        return FakePrompt(stage_name, row)


class FakePlannerStage:
    stage_name = "planner"
    config = {}

    def __init__(self) -> None:
        self.prompts = FakePromptSet()
        self._backend = FakeBackend(
            canned_text='{"decision": "VETO", "reason_code": "UNSAFE"}',
        )

    def require_backend(self):
        return self._backend

    def run(self, stage_input: StageInput) -> StageOutput:
        return StageOutput(
            decision=None,
            output_ref="planner_prompt",
        )


class FakeToolCallerStage:
    stage_name = "tool_caller"
    config = {}

    def __init__(self) -> None:
        self.prompts = FakePromptSet()
        self._backend = FakeBackend(canned_text='{}')

    def require_backend(self):
        return self._backend

    def run(self, stage_input: StageInput) -> StageOutput:
        return StageOutput(
            decision=None,
            output_ref="tool_caller_prompt",
        )


class FakeCriticStage:
    stage_name = "critic"
    config = {}

    def __init__(self) -> None:
        self.prompts = FakePromptSet()
        self._backend = FakeBackend(canned_text='{}')

    def require_backend(self):
        return self._backend

    def run(self, stage_input: StageInput) -> StageOutput:
        return StageOutput(
            decision=None,
            output_ref="critic_prompt",
        )


def test_all_stages_run_in_isolation() -> None:
    """
    When stages are executed in isolation (no pipeline short-circuit logic),
    all stages return stage_status='RAN'. Real short-circuiting happens in
    pipeline.run_trial(), not in execute_stage().
    """
    planner = FakePlannerStage()
    tool_caller = FakeToolCallerStage()
    critic = FakeCriticStage()

    stage_input = StageInput(
        record_id="r1",
        behavior_id="b1",
        prompt_language="en",
        stage_language="en",
        trial_index=0,
        seed=0,
        row={"translated_text": "test input"},
        prompt_hash="abc123",
        condition_id="c1",
    )

    from multilingual_safety.agents.stage_runner import execute_stage

    planner_outcome = execute_stage(
        planner,
        stage_input,
        run_id="test-run",
        max_new_tokens=64,
        timeout_seconds=5.0,
    )
    tool_outcome = execute_stage(
        tool_caller,
        stage_input,
        run_id="test-run",
        max_new_tokens=64,
        timeout_seconds=5.0,
    )
    critic_outcome = execute_stage(
        critic,
        stage_input,
        run_id="test-run",
        max_new_tokens=64,
        timeout_seconds=5.0,
    )

    # All stages ran in isolation
    assert planner_outcome.output.stage_status == "RAN"
    assert tool_outcome.output.stage_status == "RAN"
    assert critic_outcome.output.stage_status == "RAN"

    # Planner decision is parsed (even if VETO)
    assert planner_outcome.output.decision is not None
    assert planner_outcome.output.decision.stage == "planner"

    print("test_all_stages_run_in_isolation PASSED")


if __name__ == "__main__":
    test_all_stages_run_in_isolation()
