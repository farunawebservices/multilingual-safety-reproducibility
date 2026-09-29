"""Test stage_runner.execute_stage() with a fake backend."""

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
    def build(
        self,
        stage_name: str,
        row: Mapping[str, Any],
        **kwargs: Any,
    ) -> FakePrompt:
        return FakePrompt(stage_name, row)


class FakePlannerStage:
    stage_name = "planner"
    config = {}

    def __init__(self) -> None:
        self.prompts = FakePromptSet()
        self._backend = FakeBackend(
            canned_text='{"decision": "ALLOW", "reason_code": "SAFE"}',
        )

    def require_backend(self):
        return self._backend

    def run(self, stage_input: StageInput) -> StageOutput:
        backend = self.require_backend()
        # Minimal stub: just return a canned outcome
        return StageOutput(
            decision=None,
            output_ref="fake-ref",
            stage="planner",
            stage_status="RAN",
        )


def test_execute_stage_fake_planner() -> None:
    stage = FakePlannerStage()
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

    outcome = execute_stage(
        stage,
        stage_input,
        run_id="test-run",
        max_new_tokens=64,
        timeout_seconds=5.0,
    )

    assert isinstance(outcome, StageOutcome)
    # The decision category should be set (even if MISSING_DECISION for bad JSON)
    assert outcome.output.decision is not None
    assert outcome.output.decision.stage == "planner"
    # output_ref should be the prompt_id
    assert outcome.output.output_ref == "planner_prompt"
    print("test_execute_stage_fake_planner PASSED")


if __name__ == "__main__":
    test_execute_stage_fake_planner()
