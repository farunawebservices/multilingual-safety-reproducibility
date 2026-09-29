"""Minimal end-to-end dry run with HfLocalBackend."""

import os
import sys
sys.path.insert(0, "/workspace/multilingual-multiagent-safety/src")
os.environ["HF_HOME"] = "/workspace/.hf_home"

import torch

from multilingual_safety.agents.backends.hf_local import HfLocalBackend
from multilingual_safety.agents.backends.base import GenerationRequest
from multilingual_safety.agents.base import StageInput, StageOutput
from multilingual_safety.agents.stage_runner import execute_stage


class SimplePromptSet:
    def build(self, stage_name: str, row: dict, **kwargs):
        class P:
            prompt_id = f"{stage_name}_prompt"
            messages = [{"role": "user", "content": row.get("translated_text", "")}]
            prompt_hash = "fake_hash"
            tool_results_sha256 = None
            template_id = "fake_template"
            template_sha256 = "fake_sha"
            orthography_fingerprint = "fake_ortho"
            def to_metadata_dict(self):
                return {"prompt_id": self.prompt_id, "prompt_hash": self.prompt_hash}
        return P()


class RealPlannerStage:
    stage_name = "planner"
    config = {"split": "test"}

    def __init__(self, backend: HfLocalBackend) -> None:
        self.prompts = SimplePromptSet()
        self._backend = backend

    def require_backend(self):
        return self._backend

    def run(self, stage_input: StageInput) -> StageOutput:
        backend = self.require_backend()
        request = GenerationRequest(
            prompt_id="smoke_planner",
            role="planner",
            messages=[{"role": "user", "content": "Classify as SAFE or UNSAFE: 'Hello'"}],
            max_new_tokens=32,
            seed=42,
            timeout_seconds=10.0,
            intervention=None,
        )
        result = backend.generate(request)
        return StageOutput(
            decision=None,
            output_ref="smoke_planner",
            stage="planner",
            stage_status="RAN",
        )


def main():
    model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
    revision = "fcc320f490e08fdb4b99d935b2c58d40bf35b0d0"

    print(f"Initializing backend for {model_id}...")
    backend = HfLocalBackend(
        model_id=model_id,
        revision=revision,
        dtype=torch.float32,
        device_map="auto",
        max_input_length=256,
        max_new_tokens=32,
    )

    print("Running preflight...")
    backend.preflight()

    print("Creating stage...")
    stage = RealPlannerStage(backend)

    stage_input = StageInput(
        record_id="smoke_001",
        behavior_id="test_behavior",
        prompt_language="en",
        stage_language="en",
        trial_index=0,
        seed=42,
        row={"translated_text": "Hello, how are you?"},
        prompt_hash="abc123",
        condition_id="c1",
    )

    print("Executing stage...")
    outcome = execute_stage(
        stage,
        stage_input,
        run_id="smoke_run",
        max_new_tokens=32,
        timeout_seconds=10.0,
    )

    print(f"✓ Stage completed: {outcome.output.stage_status}")
    print(f"  Decision: {outcome.output.decision.category if outcome.output.decision else 'None'}")
    print(f"  Tokens: in={outcome.output.tokens_in}, out={outcome.output.tokens_out}")
    print("\n✓ Smoke dry-run PASSED")


if __name__ == "__main__":
    main()
