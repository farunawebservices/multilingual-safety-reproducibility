"""Tool-Caller stage: plan steps -> tool invocations against a MOCKED surface.

Not implemented: no model is run at this stage of the project.

Hard constraint: the tool surface is mocked and no real capability is wired in.
The object of study is the decision, not the effect. A backend that executes a
real tool call is out of scope for this repository (CLAUDE.md, conventions).
"""

from __future__ import annotations

from multilingual_safety.agents.base import AgentStage, StageOutput, ToolCallerInput


class ToolCaller(AgentStage):
    """Maps plan steps onto a mock tool registry. Executes nothing real."""

    stage_name = "tool_caller"

    #: Enforced invariant: real execution is never enabled.
    allow_real_execution: bool = False

    def run(self, stage_input: ToolCallerInput) -> StageOutput:
        raise NotImplementedError(
            "ToolCaller backend is not implemented. No model inference is run at this "
            "stage of the project, and the tool surface is mock-only."
        )
