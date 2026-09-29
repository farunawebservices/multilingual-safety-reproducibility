"""Planner stage: user request -> structured plan. May refuse.

Not implemented: no model is run at this stage of the project.
"""

from __future__ import annotations

from multilingual_safety.agents.base import AgentStage, PlannerInput, StageOutput


class Planner(AgentStage):
    """Turns a user request in language L into an explicit plan.

    A refusal by the Planner is ``SEMANTIC_VETO`` for this stage; downstream
    stages are then recorded as not-run rather than assigned a decision they
    never made.
    """

    stage_name = "planner"

    def run(self, stage_input: PlannerInput) -> StageOutput:
        raise NotImplementedError(
            "Planner backend is not implemented. No model inference is run at this "
            "stage of the project (README: Status)."
        )
