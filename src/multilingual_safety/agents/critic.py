"""Critic stage: review plan + proposed tool calls, emit ALLOW or VETO.

Not implemented: no model is run at this stage of the project.

``stage_language`` is independent of ``prompt_language``. Sweeping the Critic's
language over ``{user, en}`` while holding the user prompt language fixed is how
RQ3 (cross-lingual transfer of the Critic) is tested.
"""

from __future__ import annotations

from multilingual_safety.agents.base import AgentStage, CriticInput, StageOutput


class Critic(AgentStage):
    """Emits a decision on the proposed plan and tool calls.

    The classification of the Critic's output is semantic, not keyword-based: a
    refusal expressed in fluent Yoruba with no English marker is
    ``SEMANTIC_VETO``, and a response containing the literal string ``ALLOW``
    inside a refusal is not an allow (CLAUDE.md §3). Keyword matching would
    systematically misread exactly the languages under study.
    """

    stage_name = "critic"

    def run(self, stage_input: CriticInput) -> StageOutput:
        raise NotImplementedError(
            "Critic backend is not implemented. No model inference is run at this "
            "stage of the project (README: Status)."
        )
