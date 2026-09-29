"""Shared contracts for pipeline stages.

Two things this module deliberately encodes:

1. **Prompt language and stage language are separate fields.** RQ3 asks whether
   a Critic reasoning in English still vetoes an unsafe plan whose user prompt
   was Yoruba. Collapsing these into one "language" column makes that question
   unaskable.

2. **A trial's result is a vector of per-stage decisions, not a single bit.**
   RQ2 asks *where* the pipeline failed. Recording only the final outcome
   discards the answer.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from multilingual_safety.decisions import (
    DecisionCategory,
    StageDecision,
)


@dataclass(frozen=True, slots=True)
class StageInput:
    """Base input to any stage. Carries references, not corpus text.

    ``prompt_ref`` is a record ID. Text is loaded at run time from git-ignored
    local data and never written into tracked artifacts.
    """

    record_id: str
    behavior_id: str
    prompt_language: str
    stage_language: str
    trial_index: int = 0
    seed: int | None = None
    row: Mapping[str, Any] | None = None
    prompt_hash: str | None = None
    condition_id: str | None = None
    intervention: InterventionSpec | None = None


@dataclass(frozen=True, slots=True)
class PlannerInput(StageInput):
    """User request in language L. The Planner may refuse."""


@dataclass(frozen=True, slots=True)
class ToolCallerInput(StageInput):
    """A plan to be mapped onto the MOCKED tool surface."""

    plan_ref: str | None = None


@dataclass(frozen=True, slots=True)
class CriticInput(StageInput):
    """Plan plus proposed tool calls, for ALLOW/VETO review."""

    plan_ref: str | None = None
    tool_calls_ref: str | None = None


@dataclass(frozen=True, slots=True)
class StageOutput:
    """What a stage returns. The decision category is mandatory.

    There is no "unset" and no default: constructing a StageOutput forces the
    caller to have classified the outcome, including into TIMEOUT or
    NON_TERMINATION.
    """

    decision: StageDecision | None
    output_ref: str | None = None
    raw_output_path: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    stage: str | None = None
    stage_status: str | None = None
    not_run_reason: str | None = None
    finish_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def category(self) -> DecisionCategory:
        return self.decision.category


@dataclass(frozen=True, slots=True)
class StageOutcome:
    """Full internal result of one stage execution.

    `output` is the public stage decision, `record` is its text-free
    provenance record, and `prompt`/`raw_text` remain available to the run
    logger for the private logs store.
    """

    output: StageOutput
    record: Any
    prompt: Any
    raw_text: str
    parsed: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class TrialResult:
    """One behavior, one language, one trial: the full per-stage decision vector."""

    run_id: str
    record_id: str
    behavior_id: str
    arm: str
    split: str
    prompt_language: str
    trial_index: int
    seed: int | None
    stages: Sequence[StageOutput]
    intervention_id: str | None = None
    control_id: str | None = None

    def __post_init__(self) -> None:
        if self.arm not in {"harmful", "benign"}:
            raise ValueError(f"arm must be 'harmful' or 'benign', got {self.arm!r}")
        if not self.stages:
            raise ValueError("a trial with no stage outputs is degenerate (CLAUDE.md §10)")
        # An intervention arm without a matched control is not reportable
        # (CLAUDE.md §4). Catching it at record construction stops the missing
        # control from being discovered at analysis time, when the compute is
        # already spent.
        if self.intervention_id is not None and self.control_id is None:
            raise ValueError(
                f"intervention {self.intervention_id!r} recorded without a matched control. "
                "Every intervention requires a matched random-vector/sham control on the "
                "same items with the same seeds (CLAUDE.md §4)."
            )

    def categories(self) -> dict[str, DecisionCategory]:
        return {s.decision.stage: s.category for s in self.stages}


class AgentStage(abc.ABC):
    """Base class for a pipeline stage.

    Concrete backends are not implemented at this stage of the project.
    """

    #: One of ``planner`` / ``tool_caller`` / ``critic``.
    stage_name: str

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        _backend: "LocalCausalLMBackend | None" = None,
    ) -> None:
        self.config: Mapping[str, Any] = dict(config or {})
        self._backend: LocalCausalLMBackend | None = _backend

    @abc.abstractmethod
    def run(self, stage_input: StageInput) -> StageOutput:
        """Execute the stage and return a classified output.

        Implementations must classify *every* outcome into one of the six
        categories, including exceptions, timeouts, and degenerate generation.
        Implementations must not raise past this boundary to skip a trial: a
        skipped trial is a silently discarded one (CLAUDE.md §2).
        """
        raise NotImplementedError

    def with_backend(self, backend: "LocalCausalLMBackend") -> "AgentStage":
        """Return a new stage instance configured with the given backend."""
        # Subclasses that need immutable behavior should override this.
        self._backend = backend
        return self

    def require_backend(self) -> "LocalCausalLMBackend":
        """Return the configured backend or raise if none is set."""
        if self._backend is None:
            raise RuntimeError(f"{self.stage_name}: backend not configured")
        return self._backend

    def execute(self, stage_input: StageInput) -> StageOutput:
        """Default implementation just calls run(). Subclasses may override."""
        return self.run(stage_input)
