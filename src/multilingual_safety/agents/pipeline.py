"""Trial loop: Planner -> Tool-Caller -> Critic, one v2 row at a time.

Short-circuit semantics, stated because they set the denominators
-----------------------------------------------------------------
If the Planner blocks -- or fails in any of the four apparatus categories -- the
trial halts. The downstream stages are recorded with ``stage_status="NOT_RUN"``
and the matching non-enum sentinel ``decision_category="NOT_RUN"``, with the
upstream reason preserved in ``evidence["not_run_reason"]``. That is **not** a
seventh decision category: it is the absence of an attempt, carried in its own
column so per-stage denominators stay honest, and it is not a MISSING_DECISION
-- no output was produced at all. The trial is retained with its partial stage
vector; nothing is dropped (CLAUDE.md §2).

Denominators
------------
Every ``(record, trial, stage)`` the loop intended to run is emitted. A stage
that was never attempted is emitted too, flagged. There is no ``except:
continue`` anywhere in this module: a backend exception becomes a
``GenerationResult`` with ``error_type`` set and is classified like any other
outcome.

Provenance
----------
Every emitted record carries the row's ``translation_status``,
``translation_method``, ``translation_model``, ``review_required`` and
``reviewer_id`` verbatim. In the frozen v2 corpus that means every non-English
row is stamped ``MACHINE_TRANSLATED_UNCHECKED`` / ``review_required=True`` /
``reviewer_id=null``. No code here upgrades that to "reviewed", whatever a
corpus-level summary asserts (CLAUDE.md §6).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from multilingual_safety.agents.backends.base import InterventionSpec
from multilingual_safety.agents.base import (
    CriticInput,
    PlannerInput,
    StageOutcome,
    StageOutput,
    ToolCallerInput,
    TrialResult,
)
from multilingual_safety.agents.hidden_states import assert_control_present
from multilingual_safety.agents.records import ExecutionRecord
from multilingual_safety.agents.stage_runner import row_provenance
from multilingual_safety.decisions import NOT_RUN, DecisionCategory, StageDecision
from multilingual_safety.ids import record_id as make_record_id

#: Stages, in order.
STAGE_ORDER: tuple[str, ...] = ("planner", "tool_caller", "critic")

__all__ = [
    "NOT_RUN",
    "STAGE_ORDER",
    "ConditionSpec",
    "TrialArtifacts",
    "iter_trials",
    "run_trial",
]


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    """One pre-registered condition or ablation.

    ``stage_languages`` maps a role to ``"user"`` (follow the prompt language)
    or a fixed language code. That is the mechanism behind
    ``critic_language_en`` vs ``critic_language_user`` (RQ3).
    """

    id: str
    stages: tuple[str, ...] = STAGE_ORDER
    stage_languages: Mapping[str, str] = field(default_factory=dict)
    reportable: bool = True
    description: str = ""

    def language_for(self, role: str, prompt_language: str) -> str:
        setting = self.stage_languages.get(role, "user")
        return prompt_language if setting == "user" else setting

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "stages": list(self.stages),
            "stage_languages": dict(self.stage_languages),
            "reportable": self.reportable,
            "description": self.description,
        }


@dataclass(slots=True)
class TrialArtifacts:
    """One trial's complete output: decisions, records, raw text, tool calls."""

    trial: TrialResult
    records: list[ExecutionRecord]
    raw_texts: dict[str, str]                    # prompt_id -> verbatim output
    rendered: dict[str, Any]                     # prompt_id -> rendered messages
    prompts: list[Mapping[str, Any]]             # text-free prompt metadata
    tool_results: list[Mapping[str, Any]]
    tool_problems: list[str]


def _not_run_record(
    role: str,
    row: Mapping[str, Any],
    *,
    run_id: str,
    condition: ConditionSpec,
    trial_index: int,
    seed: int | None,
    model_id: str,
    model_revision: str,
    split: str,
    reason: str,
) -> ExecutionRecord:
    """A stage that was never attempted, recorded rather than omitted."""
    return ExecutionRecord(
        run_id=run_id,
        role=role,
        behavior_id=str(row["behavior_id"]),
        language=str(row["language"]),
        model_id=model_id,
        seed=seed,
        prompt_hash=str(row["prompt_hash"]),
        raw_sha256="",
        raw_n_chars=0,
        parsed=None,
        # A stage with no attempt has no decision, so it gets the non-enum
        # sentinel rather than a category it never produced. MISSING_DECISION is
        # reserved for "output produced, no decision in it" and would be false
        # twice over here -- no output was produced at all. Stamping it would
        # also make the row-level MISSING_DECISION count a deterministic
        # function of the upstream veto rate, i.e. a fabricated cross-lingual
        # apparatus gradient. NOT_RUN rows are excluded from the stage's
        # denominator, with the exclusion reported in the same table.
        decision_category=NOT_RUN,
        stage_decision_value=None,
        schema_version=None,
        termination=None,
        stage_status=NOT_RUN,
        evidence={"not_run_reason": reason},
        model_revision=model_revision,
        stage_language=condition.language_for(role, str(row["language"])),
        condition_id=condition.id,
        trial_index=trial_index,
        split=split,
        **row_provenance(row),
    )


def run_trial(
    row: Mapping[str, Any],
    *,
    stages: Mapping[str, Any],
    condition: ConditionSpec,
    run_id: str,
    trial_index: int,
    seed: int | None,
    split: str,
    model_id: str,
    model_revision: str,
    registry: Any = None,
    intervention: InterventionSpec | None = None,
    control: InterventionSpec | None = None,
) -> TrialArtifacts:
    """Run one behavior in one language for one trial. Never drops a stage."""
    # Refuse an uncontrolled intervention before anything is generated.
    assert_control_present(intervention, control)

    prompt_language = str(row["language"])
    behavior = str(row["behavior_id"])
    rec_id = make_record_id(behavior=behavior, language=prompt_language)
    arm = str(row["harm_label"])

    records: list[ExecutionRecord] = []
    outputs: list[StageOutput] = []
    raw_texts: dict[str, str] = {}
    rendered: dict[str, Any] = {}
    prompt_meta: list[Mapping[str, Any]] = []
    tool_results: list[Mapping[str, Any]] = []
    tool_problems: list[str] = []

    plan: Mapping[str, Any] | None = None
    tool_calls: list[Mapping[str, Any]] = []
    executed: list[Any] = []
    halted_reason: str | None = None

    def base_kwargs(role: str) -> dict[str, Any]:
        return {
            "record_id": rec_id,
            "behavior_id": behavior,
            "prompt_language": prompt_language,
            "stage_language": condition.language_for(role, prompt_language),
            "trial_index": trial_index,
            "seed": seed,
            "row": row,
            "prompt_hash": str(row["prompt_hash"]),
            "condition_id": condition.id,
            "intervention": (
                intervention if intervention and intervention.target_role == role else None
            ),
        }

    for role in condition.stages:
        if halted_reason is not None:
            records.append(
                _not_run_record(
                    role,
                    row,
                    run_id=run_id,
                    condition=condition,
                    trial_index=trial_index,
                    seed=seed,
                    model_id=model_id,
                    model_revision=model_revision,
                    split=split,
                    reason=halted_reason,
                )
            )
            continue

        stage = stages[role]
        if role == "planner":
            outcome: StageOutcome = stage.execute(PlannerInput(**base_kwargs(role)))
        elif role == "tool_caller":
            outcome = stage.execute(ToolCallerInput(**base_kwargs(role), plan=plan))
        else:
            outcome = stage.execute(
                CriticInput(
                    **base_kwargs(role),
                    plan=plan,
                    tool_calls=tool_calls,
                    tool_results=executed,
                )
            )

        record = outcome.record
        records.append(record)
        outputs.append(outcome.output)
        raw_texts[outcome.prompt.prompt_id] = outcome.raw_text
        rendered[outcome.prompt.prompt_id] = outcome.prompt.messages
        prompt_meta.append(outcome.prompt.to_metadata_dict())

        if role == "planner":
            plan = outcome.parsed
        elif role == "tool_caller" and outcome.parsed:
            calls, problems, results = _execute_mock_calls(registry, outcome.parsed)
            tool_calls.extend(calls)
            tool_problems.extend(problems)
            executed.extend(results)
            tool_results.extend(r.to_dict() for r in results)

        # A blocking or failed stage halts the pipeline. The remaining stages
        # are recorded as NOT_RUN rather than given a decision they never made.
        if record.category is not DecisionCategory.SEMANTIC_ALLOW:
            halted_reason = f"{role}:{record.decision_category}"

    if not outputs:                                # pragma: no cover - defensive
        raise RuntimeError("no stage produced output; a zero-stage trial is degenerate")

    trial = TrialResult(
        run_id=run_id,
        record_id=rec_id,
        behavior_id=behavior,
        arm=arm,
        split=split,
        prompt_language=prompt_language,
        trial_index=trial_index,
        seed=seed,
        stages=_stage_vector(records),
        intervention_id=intervention.id if intervention else None,
        control_id=(control.id if control else (intervention.control_of if intervention else None)),
    )
    return TrialArtifacts(
        trial=trial,
        records=records,
        raw_texts=raw_texts,
        rendered=rendered,
        prompts=prompt_meta,
        tool_results=tool_results,
        tool_problems=tool_problems,
    )


def _stage_vector(records: Sequence[ExecutionRecord]) -> list[StageOutput]:
    """Every stage in the trial, attempted or not, in order.

    A never-attempted stage is emitted with ``decision=None`` and its status,
    not with a manufactured :class:`StageDecision`. ``StageDecision.category``
    is typed ``DecisionCategory``, and there is no member meaning "never ran" --
    that is the type system enforcing the pre-registration rather than a gap in
    it.
    """
    vector: list[StageOutput] = []
    for record in records:
        if not record.attempted:
            vector.append(
                StageOutput(
                    decision=None,
                    stage=record.role,
                    stage_status=record.stage_status,
                    not_run_reason=record.evidence.get("not_run_reason"),
                    output_ref=record.prompt_id,
                    metadata={"stage_status": record.stage_status},
                )
            )
            continue
        vector.append(
            StageOutput(
                decision=StageDecision(
                    stage=record.role,
                    category=record.category,
                    detector=record.reason_code or "unrecorded",
                    evidence=dict(record.evidence),
                ),
                stage=record.role,
                stage_status=record.stage_status,
                output_ref=record.prompt_id,
                tokens_in=record.tokens_in,
                tokens_out=record.tokens_out,
                finish_reason=record.finish_reason,
                metadata={"stage_status": record.stage_status},
            )
        )
    return vector


def _execute_mock_calls(
    registry: Any, payload: Mapping[str, Any]
) -> tuple[list[Mapping[str, Any]], list[str], list[Any]]:
    """Run the Tool-Caller's decision against the MOCK registry.

    An unknown tool is a *finding*, recorded with the name it invented. It does
    not change the stage's decision category: the model did decide to call
    something, and rewriting that as a refusal would hide a real failure mode.
    """
    from multilingual_safety.agents.tools import UnknownToolError

    if registry is None or payload.get("decision") != "CALL":
        return [], [], []
    name = payload.get("tool_name")
    if not name:
        return [], ["decision=CALL with no tool_name"], []

    call = {"tool_name": name, "arguments": dict(payload.get("arguments") or {})}
    try:
        result = registry.execute(name, call["arguments"])
    except UnknownToolError as exc:
        return [call], [f"tool_hallucinated: {exc}"], []
    return [call], list(result.problems), [result]


def iter_trials(
    rows: Iterable[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
) -> Iterable[tuple[Mapping[str, Any], int, int]]:
    """Deterministic (row, trial_index, seed) enumeration.

    Sorted by (behavior_id, language) so the same config always produces the
    same order, and therefore the same batch composition.
    """
    ordered = sorted(rows, key=lambda r: (str(r["behavior_id"]), str(r["language"])))
    for row in ordered:
        for index, seed in enumerate(seeds):
            yield row, index, seed
