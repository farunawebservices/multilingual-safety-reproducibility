"""Shared execution path for the three role stages.

All three roles do the same four things and differ only in which prompt they
build and which schema they are parsed against::

    build messages   (prompts.PromptLibrary, from the v2 row)
      -> backend.generate(GenerationRequest)
      -> RawOutput(text, role, finish_reason, latency, timed_out, error_type)
      -> records.parse_agent_output(...)   -> ExecutionRecord
      -> ExecutionRecord.to_stage_decision() -> StageOutput

Keeping it in one function is deliberate. Three near-copies would be three
places for the classification path to drift, and the whole point of
:mod:`multilingual_safety.agents.decision_parser` is that there is exactly one.

The role stages themselves contain no classification logic whatsoever.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from multilingual_safety.agents.backends.base import GenerationRequest
from multilingual_safety.agents.base import StageInput, StageOutcome, StageOutput
from multilingual_safety.agents.records import RawOutput, parse_agent_output
from multilingual_safety.agents.run_log import prompt_reference
from multilingual_safety.hashing import canonical_json, text_sha256

#: v2 corpus fields copied onto every result row so the translation-quality
#: confound stays visible in the output rather than living only in a manifest
#: (CLAUDE.md §6). `reviewer_id` is carried verbatim: it is null on all 1,400
#: rows, which is what "unreviewed" looks like.
PROVENANCE_FIELDS: dict[str, str] = {
    "translation_status": "translation_status",
    "translation_method": "translation_method",
    "translation_model": "translation_model",
    "review_required": "review_required",
    "reviewer_id": "reviewer_id",
    "harm_label": "harm_label",
    "category": "category_label",
}


def row_provenance(row: Mapping[str, Any] | None) -> dict[str, Any]:
    """Extract the provenance columns from a v2 row. Never invents a value."""
    if not row:
        return {}
    return {
        record_field: row.get(row_field)
        for row_field, record_field in PROVENANCE_FIELDS.items()
        if row_field in row
    }


def execute_stage(
    stage: Any,
    stage_input: StageInput,
    *,
    run_id: str,
    max_new_tokens: int,
    timeout_seconds: float,
    tool_registry: Any = None,
    plan: Mapping[str, Any] | None = None,
    tool_calls: Sequence[Mapping[str, Any]] | None = None,
    tool_results: Sequence[Any] | None = None,
) -> StageOutcome:
    """Run one role stage end to end and return its full outcome."""
    backend = stage.require_backend()
    row = stage_input.row
    if row is None:
        raise ValueError(
            f"{stage.stage_name}: no dataset row supplied. The prompt is built from the "
            "row's translated_text; there is nothing to run without it."
        )

    described = backend.describe()
    prompt = stage.prompts.build(
        stage.stage_name,
        row,
        stage_language=stage_input.stage_language,
        model_id=described["model_id"],
        model_revision=str(described.get("model_revision") or ""),
        tool_registry=tool_registry,
        plan=plan,
        tool_calls=tool_calls,
        tool_results=tool_results,
    )

    result = backend.generate(
        GenerationRequest(
            prompt_id=prompt.prompt_id,
            messages=prompt.messages,
            role=stage.stage_name,
            max_new_tokens=max_new_tokens,
            seed=stage_input.seed if stage_input.seed is not None else 0,
            timeout_seconds=timeout_seconds,
            intervention=stage_input.intervention,
        )
    )

    raw = RawOutput(
        text=result.text,
        role=stage.stage_name,
        finish_reason=result.finish_reason,
        latency_seconds=result.latency_seconds,
        # A `length` finish means the token budget bound before a stop token.
        # That is a budget failure, not a decision, and the budget is not
        # language-neutral -- so it is classified as TIMEOUT, never as content.
        timed_out=result.timed_out or result.finish_reason in {"timeout", "length"},
        error_type=result.error_type,
        error_message=result.error_message,
        error_phase=result.error_phase,
    )

    record = parse_agent_output(
        raw,
        run_id=run_id,
        behavior_id=stage_input.behavior_id,
        language=stage_input.prompt_language,
        model_id=result.model_id,
        prompt_hash=prompt.prompt_hash,
        seed=stage_input.seed,
        prompt_id=prompt.prompt_id,
        prompt_path=prompt_reference(run_id, prompt.prompt_id),
        tool_results_sha256=prompt.tool_results_sha256,
        model_generation_config_sha256=text_sha256(
            canonical_json(described.get("effective_generation", {}))
        ),
        template_id=prompt.template_id,
        template_sha256=prompt.template_sha256,
        model_revision=result.model_revision,
        tokenizer_id=result.tokenizer_id,
        dtype=result.dtype,
        device_map=dict(result.device_map),
        stage_language=stage_input.stage_language,
        orthography_fingerprint=prompt.orthography_fingerprint,
        condition_id=stage_input.condition_id,
        trial_index=stage_input.trial_index,
        stage_status="RAN",
        split=stage.config.get("split"),
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        generation_config=dict(result.generation_config),
        generate_kwargs=dict(result.generate_kwargs),
        batch_size=result.batch_size,
        batch_index=result.batch_index,
        position_in_batch=result.position_in_batch,
        intervention_id=(
            result.intervention_applied.intervention_id if result.intervention_applied else None
        ),
        control_id=(
            result.intervention_applied.control_id if result.intervention_applied else None
        ),
        intervention_layer=(
            result.intervention_applied.layer if result.intervention_applied else None
        ),
        intervention_norm=(
            result.intervention_applied.applied_norm if result.intervention_applied else None
        ),
        **row_provenance(row),
    )

    output = StageOutput(
        decision=record.to_stage_decision(),
        output_ref=prompt.prompt_id,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        finish_reason=result.finish_reason,
        metadata=prompt.to_metadata_dict(),
        stage=stage.stage_name,
        stage_status=record.stage_status,
    )
    return StageOutcome(
        output=output, record=record, prompt=prompt, raw_text=result.text, parsed=record.parsed
    )
