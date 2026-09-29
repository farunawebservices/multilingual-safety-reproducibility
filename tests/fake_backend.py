"""Fake backend for unit-testing stage_runner and pipeline without models."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from multilingual_safety.agents.backends.base import (
    GenerationRequest,
    GenerationResult,
    InterventionRecord,
    LocalCausalLMBackend,
)


class FakeBackend(LocalCausalLMBackend):
    """Deterministic fake backend that returns canned JSON outputs."""

    def __init__(
        self,
        model_id: str = "fake/model",
        model_revision: str | None = None,
        canned_text: str | None = None,
        canned_json: Mapping[str, Any] | None = None,
        raise_on_generate: Exception | None = None,
    ) -> None:
        self._model_id = model_id
        self._model_revision = model_revision
        self._canned_text = canned_text
        self._canned_json = canned_json or {}
        self._raise_on_generate = raise_on_generate

    def describe(self) -> dict[str, Any]:
        return {
            "model_id": self._model_id,
            "model_revision": self._model_revision,
            "effective_generation": {},
        }

    def preflight(self) -> dict[str, Any]:
        """No-op preflight for tests."""
        return {}

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if self._raise_on_generate is not None:
            raise self._raise_on_generate

        text = self._canned_text or "{}"
        return GenerationResult(
            prompt_id=request.prompt_id,
            role=request.role,
            text=text,
            finish_reason="stop",
            tokens_in=10,
            tokens_out=20,
            latency_seconds=0.0,
            timed_out=False,
            model_id=self._model_id,
            model_revision=self._model_revision or "",
            tokenizer_id="fake/tokenizer",
            dtype="bfloat16",
            device_map={"cpu": "1"},
            generation_config={},
            generate_kwargs={},
            seed=request.seed,
            error_type=None,
            error_message=None,
            error_phase=None,
            batch_size=1,
            batch_index=0,
            position_in_batch=0,
            hidden_states_ref=None,
            intervention_applied=None,
        )

    def generate_batch(
        self,
        requests: Sequence[GenerationRequest],
    ) -> list[GenerationResult]:
        """Batch generate by calling generate() for each request."""
        return [self.generate(req) for req in requests]
