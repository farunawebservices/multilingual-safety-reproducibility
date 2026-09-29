from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


class BackendError(RuntimeError):
    pass


@dataclass
class InterventionSpec:
    id: str
    kind: str
    target_role: str
    layer: int
    token_positions: tuple[int, ...]
    strength: float
    vector_ref: str | None = None
    vector_sha256: str | None = None
    control_of: str | None = None
    control_seed: int | None = None

    @property
    def is_control(self) -> bool:
        return self.control_of is not None


@dataclass
class InterventionRecord:
    intervention_id: str
    control_id: str | None
    layer: int
    token_positions: tuple[int, ...]
    strength: float
    applied_norm: float
    vector_sha256: str | None
    device: str


@dataclass
class GenerationRequest:
    prompt_id: str
    messages: tuple[Mapping[str, str], ...]
    role: str
    max_new_tokens: int
    seed: int
    timeout_seconds: float = 120.0
    capture_hidden_states: bool = False
    capture_layers: tuple[int, ...] = ()
    intervention: InterventionSpec | None = None


@dataclass
class GenerationResult:
    prompt_id: str
    role: str
    text: str
    finish_reason: str
    tokens_in: int
    tokens_out: int
    latency_seconds: float
    timed_out: bool
    model_id: str
    model_revision: str
    tokenizer_id: str
    dtype: str
    device_map: Mapping[str, str]
    generation_config: Mapping[str, Any]
    generate_kwargs: Mapping[str, Any]
    seed: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_phase: str | None = None
    batch_size: int = 1
    batch_index: int = 0
    position_in_batch: int = 0
    hidden_states_ref: str | None = None
    intervention_applied: InterventionRecord | None = None


class LocalCausalLMBackend(ABC):
    @abstractmethod
    def preflight(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        pass

    @abstractmethod
    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        pass


    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return credential-free backend provenance."""
        raise NotImplementedError
