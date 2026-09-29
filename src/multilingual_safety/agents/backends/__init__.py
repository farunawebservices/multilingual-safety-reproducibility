"""Local model backend contracts and offline cache-resolution utilities."""

from .base import (
    BackendError,
    GenerationRequest,
    GenerationResult,
    InterventionRecord,
    InterventionSpec,
    LocalCausalLMBackend,
)
from .cache import (
    ModelNotCachedError,
    ModelRevisionError,
    ModelSnapshotError,
    ResolvedModel,
    assert_pinned_revision,
    resolve_model_snapshot,
)

__all__ = [
    "BackendError",
    "GenerationRequest",
    "GenerationResult",
    "InterventionRecord",
    "InterventionSpec",
    "LocalCausalLMBackend",
    "ModelNotCachedError",
    "ModelRevisionError",
    "ModelSnapshotError",
    "ResolvedModel",
    "assert_pinned_revision",
    "resolve_model_snapshot",
]
