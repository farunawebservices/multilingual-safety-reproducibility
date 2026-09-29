"""Local Hugging Face transformers backend using pinned cached snapshots."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .base import (
    GenerationRequest,
    GenerationResult,
    InterventionRecord,
    LocalCausalLMBackend,
)
from .cache import ResolvedModel, resolve_model_snapshot


class HfLocalBackend(LocalCausalLMBackend):
    """Local transformers backend that loads a pinned cached model snapshot."""

    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str | Mapping[str, int] = "auto",
        max_input_length: int = 4096,
        max_new_tokens: int = 256,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.dtype = dtype
        self.device_map = device_map
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens

        # Resolve and validate the cached snapshot (no download)
        self.resolved: ResolvedModel | None = None
        self._model: AutoModelForCausalLM | None = None
        self._tokenizer: AutoTokenizer | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.revision,
            "effective_generation": {
                "max_new_tokens": self.max_new_tokens,
                "dtype": str(self.dtype),
                "device_map": str(self.device_map),
            },
        }

    def preflight(self) -> dict[str, Any]:
        """Resolve the snapshot and optionally load model/tokenizer."""
        if self.resolved is None:
            self.resolved = resolve_model_snapshot(
                self.model_id,
                self.revision,
            )
        return {"resolved": True}

    def _ensure_loaded(self) -> None:
        if self._model is None or self._tokenizer is None:
            if self.resolved is None:
                self.preflight()
            assert self.resolved is not None
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.resolved.snapshot_dir,
                revision=self.revision,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.resolved.snapshot_dir,
                revision=self.revision,
                torch_dtype=self.dtype,
                device_map=self.device_map,
                attn_implementation="sdpa",
            )
            self._model.eval()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self._ensure_loaded()
        assert self._model is not None and self._tokenizer is not None

        # Build messages -> text (simple concatenation for now)
        text = "\n".join(m.get("content", "") for m in request.messages)
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_input_length)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        gen_config = GenerationConfig(
            max_new_tokens=request.max_new_tokens or self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        with torch.inference_mode():
            outputs = self._model.generate(**inputs, generation_config=gen_config)

        generated_ids = outputs[0, inputs["input_ids"].shape[1]:]
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        return GenerationResult(
            prompt_id=request.prompt_id,
            role=request.role,
            text=generated_text,
            finish_reason="stop",
            tokens_in=inputs["input_ids"].shape[1],
            tokens_out=generated_ids.shape[0],
            latency_seconds=0.0,  # TODO: measure
            timed_out=False,
            model_id=self.model_id,
            model_revision=self.revision,
            tokenizer_id=self._tokenizer.name_or_path,
            dtype=str(self.dtype),
            device_map={str(k): str(v) for k, v in dict(self._model.named_devices()).items()} if hasattr(self._model, "named_devices") else {"cpu": 1},
            generation_config=gen_config.to_dict(),
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
        return [self.generate(req) for req in requests]
