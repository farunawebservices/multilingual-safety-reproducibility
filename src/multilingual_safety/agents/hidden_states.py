"""Hidden-state capture and activation steering, kept off the baseline path.

Three separate concerns, deliberately not one flag:

**Baseline.** With neither capture nor intervention requested, :class:`HookManager`
registers *no hook at all* and the forward pass is byte-identical to a plain
``generate()``. This is asserted in ``tests/test_hidden_states.py`` rather than
assumed -- "the instrumentation is inert when off" is exactly the kind of claim
that quietly stops being true.

**Capture (read-only).** Forward hooks read the residual stream and return
``None``, so the module output is never replaced. Captures are written under
``logs/<run_id>/hidden/``, never ``results/``: activations derived from harmful
prompts are corpus payload. Capture belongs in its own run -- it changes memory
pressure and batch composition, so a decision rate measured under capture is not
the same measurement as one measured without it.

**Steering (write) and its matched control.** Every vector intervention requires
a control that is matched in **norm, shape, layer, and token positions**, run on
the same items with the same seeds in the same session (CLAUDE.md §4). The
control is generated here, from an explicit ``control_seed``, so "matched" is a
computed property rather than a claim. The reported effect is *intervention
minus matched control* -- never *intervention minus baseline*, because applying
any vector perturbs behavior and that is not the finding.

Device note: a 70B model is sharded across GPUs by ``device_map``, so layer *i*
may live on any device. Vectors are moved to the hook's own device at call time.
Preloading to ``cuda:0`` would either raise or, worse, silently add across
devices.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from multilingual_safety.agents.backends.base import (
    BackendError,
    InterventionRecord,
    InterventionSpec,
)


class InterventionError(BackendError):
    """An intervention is malformed, or its control does not match."""


class MissingControlError(InterventionError):
    """An intervention arm was requested with no matched control.

    Raised *before* the model loads, so a missing control costs zero GPU hours
    instead of being discovered at analysis time when the compute is spent.
    """


def decoder_layers(model: Any) -> Sequence[Any]:
    """Return the model's decoder layer modules.

    Covers ``LlamaForCausalLM`` and ``Qwen2ForCausalLM`` (both ``model.model.layers``)
    and raises for anything else rather than guessing at an attribute path.
    """
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise InterventionError(
            f"cannot locate decoder layers on {type(model).__name__}. Steering and "
            "capture need an explicit layer path; refusing to guess."
        )
    return layers


def resolve_layer(model: Any, index: int) -> Any:
    layers = decoder_layers(model)
    if not -len(layers) <= index < len(layers):
        raise InterventionError(
            f"layer {index} out of range for a {len(layers)}-layer model"
        )
    return layers[index]


def _residual(output: Any) -> Any:
    """Decoder layers return either a tensor or a tuple whose first item is one."""
    return output[0] if isinstance(output, tuple) else output


def _rewrap(output: Any, new_hidden: Any) -> Any:
    return (new_hidden, *output[1:]) if isinstance(output, tuple) else new_hidden


# ---------------------------------------------------------------------------
# Matched controls
# ---------------------------------------------------------------------------
def make_matched_control(vector: Any, *, control_seed: int) -> Any:
    """A random vector with the same shape and the same L2 norm as ``vector``.

    Deterministic in ``control_seed``, and generated on CPU before being moved
    to the target device, so the control does not depend on GPU RNG state or on
    which shard the layer happens to live on.
    """
    import torch

    if control_seed is None:
        raise InterventionError("a matched control requires an explicit control_seed")
    generator = torch.Generator(device="cpu").manual_seed(int(control_seed))
    flat = torch.randn(
        vector.shape, generator=generator, dtype=torch.float32, device="cpu"
    )
    norm = torch.linalg.vector_norm(flat)
    if float(norm) == 0.0:                     # pragma: no cover - measure-zero
        raise InterventionError("random control vector had zero norm; reseed")
    target = torch.linalg.vector_norm(vector.detach().to("cpu", torch.float32))
    scaled = flat * (target / norm)
    return scaled.to(dtype=vector.dtype)


def assert_control_matches(intervention: Any, control: Any, *, tolerance: float = 1e-3) -> None:
    """Refuse a control that is not matched in shape and norm."""
    import torch

    if tuple(intervention.shape) != tuple(control.shape):
        raise InterventionError(
            f"control shape {tuple(control.shape)} != intervention shape "
            f"{tuple(intervention.shape)}; a mismatched control is not a control."
        )
    a = float(torch.linalg.vector_norm(intervention.detach().to("cpu", torch.float32)))
    b = float(torch.linalg.vector_norm(control.detach().to("cpu", torch.float32)))
    if a == 0.0 and b == 0.0:
        return
    if not math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance):
        raise InterventionError(
            f"control L2 norm {b:.6g} != intervention L2 norm {a:.6g}. The reported "
            "effect is intervention minus MATCHED control (CLAUDE.md §4)."
        )


def assert_control_present(
    intervention: InterventionSpec | None,
    control: InterventionSpec | None,
) -> None:
    """Gate an intervention arm on having its matched control in the same run."""
    if intervention is None or intervention.is_control:
        return
    if control is None:
        raise MissingControlError(
            f"intervention {intervention.id!r} was requested with no matched control. "
            "Every intervention requires a matched random-vector/sham control on the "
            "same items with the same seeds, in the same session (CLAUDE.md §4). "
            "Refusing before the model loads."
        )
    if control.control_of != intervention.id:
        raise MissingControlError(
            f"control {control.id!r} declares control_of={control.control_of!r}, which "
            f"does not match intervention {intervention.id!r}. Do not reuse the control "
            "from another arm."
        )
    for attribute in ("target_role", "layer", "token_positions", "strength"):
        want = getattr(intervention, attribute)
        got = getattr(control, attribute)
        if want != got:
            raise MissingControlError(
                f"control {control.id!r} differs from its intervention on {attribute}: "
                f"{got!r} vs {want!r}. Matched means matched."
            )


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class CaptureBuffer:
    """Residual streams read at one layer. Text-free by construction."""

    layer: int
    tensors: list[Any]

    def stack(self) -> Any:
        import torch

        return torch.cat(self.tensors, dim=1) if self.tensors else torch.empty(0)


class HookManager:
    """Registers capture and steering hooks for the duration of one generation.

    A no-op when nothing is requested: ``__enter__`` registers no hook, so the
    baseline forward pass is untouched.
    """

    def __init__(
        self,
        model: Any,
        *,
        capture_layers: Sequence[int] = (),
        intervention: InterventionSpec | None = None,
        vector: Any | None = None,
    ) -> None:
        self.model = model
        self.capture_layers = tuple(capture_layers)
        self.intervention = intervention
        self.vector = vector
        self.captures: dict[int, CaptureBuffer] = {}
        self.applied: InterventionRecord | None = None
        self._handles: list[Any] = []

        if intervention is not None and vector is None:
            raise InterventionError(
                f"intervention {intervention.id!r} has no vector to apply"
            )

    @property
    def is_inert(self) -> bool:
        """True when this manager will register nothing at all."""
        return not self.capture_layers and self.intervention is None

    def __enter__(self) -> HookManager:
        if self.is_inert:
            return self
        for index in self.capture_layers:
            self._handles.append(
                resolve_layer(self.model, index).register_forward_hook(
                    self._make_capture_hook(index)
                )
            )
        if self.intervention is not None:
            self._handles.append(
                resolve_layer(self.model, self.intervention.layer).register_forward_hook(
                    self._make_steering_hook(self.intervention)
                )
            )
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # --- hook factories ---------------------------------------------------
    def _make_capture_hook(self, index: int) -> Callable[[Any, Any, Any], None]:
        buffer = self.captures.setdefault(index, CaptureBuffer(layer=index, tensors=[]))

        def hook(_module: Any, _args: Any, output: Any) -> None:
            # Read-only: returning None leaves the module output untouched.
            buffer.tensors.append(_residual(output).detach().to("cpu"))
            return None

        return hook

    def _make_steering_hook(self, spec: InterventionSpec) -> Callable[[Any, Any, Any], Any]:
        import torch

        positions = spec.token_positions

        source = self.vector
        if source is None:                     # pragma: no cover - guarded in __init__
            raise InterventionError(f"intervention {spec.id!r} has no vector to apply")

        def hook(module: Any, _args: Any, output: Any) -> Any:
            hidden = _residual(output)
            # Move to the hook's own device: with device_map the layer may live
            # on any shard, and a cross-device add is a silent-wrong-number bug.
            vector = source.to(device=hidden.device, dtype=hidden.dtype)
            delta = vector * spec.strength
            updated = hidden.clone()
            if positions:
                length = updated.shape[1]
                index = [p for p in positions if -length <= p < length]
                if index:
                    updated[:, index, :] = updated[:, index, :] + delta
            else:
                updated = updated + delta
            if self.applied is None:
                self.applied = InterventionRecord(
                    intervention_id=spec.id,
                    control_id=spec.control_of,
                    layer=spec.layer,
                    token_positions=positions,
                    strength=spec.strength,
                    applied_norm=float(
                        torch.linalg.vector_norm(delta.detach().to("cpu", torch.float32))
                    ),
                    vector_sha256=spec.vector_sha256,
                    device=str(hidden.device),
                )
            return _rewrap(output, updated)

        return hook


@contextmanager
def hooks(
    model: Any,
    *,
    capture_layers: Sequence[int] = (),
    intervention: InterventionSpec | None = None,
    vector: Any | None = None,
) -> Iterator[HookManager]:
    """Convenience context manager around :class:`HookManager`."""
    manager = HookManager(
        model, capture_layers=capture_layers, intervention=intervention, vector=vector
    )
    with manager:
        yield manager


def count_registered_hooks(model: Any) -> int:
    """Total forward hooks across all decoder layers. Zero on a clean model."""
    return sum(len(layer._forward_hooks) for layer in decoder_layers(model))
