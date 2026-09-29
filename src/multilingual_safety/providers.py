"""Translation providers and deterministic request/response logging.

Design rules, all enforced rather than documented:

* **Explicit selection, no fallback.** :func:`get_provider` resolves exactly the
  provider named. If it is unavailable the call raises. Silently switching
  engines would change the provenance of the corpus without changing any
  recorded field, which is the worst possible failure mode here.
* **The response log is written for every attempt**, including failures. A
  translation that errored is data; a missing row is an unexplained gap.
* **Append-only, no overwrite without ``force``.** Re-translating a row silently
  would make an earlier recorded hash unreproducible.
* **No credentials anywhere.** Providers record the environment variable *name*
  they would read, never a value (CLAUDE.md §7).
* **Raw translated text stays in git-ignored files.** The response log lives in
  ``data/interim/``; tracked manifests get hashes and lengths only.

The only provider implemented here is ``nllb_local``: local weights, no network
call to any translation API, no credential. Commercial providers are declared in
``configs/translation.yaml`` but deliberately have no client code.
"""

from __future__ import annotations

import abc
import json
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multilingual_safety.hashing import text_sha256
from multilingual_safety.languages import PIVOT_LANGUAGE, validate_language_code

#: Providers with an implemented client. Everything else is declaration only.
IMPLEMENTED_PROVIDERS: frozenset[str] = frozenset({"nllb_local"})

#: Providers that reach a third-party API. None is implemented; listed so that
#: `assert_no_network_provider` can refuse them explicitly rather than by
#: omission.
NETWORK_PROVIDERS: frozenset[str] = frozenset(
    {"deepl", "google_translate", "azure_translator", "llm"}
)

#: Keys recorded in `generation_config` for provenance but NOT passed to
#: transformers `generate()`, which rejects unknown kwargs outright.
_NON_GENERATE_KEYS: frozenset[str] = frozenset({"seed", "random_seed", "torch_seed"})

#: Rejected outright if a config tries to set them alongside max_new_tokens.
_REJECTED_GENERATE_KEYS: frozenset[str] = frozenset({"max_length", "min_length"})

#: Length controls that must never coexist with `max_new_tokens`. `max_length`
#: caps the TOTAL sequence; `max_new_tokens` budgets NEW tokens. Both set means
#: two different semantics are live and which binds depends on prompt length.
#: Project policy: `max_new_tokens` is the single control.
_CONFLICTING_LENGTH_KEYS: frozenset[str] = frozenset({"max_length", "min_length"})

#: Nested blocks a config might hide generation settings inside. Stripping only
#: the top level would let `generation: {generation_config: {seed: ...}}` slip
#: straight through to generate() and fail every request again.
_NESTED_GENERATION_KEYS: frozenset[str] = frozenset(
    {"generation_config", "model_kwargs", "generate_kwargs", "generation"}
)


def extract_seed(settings: Mapping[str, Any]) -> int | None:
    """Find the seed at any nesting depth. Returns None when absent."""
    for key in sorted(settings):
        value = settings[key]
        if key in _NON_GENERATE_KEYS and isinstance(value, int):
            return value
        if key in _NESTED_GENERATION_KEYS and isinstance(value, Mapping):
            nested = extract_seed(value)
            if nested is not None:
                return nested
    return None


def strip_non_generate_keys(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Return generate()-safe kwargs: no seed, at any depth.

    Recurses into nested generation blocks rather than trusting that a caller
    kept the config flat.
    """
    cleaned: dict[str, Any] = {}
    for key in sorted(settings):
        if key in _NON_GENERATE_KEYS:
            continue
        value = settings[key]
        if key in _NESTED_GENERATION_KEYS and isinstance(value, Mapping):
            nested = strip_non_generate_keys(value)
            if nested:
                cleaned[key] = nested
            continue
        cleaned[key] = value
    return cleaned


def assert_single_length_control(kwargs: Mapping[str, Any]) -> None:
    """Refuse generate() kwargs carrying a second, conflicting length control."""
    offenders = sorted(k for k in kwargs if k in _REJECTED_GENERATE_KEYS)
    if offenders:
        raise ProviderError(
            f"refusing to call generate() with {offenders} alongside `max_new_tokens`. "
            "Two length controls with different semantics (total sequence vs new "
            "tokens) would make the effective limit depend on prompt length. "
            "Project policy: max_new_tokens is the single control."
        )


def assert_no_seed_in_kwargs(kwargs: Mapping[str, Any]) -> None:
    """Guard immediately before generate(). Fails loudly if a seed crept back.

    A defence-in-depth check: the constructor already strips these, so reaching
    this error means a new code path built kwargs some other way.
    """
    offenders = sorted(k for k in kwargs if k in _NON_GENERATE_KEYS)
    nested = sorted(
        f"{k}.{n}"
        for k, v in kwargs.items()
        if k in _NESTED_GENERATION_KEYS and isinstance(v, Mapping)
        for n in v
        if n in _NON_GENERATE_KEYS
    )
    if offenders or nested:
        raise ProviderError(
            f"refusing to call generate() with seed-like kwarg(s) {offenders + nested}. "
            "transformers rejects unknown model_kwargs and every request would fail. "
            "Seeding is applied via torch.manual_seed before generation instead."
        )

STATUS_OK = "OK"
STATUS_ERROR = "ERROR"
STATUS_SKIPPED = "SKIPPED"


class ProviderError(RuntimeError):
    """Base class for provider failures."""


class ProviderUnavailableError(ProviderError):
    """The selected provider cannot run (missing deps, missing weights, ...).

    Raised instead of falling back to another provider.
    """


class ProviderMismatchError(ProviderError):
    """A response log already holds rows from a different provider or model."""


class ResponseLogError(ProviderError):
    """An append would overwrite an existing response without ``force``."""


# ---------------------------------------------------------------------------
def make_request_id(behavior_id: str, target_language: str) -> str:
    """Deterministic request ID: ``<behavior_id>:<language>``.

    Pure function of the pair, so the same behavior/language always correlates
    to the same request across runs, machines, and providers.
    """
    validate_language_code(target_language)
    if not behavior_id:
        raise ValueError("behavior_id is required for a request id")
    return f"{behavior_id}:{target_language}"


@dataclass(frozen=True, slots=True)
class TranslationRequest:
    """One unit of work. Carries source text; never written to a tracked file."""

    request_id: str
    behavior_id: str
    source_language: str
    target_language: str
    source_text: str
    source_text_sha256: str

    @classmethod
    def create(
        cls,
        *,
        behavior_id: str,
        target_language: str,
        source_text: str,
        source_language: str = PIVOT_LANGUAGE,
    ) -> TranslationRequest:
        validate_language_code(target_language)
        validate_language_code(source_language)
        if target_language == source_language:
            raise ValueError(
                f"refusing to translate {source_language!r} into itself; English rows "
                "are canonical source text and are never translated"
            )
        if not source_text.strip():
            raise ValueError(f"{behavior_id}: empty source_text")
        return cls(
            request_id=make_request_id(behavior_id, target_language),
            behavior_id=behavior_id,
            source_language=source_language,
            target_language=target_language,
            source_text=source_text,
            source_text_sha256=text_sha256(source_text),
        )


@dataclass(frozen=True, slots=True)
class TranslationResponse:
    """One attempt, successful or not. This is the response-log row."""

    request_id: str
    behavior_id: str
    source_language: str
    target_language: str
    provider: str
    model_id: str
    model_revision: str | None
    tokenizer_id: str | None
    device: str | None
    dtype: str | None
    generation_config: Mapping[str, Any]
    response_text: str | None
    response_sha256: str | None
    timestamp: str
    latency_seconds: float
    status: str
    error_type: str | None
    error_message: str | None
    run_id: str
    source_text_sha256: str
    attempt: int = 1

    def __post_init__(self) -> None:
        if self.status not in (STATUS_OK, STATUS_ERROR, STATUS_SKIPPED):
            raise ValueError(f"unknown status {self.status!r}")
        expected = text_sha256(self.response_text) if self.response_text else None
        if self.response_sha256 != expected:
            raise ValueError(f"{self.request_id}: response_sha256 does not match response_text")
        if self.status == STATUS_OK and not self.response_text:
            raise ValueError(f"{self.request_id}: status OK with no response text")
        if self.status == STATUS_ERROR and not self.error_type:
            raise ValueError(f"{self.request_id}: status ERROR with no error_type")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["generation_config"] = dict(sorted(self.generation_config.items()))
        return dict(sorted(payload.items()))

    def to_manifest_dict(self) -> dict[str, Any]:
        """Text-free projection for tracked artifacts."""
        payload = {k: v for k, v in self.to_dict().items() if k != "response_text"}
        payload["response_n_chars"] = None if self.response_text is None else len(self.response_text)
        return dict(sorted(payload.items()))


# ---------------------------------------------------------------------------
class TranslationProvider(abc.ABC):
    """Base provider. Subclasses must never fall back to another engine."""

    name: str
    #: Environment variable NAME this provider would read, if any. Never a value.
    api_key_env: str | None = None

    @abc.abstractmethod
    def preflight(self) -> dict[str, Any]:
        """Verify the provider can run. Raise ProviderUnavailableError if not.

        Returns a metadata dict recorded in the run report.
        """

    @abc.abstractmethod
    def translate(self, request: TranslationRequest, *, run_id: str) -> TranslationResponse:
        """Translate one request. Must return a response even on failure."""

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "api_key_env": self.api_key_env}


class NLLBLocalProvider(TranslationProvider):
    """facebook/nllb-200-distilled-600M, executed locally.

    No network translation API is contacted. Weights come from the local Hugging
    Face cache (or are downloaded once into it); after that
    ``local_files_only=True`` forbids network access entirely.

    Licence: CC-BY-NC-4.0, research use, explicitly "not released for production
    deployment" upstream. Output requires human review, and derived translations
    inherit these restrictions (THIRD_PARTY_DATASETS.md).
    """

    name = "nllb_local"
    api_key_env = None

    def __init__(self, settings: Mapping[str, Any] | None = None) -> None:
        settings = dict(settings or {})
        self.model_id: str = settings.get("model_id", "facebook/nllb-200-distilled-600M")
        self.pinned_revision: str | None = settings.get("revision")
        self.local_files_only: bool = bool(settings.get("local_files_only", False))
        self.requested_device: str = settings.get("device", "auto")
        self.dtype_name: str = settings.get("dtype", "float32")
        self.generation: dict[str, Any] = dict(
            settings.get("generation")
            or {"num_beams": 4, "do_sample": False, "max_new_tokens": 256}
        )
        # `seed` belongs in the recorded provenance but is NOT a transformers
        # generate() kwarg -- passing it through raises
        # "model_kwargs are not used by the model: ['seed']" and fails EVERY
        # request. It is applied via torch.manual_seed instead. The strip is
        # recursive so it cannot re-enter through a nested `generation_config`
        # or `model_kwargs` block either.
        self.seed: int | None = extract_seed(self.generation)
        self.generate_kwargs: dict[str, Any] = strip_non_generate_keys(self.generation)
        self.flores_codes: dict[str, str] = dict(settings.get("flores_codes") or {})
        self._model = None
        self._tokenizer = None
        self._resolved_revision: str | None = None
        self._device: str | None = None
        self._removed_length_controls: dict[str, Any] = {}

    # --- availability -----------------------------------------------------
    def probe(self) -> dict[str, Any]:
        """Cheap availability check. Does NOT load weights and never raises.

        Used for the startup banner, where loading a 2.4 GB model just to print
        one line would be absurd. It reports what it could and could not check;
        a probe saying ``usable: true`` is a *prediction*, not a guarantee --
        :meth:`preflight` remains the authority and still fails closed.
        """
        report: dict[str, Any] = {
            "provider": self.name,
            "model_id": self.model_id,
            "model_revision": self.pinned_revision,
            "dependencies_available": False,
            "weights_cached": None,
            "usable": False,
            "reason": None,
            "checked_by": "probe (no weights loaded)",
        }
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            report["reason"] = (
                f"{self.name}: transformers/torch not importable in this interpreter "
                f"({exc}). Install with: python -m pip install -e '.[translate]' -- and "
                "check you are running the interpreter you installed into."
            )
            return report
        report["dependencies_available"] = True

        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(
                self.model_id, "config.json", revision=self.pinned_revision
            )
            report["weights_cached"] = isinstance(cached, str)
        except Exception:
            report["weights_cached"] = None

        if self.local_files_only and report["weights_cached"] is False:
            report["reason"] = (
                f"{self.name}: local_files_only=True but {self.model_id!r} is not in the "
                "local cache."
            )
            return report

        report["usable"] = True
        report["reason"] = (
            f"{self.name}: dependencies present"
            + ("; weights cached" if report["weights_cached"] else "; weights not yet cached")
        )
        return report

    def preflight(self) -> dict[str, Any]:
        """Load deps + weights, and assert the FLORES codes really exist.

        Raises :class:`ProviderUnavailableError` rather than degrading.
        """
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ProviderUnavailableError(
                f"{self.name}: transformers/torch are not installed in this environment "
                f"({exc}). Install with: python -m pip install -e '.[translate]'. "
                "Refusing to fall back to another provider."
            ) from exc

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                revision=self.pinned_revision,
                local_files_only=self.local_files_only,
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id,
                revision=self.pinned_revision,
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            raise ProviderUnavailableError(
                f"{self.name}: could not load {self.model_id!r} "
                f"(revision={self.pinned_revision!r}, local_files_only={self.local_files_only}): "
                f"{type(exc).__name__}: {exc}. Refusing to fall back to another provider."
            ) from exc

        self._device = self._resolve_device()
        self._model = self._model.to(self._device)
        self._model.eval()
        self._resolved_revision = self._detect_revision()
        # Clear the model's own max_length so max_new_tokens is the sole control.
        self._removed_length_controls = self.neutralize_model_length_controls()

        missing = self._verify_flores_codes()
        if missing:
            raise ProviderUnavailableError(
                f"{self.name}: tokenizer does not know FLORES code(s) {sorted(missing)}. "
                "The configured flores_codes table is wrong for this model revision."
            )
        return self.describe()

    def _resolve_device(self) -> str:
        import torch

        if self.requested_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.requested_device == "cuda" and not torch.cuda.is_available():
            raise ProviderUnavailableError(
                f"{self.name}: device 'cuda' requested but CUDA is unavailable. "
                "Refusing to silently downgrade to CPU, which would change latency "
                "and possibly numerics without changing any recorded field."
            )
        return self.requested_device

    def _detect_revision(self) -> str | None:
        """Best-effort resolution of the cached snapshot commit sha."""
        if self.pinned_revision:
            return self.pinned_revision
        try:
            from huggingface_hub import HfApi

            return HfApi().model_info(self.model_id).sha
        except Exception:
            config = getattr(self._model, "config", None)
            return getattr(config, "_commit_hash", None)

    def _verify_flores_codes(self) -> set[str]:
        """Return configured FLORES codes the tokenizer does not recognise."""
        if not self.flores_codes or self._tokenizer is None:
            return set()
        vocabulary = set(getattr(self._tokenizer, "additional_special_tokens", []) or [])
        if not vocabulary:
            lang_map = getattr(self._tokenizer, "lang_code_to_id", None)
            vocabulary = set(lang_map) if lang_map else set()
        if not vocabulary:
            return set()  # cannot introspect; recorded rather than guessed
        return {code for code in self.flores_codes.values() if code not in vocabulary}

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model_id": self.model_id,
            "model_revision": self._resolved_revision,
            "tokenizer_id": self.model_id,
            "device": self._device,
            "dtype": self.dtype_name,
            "generation_config": dict(sorted(self.generation.items())),
            "generate_kwargs": dict(sorted(self.generate_kwargs.items())),
            "length_control": "max_new_tokens",
            "model_default_length_controls_removed": dict(
                sorted(self._removed_length_controls.items())
            ),
            "local_files_only": self.local_files_only,
            "api_key_env": None,
            "network_translation_api_called": False,
            "license": "CC-BY-NC-4.0",
            "license_note": "research use; output requires human review",
        }

    def neutralize_model_length_controls(self) -> dict[str, Any]:
        """Remove the model's baked-in ``max_length`` so exactly one control governs.

        ``facebook/nllb-200-distilled-600M`` ships ``max_length: 200`` inside its
        own ``generation_config.json``. We never pass ``max_length``, but
        transformers computes ``has_default_max_length`` from
        ``self.generation_config.max_length`` (generation/utils.py), so the
        model's own value collides with our ``max_new_tokens=256`` and every
        call logs "Both `max_new_tokens` and `max_length` seem to have been set".

        The warning is not cosmetic: 200 is a *total* sequence cap while 256 is a
        *new-token* budget, so leaving both set means two different length
        semantics are live at once and which one binds depends on prompt length.
        Clearing it makes ``max_new_tokens`` the single, unambiguous control.

        Returns what was removed, for the run record.
        """
        removed: dict[str, Any] = {}
        config = getattr(self._model, "generation_config", None)
        if config is None:
            return removed
        for attribute in _CONFLICTING_LENGTH_KEYS:
            prior = getattr(config, attribute, None)
            if prior is not None:
                removed[attribute] = prior
                setattr(config, attribute, None)
        return removed

    def apply_seed(self) -> int | None:
        """Seed the RNGs before generation. Returns the seed applied, or None.

        CUDA keeps its own generator state, so ``torch.manual_seed`` alone does
        not make a GPU run reproducible -- ``cuda.manual_seed_all`` is required
        as well, and is applied only when the model is actually on CUDA.
        """
        if self.seed is None:
            return None
        import torch

        torch.manual_seed(self.seed)
        if self._device is not None and str(self._device).startswith("cuda"):
            torch.cuda.manual_seed_all(self.seed)
        return self.seed

    # --- translation ------------------------------------------------------
    def translate(self, request: TranslationRequest, *, run_id: str) -> TranslationResponse:
        if self._model is None or self._tokenizer is None:
            raise ProviderUnavailableError(f"{self.name}: preflight() has not been run")

        target = self.flores_codes.get(request.target_language)
        source = self.flores_codes.get(request.source_language, "eng_Latn")
        started = time.perf_counter()
        common = {
            "request_id": request.request_id,
            "behavior_id": request.behavior_id,
            "source_language": request.source_language,
            "target_language": request.target_language,
            "provider": self.name,
            "model_id": self.model_id,
            "model_revision": self._resolved_revision,
            "tokenizer_id": self.model_id,
            "device": self._device,
            "dtype": self.dtype_name,
            "generation_config": dict(sorted(self.generation.items())),
            "run_id": run_id,
            "source_text_sha256": request.source_text_sha256,
        }

        if target is None:
            return TranslationResponse(
                **common,
                response_text=None,
                response_sha256=None,
                timestamp=_now(),
                latency_seconds=round(time.perf_counter() - started, 6),
                status=STATUS_ERROR,
                error_type="UnsupportedLanguage",
                error_message=f"no FLORES code configured for {request.target_language!r}",
            )

        try:
            import torch

            # Deterministic seeding happens HERE, before generation -- never by
            # handing `seed` to generate().
            self.apply_seed()
            self._tokenizer.src_lang = source
            encoded = self._tokenizer(request.source_text, return_tensors="pt").to(self._device)
            forced_bos = self._forced_bos_token_id(target)
            generate_kwargs = dict(self.generate_kwargs)
            assert_no_seed_in_kwargs(generate_kwargs)          # defence in depth
            assert_single_length_control(generate_kwargs)      # exactly one length control
            with torch.inference_mode():
                generated = self._model.generate(
                    **encoded,
                    forced_bos_token_id=forced_bos,
                    **generate_kwargs,
                )
            text = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        except Exception as exc:
            return TranslationResponse(
                **common,
                response_text=None,
                response_sha256=None,
                timestamp=_now(),
                latency_seconds=round(time.perf_counter() - started, 6),
                status=STATUS_ERROR,
                error_type=type(exc).__name__,
                error_message=str(exc)[:500],
            )

        return TranslationResponse(
            **common,
            response_text=text,
            response_sha256=text_sha256(text) if text else None,
            timestamp=_now(),
            latency_seconds=round(time.perf_counter() - started, 6),
            status=STATUS_OK if text.strip() else STATUS_ERROR,
            error_type=None if text.strip() else "EmptyGeneration",
            error_message=None if text.strip() else "model returned empty text",
        )

    def _forced_bos_token_id(self, flores_code: str) -> int:
        tokenizer = self._tokenizer
        lang_map = getattr(tokenizer, "lang_code_to_id", None)
        if lang_map and flores_code in lang_map:
            return lang_map[flores_code]
        return tokenizer.convert_tokens_to_ids(flores_code)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
def get_provider(name: str | None, config: Mapping[str, Any]) -> TranslationProvider:
    """Resolve a provider by explicit name. Never falls back.

    ``None`` is an error here: the caller must decide, not this function.
    """
    if not name:
        raise ProviderUnavailableError(
            "no provider selected. Set `provider:` in configs/translation.yaml or pass "
            "--provider explicitly. There is no default and no fallback."
        )
    if name in NETWORK_PROVIDERS:
        raise ProviderUnavailableError(
            f"provider {name!r} reaches a third-party translation API and has no client "
            "code in this repository. Only 'nllb_local' is implemented, deliberately: "
            "it keeps harmful prompt text on this machine."
        )
    if name not in IMPLEMENTED_PROVIDERS:
        raise ProviderUnavailableError(
            f"unknown provider {name!r}; implemented: {sorted(IMPLEMENTED_PROVIDERS)}"
        )
    settings = dict(config.get("nllb_local_settings") or {})
    provider_spec = (config.get("providers") or {}).get("nllb_local") or {}
    settings.setdefault("model_id", provider_spec.get("model"))
    settings["flores_codes"] = provider_spec.get("flores_codes") or {}
    return NLLBLocalProvider(settings)


def probe_provider(name: str | None, config: Mapping[str, Any]) -> dict[str, Any]:
    """Cheap availability report for ``name``. Never raises, never falls back.

    A provider that cannot be resolved reports ``usable: false`` with the reason;
    it is never substituted for a working one.
    """
    if not name:
        return {
            "provider": None,
            "usable": False,
            "reason": "no provider selected (config `provider:` is null and no --provider given)",
        }
    try:
        provider = get_provider(name, config)
    except ProviderUnavailableError as exc:
        return {"provider": name, "usable": False, "reason": str(exc)}
    probe = getattr(provider, "probe", None)
    if probe is None:
        return {"provider": name, "usable": False, "reason": "provider has no probe()"}
    return probe()


def assert_no_network_provider(name: str | None) -> None:
    """Raise if ``name`` would contact a third-party translation API."""
    if name in NETWORK_PROVIDERS:
        raise ProviderUnavailableError(
            f"provider {name!r} is a network translation API and is not permitted here"
        )


# ---------------------------------------------------------------------------
class ResponseLog:
    """Append-only JSONL response log with correlation and overwrite protection.

    Lives in git-ignored ``data/interim/``: it necessarily contains raw
    translated text.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def existing(self) -> dict[str, dict[str, Any]]:
        """Map ``request_id -> latest row``. Malformed lines raise, never skip."""
        rows: dict[str, dict[str, Any]] = {}
        if not self.path.exists():
            return rows
        for lineno, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ResponseLogError(f"{self.path}:{lineno}: malformed JSON: {exc}") from exc
            rows[row["request_id"]] = row
        return rows

    def history(self) -> dict[str, list[dict[str, Any]]]:
        """Full attempt history per request_id, in chronological (append) order.

        The audit log is append-only: every attempt is retained, including the
        ones a later retry superseded. Nothing here collapses or rewrites them.
        """
        attempts: dict[str, list[dict[str, Any]]] = {}
        if not self.path.exists():
            return attempts
        for lineno, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ResponseLogError(f"{self.path}:{lineno}: malformed JSON: {exc}") from exc
            attempts.setdefault(row["request_id"], []).append(row)
        return {k: v for k, v in sorted(attempts.items())}

    def effective(self) -> dict[str, dict[str, Any]]:
        """The single response that applies per request_id.

        Rule: **the latest successful attempt wins; if none succeeded, the latest
        attempt wins.**

        A later ERROR does not retract an earlier OK. An error is an apparatus
        failure (a crashed process, a transient CUDA fault), not a retraction of
        a translation that was already produced and hashed -- so letting it
        supersede a good result would silently drop real data while the audit log
        still showed success.
        """
        resolved: dict[str, dict[str, Any]] = {}
        for request_id, attempts in self.history().items():
            successes = [a for a in attempts if a.get("status") == STATUS_OK]
            resolved[request_id] = successes[-1] if successes else attempts[-1]
        return resolved

    def retry_summary(self) -> dict[str, Any]:
        """Retry statistics. Reports superseded attempts rather than hiding them."""
        attempts = self.history()
        effective = self.effective()
        superseded = {
            request_id: len(rows) - 1 for request_id, rows in attempts.items() if len(rows) > 1
        }
        return {
            "n_request_ids": len(attempts),
            "n_records_total": sum(len(v) for v in attempts.values()),
            "n_effective": len(effective),
            "n_request_ids_with_retries": len(superseded),
            "n_superseded_records": sum(superseded.values()),
            "max_attempts_for_one_request": max((len(v) for v in attempts.values()), default=0),
            "effective_by_status": {
                status: sum(1 for r in effective.values() if r.get("status") == status)
                for status in (STATUS_OK, STATUS_ERROR, STATUS_SKIPPED)
            },
            "attempts_by_status": {
                status: sum(
                    1 for rows in attempts.values() for r in rows if r.get("status") == status
                )
                for status in (STATUS_OK, STATUS_ERROR, STATUS_SKIPPED)
            },
            "unresolved_request_ids": sorted(
                k for k, v in effective.items() if v.get("status") != STATUS_OK
            ),
            "audit_log_is_append_only": True,
            "note": (
                "Every attempt is retained. `effective` selects the latest successful "
                "attempt per request_id (else the latest attempt); superseded records "
                "remain in the log and are never deleted or collapsed."
            ),
        }

    def check_provider_consistency(self, provider: str, model_id: str) -> None:
        """Refuse to mix providers or models within one response log."""
        for request_id, rows in self.history().items():
            row = rows[-1]
            if row.get("provider") != provider or row.get("model_id") != model_id:
                raise ProviderMismatchError(
                    f"{self.path} already contains rows from provider "
                    f"{row.get('provider')!r} / model {row.get('model_id')!r} "
                    f"(e.g. {request_id}), but this run uses {provider!r} / {model_id!r}. "
                    "Mixing engines in one log would make the corpus's provenance "
                    "unrecoverable. Use a new log path, or --force deliberately."
                )

    def pending(self, requests: Sequence[TranslationRequest], *, force: bool = False) -> list[TranslationRequest]:
        """Requests not already logged with status OK, unless ``force``."""
        if force:
            return list(requests)
        done = {rid for rid, row in self.effective().items() if row.get("status") == STATUS_OK}
        return [r for r in requests if r.request_id not in done]

    def append(self, responses: Iterable[TranslationResponse], *, force: bool = False) -> int:
        """Append responses. Refuses to duplicate an OK row without ``force``."""
        responses = list(responses)
        if not responses:
            return 0
        effective = self.effective()
        if not force:
            clashes = sorted(
                r.request_id
                for r in responses
                if effective.get(r.request_id, {}).get("status") == STATUS_OK
            )
            if clashes:
                raise ResponseLogError(
                    f"refusing to overwrite {len(clashes)} existing successful response(s) "
                    f"(e.g. {clashes[:3]}). A recorded hash must stay reproducible. "
                    "Pass --force to re-translate deliberately."
                )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            for response in sorted(responses, key=lambda r: r.request_id):
                fh.write(json.dumps(response.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
        return len(responses)


def correlate(
    requests: Sequence[TranslationRequest],
    responses: Sequence[TranslationResponse],
) -> dict[str, Any]:
    """Correlate requests to responses by request_id. Reports both directions."""
    request_ids = {r.request_id for r in requests}
    response_ids = {r.request_id for r in responses}
    mismatched_source = sorted(
        response.request_id
        for response in responses
        for request in requests
        if request.request_id == response.request_id
        and request.source_text_sha256 != response.source_text_sha256
    )
    return {
        "n_requests": len(request_ids),
        "n_responses": len(response_ids),
        "missing_responses": sorted(request_ids - response_ids),
        "orphan_responses": sorted(response_ids - request_ids),
        "source_hash_mismatches": mismatched_source,
        "fully_correlated": (
            request_ids == response_ids and not mismatched_source
        ),
    }


#: Whole words that mark a field name as credential-bearing. Matched against
#: name PARTS, not substrings: a substring test flags `tokenizer_id` (which is
#: model metadata, not a credential) while missing nothing real.
_SECRET_WORDS = frozenset(
    {"key", "apikey", "token", "secret", "password", "passwd", "credential",
     "credentials", "authorization", "auth", "cookie", "bearer", "session"}
)


def _name_parts(name: str) -> set[str]:
    return {part for part in re.split(r"[^a-z0-9]+", name.lower()) if part}


def assert_log_row_has_no_secrets(row: Mapping[str, Any]) -> None:
    """Raise if a response-log row carries anything credential-shaped."""
    offenders = sorted(k for k in row if _name_parts(k) & _SECRET_WORDS)
    if offenders:
        raise ProviderError(f"response log row carries credential-shaped field(s): {offenders}")
