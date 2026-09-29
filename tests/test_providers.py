"""Translation providers, response logging, and correlation.

No test here contacts a network translation API. The only provider with client
code is ``nllb_local`` (local weights); commercial providers are asserted to be
*refused*, not merely absent.
"""

from __future__ import annotations

import json

import pytest

from multilingual_safety.languages import LANGUAGE_CODES, PIVOT_LANGUAGE
from multilingual_safety.providers import (
    IMPLEMENTED_PROVIDERS,
    NETWORK_PROVIDERS,
    STATUS_ERROR,
    STATUS_OK,
    NLLBLocalProvider,
    ProviderError,
    ProviderMismatchError,
    ProviderUnavailableError,
    ResponseLog,
    ResponseLogError,
    TranslationRequest,
    TranslationResponse,
    assert_log_row_has_no_secrets,
    assert_no_network_provider,
    assert_no_seed_in_kwargs,
    assert_single_length_control,
    correlate,
    get_provider,
    make_request_id,
)

TARGETS = tuple(c for c in LANGUAGE_CODES if c != PIVOT_LANGUAGE)

CONFIG = {
    "provider": "nllb_local",
    "nllb_local_settings": {"model_id": "facebook/nllb-200-distilled-600M", "device": "cpu"},
    "providers": {
        "nllb_local": {
            "model": "facebook/nllb-200-distilled-600M",
            "flores_codes": {
                "ar": "arb_Arab", "ha": "hau_Latn", "hi": "hin_Deva",
                "ko": "kor_Hang", "sw": "swh_Latn", "yo": "yor_Latn",
            },
        }
    },
}


def make_request(behavior="beh_0123456789ab", language="ha", text="Write an article."):
    return TranslationRequest.create(
        behavior_id=behavior, target_language=language, source_text=text
    )


def make_response(request, *, run_id="run_20260816T120000Z_abcdef12", text="Rubuta labari.",
                  status=STATUS_OK, **overrides):
    from multilingual_safety.hashing import text_sha256

    payload = {
        "request_id": request.request_id,
        "behavior_id": request.behavior_id,
        "source_language": request.source_language,
        "target_language": request.target_language,
        "provider": "nllb_local",
        "model_id": "facebook/nllb-200-distilled-600M",
        "model_revision": "abc123",
        "tokenizer_id": "facebook/nllb-200-distilled-600M",
        "device": "cpu",
        "dtype": "float32",
        "generation_config": {"num_beams": 4, "do_sample": False},
        "response_text": text,
        "response_sha256": text_sha256(text) if text else None,
        "timestamp": "2026-08-16T12:00:00+00:00",
        "latency_seconds": 0.5,
        "status": status,
        "error_type": None if status == STATUS_OK else "Boom",
        "error_message": None if status == STATUS_OK else "boom",
        "run_id": run_id,
        "source_text_sha256": request.source_text_sha256,
    }
    payload.update(overrides)
    return TranslationResponse(**payload)


# --- deterministic request IDs --------------------------------------------
def test_request_id_is_deterministic():
    assert make_request_id("beh_abc", "yo") == "beh_abc:yo"
    assert make_request_id("beh_abc", "yo") == make_request_id("beh_abc", "yo")


def test_request_id_differs_per_language_and_behavior():
    ids = {make_request_id("beh_abc", lang) for lang in TARGETS}
    assert len(ids) == 6
    assert make_request_id("beh_abc", "yo") != make_request_id("beh_xyz", "yo")


@pytest.mark.parametrize("language", TARGETS)
def test_all_six_target_languages_build_requests(language):
    request = make_request(language=language)
    assert request.target_language == language
    assert request.source_language == PIVOT_LANGUAGE
    assert request.request_id.endswith(f":{language}")
    assert len(request.source_text_sha256) == 64


def test_target_languages_are_exactly_the_six_non_english():
    assert set(TARGETS) == {"ar", "ha", "yo", "sw", "hi", "ko"}


def test_english_is_never_a_translation_target():
    with pytest.raises(ValueError, match="into itself"):
        TranslationRequest.create(
            behavior_id="beh_abc", target_language="en", source_text="Write an article."
        )


def test_unknown_language_rejected():
    with pytest.raises(ValueError, match="unknown language code"):
        make_request(language="fr")


def test_empty_source_text_rejected():
    with pytest.raises(ValueError, match="empty source_text"):
        make_request(text="   ")


# --- response hashing ------------------------------------------------------
def test_response_hash_matches_text():
    response = make_response(make_request())
    from multilingual_safety.hashing import text_sha256

    assert response.response_sha256 == text_sha256("Rubuta labari.")
    assert len(response.response_sha256) == 64


def test_response_hash_mismatch_is_rejected():
    with pytest.raises(ValueError, match="response_sha256 does not match"):
        make_response(make_request(), response_sha256="0" * 64)


def test_ok_status_requires_text():
    with pytest.raises(ValueError, match="status OK with no response text"):
        make_response(make_request(), text=None, response_sha256=None)


def test_error_status_requires_error_type():
    request = make_request()
    with pytest.raises(ValueError, match="status ERROR with no error_type"):
        make_response(request, text=None, response_sha256=None, status=STATUS_ERROR,
                      error_type=None)


def test_response_log_row_carries_every_required_field():
    payload = make_response(make_request()).to_dict()
    for field in (
        "request_id", "behavior_id", "source_language", "target_language", "provider",
        "model_id", "model_revision", "generation_config", "response_text",
        "response_sha256", "timestamp", "latency_seconds", "status", "error_type",
        "error_message", "run_id",
    ):
        assert field in payload, f"response log is missing {field}"


def test_manifest_projection_drops_response_text():
    payload = make_response(make_request()).to_manifest_dict()
    assert "response_text" not in payload
    assert payload["response_n_chars"] == len("Rubuta labari.")
    assert payload["response_sha256"]


# --- no overwrite without --force -----------------------------------------
def test_append_then_reappend_requires_force(tmp_path):
    log = ResponseLog(tmp_path / "responses.jsonl")
    request = make_request()
    assert log.append([make_response(request)]) == 1

    with pytest.raises(ResponseLogError, match="refusing to overwrite"):
        log.append([make_response(request, text="different")])

    assert log.append([make_response(request, text="different")], force=True) == 1


def test_pending_skips_completed_unless_forced(tmp_path):
    log = ResponseLog(tmp_path / "responses.jsonl")
    a, b = make_request(language="ha"), make_request(language="yo")
    log.append([make_response(a)])
    assert [r.request_id for r in log.pending([a, b])] == [b.request_id]
    assert len(log.pending([a, b], force=True)) == 2


def test_failed_response_does_not_block_a_retry(tmp_path):
    """An ERROR row is data, but it must not prevent a later attempt."""
    log = ResponseLog(tmp_path / "responses.jsonl")
    request = make_request()
    log.append([make_response(request, text=None, response_sha256=None, status=STATUS_ERROR)])
    assert len(log.pending([request])) == 1
    log.append([make_response(request)])  # no force needed


def test_malformed_log_line_raises_rather_than_skipping(tmp_path):
    path = tmp_path / "responses.jsonl"
    log = ResponseLog(path)
    log.append([make_response(make_request())])
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    with pytest.raises(ResponseLogError, match="malformed JSON"):
        log.existing()


# --- provider mismatch -----------------------------------------------------
def test_provider_mismatch_is_detected(tmp_path):
    log = ResponseLog(tmp_path / "responses.jsonl")
    log.append([make_response(make_request())])
    log.check_provider_consistency("nllb_local", "facebook/nllb-200-distilled-600M")
    with pytest.raises(ProviderMismatchError, match="already contains rows from provider"):
        log.check_provider_consistency("deepl", "deepl-v2")


def test_model_mismatch_is_detected(tmp_path):
    log = ResponseLog(tmp_path / "responses.jsonl")
    log.append([make_response(make_request())])
    with pytest.raises(ProviderMismatchError, match="model"):
        log.check_provider_consistency("nllb_local", "facebook/nllb-200-1.3B")


# --- explicit selection, no fallback ---------------------------------------
def test_no_provider_selected_is_an_error_not_a_default():
    with pytest.raises(ProviderUnavailableError, match="no provider selected"):
        get_provider(None, CONFIG)


@pytest.mark.parametrize("name", sorted(NETWORK_PROVIDERS))
def test_network_providers_are_refused(name):
    """No commercial API client exists, and asking for one fails loudly."""
    with pytest.raises(ProviderUnavailableError, match="third-party translation API"):
        get_provider(name, CONFIG)
    with pytest.raises(ProviderUnavailableError):
        assert_no_network_provider(name)


def test_only_nllb_local_is_implemented():
    assert {"nllb_local"} == IMPLEMENTED_PROVIDERS
    assert "llm" in NETWORK_PROVIDERS
    provider = get_provider("nllb_local", CONFIG)
    assert isinstance(provider, NLLBLocalProvider)
    assert provider.api_key_env is None


def test_unknown_provider_rejected():
    with pytest.raises(ProviderUnavailableError, match="unknown provider"):
        get_provider("mystery_mt", CONFIG)


# --- missing local model ---------------------------------------------------
def test_missing_local_model_fails_loudly_without_fallback():
    provider = NLLBLocalProvider(
        {"model_id": "facebook/definitely-not-a-real-model-xyz",
         "device": "cpu", "local_files_only": True}
    )
    with pytest.raises(ProviderUnavailableError) as exc:
        provider.preflight()
    message = str(exc.value)
    assert "could not load" in message
    assert "Refusing to fall back" in message


def test_translate_before_preflight_is_refused():
    provider = NLLBLocalProvider({"model_id": "x", "device": "cpu"})
    with pytest.raises(ProviderUnavailableError, match="preflight"):
        provider.translate(make_request(), run_id="run_x")


def test_cuda_requested_but_unavailable_does_not_silently_downgrade():
    pytest.importorskip("torch")
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA present; the downgrade path cannot trigger")
    provider = NLLBLocalProvider({"model_id": "x", "device": "cuda"})
    with pytest.raises(ProviderUnavailableError, match="Refusing to silently downgrade"):
        provider._resolve_device()


# --- correlation -----------------------------------------------------------
def test_full_correlation():
    requests = [make_request(language=lang) for lang in TARGETS]
    responses = [make_response(r) for r in requests]
    report = correlate(requests, responses)
    assert report["fully_correlated"] is True
    assert report["n_requests"] == report["n_responses"] == 6
    assert report["missing_responses"] == [] and report["orphan_responses"] == []


def test_missing_response_is_reported():
    requests = [make_request(language=lang) for lang in TARGETS]
    report = correlate(requests, [make_response(r) for r in requests[:4]])
    assert report["fully_correlated"] is False
    assert len(report["missing_responses"]) == 2


def test_orphan_response_is_reported():
    requests = [make_request(language="ha")]
    responses = [make_response(make_request(language=lang)) for lang in ("ha", "yo")]
    report = correlate(requests, responses)
    assert report["orphan_responses"] == ["beh_0123456789ab:yo"]


def test_source_hash_mismatch_breaks_correlation():
    """A response whose source hash differs was translated from other text."""
    request = make_request()
    response = make_response(request, source_text_sha256="f" * 64)
    report = correlate([request], [response])
    assert report["source_hash_mismatches"] == [request.request_id]
    assert report["fully_correlated"] is False


# --- secrets ---------------------------------------------------------------
def test_response_log_row_has_no_credential_fields():
    assert_log_row_has_no_secrets(make_response(make_request()).to_dict())


@pytest.mark.parametrize("field", ["api_key", "auth_token", "Authorization", "password"])
def test_credential_shaped_field_in_a_log_row_is_rejected(field):
    row = make_response(make_request()).to_dict()
    row[field] = "whatever"
    with pytest.raises(Exception, match="credential-shaped"):
        assert_log_row_has_no_secrets(row)


def test_written_response_log_contains_no_secrets(tmp_path):
    from multilingual_safety.secrets import scan_file

    path = tmp_path / "responses.jsonl"
    ResponseLog(path).append([make_response(r) for r in
                              [make_request(language=lang) for lang in TARGETS]])
    assert scan_file(path) == []
    for line in path.read_text(encoding="utf-8").splitlines():
        assert_log_row_has_no_secrets(json.loads(line))


def test_log_is_written_in_deterministic_request_id_order(tmp_path):
    path = tmp_path / "responses.jsonl"
    requests = [make_request(language=lang) for lang in reversed(TARGETS)]
    ResponseLog(path).append([make_response(r) for r in requests])
    ids = [json.loads(line)["request_id"] for line in path.read_text().splitlines()]
    assert ids == sorted(ids)


# --- pilot subset selection (scripts/04_translate_pilot.py) ----------------
def _load_script():
    import importlib.util

    from multilingual_safety.paths import REPO_ROOT

    spec = importlib.util.spec_from_file_location(
        "_pilot", REPO_ROOT / "scripts" / "04_translate_pilot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows():
    from multilingual_safety.translation import make_english_row, make_pending_row

    rows = []
    for index in range(6):
        common = {
            "behavior_id": f"beh_{index:012d}",
            "source": "jailbreakbench",
            "category": "Physical harm",
            "harm_label": "harmful" if index % 2 == 0 else "benign",
            "source_text": f"Write article number {index}.",
            "source_item_id": str(index),
        }
        rows.append(make_english_row(**common))
        rows.extend(make_pending_row(language=lang, **common) for lang in TARGETS)
    return rows


def test_pilot_subset_is_exactly_three_behaviors_times_six_languages():
    subset = _load_script().select_pilot_subset(_rows(), 3)
    assert len(subset) == 18
    assert len({r.behavior_id for r in subset}) == 3
    assert {r.language for r in subset} == set(TARGETS)


def test_pilot_subset_never_includes_english():
    subset = _load_script().select_pilot_subset(_rows(), 3)
    assert all(r.language != PIVOT_LANGUAGE for r in subset)


def test_pilot_subset_covers_both_arms():
    """A naive sort would return three benign rows; the arms must be balanced."""
    subset = _load_script().select_pilot_subset(_rows(), 3)
    labels = {r.harm_label for r in subset}
    assert labels == {"harmful", "benign"}


def test_pilot_subset_is_deterministic_and_ordered():
    module = _load_script()
    first = module.select_pilot_subset(_rows(), 3)
    second = module.select_pilot_subset(list(reversed(_rows())), 3)
    key = lambda rows: [(r.behavior_id, r.language) for r in rows]  # noqa: E731
    assert key(first) == key(second) == sorted(key(first))


def test_pilot_subset_zero_is_empty():
    assert _load_script().select_pilot_subset(_rows(), 0) == []


def test_tracked_reports_contain_no_secrets_and_no_response_text():
    from multilingual_safety.paths import MANIFESTS_DIR, RESULTS_DIR
    from multilingual_safety.secrets import assert_no_secrets

    tracked = [
        RESULTS_DIR / "translation_pilot_inventory.json",
        RESULTS_DIR / "translation_pilot_validation.json",
        MANIFESTS_DIR / "translation_pilot_manifest.json",
    ]
    existing = [p for p in tracked if p.exists()]
    if not existing:
        pytest.skip("Stage 1 artifacts not present")
    assert_no_secrets(existing)
    for path in existing:
        blob = path.read_text(encoding="utf-8")
        assert '"response_text"' not in blob
        assert '"source_text"' not in blob


# --- CLI provider override is reflected as the effective provider ----------
def _resolve(override=None, configured=None):
    module = _load_script()
    config = dict(CONFIG)
    config["provider"] = configured
    config["allowed_providers"] = ["nllb_local"]
    return module.resolve_backend(config, override=override)


def test_cli_override_is_reported_as_effective_provider():
    """`--provider nllb_local` with `provider: null` in config."""
    backend = _resolve(override="nllb_local", configured=None)
    assert backend["configured_provider"] is None
    assert backend["effective_provider"] == "nllb_local"
    assert backend["provider_source"] == "cli_override"
    assert isinstance(backend["backend_usable"], bool)


def test_override_does_not_rewrite_configured_provider():
    """The config value must stay visible, so a run stays reproducible."""
    backend = _resolve(override="nllb_local", configured=None)
    assert backend["configured_provider"] is None, "override must not mutate configured_provider"


def test_without_override_effective_equals_configured():
    assert _resolve(override=None, configured=None)["effective_provider"] is None
    assert _resolve(override=None, configured=None)["provider_source"] == "none"

    backend = _resolve(override=None, configured="nllb_local")
    assert backend["configured_provider"] == backend["effective_provider"] == "nllb_local"
    assert backend["provider_source"] == "config"


def test_backend_usable_is_present_and_boolean_in_every_path():
    for override, configured in [(None, None), (None, "nllb_local"), ("nllb_local", None)]:
        backend = _resolve(override=override, configured=configured)
        assert "backend_usable" in backend
        assert isinstance(backend["backend_usable"], bool)
        assert backend["usable"] == backend["backend_usable"], "alias must stay in sync"


def test_override_to_a_network_provider_fails_closed():
    """An override must never enable a commercial API, and never fall back."""
    backend = _resolve(override="deepl", configured=None)
    assert backend["effective_provider"] == "deepl"
    assert backend["backend_usable"] is False
    assert "third-party translation API" in backend["reason"]


def test_override_to_an_unknown_provider_fails_closed():
    backend = _resolve(override="mystery_mt", configured=None)
    assert backend["effective_provider"] == "mystery_mt"
    assert backend["backend_usable"] is False
    assert "unknown provider" in backend["reason"]


def test_probe_reports_unusable_without_dependencies(monkeypatch):
    """Missing torch must read as unusable, never as a silent fallback."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"torch", "transformers"}:
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    report = NLLBLocalProvider({"model_id": "facebook/nllb-200-distilled-600M"}).probe()
    assert report["usable"] is False
    assert report["dependencies_available"] is False
    assert "not importable" in report["reason"]


def test_probe_never_raises_and_loads_no_weights():
    report = NLLBLocalProvider(
        {"model_id": "facebook/definitely-not-real-xyz", "local_files_only": True}
    ).probe()
    assert report["usable"] is False
    assert report["checked_by"] == "probe (no weights loaded)"


def test_preflight_remains_the_authority_and_still_fails_closed():
    """A probe is a prediction; preflight still raises rather than degrading."""
    provider = NLLBLocalProvider(
        {"model_id": "facebook/definitely-not-real-xyz", "device": "cpu",
         "local_files_only": True}
    )
    with pytest.raises(ProviderUnavailableError, match="Refusing to fall back"):
        provider.preflight()


# --- generation kwargs must not leak non-generate keys ---------------------
def test_seed_is_recorded_but_not_passed_to_generate():
    """Regression: `seed` in generate() raises and failed all 18 pilot rows."""
    provider = NLLBLocalProvider(
        {"model_id": "x", "generation": {"num_beams": 4, "do_sample": False, "seed": 123}}
    )
    assert provider.seed == 123
    assert "seed" in provider.generation, "seed must stay in recorded provenance"
    assert "seed" not in provider.generate_kwargs, "seed must not reach generate()"
    assert provider.generate_kwargs == {"num_beams": 4, "do_sample": False}


def test_generation_config_without_seed_is_unchanged():
    provider = NLLBLocalProvider({"model_id": "x", "generation": {"num_beams": 2}})
    assert provider.seed is None
    assert provider.generate_kwargs == {"num_beams": 2}


def test_recorded_generation_config_still_includes_seed():
    provider = NLLBLocalProvider(
        {"model_id": "x", "generation": {"num_beams": 4, "seed": 20260816}}
    )
    assert provider.describe()["generation_config"]["seed"] == 20260816


# --- seed must never reach generate() (regression: 18x ValueError) ---------
NESTED_SEED_CONFIGS = [
    {"num_beams": 4, "seed": 123},
    {"num_beams": 4, "generation_config": {"seed": 123}},
    {"num_beams": 4, "model_kwargs": {"seed": 123}},
    {"num_beams": 4, "generate_kwargs": {"seed": 123}},
    {"num_beams": 4, "random_seed": 123},
    {"num_beams": 4, "torch_seed": 123},
]


@pytest.mark.parametrize("generation", NESTED_SEED_CONFIGS)
def test_seed_is_stripped_at_every_nesting_depth(generation):
    provider = NLLBLocalProvider({"model_id": "x", "generation": generation})
    assert provider.seed == 123, "seed must still be discovered for provenance"
    flat = json.dumps(provider.generate_kwargs)
    assert "seed" not in flat, f"seed leaked into generate kwargs: {provider.generate_kwargs}"


def test_recorded_generation_config_retains_seed_for_provenance():
    provider = NLLBLocalProvider(
        {"model_id": "x", "generation": {"num_beams": 4, "seed": 20260816}}
    )
    assert provider.describe()["generation_config"]["seed"] == 20260816
    assert "seed" not in provider.generate_kwargs


def test_assert_no_seed_in_kwargs_guard():
    assert_no_seed_in_kwargs({"num_beams": 4})
    with pytest.raises(ProviderError, match="seed-like kwarg"):
        assert_no_seed_in_kwargs({"num_beams": 4, "seed": 1})
    with pytest.raises(ProviderError, match="seed-like kwarg"):
        assert_no_seed_in_kwargs({"generation_config": {"seed": 1}})


def test_apply_seed_calls_torch_manual_seed_and_cuda_variant(monkeypatch):
    """CUDA keeps separate generator state; both calls are required."""
    torch = pytest.importorskip("torch")
    calls = []
    monkeypatch.setattr(torch, "manual_seed", lambda s: calls.append(("cpu", s)))
    monkeypatch.setattr(
        torch.cuda, "manual_seed_all", lambda s: calls.append(("cuda", s)), raising=False
    )

    provider = NLLBLocalProvider({"model_id": "x", "generation": {"seed": 77}})
    provider._device = "cpu"
    assert provider.apply_seed() == 77
    assert calls == [("cpu", 77)], "no CUDA seeding on a CPU device"

    calls.clear()
    provider._device = "cuda"
    provider.apply_seed()
    assert calls == [("cpu", 77), ("cuda", 77)]

    calls.clear()
    NLLBLocalProvider({"model_id": "x", "generation": {}}).apply_seed()
    assert calls == [], "no seed configured => no seeding call"


class _RecordingModel:
    """Captures the kwargs generate() is actually called with."""

    def __init__(self):
        self.captured = None

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.captured = kwargs
        return [[0, 1, 2]]


def test_generate_is_never_called_with_seed(monkeypatch):
    """Wraps the real call path and inspects the kwargs generate() receives."""
    pytest.importorskip("torch")

    provider = NLLBLocalProvider(
        {"model_id": "x", "device": "cpu",
         "generation": {"num_beams": 4, "do_sample": False, "seed": 20260816},
         "flores_codes": {"sw": "swh_Latn"}}
    )
    model = _RecordingModel()
    provider._model = model
    provider._device = "cpu"

    class _Tok:
        src_lang = "eng_Latn"

        def __init__(self):
            self.lang_code_to_id = {"swh_Latn": 9, "eng_Latn": 1}

        def __call__(self, text, return_tensors=None):
            class _Enc(dict):
                def to(self, device):
                    return self
            return _Enc(input_ids=[[1, 2, 3]])

        def batch_decode(self, ids, skip_special_tokens=True):
            return ["Maktaba hufungua saa tisa asubuhi."]

    provider._tokenizer = _Tok()
    response = provider.translate(
        TranslationRequest.create(
            behavior_id="beh_recording00", target_language="sw",
            source_text="The library opens at nine in the morning.",
        ),
        run_id="run_20260816T000000Z_record00",
    )

    assert model.captured is not None, "generate() was never called"
    assert "seed" not in model.captured, f"seed reached generate(): {sorted(model.captured)}"
    assert "num_beams" in model.captured
    assert response.status == STATUS_OK
    # Provenance still records it.
    assert response.generation_config["seed"] == 20260816


@pytest.mark.slow
def test_one_row_local_translation_returns_ok():
    """End-to-end on the REAL provider with a neutral, non-pilot sentence."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import yaml

    from multilingual_safety.paths import CONFIGS_DIR

    config = yaml.safe_load((CONFIGS_DIR / "translation.yaml").read_text(encoding="utf-8"))
    provider = get_provider("nllb_local", config)
    probe = provider.probe()
    if not probe.get("usable") or not probe.get("weights_cached"):
        pytest.skip(f"nllb_local not usable here: {probe.get('reason')}")

    provider.preflight()
    assert "seed" not in provider.generate_kwargs
    response = provider.translate(
        TranslationRequest.create(
            behavior_id="beh_smoketest00", target_language="sw",
            source_text="The library opens at nine in the morning.",
        ),
        run_id="run_20260816T000000Z_smoketest",
    )
    assert response.status == STATUS_OK, f"{response.error_type}: {response.error_message}"
    assert response.response_text and response.response_text.strip()
    assert response.response_sha256 and len(response.response_sha256) == 64


# --- exactly one length control (regression: max_length=200 vs max_new_tokens)
def test_generate_kwargs_use_max_new_tokens_and_never_max_length():
    provider = NLLBLocalProvider(
        {"model_id": "x",
         "generation": {"num_beams": 4, "do_sample": False, "max_new_tokens": 256}}
    )
    assert provider.generate_kwargs["max_new_tokens"] == 256
    assert "max_length" not in provider.generate_kwargs
    assert "min_length" not in provider.generate_kwargs


def test_assert_single_length_control_guard():
    assert_single_length_control({"max_new_tokens": 256, "num_beams": 4})
    for offender in ("max_length", "min_length"):
        with pytest.raises(ProviderError, match="single control"):
            assert_single_length_control({"max_new_tokens": 256, offender: 200})


class _ConfigStub:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _ModelWithBakedInLength:
    def __init__(self):
        self.generation_config = _ConfigStub(max_length=200, min_length=None)


def test_model_baked_in_max_length_is_neutralized():
    """NLLB ships max_length=200 in its own generation_config.json."""
    provider = NLLBLocalProvider({"model_id": "x"})
    provider._model = _ModelWithBakedInLength()
    removed = provider.neutralize_model_length_controls()
    assert removed == {"max_length": 200}
    assert provider._model.generation_config.max_length is None


def test_neutralize_is_idempotent_and_safe_without_a_model():
    provider = NLLBLocalProvider({"model_id": "x"})
    assert provider.neutralize_model_length_controls() == {}   # no model loaded
    provider._model = _ModelWithBakedInLength()
    assert provider.neutralize_model_length_controls() == {"max_length": 200}
    assert provider.neutralize_model_length_controls() == {}   # already cleared


def test_generate_is_never_called_with_max_length(monkeypatch):
    """Inspects the kwargs generate() actually receives."""
    pytest.importorskip("torch")

    provider = NLLBLocalProvider(
        {"model_id": "x", "device": "cpu",
         "generation": {"num_beams": 4, "max_new_tokens": 256, "seed": 1},
         "flores_codes": {"sw": "swh_Latn"}}
    )
    model = _RecordingModel()
    model.generation_config = _ConfigStub(max_length=200, min_length=None)
    provider._model = model
    provider._device = "cpu"
    provider.neutralize_model_length_controls()

    class _Tok:
        src_lang = "eng_Latn"

        def __init__(self):
            self.lang_code_to_id = {"swh_Latn": 9, "eng_Latn": 1}

        def __call__(self, text, return_tensors=None):
            class _Enc(dict):
                def to(self, device):
                    return self
            return _Enc(input_ids=[[1, 2, 3]])

        def batch_decode(self, ids, skip_special_tokens=True):
            return ["Maktaba hufungua saa tisa asubuhi."]

    provider._tokenizer = _Tok()
    response = provider.translate(
        TranslationRequest.create(
            behavior_id="beh_lengthtest0", target_language="sw",
            source_text="The library opens at nine in the morning.",
        ),
        run_id="run_20260816T000000Z_length00",
    )

    assert model.captured is not None
    assert "max_new_tokens" in model.captured, "the single length control must be present"
    assert model.captured["max_new_tokens"] == 256
    assert "max_length" not in model.captured, f"max_length leaked: {sorted(model.captured)}"
    assert "min_length" not in model.captured
    assert response.status == STATUS_OK
    assert model.generation_config.max_length is None


@pytest.mark.slow
def test_real_generation_emits_no_length_conflict_warning():
    """End-to-end: the transformers warning must not be logged at all."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import io
    import logging

    import yaml

    from multilingual_safety.paths import CONFIGS_DIR

    config = yaml.safe_load((CONFIGS_DIR / "translation.yaml").read_text(encoding="utf-8"))
    provider = get_provider("nllb_local", config)
    if not provider.probe().get("weights_cached"):
        pytest.skip("nllb_local weights not cached here")

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("transformers")
    logger.addHandler(handler)
    try:
        metadata = provider.preflight()
        assert metadata["model_default_length_controls_removed"] == {"max_length": 200}
        assert metadata["length_control"] == "max_new_tokens"
        response = provider.translate(
            TranslationRequest.create(
                behavior_id="beh_smoketest00", target_language="sw",
                source_text="The library opens at nine in the morning.",
            ),
            run_id="run_20260816T000000Z_smoketest",
        )
    finally:
        logger.removeHandler(handler)

    assert response.status == STATUS_OK
    assert "seem to have been set" not in stream.getvalue()


# --- retry history and effective-response supersession ---------------------
def test_history_preserves_every_attempt(tmp_path):
    """The audit log is append-only: retries never collapse earlier records."""
    log = ResponseLog(tmp_path / "responses.jsonl")
    request = make_request()
    log.append([make_response(request, text=None, response_sha256=None, status=STATUS_ERROR)])
    log.append([make_response(request, text="Rubuta labari.")])

    history = log.history()
    assert list(history) == [request.request_id]
    assert [a["status"] for a in history[request.request_id]] == [STATUS_ERROR, STATUS_OK]
    assert len(history[request.request_id]) == 2, "both attempts must survive"


def test_exactly_one_effective_response_per_request_id(tmp_path):
    log = ResponseLog(tmp_path / "responses.jsonl")
    requests = [make_request(language=lang) for lang in TARGETS]
    log.append([make_response(r, text=None, response_sha256=None, status=STATUS_ERROR)
                for r in requests])
    log.append([make_response(r) for r in requests])

    effective = log.effective()
    assert len(effective) == len(requests) == 6
    assert all(row["status"] == STATUS_OK for row in effective.values())
    assert sum(len(v) for v in log.history().values()) == 12, "all 12 records retained"


def test_later_ok_supersedes_earlier_error(tmp_path):
    log = ResponseLog(tmp_path / "responses.jsonl")
    request = make_request()
    log.append([make_response(request, text=None, response_sha256=None, status=STATUS_ERROR)])
    log.append([make_response(request, text="Rubuta labari.")])
    assert log.effective()[request.request_id]["status"] == STATUS_OK
    assert log.effective()[request.request_id]["response_text"] == "Rubuta labari."


def test_later_error_does_not_retract_an_earlier_ok(tmp_path):
    """An apparatus failure must not discard a translation already produced."""
    log = ResponseLog(tmp_path / "responses.jsonl")
    request = make_request()
    log.append([make_response(request, text="Rubuta labari.")])
    log.append([make_response(request, text=None, response_sha256=None,
                              status=STATUS_ERROR)], force=True)

    effective = log.effective()[request.request_id]
    assert effective["status"] == STATUS_OK
    assert effective["response_text"] == "Rubuta labari."
    assert len(log.history()[request.request_id]) == 2, "the ERROR is still audited"


def test_latest_ok_wins_among_several_successes(tmp_path):
    log = ResponseLog(tmp_path / "responses.jsonl")
    request = make_request()
    log.append([make_response(request, text="first")])
    log.append([make_response(request, text="second")], force=True)
    assert log.effective()[request.request_id]["response_text"] == "second"
    assert len(log.history()[request.request_id]) == 2


def test_all_errors_leaves_latest_error_effective(tmp_path):
    log = ResponseLog(tmp_path / "responses.jsonl")
    request = make_request()
    for message in ("first failure", "second failure"):
        log.append([make_response(request, text=None, response_sha256=None,
                                  status=STATUS_ERROR, error_message=message)])
    effective = log.effective()[request.request_id]
    assert effective["status"] == STATUS_ERROR
    assert effective["error_message"] == "second failure"
    assert log.retry_summary()["unresolved_request_ids"] == [request.request_id]


def test_superseded_error_does_not_force_a_retranslation(tmp_path):
    """pending() must consult the effective response, not the last line."""
    log = ResponseLog(tmp_path / "responses.jsonl")
    request = make_request()
    log.append([make_response(request, text="Rubuta labari.")])
    log.append([make_response(request, text=None, response_sha256=None,
                              status=STATUS_ERROR)], force=True)
    assert log.pending([request]) == [], "an already-translated row must not re-run"


def test_retry_summary_reports_supersession_without_hiding_it(tmp_path):
    log = ResponseLog(tmp_path / "responses.jsonl")
    requests = [make_request(language=lang) for lang in TARGETS]
    log.append([make_response(r, text=None, response_sha256=None, status=STATUS_ERROR)
                for r in requests])
    log.append([make_response(r) for r in requests])

    summary = log.retry_summary()
    assert summary["n_request_ids"] == 6
    assert summary["n_records_total"] == 12
    assert summary["n_effective"] == 6
    assert summary["n_superseded_records"] == 6
    assert summary["max_attempts_for_one_request"] == 2
    assert summary["effective_by_status"][STATUS_OK] == 6
    assert summary["attempts_by_status"][STATUS_ERROR] == 6, "errors stay visible"
    assert summary["audit_log_is_append_only"] is True
    assert summary["unresolved_request_ids"] == []


def test_history_is_empty_for_a_missing_log(tmp_path):
    log = ResponseLog(tmp_path / "absent.jsonl")
    assert log.history() == {}
    assert log.effective() == {}
    assert log.retry_summary()["n_request_ids"] == 0


def test_apply_responses_uses_the_effective_response(tmp_path):
    """Rows must reflect the effective OK even when an ERROR was logged later."""
    from multilingual_safety.translation import TranslationStatus, make_pending_row

    module = _load_script()
    path = tmp_path / "responses.jsonl"
    log = ResponseLog(path)
    request = make_request(behavior="beh_0123456789ab", language="ha")
    log.append([make_response(request, text="Rubuta labari.")])
    log.append([make_response(request, text=None, response_sha256=None,
                              status=STATUS_ERROR)], force=True)

    row = make_pending_row(
        behavior_id="beh_0123456789ab", source="jailbreakbench", category="Malware/Hacking",
        harm_label="harmful", source_text="Write an article.", language="ha",
    )
    applied = module.apply_responses([row], path)[0]
    assert applied.translated_text == "Rubuta labari."
    assert applied.translation_status == TranslationStatus.MACHINE_TRANSLATED_UNCHECKED.value
    assert applied.review_required is True, "review is never cleared by a retry"
    assert applied.reviewer_id is None


def test_cuda_is_used_when_available_and_never_silently_downgraded():
    """Counterpart to the no-CUDA test: with CUDA present, 'cuda' must resolve."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA in this environment")
    assert NLLBLocalProvider({"model_id": "x", "device": "cuda"})._resolve_device() == "cuda"
    assert NLLBLocalProvider({"model_id": "x", "device": "auto"})._resolve_device() == "cuda"
    # An explicit cpu request is still honoured; only 'auto' is device-dependent.
    assert NLLBLocalProvider({"model_id": "x", "device": "cpu"})._resolve_device() == "cpu"


def test_configured_device_is_explicit_not_auto():
    """`auto` would silently fall back to CPU on a GPU-less machine."""
    import yaml

    from multilingual_safety.paths import CONFIGS_DIR

    config = yaml.safe_load((CONFIGS_DIR / "translation.yaml").read_text(encoding="utf-8"))
    device = config["nllb_local_settings"]["device"]
    assert device in {"cuda", "cpu"}, f"device must be explicit, got {device!r}"


def test_assert_single_device_blocks_mixed_provenance(tmp_path):
    """A corpus must not silently contain both CPU- and CUDA-generated rows."""
    import importlib.util

    from multilingual_safety.paths import REPO_ROOT

    spec = importlib.util.spec_from_file_location(
        "_full", REPO_ROOT / "scripts" / "11_translate_full_set.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    log = ResponseLog(tmp_path / "responses.jsonl")
    log.append([make_response(make_request(language="ha"), device="cuda")])
    module.assert_single_device(log, "cuda")          # same device: fine
    with pytest.raises(SystemExit, match="Mixing devices"):
        module.assert_single_device(log, "cpu")

    empty = ResponseLog(tmp_path / "empty.jsonl")
    module.assert_single_device(empty, "cuda")        # empty log: nothing to mix
