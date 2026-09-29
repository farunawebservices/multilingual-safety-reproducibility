"""Dataset-v2 translation provenance: the right seed, the whole pilot.

Two defects motivated this file.

**Seed.** ``configs/dataset_v2_rebuild.yaml`` declares ``translation.seed:
20260818``, but the v2 pilot ran against ``configs/translation.yaml``, whose seed
is v1's ``20260816``. The rows were produced under one seed while the protocol
recorded another, and nothing caught it: the seed reached only the inventory,
nested inside the provider metadata, and never the manifest.

**Subset.** ``pilot.behaviors_per_language: 3`` in the v1 config translated 3 of
the 20 selected behaviors -- 18 rows done, 102 left pending.

``configs/translation_v2.yaml`` fixes both. It is a copy of the v1 config
differing in exactly those two scalars, and
:func:`test_v2_matches_v1_except_seed_and_pilot_size` holds it to that: an
unmirrored edit to either file fails here rather than quietly splitting the two
pipelines.

No model is loaded and nothing is translated anywhere in this module.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

import pytest
import yaml

from multilingual_safety.hashing import text_sha256
from multilingual_safety.languages import LANGUAGE_CODES, PIVOT_LANGUAGE
from multilingual_safety.paths import CONFIGS_DIR, REPO_ROOT
from multilingual_safety.providers import NLLBLocalProvider, extract_seed
from multilingual_safety.translation import make_english_row, make_pending_row

V1_CONFIG = CONFIGS_DIR / "translation.yaml"
V2_CONFIG = CONFIGS_DIR / "translation_v2.yaml"
PROTOCOL = CONFIGS_DIR / "dataset_v2_rebuild.yaml"
SCRIPT = REPO_ROOT / "scripts" / "04_translate_pilot.py"

V1_SEED = 20260816
V2_SEED = 20260818

#: The only leaf keys permitted to differ between the two configs.
PERMITTED_DIFFERENCES = {
    "nllb_local_settings.generation.seed",
    "pilot.behaviors_per_language",
}

# Invented placeholder, deliberately not drawn from any benchmark.
FIXTURE_TEXT = "Describe how a public library organises its catalogue."


def _load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _leaves(node, prefix=""):
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            out.update(_leaves(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: node}


def _script_module():
    """Load scripts/04_translate_pilot.py (a digit-leading filename)."""
    spec = importlib.util.spec_from_file_location("_pilot_writer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
def test_v2_config_declares_the_v2_seed():
    assert extract_seed(_load(V2_CONFIG)["nllb_local_settings"]) == V2_SEED


def test_v1_config_seed_is_unchanged():
    """v1 default behavior is preserved when the v2 config is not used."""
    assert extract_seed(_load(V1_CONFIG)["nllb_local_settings"]) == V1_SEED


def test_v2_seed_matches_the_protocol_declaration():
    """The mismatch that started this: protocol and config must agree."""
    assert _load(PROTOCOL)["translation"]["seed"] == V2_SEED
    assert extract_seed(_load(V2_CONFIG)["nllb_local_settings"]) == V2_SEED


def test_provider_would_apply_the_v2_seed():
    settings = dict(_load(V2_CONFIG)["nllb_local_settings"])
    provider = NLLBLocalProvider(settings)
    assert provider.seed == V2_SEED
    # Recorded for provenance, never handed to generate().
    assert "seed" not in provider.generate_kwargs


def test_provider_would_apply_the_v1_seed():
    provider = NLLBLocalProvider(dict(_load(V1_CONFIG)["nllb_local_settings"]))
    assert provider.seed == V1_SEED


# ---------------------------------------------------------------------------
# The two configs may differ in exactly two places
# ---------------------------------------------------------------------------
def test_v2_matches_v1_except_seed_and_pilot_size():
    v1, v2 = _leaves(_load(V1_CONFIG)), _leaves(_load(V2_CONFIG))
    assert set(v1) == set(v2), "config shapes diverged"
    differing = {k for k in v1 if v1[k] != v2[k]}
    assert differing == PERMITTED_DIFFERENCES, (
        f"unexpected divergence between the v1 and v2 translation configs: "
        f"{sorted(differing - PERMITTED_DIFFERENCES)}"
    )


@pytest.mark.parametrize(
    "key",
    [
        "nllb_local_settings.model_id",
        "nllb_local_settings.revision",
        "nllb_local_settings.device",
        "nllb_local_settings.dtype",
        "nllb_local_settings.generation.num_beams",
        "nllb_local_settings.generation.do_sample",
        "nllb_local_settings.generation.max_new_tokens",
        "nllb_local_settings.generation.no_repeat_ngram_size",
        "pilot.translate_english",
    ],
)
def test_revision_beams_and_decoding_are_untouched(key):
    """Explicitly out of scope for this change; asserted, not assumed."""
    assert _leaves(_load(V1_CONFIG))[key] == _leaves(_load(V2_CONFIG))[key]


def test_flores_codes_are_identical():
    v1 = _load(V1_CONFIG)["providers"]["nllb_local"]["flores_codes"]
    assert v1 == _load(V2_CONFIG)["providers"]["nllb_local"]["flores_codes"]


def test_v2_still_requires_an_explicit_provider():
    """Fail-closed is preserved: no bare invocation can start generating."""
    assert _load(V2_CONFIG)["provider"] is None


# ---------------------------------------------------------------------------
# Pilot subset size
# ---------------------------------------------------------------------------
def test_v2_pilot_covers_all_twenty_behaviors():
    assert _load(V2_CONFIG)["pilot"]["behaviors_per_language"] == 20


def test_v1_pilot_size_is_unchanged():
    assert _load(V1_CONFIG)["pilot"]["behaviors_per_language"] == 3


def _twenty_behavior_rows():
    rows = []
    for index in range(20):
        common = {
            "behavior_id": f"beh_{index:012x}",
            "source": "jailbreakbench",
            "category": "Disinformation",
            "harm_label": "harmful" if index % 2 else "benign",
            "source_text": f"{FIXTURE_TEXT} ({index})",
            "source_item_id": str(index),
        }
        rows.append(make_english_row(**common))
        rows += [
            make_pending_row(language=code, **common)
            for code in LANGUAGE_CODES
            if code != PIVOT_LANGUAGE
        ]
    return rows


def test_twenty_behaviors_yield_140_rows_and_120_translations():
    """20 x 7 = 140 rows; 20 x 6 = 120 non-English requests."""
    rows = _twenty_behavior_rows()
    assert len(rows) == 140

    subset = _script_module().select_pilot_subset(rows, 20)
    assert len(subset) == 120
    assert len({r.behavior_id for r in subset}) == 20
    assert {r.language for r in subset} == set(LANGUAGE_CODES) - {PIVOT_LANGUAGE}
    # English is canonical source text and is never translated.
    assert not [r for r in subset if r.language == PIVOT_LANGUAGE]


def test_v1_limit_of_three_still_yields_eighteen_rows():
    subset = _script_module().select_pilot_subset(_twenty_behavior_rows(), 3)
    assert len(subset) == 18
    assert len({r.behavior_id for r in subset}) == 3


# ---------------------------------------------------------------------------
# effective_seed
# ---------------------------------------------------------------------------
def test_effective_seed_prefers_what_the_provider_reported():
    module = _script_module()
    config = {"nllb_local_settings": {"generation": {"seed": V1_SEED}}}
    report = {"provider_metadata": {"generation_config": {"seed": V2_SEED}}}
    assert module.effective_seed(config, report) == V2_SEED


def test_effective_seed_falls_back_to_the_config_when_nothing_translated():
    module = _script_module()
    config = {"nllb_local_settings": {"generation": {"seed": V2_SEED}}}
    assert module.effective_seed(config, {"attempted": False}) == V2_SEED


def test_effective_seed_is_none_when_no_seed_is_configured():
    """Recorded as null rather than invented."""
    assert _script_module().effective_seed({}, {}) is None


# ---------------------------------------------------------------------------
# End to end: the manifest records the seed the run used
# ---------------------------------------------------------------------------
@pytest.fixture
def selection_file(tmp_path):
    path = tmp_path / "selection.jsonl"
    row = {
        "behavior_id": "beh_0123456789ab",
        "source": "jailbreakbench",
        "source_item_id": "0",
        "category": "Disinformation",
        "harm_label": "benign",
        "source_text": FIXTURE_TEXT,
        "source_text_sha256": text_sha256(FIXTURE_TEXT),
    }
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _run_writer(config, selection_file, tmp_path):
    out = {name: tmp_path / f"{name}.out" for name in
           ("requests", "outputs", "queue", "manifest", "inventory", "responses")}
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--config", str(config), "--selection", str(selection_file),
         *[arg for name, path in out.items() for arg in (f"--{name}", str(path))]],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(out["manifest"].read_text(encoding="utf-8"))


@pytest.mark.parametrize("config,expected", [(V2_CONFIG, V2_SEED), (V1_CONFIG, V1_SEED)])
def test_manifest_records_the_effective_seed(config, expected, selection_file, tmp_path):
    manifest = _run_writer(config, selection_file, tmp_path)
    assert manifest["generation_seed"] == expected
    assert manifest["generation_seed_source"] == "config"   # nothing translated here
    assert manifest["config_path"] == f"configs/{config.name}"


def test_manifest_stays_text_free(selection_file, tmp_path):
    manifest = _run_writer(V2_CONFIG, selection_file, tmp_path)
    assert manifest["contains_prompt_text"] is False
    blob = json.dumps(manifest, ensure_ascii=False)
    assert FIXTURE_TEXT not in blob
    for field in ("source_text", "translated_text"):
        assert f'"{field}":' not in blob
