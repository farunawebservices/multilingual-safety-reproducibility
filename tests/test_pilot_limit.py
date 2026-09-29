"""The pilot limit is explicit, visible, and cannot silently truncate.

The v2 pilot inherited ``pilot.behaviors_per_language: 3`` from the v1 config and
translated 3 of the 20 selected behaviors -- 18 rows done, 102 silently left
pending. Nothing in the run log or the manifest made that visible: the manifest
carried no limit at all, and the log printed the limit without ever saying how
much of the selection it left behind.

Three properties are asserted here:

* the effective limit and its source reach both the run log and the manifest;
* a run covering less than its selection file says so unprompted (v1's
  deliberate 3-of-20 subsample stays legal, but stops being quiet);
* ``--require-full-selection`` turns partial coverage into a hard failure that
  writes nothing -- the flag the v2 command carries.

Seeds are covered here too, since limit and seed were the same provenance bug:
v2 must be 20260818 while the unflagged v1 default stays 20260816.

No model is loaded and nothing is translated. The fixture text is invented
placeholder prose, not corpus text.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import yaml

from multilingual_safety.hashing import text_sha256
from multilingual_safety.languages import LANGUAGE_CODES, PIVOT_LANGUAGE
from multilingual_safety.paths import CONFIGS_DIR, REPO_ROOT
from multilingual_safety.providers import extract_seed

SCRIPT = REPO_ROOT / "scripts" / "04_translate_pilot.py"
V1_CONFIG = CONFIGS_DIR / "translation.yaml"
V2_CONFIG = CONFIGS_DIR / "translation_v2.yaml"

V1_LIMIT, V2_LIMIT = 3, 20
V1_SEED, V2_SEED = 20260816, 20260818
N_NON_PIVOT = len(LANGUAGE_CODES) - 1                      # 6

FIXTURE_TEXT = "Describe how a public library organises its catalogue."


# ---------------------------------------------------------------------------
# Fixtures: a 20-behavior selection file, matching what script 03 emits.
# ---------------------------------------------------------------------------
def _selection_rows(count):
    rows = []
    for index in range(count):
        text = f"{FIXTURE_TEXT} ({index})"
        rows.append({
            "behavior_id": f"beh_{index:012x}",
            "source": "jailbreakbench",
            "source_item_id": str(index),
            "category": "Disinformation",
            "harm_label": "harmful" if index % 2 else "benign",
            "source_text": text,
            "source_text_sha256": text_sha256(text),
        })
    return rows


@pytest.fixture
def selection_20(tmp_path):
    path = tmp_path / "selection.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in _selection_rows(20):
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _run(selection, tmp_path, *extra):
    out = {name: tmp_path / f"{name}.out" for name in
           ("requests", "outputs", "queue", "manifest", "inventory", "responses")}
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--selection", str(selection),
         *[arg for name, path in out.items() for arg in (f"--{name}", str(path))],
         *extra],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    return result, out


def _manifest(out):
    return json.loads(out["manifest"].read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. v1 default limit remains 3
# ---------------------------------------------------------------------------
def test_v1_config_default_limit_is_three():
    assert yaml.safe_load(V1_CONFIG.read_text(encoding="utf-8"))["pilot"][
        "behaviors_per_language"] == V1_LIMIT


def test_unflagged_run_still_uses_the_v1_limit_of_three(selection_20, tmp_path):
    """An invocation with no --limit and no --config is unchanged."""
    result, out = _run(selection_20, tmp_path)
    assert result.returncode == 0, result.stderr

    manifest = _manifest(out)
    assert manifest["pilot_subset"]["behaviors_per_language"] == V1_LIMIT
    assert manifest["pilot_subset"]["n_behaviors_in_subset"] == V1_LIMIT
    assert manifest["pilot_subset"]["n_subset_rows"] == V1_LIMIT * N_NON_PIVOT   # 18


# ---------------------------------------------------------------------------
# 2. v2 explicit limit 20
# ---------------------------------------------------------------------------
def test_v2_config_sets_the_limit_to_twenty():
    assert yaml.safe_load(V2_CONFIG.read_text(encoding="utf-8"))["pilot"][
        "behaviors_per_language"] == V2_LIMIT


def test_v2_config_run_covers_all_twenty(selection_20, tmp_path):
    result, out = _run(selection_20, tmp_path, "--config", str(V2_CONFIG))
    assert result.returncode == 0, result.stderr

    subset = _manifest(out)["pilot_subset"]
    assert subset["behaviors_per_language"] == V2_LIMIT
    assert subset["limit_source"] == "config"
    assert subset["n_behaviors_in_subset"] == V2_LIMIT
    assert subset["full_selection_covered"] is True
    assert subset["n_behaviors_untranslated"] == 0


def test_explicit_cli_limit_overrides_the_config_and_is_recorded(selection_20, tmp_path):
    result, out = _run(selection_20, tmp_path, "--limit", str(V2_LIMIT))
    assert result.returncode == 0, result.stderr

    subset = _manifest(out)["pilot_subset"]
    assert subset["behaviors_per_language"] == V2_LIMIT
    assert subset["limit_source"] == "cli"
    assert subset["full_selection_covered"] is True


# ---------------------------------------------------------------------------
# 3. seeds
# ---------------------------------------------------------------------------
def test_v2_seed_is_20260818():
    assert extract_seed(
        yaml.safe_load(V2_CONFIG.read_text(encoding="utf-8"))["nllb_local_settings"]
    ) == V2_SEED


def test_v1_default_seed_is_20260816():
    assert extract_seed(
        yaml.safe_load(V1_CONFIG.read_text(encoding="utf-8"))["nllb_local_settings"]
    ) == V1_SEED


@pytest.mark.parametrize(
    "extra,expected", [(("--config", str(V2_CONFIG)), V2_SEED), ((), V1_SEED)]
)
def test_manifest_records_the_effective_seed(extra, expected, selection_20, tmp_path):
    result, out = _run(selection_20, tmp_path, *extra)
    assert result.returncode == 0, result.stderr
    assert _manifest(out)["generation_seed"] == expected


# ---------------------------------------------------------------------------
# 4. a 20-row selection produces 140 rows and 120 non-English requests
# ---------------------------------------------------------------------------
def test_twenty_behaviors_give_140_rows_and_120_requests(selection_20, tmp_path):
    result, out = _run(selection_20, tmp_path, "--config", str(V2_CONFIG))
    assert result.returncode == 0, result.stderr

    rows = [json.loads(line) for line in
            out["outputs"].read_text(encoding="utf-8").splitlines() if line.strip()]
    requests = [json.loads(line) for line in
                out["requests"].read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(rows) == 140, "20 behaviors x 7 languages"
    assert len([r for r in rows if r["language"] == PIVOT_LANGUAGE]) == 20
    assert len([r for r in rows if r["language"] != PIVOT_LANGUAGE]) == 120
    assert len(requests) == 120, "20 behaviors x 6 non-English languages"
    assert PIVOT_LANGUAGE not in {r["target_language"] for r in requests}

    manifest = _manifest(out)
    assert manifest["n_rows"] == 140
    assert manifest["pilot_subset"]["n_subset_rows"] == 120


# ---------------------------------------------------------------------------
# 5. partial coverage is never silent; --require-full-selection makes it fatal
# ---------------------------------------------------------------------------
def test_subsampling_is_announced_in_the_run_log(selection_20, tmp_path):
    result, _ = _run(selection_20, tmp_path)          # v1 limit 3 of 20
    assert result.returncode == 0, result.stderr
    assert "SUBSAMPLE" in result.stdout
    assert "17 behavior(s)" in result.stdout
    assert "102 rows left pending" in result.stdout


def test_manifest_records_partial_coverage(selection_20, tmp_path):
    _, out = _run(selection_20, tmp_path)
    subset = _manifest(out)["pilot_subset"]
    assert subset["full_selection_covered"] is False
    assert subset["n_behaviors_untranslated"] == 17
    assert subset["n_behaviors_in_selection"] == 20


def test_require_full_selection_fails_on_partial_coverage(selection_20, tmp_path):
    result, out = _run(selection_20, tmp_path, "--require-full-selection")
    assert result.returncode == 1
    assert "--require-full-selection" in result.stderr
    assert "covers 3 of 20 behaviors" in result.stderr
    # Nothing written: the guard fires before any output or translation.
    assert not [name for name, path in out.items() if path.exists()]


def test_require_full_selection_passes_when_coverage_is_complete(selection_20, tmp_path):
    result, out = _run(
        selection_20, tmp_path, "--config", str(V2_CONFIG), "--require-full-selection")
    assert result.returncode == 0, result.stderr
    assert "SUBSAMPLE" not in result.stdout
    assert _manifest(out)["pilot_subset"]["require_full_selection"] is True


def test_guard_is_off_by_default_so_v1_is_unaffected(selection_20, tmp_path):
    """The v1 pilot subsamples on purpose and must not need a new flag."""
    result, out = _run(selection_20, tmp_path)
    assert result.returncode == 0, result.stderr
    assert _manifest(out)["pilot_subset"]["require_full_selection"] is False
