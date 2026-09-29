"""Argument routing and path safety for the full-set translation stages.

``scripts/11_translate_full_set.py`` hardcoded every path under
``data/interim/full/`` and read ``configs/translation.yaml``. That made a
dataset-v2 run impossible to do safely: the only writable response path was
inside the read-only recovered request area, and the only readable config was
v1's. The paths are now parameterised, with the v1 paths as defaults.

These tests assert both halves:

* **routing** -- ``--config``/``--requests``/``--responses``/``--pilot-responses``
  are actually honoured, and the v1 defaults are unchanged;
* **path safety** -- ``assert_output_path_is_safe`` refuses the writes that
  would destroy a read-only input, a frozen dataset, or a reviewed pilot log.

The fixture text below is invented placeholder prose -- no corpus prompt text
appears in this file.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import pytest

from multilingual_safety.paths import (
    DATA_DIR,
    FULL_REQUESTS_PATH,
    INTERIM_DIR,
    PROCESSED_DIR,
    REPO_ROOT,
    V2_INTERIM_DIR,
    V2_MANIFESTS_DIR,
    V2_PROCESSED_DIR,
    PathSafetyError,
    assert_output_path_is_safe,
)

SCRIPT = REPO_ROOT / "scripts" / "11_translate_full_set.py"
APPLY_SCRIPT = REPO_ROOT / "scripts" / "13_apply_full_responses.py"

V1_CONFIG = "configs/translation.yaml"
V2_CONFIG = "configs/translation_v2.yaml"
V2_RESPONSES = "data/v2/interim/full/full_translation_responses_v2.jsonl"
V2_PILOT_RESPONSES = "data/v2/interim/full/pilot_responses_v2.jsonl"

#: Where a v2 run puts its manifest when --manifest is not given.
V2_MANIFEST = V2_MANIFESTS_DIR / "full_translation_manifest_v2.json"

# Invented placeholder, deliberately not drawn from any benchmark.
_FIXTURE_TEXT = "Explain how a municipal recycling centre sorts glass."

FIXTURE_REQUESTS = [
    {"behavior_id": "beh_0123456789ab", "target_language": "ha", "source_text": _FIXTURE_TEXT},
    {"behavior_id": "beh_0123456789ab", "target_language": "yo", "source_text": _FIXTURE_TEXT},
]


@pytest.fixture
def requests_file(tmp_path):
    path = tmp_path / "requests.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in FIXTURE_REQUESTS),
        encoding="utf-8",
    )
    return path


def _run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


# ---------------------------------------------------------------------------
# Path safety -- the rules, unit-tested directly.
# ---------------------------------------------------------------------------
def test_v2_corpus_output_must_be_under_v2_interim():
    """data/v2/manifests/ and data/v2/processed/ are not corpus destinations."""
    with pytest.raises(PathSafetyError, match="data/v2/interim"):
        assert_output_path_is_safe(V2_MANIFESTS_DIR / "responses.jsonl")


def test_v2_processed_is_refused():
    """Translated rows do not go to data/v2/processed/ at this stage."""
    with pytest.raises(PathSafetyError, match="freeze"):
        assert_output_path_is_safe(V2_PROCESSED_DIR / "frozen_dataset_v2.jsonl")


def test_v1_processed_is_refused():
    with pytest.raises(PathSafetyError, match="data/processed/"):
        assert_output_path_is_safe(PROCESSED_DIR / "frozen_dataset_v1.jsonl")


def test_recovered_request_set_is_read_only():
    with pytest.raises(PathSafetyError, match="read-only input"):
        assert_output_path_is_safe(FULL_REQUESTS_PATH)


def test_output_equal_to_an_input_is_refused(tmp_path):
    """The one mistake that destroys the thing the output was derived from."""
    somewhere = tmp_path / "requests.jsonl"
    with pytest.raises(PathSafetyError, match="read-only input"):
        assert_output_path_is_safe(somewhere, inputs=(somewhere,))


@pytest.mark.parametrize(
    "artifact",
    [
        INTERIM_DIR / "translation_pilot_responses.jsonl",
        V2_INTERIM_DIR / "pilot_responses_v2.jsonl",
        V2_MANIFESTS_DIR / "pilot_manifest_v2.json",
    ],
)
def test_reviewed_pilot_artifacts_are_refused(artifact):
    kind = "metadata" if artifact.suffix == ".json" else "corpus"
    with pytest.raises(PathSafetyError, match="pilot"):
        assert_output_path_is_safe(artifact, kind=kind)


def test_arbitrary_location_is_refused_for_corpus(tmp_path):
    """Translated text may only land under a declared write root."""
    with pytest.raises(PathSafetyError, match="may only"):
        assert_output_path_is_safe(tmp_path / "responses.jsonl")


@pytest.mark.parametrize(
    "path",
    [
        DATA_DIR / "interim" / "full" / "full_translation_responses.jsonl",   # v1 default
        V2_INTERIM_DIR / "full" / "full_translation_responses_v2.jsonl",      # v2
    ],
)
def test_legitimate_corpus_paths_are_allowed(path):
    assert assert_output_path_is_safe(path) == path.resolve()


def test_metadata_is_not_confined_to_a_write_root(tmp_path):
    """A text-free manifest carries no payload, so a root would protect nothing."""
    target = tmp_path / "manifest.json"
    assert assert_output_path_is_safe(target, kind="metadata") == target.resolve()
    assert assert_output_path_is_safe(V2_MANIFEST, kind="metadata") == V2_MANIFEST.resolve()


def test_metadata_still_cannot_clobber_protected_artifacts():
    with pytest.raises(PathSafetyError):
        assert_output_path_is_safe(PROCESSED_DIR / "report.json", kind="metadata")


# ---------------------------------------------------------------------------
# Path safety -- enforced by the CLI, not just by the library.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "responses,expected",
    [
        ("data/processed/x.jsonl", "data/processed/"),
        ("data/v2/processed/x.jsonl", "freeze"),
        ("data/v2/manifests/x.jsonl", "data/v2/interim"),
        ("data/interim/full/full_translation_requests.jsonl", "read-only input"),
        ("data/v2/interim/pilot_responses_v2.jsonl", "pilot"),
    ],
)
def test_cli_refuses_unsafe_response_paths(requests_file, responses, expected):
    result = _run("--plan", "--requests", str(requests_file), "--responses", responses)
    assert result.returncode != 0, f"{responses} was accepted"
    assert expected in result.stderr, result.stderr


def test_cli_refuses_writing_responses_over_its_own_requests(requests_file):
    result = _run(
        "--plan", "--requests", str(requests_file), "--responses", str(requests_file)
    )
    assert result.returncode != 0
    assert "read-only input" in result.stderr


def test_cli_refuses_a_manifest_over_a_protected_artifact(requests_file):
    result = _run(
        "--plan", "--requests", str(requests_file), "--responses", V2_RESPONSES,
        "--manifest", "data/v2/manifests/pilot_manifest_v2.json",
    )
    assert result.returncode != 0
    assert "pilot" in result.stderr


# ---------------------------------------------------------------------------
# Argument routing.
# ---------------------------------------------------------------------------
def test_v1_defaults_are_unchanged():
    """An unflagged invocation must behave exactly as it did before."""
    help_text = _run("--help").stdout
    for expected in (
        V1_CONFIG,
        "data/interim/full/full_translation_requests.jsonl",
        "data/interim/full/full_translation_responses.jsonl",
        "data/interim/translation_pilot_responses.jsonl",
    ):
        assert expected in help_text, f"default for {expected} changed"


def test_config_argument_is_actually_read(requests_file):
    """--config routes: the v2 config's seed differs from v1's."""
    v1 = _run("--plan", "--requests", str(requests_file), "--no-manifest")
    v2 = _run(
        "--plan", "--requests", str(requests_file), "--config", V2_CONFIG,
        "--responses", V2_RESPONSES, "--no-manifest",
    )
    assert v1.returncode == 0, v1.stderr
    assert v2.returncode == 0, v2.stderr
    assert "configured seed   : 20260816" in v1.stdout
    assert "configured seed   : 20260818" in v2.stdout


def test_v2_paths_are_reported_and_nothing_is_written(requests_file):
    before = _digest(REPO_ROOT / V2_RESPONSES)
    result = _run(
        "--plan",
        "--config", V2_CONFIG,
        "--requests", str(requests_file),
        "--responses", V2_RESPONSES,
        "--pilot-responses", V2_PILOT_RESPONSES,
        "--no-manifest",
    )
    assert result.returncode == 0, result.stderr
    assert "dataset v2" in result.stdout
    for path in (V2_CONFIG, V2_RESPONSES, V2_PILOT_RESPONSES):
        assert path in result.stdout, f"{path} not routed"
    assert "PLAN ONLY" in result.stdout
    assert _digest(REPO_ROOT / V2_RESPONSES) == before, "--plan wrote a response log"


def test_v2_plan_writes_a_text_free_manifest(requests_file, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    result = _run(
        "--plan",
        "--config", V2_CONFIG,
        "--requests", str(requests_file),
        "--responses", V2_RESPONSES,
        "--pilot-responses", V2_PILOT_RESPONSES,
        "--manifest", str(manifest_path),
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["config_path"] == V2_CONFIG
    assert manifest["model_revision_configured"] == "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
    assert manifest["seed_configured"] == 20260818
    assert manifest["device_configured"] == "cuda"
    assert manifest["dtype_configured"] == "float32"
    assert manifest["requests_sha256"] == hashlib.sha256(
        requests_file.read_bytes()
    ).hexdigest()
    assert manifest["responses_path"] == V2_RESPONSES
    assert manifest["responses_path_is_under_v2_interim"] is True
    assert manifest["dataset_version"] == "v2"
    assert manifest["mode"] == "plan"
    assert manifest["translation_attempted"] is False
    assert manifest["translated_rows_written_to_processed"] is False

    assert manifest["contains_prompt_text"] is False
    assert _FIXTURE_TEXT not in manifest_path.read_text(encoding="utf-8")


def test_v1_run_writes_no_manifest(requests_file):
    """v1's on-disk footprint is exactly what it was: no new artifact."""
    before = _digest(V2_MANIFEST)
    result = _run("--plan", "--requests", str(requests_file))
    assert result.returncode == 0, result.stderr
    assert "manifest  : (none)" in result.stdout
    assert _digest(V2_MANIFEST) == before


def test_seed_pilot_fails_loudly_when_the_pilot_log_is_absent(requests_file, tmp_path):
    """Reusing zero rows silently looks identical to reusing all of them."""
    absent = tmp_path / "no_such_pilot_log.jsonl"
    result = _run(
        "--plan", "--requests", str(requests_file), "--responses", V2_RESPONSES,
        "--pilot-responses", str(absent), "--seed-pilot", "--no-manifest",
    )
    assert result.returncode != 0
    assert "does not exist" in result.stderr


def test_translate_refuses_a_request_set_the_protocol_does_not_pin(requests_file):
    """The materialization precondition is re-checked before translation."""
    result = _run(
        "--translate", "--config", V2_CONFIG, "--requests", str(requests_file),
        "--responses", V2_RESPONSES, "--no-manifest",
    )
    assert result.returncode != 0
    assert "does not match the hash pinned" in result.stderr


def test_real_request_set_matches_the_pinned_precondition():
    """The recovered request set is the one the v2 protocol declares."""
    if not FULL_REQUESTS_PATH.exists():
        pytest.skip("full request set not materialized")
    result = _run("--plan", "--config", V2_CONFIG, "--responses", V2_RESPONSES,
                  "--no-manifest")
    assert result.returncode == 0, result.stderr
    assert "matches protocol precondition: True" in result.stdout
    assert "requests loaded: 1200" in result.stdout


# ---------------------------------------------------------------------------
# Stage 2.3 uses the same gate.
# ---------------------------------------------------------------------------
def test_apply_stage_exposes_provenance_and_report_paths():
    help_text = subprocess.run(
        [sys.executable, str(APPLY_SCRIPT), "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    ).stdout
    assert "--provenance" in help_text
    assert "--report" in help_text
    # v1 defaults unchanged.
    assert "full_translation_provenance.jsonl" in help_text
    assert "full_apply_report.json" in help_text
