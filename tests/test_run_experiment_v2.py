"""``scripts/run_experiment_v2.py`` -- the dev-only v2 preflight runner.

What is pinned here:

* the v2 config parses and is rejected when it is not a v2 config;
* all three hashes -- dataset, splits file, and ``split_hash`` -- are verified,
  and each one, tampered with independently, aborts the run;
* the dev selection is exactly the 140 frozen dev behaviors, 980 rows, seven
  languages, 70/70 by arm;
* ``final_test`` is refused without ``--allow-test``, at the CLI gate *and*
  again at the library boundary;
* ``--dry-run`` writes nothing and runs no inference, and a non-dry run fails
  with a configuration error naming the missing backends rather than emitting
  an empty result table.

No test here reads the holdout. The permitted-read path (``--allow-test``) is
deliberately **not** exercised: the split is frozen and unread, and a test suite
that opens it on every CI run would be the first thing to break that
(CLAUDE.md §1). What is tested is that the door is shut.
"""

from __future__ import annotations

import functools
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

from multilingual_safety.agents.tool_caller import ToolCaller
from multilingual_safety.paths import REPO_ROOT
from multilingual_safety.splits import SplitAccessError

SCRIPT = REPO_ROOT / "scripts" / "run_experiment_v2.py"
CONFIG = REPO_ROOT / "configs" / "experiment_v2.example.yaml"

DATASET = REPO_ROOT / "data" / "v2" / "processed" / "frozen_dataset_v2.jsonl"
SPLITS = REPO_ROOT / "data" / "v2" / "processed" / "splits_v2.json"

#: configs/experiment_v2.example.yaml, `data`. Restated rather than read from
#: the config so a silent edit to the config cannot make these tests agree with it.
DATASET_SHA = "05b5abb311a3aa973b0b5dbc3bc96637be8039e99d4f42bd18ef86f608c44fb2"
SPLITS_SHA = "14535a2a32dbf2ed2cb7037ff5ea05001be96039709db00aba8b360c601640ac"
SPLIT_HASH = "01bbf0c609450e640197b6d4328573f378b6727459a2869ccfc7c05bff6703d6"

EXPECTED_DEV = 140
EXPECTED_LANGUAGES = 7
EXPECTED_DEV_ROWS = EXPECTED_DEV * EXPECTED_LANGUAGES  # 980

#: A dry run must never print a behavior id (it would name split members) or
#: corpus text.
BEHAVIOR_ID_RE = re.compile(r"beh_[0-9a-f]{12}")

requires_v2_corpus = pytest.mark.skipif(
    not (DATASET.exists() and SPLITS.exists()),
    reason="v2 frozen corpus/splits not materialized",
)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )


def _dry(*args: str) -> subprocess.CompletedProcess[str]:
    return _run("--config", str(CONFIG), "--dry-run", *args)


@functools.lru_cache(maxsize=1)
def _module():
    """Import the CLI as a module so its pure functions can be unit-tested.

    Registered in ``sys.modules`` before execution: ``@dataclass(slots=True)``
    rebuilds the class and resolves its annotations through
    ``sys.modules[cls.__module__]``, which is absent for a module loaded purely
    from a path.
    """
    spec = importlib.util.spec_from_file_location("run_experiment_v2", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tree_snapshot() -> set[str]:
    """Every path under the artifact roots a run could plausibly touch."""
    roots = (REPO_ROOT / "results", REPO_ROOT / "logs", REPO_ROOT / "data" / "v2")
    return {
        str(p.relative_to(REPO_ROOT))
        for root in roots
        if root.exists()
        for p in root.rglob("*")
    }


def _tampered_config(tmp_path: Path, field: str, value: str) -> Path:
    """Copy the example config with one `data` field replaced."""
    original = CONFIG.read_text(encoding="utf-8")
    pattern = re.compile(rf"^(\s*{re.escape(field)}:\s*)\S+", re.MULTILINE)
    patched, n = pattern.subn(rf"\g<1>{value}", original, count=1)
    assert n == 1, f"{field} not found in {CONFIG}"
    path = tmp_path / "tampered.yaml"
    path.write_text(patched, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------
def test_cli_exposes_the_required_flags():
    help_text = _run("--help").stdout
    for flag in ("--config", "--split", "--dry-run", "--limit", "--allow-test"):
        assert flag in help_text, f"{flag} missing"


def test_default_split_is_dev():
    args = _module().build_parser().parse_args([])
    assert args.split == "dev"
    assert args.allow_test is False
    assert args.dry_run is False
    assert args.limit is None


# ---------------------------------------------------------------------------
# 1. Config loading
# ---------------------------------------------------------------------------
def test_example_config_is_a_valid_v2_config():
    config = _module().load_experiment_config(CONFIG)
    data = config["data"]
    assert data["dataset_version"] == "v2"
    assert data["split"] == "dev"
    assert data["dataset"] == "data/v2/processed/frozen_dataset_v2.jsonl"
    assert data["splits"] == "data/v2/processed/splits_v2.json"
    assert data["split_hash"] == SPLIT_HASH


def test_config_split_defaults_to_dev_when_unset():
    module = _module()
    assert module.config_split({"data": {}}) == "dev"


def test_missing_config_file_is_a_configuration_error(tmp_path):
    module = _module()
    with pytest.raises(module.ConfigurationError, match="config not found"):
        module.load_experiment_config(tmp_path / "absent.yaml")


def test_config_without_a_data_section_is_refused(tmp_path):
    module = _module()
    path = tmp_path / "no_data.yaml"
    path.write_text("experiment:\n  name: x\n", encoding="utf-8")
    with pytest.raises(module.ConfigurationError, match="no `data:` section"):
        module.load_experiment_config(path)


@pytest.mark.parametrize("field", ["dataset", "dataset_sha256", "splits_sha256", "split_hash"])
def test_config_missing_a_required_data_key_is_refused(tmp_path, field):
    module = _module()
    path = _tampered_config(tmp_path, field, "null")
    with pytest.raises(module.ConfigurationError, match=f"missing required data keys.*{field}"):
        module.load_experiment_config(path)


def test_a_v1_config_is_refused(tmp_path):
    """v1 and v2 results are never merged; this runner takes v2 only."""
    module = _module()
    path = _tampered_config(tmp_path, "dataset_version", "v1")
    with pytest.raises(module.ConfigurationError, match="only accepts 'v2'"):
        module.load_experiment_config(path)


# ---------------------------------------------------------------------------
# 2. Hash checks
# ---------------------------------------------------------------------------
@requires_v2_corpus
def test_verify_hashes_matches_the_frozen_artifacts():
    module = _module()
    hashes = module.verify_hashes(module.load_experiment_config(CONFIG))
    assert hashes == {
        "dataset_sha256": DATASET_SHA,
        "splits_sha256": SPLITS_SHA,
        "split_hash": SPLIT_HASH,
    }


@requires_v2_corpus
@pytest.mark.parametrize(
    ("field", "expected_message"),
    [
        ("dataset_sha256", "dataset hash mismatch"),
        ("splits_sha256", "splits hash mismatch"),
        ("split_hash", "split_hash mismatch"),
    ],
)
def test_each_hash_is_checked_independently(tmp_path, field, expected_message):
    """Tampering with any one of the three aborts, and names which one."""
    module = _module()
    path = _tampered_config(tmp_path, field, "0" * 64)
    with pytest.raises(module.ConfigurationError, match=expected_message):
        module.verify_hashes(module.load_experiment_config(path))


@requires_v2_corpus
def test_a_hash_mismatch_aborts_the_cli_before_selection(tmp_path):
    path = _tampered_config(tmp_path, "dataset_sha256", "0" * 64)
    result = _run("--config", str(path), "--dry-run")
    assert result.returncode == 2
    assert "dataset hash mismatch" in result.stderr
    assert "behaviors" not in result.stdout


@requires_v2_corpus
def test_a_missing_dataset_file_is_a_configuration_error(tmp_path):
    module = _module()
    path = _tampered_config(tmp_path, "dataset", "data/v2/processed/does_not_exist.jsonl")
    with pytest.raises(module.ConfigurationError, match="dataset file not found"):
        module.verify_hashes(module.load_experiment_config(path))


# ---------------------------------------------------------------------------
# 3. Dev selection
# ---------------------------------------------------------------------------
@requires_v2_corpus
def test_dev_selection_is_the_140_frozen_behaviors():
    selected, total = _module().select_behaviors(SPLITS, "dev")
    assert total == EXPECTED_DEV
    assert len(selected) == EXPECTED_DEV
    assert len(set(selected)) == EXPECTED_DEV
    assert selected == sorted(selected)


@requires_v2_corpus
def test_limit_truncates_deterministically():
    module = _module()
    first, total = module.select_behaviors(SPLITS, "dev", limit=5)
    second, _ = module.select_behaviors(SPLITS, "dev", limit=5)
    full, _ = module.select_behaviors(SPLITS, "dev")
    assert total == EXPECTED_DEV, "the split total is reported, not the limit"
    assert first == second == full[:5]


@requires_v2_corpus
def test_limit_below_one_is_refused():
    module = _module()
    with pytest.raises(module.ConfigurationError, match="--limit must be >= 1"):
        module.select_behaviors(SPLITS, "dev", limit=0)


@requires_v2_corpus
def test_only_the_selected_behaviors_rows_are_loaded():
    """980 dev rows out of 1,400 -- the holdout's 420 are scanned past, not kept."""
    module = _module()
    selected, _ = module.select_behaviors(SPLITS, "dev")
    selection = module.load_selected_rows(DATASET, selected)

    assert selection.n_lines_scanned == 1400
    assert selection.n_rows == EXPECTED_DEV_ROWS
    assert selection.behaviors_found == EXPECTED_DEV
    assert len(selection.by_language) == EXPECTED_LANGUAGES
    assert set(selection.by_language.values()) == {EXPECTED_DEV}
    assert selection.by_arm == {"benign": EXPECTED_DEV_ROWS // 2,
                                "harmful": EXPECTED_DEV_ROWS // 2}
    assert len(selection.by_category) == 10


@requires_v2_corpus
def test_loaded_rows_carry_no_corpus_text():
    """The preflight needs shape, not prompts. Text never enters the report."""
    module = _module()
    selected, _ = module.select_behaviors(SPLITS, "dev", limit=3)
    selection = module.load_selected_rows(DATASET, selected)
    for row in selection.rows:
        assert set(row) == set(module.RETAINED_FIELDS)
        assert "source_text" not in row
        assert "translated_text" not in row


@requires_v2_corpus
def test_a_behavior_absent_from_the_corpus_is_a_stop_condition(tmp_path):
    module = _module()
    with pytest.raises(module.ConfigurationError, match="have no rows in"):
        module.load_selected_rows(DATASET, ["beh_000000000000"])


# ---------------------------------------------------------------------------
# 4. final_test refusal
# ---------------------------------------------------------------------------
def test_resolve_split_refuses_a_holdout_without_opt_in():
    """A pure function: the refusal fires before any file is opened."""
    module = _module()
    for holdout in ("final_test", "test"):
        with pytest.raises(SplitAccessError, match="refusing to read"):
            module.resolve_split(holdout, allow_test=False)


def test_resolve_split_passes_dev_through():
    assert _module().resolve_split("dev", allow_test=False) == "dev"


@requires_v2_corpus
def test_select_behaviors_refuses_a_holdout_at_the_library_boundary():
    """The second gate: no call path reaches the holdout without the flag."""
    with pytest.raises(SplitAccessError):
        _module().select_behaviors(SPLITS, "final_test", allow_test=False)


@requires_v2_corpus
def test_cli_refuses_final_test_without_allow_test():
    result = _dry("--split", "final_test")
    assert result.returncode == 2
    assert "refusing to read the 'final_test' split" in result.stderr
    assert "CLAUDE.md §1" in result.stderr
    assert result.stdout == "", "a refused run must report nothing about the holdout"
    assert not BEHAVIOR_ID_RE.search(result.stderr), "the refusal named split members"


@requires_v2_corpus
def test_refusing_final_test_writes_nothing():
    before = _tree_snapshot()
    _dry("--split", "final_test")
    assert _tree_snapshot() == before


@requires_v2_corpus
def test_an_unknown_split_is_a_configuration_error_not_a_traceback():
    result = _dry("--split", "nope")
    assert result.returncode == 2
    assert "unknown split 'nope'" in result.stderr
    assert "Traceback" not in result.stderr


# ---------------------------------------------------------------------------
# 5. Dry-run behavior
# ---------------------------------------------------------------------------
@requires_v2_corpus
def test_dry_run_reports_140_dev_behaviors():
    result = _dry()
    assert result.returncode == 0, result.stderr
    assert f"behaviors       : {EXPECTED_DEV} of {EXPECTED_DEV} in split" in result.stdout
    assert f"rows loaded     : {EXPECTED_DEV_ROWS}" in result.stdout
    assert "split           : dev" in result.stdout


@requires_v2_corpus
def test_dry_run_reports_all_three_verified_hashes():
    stdout = _dry().stdout
    for digest in (DATASET_SHA, SPLITS_SHA, SPLIT_HASH):
        assert f"{digest}  OK" in stdout


@requires_v2_corpus
def test_dry_run_runs_no_inference_and_writes_nothing():
    before = _tree_snapshot()
    result = _dry()
    assert result.returncode == 0, result.stderr
    assert "model inference : none (dry run)" in result.stdout
    assert "artifacts       : none written" in result.stdout
    assert "Dry run: nothing written" in result.stdout
    assert _tree_snapshot() == before, "--dry-run touched an artifact tree"


@requires_v2_corpus
def test_dry_run_prints_no_behavior_ids():
    """Split membership is not console output."""
    assert not BEHAVIOR_ID_RE.search(_dry().stdout)


@requires_v2_corpus
def test_dry_run_reports_a_limit_as_a_reduced_denominator():
    stdout = _dry("--limit", "5").stdout
    assert f"behaviors       : 5 of {EXPECTED_DEV} in split" in stdout
    assert "REDUCED DENOMINATOR" in stdout
    assert "rows loaded     : 35" in stdout


# ---------------------------------------------------------------------------
# 6. A non-dry run refuses, and the tool surface stays mocked
# ---------------------------------------------------------------------------
@requires_v2_corpus
def test_non_dry_run_fails_with_a_backend_configuration_error():
    before = _tree_snapshot()
    result = _run("--config", str(CONFIG))
    assert result.returncode == 2
    assert "configuration error:" in result.stderr
    assert "Planner and Critic backends are not implemented" in result.stderr
    assert result.stdout == "", "no experiment output may be produced"
    assert _tree_snapshot() == before, "a refused run created an artifact"


def test_tool_caller_stays_mocked():
    assert ToolCaller.allow_real_execution is False
    _module().assert_tool_caller_is_mocked()  # does not raise


def test_backend_gate_names_planner_and_critic():
    module = _module()
    assert module.UNIMPLEMENTED_BACKENDS == ("planner", "critic")
    with pytest.raises(module.ConfigurationError, match="Planner and Critic"):
        module.assert_backends_available()
