"""Stage 3.0 path/salt routing for dataset v2.

``scripts/15_freeze_splits.py`` hard-coded ``data/processed/splits_v1.json`` and
``results/split_report.json``. A v2 freeze would therefore have overwritten v1's
split file and v1's tracked report -- the least reversible artifacts in the
project. The destinations are now parameterised, with the v1 paths as defaults.

Every test here runs with ``--dry-run``. A real freeze is never triggered: a
non-dry v1 invocation would write an actual frozen dataset, and a v2 one would
write a split file that every later result must reference.

What is pinned:

* v1 defaults are unchanged, down to the recorded ``split_hash``;
* v2 destinations are accepted and produce 140 dev / 60 final_test;
* v2 refuses v1's salt, which would reproduce v1's assignment exactly;
* v2 refuses to write under ``data/processed/`` or ``results/``, and refuses to
  write over its own read-only input.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from multilingual_safety.paths import (
    INTERIM_DIR,
    PROCESSED_DIR,
    REPO_ROOT,
    RESULTS_DIR,
    V2_MANIFESTS_DIR,
    V2_PROCESSED_DIR,
)

SCRIPT = REPO_ROOT / "scripts" / "15_freeze_splits.py"

V1_OUTPUTS = INTERIM_DIR / "full" / "full_translation_outputs.jsonl"
V1_SPLITS = PROCESSED_DIR / "splits_v1.json"
V1_REPORT = RESULTS_DIR / "split_report.json"

V2_INPUT = V2_PROCESSED_DIR / "frozen_dataset_v2.jsonl"
V2_SPLITS = V2_PROCESSED_DIR / "splits_v2.json"
V2_REPORT = V2_MANIFESTS_DIR / "split_report_v2.json"

V1_SALT = "mls-splits-v1"
V2_SALT = "mls-v2-splits-20260818"

#: configs/dataset_v2_rebuild.yaml states this v1 hash and forbids reproducing it.
V1_SPLIT_HASH = "9cc1a7b87cc6f83c7bd66b92f2111f968497d00e7e464b481f9abc1cc4cdb90e"

#: configs/dataset_v2_rebuild.yaml, split.expected_behavior_counts.
EXPECTED_DEV = 140
EXPECTED_FINAL_TEST = 60


def _run(*args):
    """Always --dry-run: these tests must never freeze anything."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run", *args],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )


def _v2(*extra, salt=V2_SALT, splits=V2_SPLITS, report=V2_REPORT, outputs=V2_INPUT):
    return _run(
        "--outputs", str(outputs),
        "--splits-output", str(splits),
        "--report", str(report),
        *(("--salt", salt) if salt is not None else ()),
        *extra,
    )


requires_v2_input = pytest.mark.skipif(
    not V2_INPUT.exists(), reason="v2 translated artifact not materialized"
)
requires_v1_input = pytest.mark.skipif(
    not V1_OUTPUTS.exists(), reason="v1 long-format outputs not materialized"
)


# ---------------------------------------------------------------------------
# v1 defaults unchanged.
# ---------------------------------------------------------------------------
def test_v1_defaults_are_unchanged():
    help_text = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    ).stdout
    assert "data/processed/splits_v1.json" in help_text
    assert "results/split_report.json" in help_text
    for preserved in ("--outputs", "--salt", "--dry-run", "--refreeze"):
        assert preserved in help_text, f"{preserved} disappeared"


@requires_v1_input
def test_v1_dry_run_still_produces_the_v1_split_hash():
    """The v1 assignment must be bit-for-bit what it always was."""
    result = _run()
    assert result.returncode == 0, result.stderr
    assert f"split_hash        : {V1_SPLIT_HASH}" in result.stdout
    assert "dataset v1" in result.stdout
    assert "Dry run: nothing written" in result.stdout


# ---------------------------------------------------------------------------
# v2 destinations.
# ---------------------------------------------------------------------------
@requires_v2_input
def test_v2_paths_are_accepted():
    before = (V2_SPLITS.exists(), V2_REPORT.exists())
    result = _v2()
    assert result.returncode == 0, result.stderr
    assert "dataset v2" in result.stdout
    assert "(not written; the input IS the frozen dataset)" in result.stdout
    assert (V2_SPLITS.exists(), V2_REPORT.exists()) == before, "--dry-run wrote something"


@requires_v2_input
def test_v2_counts_are_140_dev_and_60_final_test():
    result = _v2()
    assert result.returncode == 0, result.stderr
    assert f"'dev': {EXPECTED_DEV}" in result.stdout
    assert f"'final_test': {EXPECTED_FINAL_TEST}" in result.stdout
    assert "dev         harmful= 70  benign= 70" in result.stdout
    assert "final_test  harmful= 30  benign= 30" in result.stdout


@requires_v2_input
def test_v2_split_hash_differs_from_v1():
    """A v2 hash equal to v1's would let v2 results be mistaken for v1's."""
    result = _v2()
    assert result.returncode == 0, result.stderr
    assert V1_SPLIT_HASH not in result.stdout


# ---------------------------------------------------------------------------
# v2 salt.
# ---------------------------------------------------------------------------
@requires_v2_input
def test_v2_refuses_the_v1_salt():
    result = _v2(salt=V1_SALT)
    assert result.returncode != 0
    assert "is v1's" in result.stderr
    assert V2_SALT in result.stderr, "the refusal should name the salt to use"
    assert "Nothing written" in result.stderr


@requires_v2_input
def test_v2_refuses_the_default_salt():
    """The default salt IS v1's, so a v2 run that forgets --salt must fail."""
    result = _v2(salt=None)
    assert result.returncode != 0
    assert "is v1's" in result.stderr


@requires_v2_input
def test_v2_warns_when_the_salt_is_not_the_preregistered_one():
    result = _v2(salt="some-other-salt")
    assert result.returncode == 0, result.stderr
    assert "not the pre-registered v2 salt" in result.stdout


# ---------------------------------------------------------------------------
# v2 output/report path safety.
# ---------------------------------------------------------------------------
@requires_v2_input
@pytest.mark.parametrize(
    "splits,report,expected",
    [
        (V1_SPLITS, V2_REPORT, "under data/processed/"),
        (V2_SPLITS, V1_REPORT, "under results/"),
        (V2_PROCESSED_DIR / "splits_v2.json", V2_PROCESSED_DIR / "report.json",
         "--report must be under data/v2/manifests/"),
        (V2_MANIFESTS_DIR / "splits_v2.json", V2_REPORT,
         "--splits-output must be under data/v2/processed/"),
    ],
)
def test_v2_refuses_unsafe_destinations(splits, report, expected):
    result = _v2(splits=splits, report=report)
    assert result.returncode != 0, f"accepted {splits} / {report}"
    assert expected in result.stderr, result.stderr
    assert "Nothing written" in result.stderr


@requires_v2_input
def test_v2_refuses_writing_over_its_own_input():
    """The v2 translated corpus is read-only at this stage."""
    result = _v2(splits=V2_INPUT)
    assert result.returncode != 0
    assert "is the input artifact" in result.stderr


@requires_v2_input
def test_v2_defaults_would_target_v1_and_are_refused():
    """--outputs alone is not enough: the destinations still default to v1's."""
    result = _run("--outputs", str(V2_INPUT), "--salt", V2_SALT)
    assert result.returncode != 0
    assert "under data/processed/" in result.stderr
    assert "under results/" in result.stderr
