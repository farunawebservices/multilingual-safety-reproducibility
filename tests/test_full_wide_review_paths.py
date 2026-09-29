"""Stage 2.4 reports its output paths without crashing on a relative CLI path.

``scripts/14_build_full_wide_review.py`` reported both of its outputs with
``Path.relative_to(REPO_ROOT)``. That is a pure string operation: a perfectly
valid relative argument such as ``--csv data/review/x.csv`` is not lexically
under the absolute ``REPO_ROOT``, so it raised ``ValueError`` -- *after* the CSV
had been written and *before* the metadata sidecar was, losing the sidecar and
returning a non-zero exit for a run whose validation had fully passed.

The fix routes both paths through ``multilingual_safety.paths.display_path``,
which resolves first and falls back to the caller's string. These tests pin the
regression and the unchanged behavior for an absolute path.

The fixture corpus below is invented placeholder prose -- no corpus prompt text
appears in this file.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from multilingual_safety.hashing import text_sha256
from multilingual_safety.languages import LANGUAGE_CODES, PIVOT_LANGUAGE
from multilingual_safety.paths import REPO_ROOT, V2_INTERIM_DIR

SCRIPT = REPO_ROOT / "scripts" / "14_build_full_wide_review.py"

#: The stage's own hard expectations: 200 behaviors, 10 categories, 100/100 arms.
N_BEHAVIORS = 200
CATEGORIES = [f"Category {i}" for i in range(10)]

# Invented placeholder, deliberately not drawn from any benchmark.
_SOURCE = "Summarise how a regional bus timetable is published."
_TRANSLATED = "Placeholder rendering of the timetable sentence."


def _row(index: int, language: str) -> dict:
    behavior_id = f"beh_{index:012x}"
    pivot = language == PIVOT_LANGUAGE
    return {
        "behavior_id": behavior_id,
        "source": "jailbreakbench",
        "source_item_id": str(index),
        "category": CATEGORIES[index % len(CATEGORIES)],
        "harm_label": "benign" if index % 2 == 0 else "harmful",
        "source_text": _SOURCE,
        "language": language,
        # English carries the canonical source verbatim; it is never translated.
        "translated_text": _SOURCE if pivot else f"{_TRANSLATED} [{language}/{index}]",
        "translation_method": "original" if pivot else "machine_unreviewed",
        "translation_model": None if pivot else "fixture-model",
        "translation_timestamp": None if pivot else "2026-01-01T00:00:00+00:00",
        "translation_status": "CANONICAL_SOURCE" if pivot else "MACHINE_TRANSLATED_UNCHECKED",
        "review_required": not pivot,
        "prompt_hash": text_sha256(_SOURCE if pivot else f"{_TRANSLATED} [{language}/{index}]"),
        "split_placeholder": "UNASSIGNED",
        "reviewer_id": None,
        "issues": [],
    }


@pytest.fixture
def long_manifest(tmp_path):
    """A complete long-format manifest: 200 behaviors x 7 languages."""
    path = tmp_path / "outputs.jsonl"
    path.write_text(
        "".join(
            json.dumps(_row(i, lang), sort_keys=True) + "\n"
            for i in range(N_BEHAVIORS)
            for lang in LANGUAGE_CODES
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def scratch_outputs():
    """Output paths inside the repo, in a git-ignored directory.

    They must be inside the repo: the stage refuses to finish if its CSV is not
    git-ignored, and it carries prompt text, so that refusal is correct. Removed
    afterwards either way.
    """
    csv_path = V2_INTERIM_DIR / "tmp_test_full_wide_review.csv"
    metadata_path = V2_INTERIM_DIR / "tmp_test_full_wide_review_metadata.json"
    try:
        yield csv_path, metadata_path
    finally:
        for path in (csv_path, metadata_path):
            path.unlink(missing_ok=True)


def _run(manifest, csv_path, metadata_path):
    return subprocess.run(
        [sys.executable, str(SCRIPT),
         "--outputs", str(manifest),
         "--csv", str(csv_path),
         "--metadata", str(metadata_path)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )


def test_relative_csv_path_does_not_crash_after_writing(long_manifest, scratch_outputs):
    """The regression: a relative --csv used to raise ValueError post-write."""
    csv_path, metadata_path = scratch_outputs
    result = _run(
        long_manifest,
        csv_path.relative_to(REPO_ROOT),
        metadata_path.relative_to(REPO_ROOT),
    )

    assert "ValueError" not in result.stderr, result.stderr
    assert result.returncode == 0, result.stderr
    assert csv_path.exists()
    # The sidecar is written after the reporting line that used to raise, so its
    # presence is what proves the crash is gone rather than merely relocated.
    assert metadata_path.exists(), "metadata sidecar was not written"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["csv_path"] == str(csv_path.relative_to(REPO_ROOT))
    assert f"wrote {csv_path.relative_to(REPO_ROOT)}" in result.stdout
    assert f"wrote {metadata_path.relative_to(REPO_ROOT)}" in result.stdout


def test_absolute_path_reporting_is_unchanged(long_manifest, scratch_outputs):
    """v1 defaults are absolute; they must report exactly as they did before."""
    csv_path, metadata_path = scratch_outputs
    result = _run(long_manifest, csv_path, metadata_path)

    assert result.returncode == 0, result.stderr
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["csv_path"] == str(csv_path.relative_to(REPO_ROOT))


def test_validation_results_are_untouched(long_manifest, scratch_outputs):
    """The fix is reporting-only: the review data and its checks are unchanged."""
    csv_path, metadata_path = scratch_outputs
    result = _run(
        long_manifest,
        csv_path.relative_to(REPO_ROOT),
        metadata_path.relative_to(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["validation_passed"] is True
    assert metadata["validation_failures"] == []
    assert metadata["row_count"] == N_BEHAVIORS
    assert metadata["n_cells"] == N_BEHAVIORS * len(LANGUAGE_CODES)
    assert metadata["human_reviewed_rows"] == 0
    assert all(
        metadata["language_counts"][lang]["empty"] == 0 for lang in LANGUAGE_CODES
    )
