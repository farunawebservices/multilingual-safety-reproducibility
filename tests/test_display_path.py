"""Path reporting must never crash after the output file is already written.

``Path.relative_to`` is a pure string operation: it does not resolve. So a valid
relative CLI argument (``--out data/v2/interim/x.jsonl``) is not lexically under
the absolute ``REPO_ROOT``, and reporting it raised ``ValueError`` *after* the
file had been written successfully -- a cosmetic failure that still returned a
non-zero exit status and looked like a data-loss bug.

Covers :func:`multilingual_safety.paths.display_path` and the two scripts whose
final report line used the unsafe form.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import pytest

from multilingual_safety.paths import REPO_ROOT, display_path
from multilingual_safety.translation import make_english_row, make_pending_row

SELECT_SCRIPT = REPO_ROOT / "scripts" / "03_select_pilot_behaviors.py"
VALIDATE_SCRIPT = REPO_ROOT / "scripts" / "05_validate_translation_pilot.py"
WIDE_REVIEW_SCRIPT = REPO_ROOT / "scripts" / "06_build_wide_review.py"

#: git-ignored (dataset-v2 storage policy), so in-repo test artifacts cannot be
#: committed by accident. A relative path must be exercised with cwd=REPO_ROOT,
#: which is why these cases cannot live entirely in tmp_path.
SCRATCH = REPO_ROOT / "data" / "v2" / "tmp"

LANGUAGES = ("en", "ar", "ha", "yo", "sw", "hi", "ko")

# Invented placeholder, deliberately not drawn from any benchmark.
FIXTURE_TEXT = "Describe how a public library organises its catalogue."
FIXTURE_COMMON = {
    "behavior_id": "beh_0123456789ab",
    "source": "jailbreakbench",
    "category": "Disinformation",
    "harm_label": "benign",
    "source_text": FIXTURE_TEXT,
    "source_item_id": "0",
}


@pytest.fixture
def scratch_dir():
    """A unique git-ignored directory inside the repo, removed afterwards."""
    import shutil
    import uuid

    path = SCRATCH / f"pytest-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# display_path unit behavior
# ---------------------------------------------------------------------------
def test_absolute_path_inside_repo_is_reported_relative():
    assert display_path(REPO_ROOT / "configs" / "languages.yaml") == "configs/languages.yaml"


def test_relative_path_inside_repo_does_not_raise(monkeypatch):
    """The regression: this input used to raise ValueError."""
    monkeypatch.chdir(REPO_ROOT)
    assert display_path("configs/languages.yaml") == "configs/languages.yaml"


def test_nonexistent_path_inside_repo_is_still_reported_relative():
    """Reporting happens for files just written; it must not require existence."""
    target = REPO_ROOT / "data" / "v2" / "interim" / "does-not-exist.jsonl"
    assert not target.exists()
    assert display_path(target) == "data/v2/interim/does-not-exist.jsonl"


def test_path_outside_repo_falls_back_to_the_original_string(tmp_path):
    outside = tmp_path / "elsewhere.jsonl"
    assert display_path(outside) == str(outside)


def test_relative_path_resolving_outside_repo_falls_back(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert display_path("elsewhere.jsonl") == "elsewhere.jsonl"


def test_accepts_str_as_well_as_path():
    assert display_path(str(REPO_ROOT / "LICENSE")) == "LICENSE"


@pytest.mark.parametrize(
    "value",
    ["", ".", "..", "data/v2", "/", "relative/../weird/path.jsonl"],
)
def test_never_raises_on_odd_inputs(value):
    assert isinstance(display_path(value), str)


# ---------------------------------------------------------------------------
# scripts/03 -- selection
# ---------------------------------------------------------------------------
def _run(script, args, cwd=REPO_ROOT):
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=cwd, capture_output=True, text=True, check=False,
    )


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "raw" / "jailbreakbench" / "harmful-behaviors.csv").exists(),
    reason="JailbreakBench source CSVs are not materialised",
)
def test_select_pilot_accepts_a_relative_out_path(scratch_dir):
    relative = (scratch_dir / "selection.jsonl").relative_to(REPO_ROOT)
    result = _run(SELECT_SCRIPT, ["--out", str(relative)])

    assert "ValueError" not in result.stderr, result.stderr
    assert "Traceback" not in result.stderr, result.stderr
    assert result.returncode == 0, result.stderr
    assert f"-> {relative}" in result.stdout
    assert (REPO_ROOT / relative).exists()


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "raw" / "jailbreakbench" / "harmful-behaviors.csv").exists(),
    reason="JailbreakBench source CSVs are not materialised",
)
def test_relative_and_absolute_out_produce_identical_files(scratch_dir):
    """Contents and destination are unchanged -- only the report line differed."""
    via_relative = scratch_dir / "relative.jsonl"
    via_absolute = scratch_dir / "absolute.jsonl"

    assert _run(SELECT_SCRIPT, ["--out", str(via_relative.relative_to(REPO_ROOT))]).returncode == 0
    assert _run(SELECT_SCRIPT, ["--out", str(via_absolute)]).returncode == 0

    digest = hashlib.sha256(via_relative.read_bytes()).hexdigest()
    assert digest == hashlib.sha256(via_absolute.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# scripts/05 -- pilot validation
# ---------------------------------------------------------------------------
@pytest.fixture
def pilot_outputs(scratch_dir):
    """A minimal but schema-valid outputs file: one behavior, all seven languages.

    Built with the library row constructors rather than copied from a real
    artifact, so the test carries no corpus text and does not depend on whether
    a pilot has been run locally.
    """
    rows = [make_english_row(**FIXTURE_COMMON)]
    rows += [
        make_pending_row(language=code, **FIXTURE_COMMON)
        for code in LANGUAGES
        if code != "en"
    ]
    path = scratch_dir / "outputs.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
    return path


def test_validate_pilot_accepts_a_relative_out_path(scratch_dir, pilot_outputs):
    relative_out = (scratch_dir / "validation.json").relative_to(REPO_ROOT)
    result = _run(VALIDATE_SCRIPT, [
        "--outputs", str(pilot_outputs.relative_to(REPO_ROOT)),
        "--out", str(relative_out),
    ])

    assert "ValueError" not in result.stderr, result.stderr
    assert "Traceback" not in result.stderr, result.stderr
    assert f"wrote {relative_out}" in result.stdout
    assert (REPO_ROOT / relative_out).exists()


def test_validate_pilot_out_outside_the_repo_is_reported_verbatim(tmp_path, pilot_outputs):
    out = tmp_path / "validation.json"
    result = _run(VALIDATE_SCRIPT, [
        "--outputs", str(pilot_outputs), "--out", str(out),
    ])

    assert "ValueError" not in result.stderr, result.stderr
    assert "Traceback" not in result.stderr, result.stderr
    assert f"wrote {out}" in result.stdout
    assert out.exists()


# ---------------------------------------------------------------------------
# scripts/06 -- wide review
#
# Worse than cosmetic here: the first unsafe call sat in the metadata payload
# (`csv_path`), which is built AFTER the CSV is written but BEFORE the metadata
# sidecar. A relative --csv therefore left a CSV on disk with no metadata beside
# it, and scripts/07 validates the CSV *against* that metadata.
# ---------------------------------------------------------------------------
@pytest.fixture
def wide_review_inputs(scratch_dir):
    """Pilot outputs plus an empty response log: enough to build a sheet."""
    rows = [make_english_row(**FIXTURE_COMMON)]
    rows += [
        make_pending_row(language=code, **FIXTURE_COMMON)
        for code in LANGUAGES
        if code != "en"
    ]
    outputs = scratch_dir / "outputs.jsonl"
    with outputs.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
    responses = scratch_dir / "responses.jsonl"
    responses.write_text("", encoding="utf-8")
    return outputs, responses


def _run_wide_review(inputs, csv_path, metadata_path, scratch_dir, relative):
    outputs, responses = inputs

    def arg(path):
        return str(path.relative_to(REPO_ROOT)) if relative else str(path)

    return _run(WIDE_REVIEW_SCRIPT, [
        "--outputs", arg(outputs),
        "--responses", arg(responses),
        "--decisions", arg(scratch_dir / "decisions.jsonl"),
        "--csv", arg(csv_path),
        "--metadata", arg(metadata_path),
        "--allow-incomplete",
    ])


def test_wide_review_accepts_relative_paths(wide_review_inputs, scratch_dir):
    csv_path = scratch_dir / "wide.csv"
    metadata_path = scratch_dir / "wide_metadata.json"
    result = _run_wide_review(
        wide_review_inputs, csv_path, metadata_path, scratch_dir, relative=True)

    assert "ValueError" not in result.stderr, result.stderr
    assert "Traceback" not in result.stderr, result.stderr
    assert result.returncode == 0, result.stderr
    assert f"wrote {csv_path.relative_to(REPO_ROOT)}" in result.stdout
    assert f"wrote {metadata_path.relative_to(REPO_ROOT)}" in result.stdout


def test_wide_review_writes_the_metadata_sidecar_not_just_the_csv(
    wide_review_inputs, scratch_dir
):
    """The regression: the crash used to abort between the two writes."""
    csv_path = scratch_dir / "wide.csv"
    metadata_path = scratch_dir / "wide_metadata.json"
    _run_wide_review(wide_review_inputs, csv_path, metadata_path, scratch_dir, relative=True)

    assert csv_path.exists(), "CSV missing"
    assert metadata_path.exists(), "metadata sidecar missing -- run aborted between writes"


def test_wide_review_metadata_records_repo_relative_paths(wide_review_inputs, scratch_dir):
    csv_path = scratch_dir / "wide.csv"
    metadata_path = scratch_dir / "wide_metadata.json"
    _run_wide_review(wide_review_inputs, csv_path, metadata_path, scratch_dir, relative=True)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["csv_path"] == str(csv_path.relative_to(REPO_ROOT))
    assert metadata["decisions_source"] == str(
        (scratch_dir / "decisions.jsonl").relative_to(REPO_ROOT))


def test_wide_review_output_is_identical_via_relative_and_absolute_paths(
    wide_review_inputs, scratch_dir
):
    """Only the report changed; contents and destinations did not."""
    csv_path = scratch_dir / "wide.csv"
    rel_meta = scratch_dir / "meta_rel.json"
    abs_meta = scratch_dir / "meta_abs.json"

    _run_wide_review(wide_review_inputs, csv_path, rel_meta, scratch_dir, relative=True)
    first = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    _run_wide_review(wide_review_inputs, csv_path, abs_meta, scratch_dir, relative=False)

    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == first

    a = json.loads(rel_meta.read_text(encoding="utf-8"))
    b = json.loads(abs_meta.read_text(encoding="utf-8"))
    # An absolute in-repo path resolves to the same repo-relative string, so the
    # recorded values are unchanged; only run-scoped fields may differ.
    assert a["csv_path"] == b["csv_path"]
    assert a["decisions_source"] == b["decisions_source"]
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} <= {"run_id", "generated_at"}


def test_wide_review_paths_outside_the_repo_are_reported_verbatim(
    wide_review_inputs, scratch_dir, tmp_path
):
    csv_path = tmp_path / "wide.csv"
    metadata_path = tmp_path / "wide_metadata.json"
    result = _run_wide_review(
        wide_review_inputs, csv_path, metadata_path, scratch_dir, relative=False)

    assert "Traceback" not in result.stderr, result.stderr
    assert result.returncode == 0, result.stderr
    assert f"wrote {csv_path}" in result.stdout
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["csv_path"] == str(csv_path)
