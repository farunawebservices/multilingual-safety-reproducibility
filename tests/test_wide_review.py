"""Wide-format review CSV: schema, completeness, and sourcing.

These tests read the generated sheet from disk and re-derive what it should
contain from the response log, so a bug in the builder cannot certify itself.
They skip cleanly when the sheet has not been generated in this working tree.
"""

from __future__ import annotations

import csv
import json

import pytest

from multilingual_safety.languages import LANGUAGE_CODES, PIVOT_LANGUAGE
from multilingual_safety.paths import DATA_DIR, INTERIM_DIR
from multilingual_safety.providers import STATUS_OK, ResponseLog, make_request_id

CSV_PATH = DATA_DIR / "review" / "translation_pilot_wide_review.csv"
METADATA_PATH = DATA_DIR / "review" / "translation_pilot_wide_review_metadata.json"
OUTPUTS_PATH = INTERIM_DIR / "translation_pilot_outputs.jsonl"
RESPONSES_PATH = INTERIM_DIR / "translation_pilot_responses.jsonl"

EXPECTED_COLUMNS = [
    "behavior_id", "source", "source_item_id", "category", "harm_label",
    "en", "ar", "ha", "yo", "sw", "hi", "ko",
]
CONSISTENCY_FIELDS = ("source", "source_item_id", "category", "harm_label")


@pytest.fixture(scope="module")
def sheet() -> list[dict[str, str]]:
    if not CSV_PATH.exists():
        pytest.skip("wide review CSV has not been generated")
    with CSV_PATH.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def raw_rows() -> list[list[str]]:
    if not CSV_PATH.exists():
        pytest.skip("wide review CSV has not been generated")
    with CSV_PATH.open(encoding="utf-8", newline="") as fh:
        return list(csv.reader(fh))


@pytest.fixture(scope="module")
def effective() -> dict:
    if not RESPONSES_PATH.exists():
        pytest.skip("no response log")
    return ResponseLog(RESPONSES_PATH).effective()


# --- schema ----------------------------------------------------------------
def test_columns_are_exactly_the_twelve(raw_rows):
    assert raw_rows[0] == EXPECTED_COLUMNS
    assert len(raw_rows[0]) == 12


def test_exactly_twenty_rows(sheet):
    assert len(sheet) == 20


def test_one_row_per_behavior_id(sheet):
    ids = [row["behavior_id"] for row in sheet]
    assert len(ids) == len(set(ids)) == 20


def test_every_row_has_twelve_fields(raw_rows):
    assert {len(row) for row in raw_rows} == {12}


def test_file_is_utf8_without_bom():
    raw = CSV_PATH.read_bytes()
    raw.decode("utf-8")
    assert not raw.startswith(b"\xef\xbb\xbf")


# --- completeness ----------------------------------------------------------
def test_no_language_cell_is_empty(sheet):
    empty = [
        f"{row['behavior_id']}:{lang}"
        for row in sheet
        for lang in LANGUAGE_CODES
        if not row[lang].strip()
    ]
    assert not empty, f"empty language cells: {empty[:6]}"


def test_all_seven_languages_present(sheet):
    assert set(LANGUAGE_CODES) <= set(sheet[0])
    assert len(LANGUAGE_CODES) == 7


def test_every_behavior_has_seven_language_rows_upstream():
    if not OUTPUTS_PATH.exists():
        pytest.skip("no pilot outputs")
    groups: dict[str, set[str]] = {}
    for line in OUTPUTS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            groups.setdefault(row["behavior_id"], set()).add(row["language"])
    wrong = {b: sorted(v) for b, v in groups.items() if len(v) != len(LANGUAGE_CODES)}
    assert not wrong, f"behaviors without 7 language rows: {wrong}"
    assert len(groups) == 20


# --- sourcing --------------------------------------------------------------
def test_english_comes_from_the_canonical_source(sheet):
    if not OUTPUTS_PATH.exists():
        pytest.skip("no pilot outputs")
    canonical = {}
    for line in OUTPUTS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["language"] == PIVOT_LANGUAGE:
                canonical[row["behavior_id"]] = row["source_text"]
    for row in sheet:
        assert row["en"] == canonical[row["behavior_id"]]


def test_non_english_cells_match_the_latest_successful_response(sheet, effective):
    for row in sheet:
        for lang in LANGUAGE_CODES:
            if lang == PIVOT_LANGUAGE:
                continue
            response = effective[make_request_id(row["behavior_id"], lang)]
            assert response["status"] == STATUS_OK
            assert row[lang] == response["response_text"]


def test_no_error_response_reaches_the_sheet(effective):
    """Superseded ERROR records must not contribute any cell."""
    assert all(r["status"] == STATUS_OK for r in effective.values())
    log = ResponseLog(RESPONSES_PATH)
    summary = log.retry_summary()
    assert summary["attempts_by_status"]["ERROR"] > 0, "the ERROR history still exists"
    assert summary["effective_by_status"]["ERROR"] == 0


# --- behavior-level metadata consistency -----------------------------------
def test_metadata_is_consistent_across_a_behaviors_languages():
    if not OUTPUTS_PATH.exists():
        pytest.skip("no pilot outputs")
    groups: dict[str, list[dict]] = {}
    for line in OUTPUTS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            groups.setdefault(row["behavior_id"], []).append(row)
    inconsistent = {
        bid: field
        for bid, group in groups.items()
        for field in CONSISTENCY_FIELDS
        if len({str(r.get(field)) for r in group}) > 1
    }
    assert not inconsistent, f"inconsistent metadata: {inconsistent}"


def test_sheet_metadata_matches_the_source_rows(sheet):
    if not OUTPUTS_PATH.exists():
        pytest.skip("no pilot outputs")
    anchors = {}
    for line in OUTPUTS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["language"] == PIVOT_LANGUAGE:
                anchors[row["behavior_id"]] = row
    for row in sheet:
        anchor = anchors[row["behavior_id"]]
        assert row["source"] == anchor["source"]
        assert row["source_item_id"] == str(anchor["source_item_id"])
        assert row["category"] == anchor["category"]
        assert row["harm_label"] == anchor["harm_label"]


# --- companion metadata and placement --------------------------------------
def test_companion_metadata_records_counts_and_hashes():
    if not METADATA_PATH.exists():
        pytest.skip("metadata not generated")
    payload = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert payload["row_count"] == 20
    assert payload["columns"] == EXPECTED_COLUMNS
    assert payload["column_count"] == 12
    for lang in LANGUAGE_CODES:
        assert payload["language_counts"][lang]["non_empty"] == 20
        assert payload["language_counts"][lang]["empty"] == 0
    hashes = payload["source_hashes"]
    for key in ("outputs_jsonl", "responses_jsonl", "csv_sha256"):
        assert hashes.get(key) and len(hashes[key]) == 64
    assert len(payload["row_hashes"]) == 20


def test_metadata_csv_hash_is_current():
    if not METADATA_PATH.exists():
        pytest.skip("metadata not generated")
    from multilingual_safety.hashing import file_sha256

    payload = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert payload["source_hashes"]["csv_sha256"] == file_sha256(CSV_PATH)


def test_no_row_is_marked_human_reviewed():
    if not METADATA_PATH.exists():
        pytest.skip("metadata not generated")
    payload = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert payload["human_reviewed_rows"] == 0
    if OUTPUTS_PATH.exists():
        statuses = [
            json.loads(line)["translation_status"]
            for line in OUTPUTS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert "HUMAN_REVIEWED" not in statuses


def test_csv_is_git_ignored():
    import subprocess

    from multilingual_safety.paths import REPO_ROOT

    for path in (CSV_PATH, METADATA_PATH):
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(path)], cwd=REPO_ROOT, check=False
        )
        assert result.returncode == 0, f"{path} must be git-ignored: it contains prompt text"
