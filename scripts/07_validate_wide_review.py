#!/usr/bin/env python3
"""Stage 1.4 -- validate the wide review CSV independently of the writer.

Re-reads the CSV from disk with a plain ``csv.reader`` and checks it against the
source JSONL and the response log. Deliberately independent: it recomputes the
expected values rather than trusting the metadata file the builder emitted, so a
bug in the builder cannot certify itself.

Reports rows, columns, non-empty and pending counts per language, duplicate
behavior IDs, and inconsistent metadata groups.

Usage
-----
    python scripts/07_validate_wide_review.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multilingual_safety.hashing import file_sha256
from multilingual_safety.languages import LANGUAGE_CODES, PIVOT_LANGUAGE
from multilingual_safety.paths import DATA_DIR, INTERIM_DIR, REPO_ROOT
from multilingual_safety.providers import STATUS_OK, ResponseLog, make_request_id

REVIEW_DIR = DATA_DIR / "review"
CSV_PATH = REVIEW_DIR / "translation_pilot_wide_review.csv"
METADATA_PATH = REVIEW_DIR / "translation_pilot_wide_review_metadata.json"
OUTPUTS_PATH = INTERIM_DIR / "translation_pilot_outputs.jsonl"
RESPONSES_PATH = INTERIM_DIR / "translation_pilot_responses.jsonl"

NON_PIVOT_LANGUAGES = tuple(c for c in LANGUAGE_CODES if c != PIVOT_LANGUAGE)

#: The original 12 columns must remain an exact PREFIX of the schema.
BASE_COLUMNS = [
    "behavior_id", "source", "source_item_id", "category", "harm_label",
    *LANGUAGE_CODES,
]
#: The sheet carries the twelve content columns only; review decisions and
#: reviewer metadata live in the companion metadata file.
EXPECTED_COLUMNS = list(BASE_COLUMNS)
CONSISTENCY_FIELDS = ("source", "source_item_id", "category", "harm_label")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--outputs", type=Path, default=OUTPUTS_PATH)
    parser.add_argument("--responses", type=Path, default=RESPONSES_PATH)
    args = parser.parse_args()
    # Resolve so a relative path from the CLI still renders against REPO_ROOT.
    for name in ("csv", "metadata", "outputs", "responses"):
        setattr(args, name, getattr(args, name).resolve())

    if not args.csv.exists():
        print(f"missing {args.csv}; run scripts/06_build_wide_review.py first", file=sys.stderr)
        return 1

    failures: list[str] = []

    # --- read the CSV back with a plain reader ---------------------------
    with args.csv.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        data_rows = list(reader)

    print(f"csv        : {args.csv.relative_to(REPO_ROOT)}")
    print("encoding   : utf-8")
    print(f"rows       : {len(data_rows)}")
    print(f"columns    : {len(header)}  -> {', '.join(header)}")

    if header != EXPECTED_COLUMNS:
        failures.append(f"column mismatch: expected {EXPECTED_COLUMNS}, got {header}")
    if len(header) != 12:
        failures.append(f"expected exactly 12 columns, got {len(header)}")
    widths = {len(row) for row in data_rows}
    if widths - {len(header)}:
        failures.append(f"ragged rows: field counts {sorted(widths)} vs header {len(header)}")

    records = [dict(zip(header, row, strict=False)) for row in data_rows]

    # --- duplicates -------------------------------------------------------
    ids = [r["behavior_id"] for r in records]
    duplicates = sorted({i for i, n in Counter(ids).items() if n > 1})
    print(f"\nduplicate behavior IDs: {len(duplicates)}  {duplicates or ''}")
    if duplicates:
        failures.append(f"duplicate behavior_id: {duplicates}")

    # --- per-language counts ---------------------------------------------
    non_empty = {lang: sum(1 for r in records if r[lang].strip()) for lang in LANGUAGE_CODES}
    pending = {lang: sum(1 for r in records if not r[lang].strip()) for lang in LANGUAGE_CODES}
    print("\nper-language:")
    print(f"  {'lang':<6} {'non-empty':>10} {'pending':>9}")
    for lang in LANGUAGE_CODES:
        marker = "  (canonical source)" if lang == PIVOT_LANGUAGE else ""
        print(f"  {lang:<6} {non_empty[lang]:>10} {pending[lang]:>9}{marker}")

    if non_empty[PIVOT_LANGUAGE] != len(records):
        failures.append(
            f"{pending[PIVOT_LANGUAGE]} row(s) have no English canonical text"
        )

    # --- metadata consistency across the source JSONL ---------------------
    inconsistent: dict[str, dict[str, list[str]]] = {}
    if args.outputs.exists():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for line in args.outputs.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                groups[row["behavior_id"]].append(row)
        for behavior_id, group in sorted(groups.items()):
            problems = {}
            for field in CONSISTENCY_FIELDS:
                values = sorted({str(r.get(field)) for r in group})
                if len(values) > 1:
                    problems[field] = values
            if problems:
                inconsistent[behavior_id] = problems
    print(f"\ninconsistent metadata groups: {len(inconsistent)}  "
          f"{sorted(inconsistent) if inconsistent else ''}")
    if inconsistent:
        failures.append(f"inconsistent metadata in {len(inconsistent)} group(s)")

    # --- independent recomputation against the response log ---------------
    if args.responses.exists():
        effective = ResponseLog(args.responses).effective()
        mismatches: list[str] = []
        error_leaks: list[str] = []
        for record in records:
            for lang in LANGUAGE_CODES:
                if lang == PIVOT_LANGUAGE:
                    continue
                response = effective.get(make_request_id(record["behavior_id"], lang))
                cell = record[lang]
                if response is None or response.get("status") != STATUS_OK:
                    # No effective OK => the cell must be empty.
                    if cell.strip():
                        error_leaks.append(f"{record['behavior_id']}:{lang}")
                elif cell != (response.get("response_text") or ""):
                    mismatches.append(f"{record['behavior_id']}:{lang}")
        print(f"cells disagreeing with the response log: {len(mismatches)}")
        print(f"non-OK responses leaked into cells     : {len(error_leaks)}")
        if mismatches:
            failures.append(f"{len(mismatches)} cell(s) disagree with the effective response")
        if error_leaks:
            failures.append(f"{len(error_leaks)} non-OK response(s) leaked into the CSV")

    # --- completeness: no language cell may be empty ----------------------
    empty = [
        f"{r['behavior_id']}:{lang}"
        for r in records
        for lang in LANGUAGE_CODES
        if not r[lang].strip()
    ]
    print(f"\nempty language cells: {len(empty)} of {len(records) * len(LANGUAGE_CODES)}")
    if empty:
        failures.append(f"{len(empty)} empty language cell(s): {empty[:6]}")

    # --- every behavior has exactly seven language rows in the source ------
    if args.outputs.exists():
        counts = {
            bid: len({r["language"] for r in group})
            for bid, group in sorted(groups.items())
        }
        wrong = {b: n for b, n in counts.items() if n != len(LANGUAGE_CODES)}
        print(f"behaviors with exactly 7 language rows: "
              f"{len(counts) - len(wrong)} / {len(counts)}")
        if wrong:
            failures.append(f"{len(wrong)} behavior(s) lack 7 language rows: {wrong}")

    # --- review layer (from metadata, not from the sheet) ------------------
    if args.outputs.exists():
        reviewed = sum(
            1 for group in groups.values()
            for r in group
            if r.get("translation_status") == "HUMAN_REVIEWED"
        )
        print(f"rows marked HUMAN_REVIEWED             : {reviewed}")
        if reviewed:
            failures.append(f"{reviewed} row(s) marked HUMAN_REVIEWED")

    # --- companion metadata ----------------------------------------------
    if args.metadata.exists():
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        print(f"\nmetadata   : {args.metadata.relative_to(REPO_ROOT)}")
        print(f"  row_count matches      : {metadata.get('row_count') == len(records)}")
        print(f"  columns match          : {metadata.get('columns') == header}")
        print(f"  csv sha256 matches     : "
              f"{metadata.get('source_hashes', {}).get('csv_sha256') == file_sha256(args.csv)}")
        print(f"  human_reviewed_rows    : {metadata.get('human_reviewed_rows')}")
        if metadata.get("row_count") != len(records):
            failures.append("metadata row_count disagrees with the CSV")
        if metadata.get("columns") != header:
            failures.append("metadata columns disagree with the CSV")
        if metadata.get("source_hashes", {}).get("csv_sha256") != file_sha256(args.csv):
            failures.append("metadata csv_sha256 is stale")
        if metadata.get("human_reviewed_rows"):
            failures.append("a row is marked HUMAN_REVIEWED; that must not happen here")
    else:
        failures.append(f"missing companion metadata {args.metadata}")

    # --- git-ignored ------------------------------------------------------
    import subprocess

    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(args.csv)], cwd=REPO_ROOT, check=False
    ).returncode == 0
    print(f"\ncsv is git-ignored: {ignored}")
    if not ignored:
        failures.append("the review CSV is NOT git-ignored; it contains prompt text")

    print("\nRESULT")
    if failures:
        for failure in failures:
            print(f"  FAIL: {failure}")
        print(f"\n{len(failures)} validation failure(s).", file=sys.stderr)
        return 1
    print("  all validation checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
