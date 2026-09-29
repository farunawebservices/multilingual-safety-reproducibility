#!/usr/bin/env python3
"""Stage 2.4 -- build the full side-by-side review CSV, then validate it.

Pivots ``data/interim/full/full_translation_outputs.jsonl`` (1,400 long-format
rows) into 200 wide rows: one per behavior, one column per language.

Cell sourcing
-------------
* ``en`` -- the ``CANONICAL_SOURCE`` row's text. Never a translation.
* non-English -- only from a row whose status is
  ``MACHINE_TRANSLATED_UNCHECKED`` with non-empty ``translated_text``. Any other
  status contributes nothing, and a missing cell is a hard failure rather than a
  blank.

The long-format manifest is opened **read-only**; this script writes only the
CSV and its metadata sidecar.

**The CSV contains prompt text in all seven languages** and lives under
git-ignored ``data/review/``. A wide sheet is the most copy-pasteable form the
corpus takes, so it is never committed (THIRD_PARTY_DATASETS.md).

No row is marked ``HUMAN_REVIEWED``: this is the artifact a reviewer works
*from*, not the record of a verdict.

Outputs
-------
    data/review/full_translation_wide_review.csv           (git-ignored)
    data/review/full_translation_wide_review_metadata.json (git-ignored)

Usage
-----
    python scripts/14_build_full_wide_review.py --dry-run
    python scripts/14_build_full_wide_review.py
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multilingual_safety.hashing import file_sha256, text_sha256
from multilingual_safety.languages import LANGUAGE_CODES, PIVOT_LANGUAGE
from multilingual_safety.paths import DATA_DIR, INTERIM_DIR, REPO_ROOT, display_path
from multilingual_safety.run_context import stage_report_header
from multilingual_safety.translation import TranslationRow, TranslationStatus

OUTPUTS_PATH = INTERIM_DIR / "full" / "full_translation_outputs.jsonl"
REVIEW_DIR = DATA_DIR / "review"
CSV_PATH = REVIEW_DIR / "full_translation_wide_review.csv"
METADATA_PATH = REVIEW_DIR / "full_translation_wide_review_metadata.json"

COLUMNS: tuple[str, ...] = (
    "behavior_id", "source", "source_item_id", "category", "harm_label", *LANGUAGE_CODES,
)
NON_PIVOT_LANGUAGES = tuple(c for c in LANGUAGE_CODES if c != PIVOT_LANGUAGE)
CONSISTENCY_FIELDS = ("source", "source_item_id", "category", "harm_label")

EXPECTED_BEHAVIORS = 200
EXPECTED_CELLS = EXPECTED_BEHAVIORS * len(LANGUAGE_CODES)   # 1400
EXPECTED_CATEGORIES = 10


def load_rows(path: Path) -> list[TranslationRow]:
    rows: list[TranslationRow] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(TranslationRow.from_dict(json.loads(line)))
        except Exception as exc:
            raise SystemExit(f"{path.name}:{lineno}: cannot parse row: {exc}") from exc
    return rows


def build(rows: list[TranslationRow]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    groups: dict[str, list[TranslationRow]] = defaultdict(list)
    for row in rows:
        groups[row.behavior_id].append(row)

    records: list[dict[str, str]] = []
    missing_cells: list[str] = []
    duplicate_languages: dict[str, list[str]] = {}
    inconsistent: dict[str, dict[str, list[str]]] = {}
    wrong_status: list[str] = []

    for behavior_id in sorted(groups):
        group = groups[behavior_id]

        seen = Counter(r.language for r in group)
        repeated = sorted(lang for lang, n in seen.items() if n > 1)
        if repeated:
            duplicate_languages[behavior_id] = repeated

        problems = {
            field: sorted({str(getattr(r, field)) for r in group})
            for field in CONSISTENCY_FIELDS
            if len({str(getattr(r, field)) for r in group}) > 1
        }
        if problems:
            inconsistent[behavior_id] = problems

        by_language = {r.language: r for r in group}
        anchor = by_language.get(PIVOT_LANGUAGE) or group[0]
        record: dict[str, str] = {
            "behavior_id": behavior_id,
            "source": anchor.source,
            "source_item_id": anchor.source_item_id or "",
            "category": anchor.category,
            "harm_label": anchor.harm_label,
        }

        for language in LANGUAGE_CODES:
            row = by_language.get(language)
            if row is None:
                missing_cells.append(f"{behavior_id}:{language}")
                record[language] = ""
                continue
            if language == PIVOT_LANGUAGE:
                if row.translation_status != TranslationStatus.CANONICAL_SOURCE.value:
                    wrong_status.append(f"{behavior_id}:en={row.translation_status}")
                value = row.translated_text or row.source_text
            else:
                # Only a successful machine translation contributes a cell.
                if row.translation_status != TranslationStatus.MACHINE_TRANSLATED_UNCHECKED.value:
                    wrong_status.append(f"{behavior_id}:{language}={row.translation_status}")
                    value = ""
                else:
                    value = row.translated_text or ""
            if not value.strip():
                missing_cells.append(f"{behavior_id}:{language}")
            record[language] = value

        records.append(record)

    diagnostics = {
        "missing_cells": missing_cells,
        "duplicate_languages": duplicate_languages,
        "inconsistent_metadata": inconsistent,
        "wrong_status": wrong_status,
    }
    return records, diagnostics


def validate(csv_path: Path, records: list[dict[str, str]]) -> tuple[dict[str, Any], list[str]]:
    """Re-read the written CSV and check it independently of the builder."""
    with csv_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        data = list(reader)

    parsed = [dict(zip(header, row, strict=False)) for row in data]
    non_empty = {lang: sum(1 for r in parsed if r[lang].strip()) for lang in LANGUAGE_CODES}
    empty = {lang: sum(1 for r in parsed if not r[lang].strip()) for lang in LANGUAGE_CODES}
    ids = [r["behavior_id"] for r in parsed]
    duplicates = sorted({i for i, n in Counter(ids).items() if n > 1})
    categories = sorted({r["category"] for r in parsed})
    arms = Counter(r["harm_label"] for r in parsed)

    report = {
        "rows": len(data),
        "columns": header,
        "n_columns": len(header),
        "non_empty_per_language": non_empty,
        "empty_per_language": empty,
        "duplicate_behavior_ids": len(duplicates),
        "n_categories": len(categories),
        "categories": categories,
        "counts_by_arm": dict(sorted(arms.items())),
        "ragged_rows": sorted({len(r) for r in data} - {len(header)}),
    }

    failures: list[str] = []
    if len(data) != EXPECTED_BEHAVIORS:
        failures.append(f"rows: expected {EXPECTED_BEHAVIORS}, got {len(data)}")
    if header != list(COLUMNS):
        failures.append(f"columns: expected {list(COLUMNS)}, got {header}")
    for lang in LANGUAGE_CODES:
        if empty[lang]:
            failures.append(f"{lang}: {empty[lang]} empty cell(s)")
    if duplicates:
        failures.append(f"duplicate behavior_ids: {duplicates[:5]}")
    if len(categories) != EXPECTED_CATEGORIES:
        failures.append(f"categories: expected {EXPECTED_CATEGORIES}, got {len(categories)}")
    if arms.get("harmful") != 100 or arms.get("benign") != 100:
        failures.append(f"arms: expected 100/100, got {dict(arms)}")
    if report["ragged_rows"]:
        failures.append(f"ragged rows: {report['ragged_rows']}")
    if len(parsed) != len(records):
        failures.append(f"round-trip row count differs: {len(records)} -> {len(parsed)}")
    return report, failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--outputs", type=Path, default=OUTPUTS_PATH)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.outputs.exists():
        print(f"missing {args.outputs}", file=sys.stderr)
        return 1

    header = stage_report_header("stage2.4_build_full_wide_review", {"columns": list(COLUMNS)})
    print(f"run_id:     {header['run_id']}")
    print(f"git commit: {header['git_commit']}  (dirty={header['git_dirty']})")

    manifest_before = file_sha256(args.outputs)
    rows = load_rows(args.outputs)
    records, diagnostics = build(rows)

    print(f"\nlong rows in : {len(rows)}")
    print(f"wide records : {len(records)}  (expected {EXPECTED_BEHAVIORS})")
    print(f"cells        : {len(records) * len(LANGUAGE_CODES)}  (expected {EXPECTED_CELLS})")
    print(f"missing cells: {len(diagnostics['missing_cells'])}")
    print(f"duplicate language rows : {len(diagnostics['duplicate_languages'])}")
    print(f"inconsistent metadata   : {len(diagnostics['inconsistent_metadata'])}")
    print(f"rows with wrong status  : {len(diagnostics['wrong_status'])}")

    failures: list[str] = []
    if len(records) != EXPECTED_BEHAVIORS:
        failures.append(f"expected {EXPECTED_BEHAVIORS} behaviors, got {len(records)}")
    if diagnostics["missing_cells"]:
        failures.append(
            f"{len(diagnostics['missing_cells'])} missing cell(s): "
            f"{diagnostics['missing_cells'][:6]}"
        )
    if diagnostics["duplicate_languages"]:
        failures.append(
            f"{len(diagnostics['duplicate_languages'])} behavior(s) with duplicate language rows"
        )
    if diagnostics["inconsistent_metadata"]:
        failures.append(
            f"{len(diagnostics['inconsistent_metadata'])} behavior(s) with inconsistent metadata"
        )
    if diagnostics["wrong_status"]:
        failures.append(f"{len(diagnostics['wrong_status'])} row(s) with an unusable status")

    for failure in failures:
        print(f"  FAIL: {failure}", file=sys.stderr)
    if failures:
        print("\nNothing written.", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\nDry run: nothing written.")
        return 0

    # --- write ------------------------------------------------------------
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(COLUMNS), quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n"
        )
        writer.writeheader()
        writer.writerows(records)

    validation, validation_failures = validate(args.csv, records)

    print("\nVALIDATION")
    print(f"  rows                  : {validation['rows']}")
    print(f"  columns               : {validation['n_columns']}  {validation['columns']}")
    print(f"  duplicate behavior ids: {validation['duplicate_behavior_ids']}")
    print(f"  categories            : {validation['n_categories']}")
    print(f"  counts by arm         : {validation['counts_by_arm']}")
    print("  per language (non-empty / empty):")
    for lang in LANGUAGE_CODES:
        marker = "  (canonical source)" if lang == PIVOT_LANGUAGE else ""
        print(f"    {lang:<4} {validation['non_empty_per_language'][lang]:>4} / "
              f"{validation['empty_per_language'][lang]:<4}{marker}")

    metadata = {
        **header,
        # display_path, not relative_to: a relative CLI path such as
        # `--csv data/review/x.csv` is not lexically under the absolute
        # REPO_ROOT, so relative_to raised here -- after the CSV had been
        # written and before the metadata sidecar was.
        "csv_path": display_path(args.csv),
        "git_ignored": True,
        "contains_prompt_text": True,
        "encoding": "utf-8",
        "csv_dialect": {
            "quoting": "QUOTE_MINIMAL", "delimiter": ",", "quotechar": '"',
            "lineterminator": "\\r\\n", "rfc": "RFC 4180",
        },
        "row_count": len(records),
        "column_count": len(COLUMNS),
        "columns": list(COLUMNS),
        "languages": list(LANGUAGE_CODES),
        "pivot_language": PIVOT_LANGUAGE,
        "validation": validation,
        "validation_passed": not validation_failures,
        "validation_failures": validation_failures,
        "language_counts": {
            lang: {
                "non_empty": validation["non_empty_per_language"][lang],
                "empty": validation["empty_per_language"][lang],
            }
            for lang in LANGUAGE_CODES
        },
        "n_cells": len(records) * len(LANGUAGE_CODES),
        "n_translated_cells": len(records) * len(NON_PIVOT_LANGUAGES),
        "n_canonical_source_cells": len(records),
        "human_reviewed_rows": 0,
        "long_manifest_unchanged": file_sha256(args.outputs) == manifest_before,
        "source_hashes": {
            "outputs_jsonl": manifest_before,
            "csv_sha256": file_sha256(args.csv),
        },
        "row_hashes": {
            r["behavior_id"]: text_sha256("\x1f".join(r[c] for c in COLUMNS)) for r in records
        },
        "notes": [
            "English is canonical source text, not a translation.",
            "Non-English cells come only from MACHINE_TRANSLATED_UNCHECKED rows with "
            "non-empty text; any other status yields a hard failure, never a blank.",
            "No row is HUMAN_REVIEWED; every non-English row still requires review.",
            "The long-format manifest was opened read-only.",
        ],
    }
    args.metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(args.csv)], cwd=REPO_ROOT, check=False
    ).returncode == 0
    print(f"\n  csv git-ignored       : {ignored}")
    print(f"  long manifest unchanged: {metadata['long_manifest_unchanged']}")
    if not ignored:
        validation_failures.append("the review CSV is NOT git-ignored; it contains prompt text")
    if not metadata["long_manifest_unchanged"]:
        validation_failures.append("the long-format manifest changed")

    print(f"\nwrote {display_path(args.csv)}  ({args.csv.stat().st_size} bytes)")
    print(f"wrote {display_path(args.metadata)}")

    print("\nRESULT")
    if validation_failures:
        for failure in validation_failures:
            print(f"  FAIL: {failure}")
        return 1
    print("  all validation checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
