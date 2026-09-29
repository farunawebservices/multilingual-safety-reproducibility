#!/usr/bin/env python3
"""Stage 1.3 -- build a wide-format CSV for manual review.

One row per behavior, one column per language. This is the shape a human
reviewer actually wants: the English source and all six renderings side by side,
so a translation can be judged against its source without scrolling through
JSONL.

    behavior_id, source, source_item_id, category, harm_label, en, ar, ha, yo, sw, hi, ko

Where each value comes from
---------------------------
* **en** -- the canonical source row (``CANONICAL_SOURCE``). Never translated.
* **non-English** -- the **latest effective response** for
  ``<behavior_id>:<language>`` in the append-only response log, and *only* when
  that response is ``OK``. An ``ERROR`` is not a translation, and a request whose
  history is ``ERROR -> OK`` contributes only the ``OK``
  (``ResponseLog.effective()``).
* **still pending** -- an empty cell. Empty means "no translation exists",
  which is a different claim from "the translation is an empty string", and the
  metadata file records the two separately.

**This file contains prompt text in all seven languages** and lives under
git-ignored ``data/review/``. A wide CSV is the most copy-pasteable form the
corpus ever takes, so it must never be committed (THIRD_PARTY_DATASETS.md).

Canonical manifests are not touched. No row is marked ``HUMAN_REVIEWED``: this
script produces the artifact a reviewer works *from*, not the record of their
verdict.

Usage
-----
    python scripts/06_build_wide_review.py
    python scripts/06_build_wide_review.py --dry-run
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

from multilingual_safety.hashing import file_sha256, text_sha256
from multilingual_safety.languages import LANGUAGE_CODES, PIVOT_LANGUAGE
from multilingual_safety.paths import DATA_DIR, INTERIM_DIR, display_path
from multilingual_safety.providers import STATUS_OK, ResponseLog, make_request_id
from multilingual_safety.run_context import stage_report_header
from multilingual_safety.translation import TranslationRow, TranslationStatus

OUTPUTS_PATH = INTERIM_DIR / "translation_pilot_outputs.jsonl"
RESPONSES_PATH = INTERIM_DIR / "translation_pilot_responses.jsonl"
REVIEW_DIR = DATA_DIR / "review"
CSV_PATH = REVIEW_DIR / "translation_pilot_wide_review.csv"
METADATA_PATH = REVIEW_DIR / "translation_pilot_wide_review_metadata.json"
DECISIONS_JSONL = REVIEW_DIR / "pilot_review_decisions.jsonl"

#: Exact column order. Metadata first, then languages in canonical order.
METADATA_COLUMNS: tuple[str, ...] = (
    "behavior_id",
    "source",
    "source_item_id",
    "category",
    "harm_label",
)
NON_PIVOT_LANGUAGES: tuple[str, ...] = tuple(c for c in LANGUAGE_CODES if c != PIVOT_LANGUAGE)

#: Per-language verdict columns, e.g. ``ar_decision``. Empty when the row is
#: untranslated or carries no decision.
DECISION_COLUMNS: tuple[str, ...] = tuple(f"{lang}_decision" for lang in NON_PIVOT_LANGUAGES)

#: Behavior-level review metadata.
REVIEW_COLUMNS: tuple[str, ...] = (
    "reviewer_id",
    "reviewer_competence",
    "n_translated",
    "n_accepted",
    "review_required",
    "human_reviewed",
)

#: The CSV carries the twelve content columns only. Review decisions and
#: reviewer metadata live in the companion metadata file, not in the sheet.
BASE_COLUMNS: tuple[str, ...] = (*METADATA_COLUMNS, *LANGUAGE_CODES)
COLUMNS: tuple[str, ...] = BASE_COLUMNS

#: Behavior-level fields that must agree across every language row.
CONSISTENCY_FIELDS: tuple[str, ...] = ("source", "source_item_id", "category", "harm_label")


def load_rows(path: Path) -> list[TranslationRow]:
    """Parse the pilot rows. A malformed line raises rather than being skipped."""
    rows: list[TranslationRow] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(TranslationRow.from_dict(json.loads(line)))
        except Exception as exc:
            raise SystemExit(f"{path.name}:{lineno}: cannot parse row: {exc}") from exc
    return rows


def load_decisions(path: Path) -> dict[str, dict[str, Any]]:
    """Map ``request_id -> decision record`` from the imported review layer.

    Absent file means no review has been imported yet; every decision cell is
    then empty, which is honest rather than defaulted.
    """
    decisions: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return decisions
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path.name}:{lineno}: malformed JSON: {exc}") from exc
        decisions[row["request_id"]] = row
    return decisions


def check_consistency(group: list[TranslationRow]) -> dict[str, list[str]]:
    """Behavior-level metadata must not differ between a behavior's languages."""
    inconsistent: dict[str, list[str]] = {}
    for field in CONSISTENCY_FIELDS:
        values = sorted({str(getattr(row, field)) for row in group})
        if len(values) > 1:
            inconsistent[field] = values
    return inconsistent


def build(
    rows: list[TranslationRow],
    effective: dict[str, dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Pivot long rows into wide records, plus a diagnostics payload."""
    by_behavior: dict[str, list[TranslationRow]] = defaultdict(list)
    for row in rows:
        by_behavior[row.behavior_id].append(row)

    records: list[dict[str, str]] = []
    inconsistent_groups: dict[str, dict[str, list[str]]] = {}
    non_empty = Counter()
    pending = Counter()
    from_error = Counter()
    decision_counts: dict[str, Counter] = {lang: Counter() for lang in NON_PIVOT_LANGUAGES}
    review_rows: list[dict[str, Any]] = []
    language_row_counts: dict[str, int] = {}
    disagreements: list[dict[str, str]] = []

    for behavior_id in sorted(by_behavior):
        group = by_behavior[behavior_id]
        problems = check_consistency(group)
        if problems:
            inconsistent_groups[behavior_id] = problems

        language_row_counts[behavior_id] = len({r.language for r in group})
        anchor = next((r for r in group if r.language == PIVOT_LANGUAGE), group[0])
        record: dict[str, str] = {
            "behavior_id": behavior_id,
            "source": anchor.source,
            "source_item_id": anchor.source_item_id or "",
            "category": anchor.category,
            "harm_label": anchor.harm_label,
        }
        by_language = {row.language: row for row in group}

        for language in LANGUAGE_CODES:
            row = by_language.get(language)
            if language == PIVOT_LANGUAGE:
                # Canonical source text, used verbatim; never a translation.
                value = (row.translated_text or row.source_text) if row else ""
            else:
                response = effective.get(make_request_id(behavior_id, language))
                if response is None:
                    value = ""                              # never attempted
                elif response.get("status") != STATUS_OK:
                    value = ""                              # ERROR is not a translation
                    from_error[language] += 1
                else:
                    value = response.get("response_text") or ""

                # Cross-check the applied row against the response log. They
                # should agree; a disagreement means outputs.jsonl drifted from
                # the audit log and is reported rather than silently preferred.
                applied = row.translated_text if row else None
                if applied and value and applied != value:
                    disagreements.append(
                        {"behavior_id": behavior_id, "language": language,
                         "applied_sha256": text_sha256(applied)[:16],
                         "effective_sha256": text_sha256(value)[:16]}
                    )

            record[language] = value
            if value:
                non_empty[language] += 1
            else:
                pending[language] += 1

        # --- review layer -------------------------------------------------
        reviewers: set[str] = set()
        competences: set[str] = set()
        n_accepted = 0
        for language in NON_PIVOT_LANGUAGES:
            decision = decisions.get(make_request_id(behavior_id, language))
            # A verdict is shown only for a row that actually has a translation.
            # A verdict counts only where a translation actually exists.
            verdict = decision["decision"] if decision and record[language] else ""
            if verdict:
                decision_counts[language][verdict] += 1
                reviewers.add(decision["reviewer_id"])
                competences.add(decision["reviewer_language_competence"])
                if verdict == "ACCEPT":
                    n_accepted += 1

        n_translated = sum(1 for lang in NON_PIVOT_LANGUAGES if record[lang])
        still_required = [
            r for r in group if r.language != PIVOT_LANGUAGE and r.review_required
        ]
        review_rows.append({
            "behavior_id": behavior_id,
            "reviewer_ids": sorted(reviewers),
            "competences": sorted(competences),
            "n_translated": n_translated,
            "n_accepted": n_accepted,
            "review_required": bool(still_required),
            "human_reviewed": any(
                r.translation_status == TranslationStatus.HUMAN_REVIEWED.value for r in group
            ),
        })

        records.append(record)

    diagnostics = {
        "non_empty_per_language": {lang: non_empty[lang] for lang in LANGUAGE_CODES},
        "empty_per_language": {lang: pending[lang] for lang in LANGUAGE_CODES},
        "suppressed_error_responses_per_language": {
            lang: from_error[lang] for lang in LANGUAGE_CODES
        },
        "inconsistent_metadata_groups": inconsistent_groups,
        "applied_vs_effective_disagreements": disagreements,
        "decision_counts_per_language": {
            lang: dict(sorted(decision_counts[lang].items())) for lang in NON_PIVOT_LANGUAGES
        },
        "n_decisions": sum(sum(c.values()) for c in decision_counts.values()),
        "review_rows": review_rows,
        "language_row_counts": language_row_counts,
        "behaviors_with_wrong_language_row_count": {
            bid: n for bid, n in sorted(language_row_counts.items()) if n != len(LANGUAGE_CODES)
        },
    }
    return records, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--outputs", type=Path, default=OUTPUTS_PATH)
    parser.add_argument("--responses", type=Path, default=RESPONSES_PATH)
    parser.add_argument("--decisions", type=Path, default=DECISIONS_JSONL)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="Write the sheet even though some language cells are empty. Without this "
             "flag an empty cell is a hard failure and nothing is written.",
    )
    args = parser.parse_args()

    if not args.outputs.exists():
        print(f"missing {args.outputs}; run scripts/04_translate_pilot.py first", file=sys.stderr)
        return 1

    header = stage_report_header("stage1.3_build_wide_review", {"columns": list(COLUMNS)})
    print(f"run_id:     {header['run_id']}")
    print(f"git commit: {header['git_commit']}  (dirty={header['git_dirty']})")

    rows = load_rows(args.outputs)
    log = ResponseLog(args.responses)
    effective = log.effective()
    retry = log.retry_summary()
    decisions = load_decisions(args.decisions)
    records, diagnostics = build(rows, effective, decisions)

    print(f"\nrows in       : {len(rows)}  ({len({r.behavior_id for r in rows})} behaviors)")
    print(f"response log  : {retry['n_records_total']} records -> "
          f"{retry['n_effective']} effective ({retry['n_superseded_records']} superseded)")
    print(f"wide records  : {len(records)}")
    print(f"columns ({len(COLUMNS)}) : {', '.join(COLUMNS)}")
    print(f"\nnon-empty per language: {diagnostics['non_empty_per_language']}")
    print(f"empty per language    : {diagnostics['empty_per_language']}")
    print(f"review decisions      : {diagnostics['n_decisions']} "
          f"({len(decisions)} imported)")
    for lang in NON_PIVOT_LANGUAGES:
        counts = diagnostics["decision_counts_per_language"][lang]
        if counts:
            print(f"    {lang}: {counts}")

    failures: list[str] = []

    # --- every behavior must have exactly seven language rows -------------
    wrong = diagnostics["behaviors_with_wrong_language_row_count"]
    print(f"\nbehaviors with exactly {len(LANGUAGE_CODES)} language rows: "
          f"{len(records) - len(wrong)} / {len(records)}")
    if wrong:
        failures.append(f"{len(wrong)} behavior(s) do not have 7 language rows: {wrong}")

    # --- completeness gate -------------------------------------------------
    empty_cells = [
        f"{record['behavior_id']}:{lang}"
        for record in records
        for lang in LANGUAGE_CODES
        if not record[lang].strip()
    ]
    print(f"empty language cells: {len(empty_cells)} "
          f"of {len(records) * len(LANGUAGE_CODES)}")
    if empty_cells:
        message = (
            f"{len(empty_cells)} empty language cell(s) across "
            f"{len({c.split(':')[0] for c in empty_cells})} behavior(s); "
            f"first few: {empty_cells[:6]}"
        )
        if args.allow_incomplete:
            print(f"  WARNING: {message}")
            print("  (--allow-incomplete given: writing the sheet anyway)")
        else:
            failures.append(message)

    ids = [r["behavior_id"] for r in records]
    if len(ids) != len(set(ids)):
        failures.append(f"duplicate behavior_id rows: {len(ids) - len(set(ids))}")
    if diagnostics["inconsistent_metadata_groups"]:
        failures.append(
            f"inconsistent metadata in {len(diagnostics['inconsistent_metadata_groups'])} group(s)"
        )
    if diagnostics["applied_vs_effective_disagreements"]:
        failures.append(
            f"{len(diagnostics['applied_vs_effective_disagreements'])} row(s) disagree with "
            "the response log"
        )
    if not records:
        failures.append("zero records (stop condition)")

    for failure in failures:
        print(f"  FAIL: {failure}", file=sys.stderr)
    if failures:
        return 1

    if args.dry_run:
        print("\nDry run: nothing written.")
        return 0

    # --- write CSV --------------------------------------------------------
    # newline="" + csv.writer handles commas, quotes, tabs and embedded line
    # breaks per RFC 4180; QUOTE_MINIMAL quotes only the fields that need it.
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(COLUMNS), quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n"
        )
        writer.writeheader()
        writer.writerows(records)

    metadata: dict[str, Any] = {
        **header,
        "csv_path": display_path(args.csv),
        "git_ignored": True,
        "contains_prompt_text": True,
        "row_count": len(records),
        "column_count": len(COLUMNS),
        "columns": list(COLUMNS),
        "languages": list(LANGUAGE_CODES),
        "pivot_language": PIVOT_LANGUAGE,
        "encoding": "utf-8",
        "csv_dialect": {
            "quoting": "QUOTE_MINIMAL",
            "lineterminator": "\\r\\n",
            "delimiter": ",",
            "quotechar": '"',
            "rfc": "RFC 4180",
        },
        "language_counts": {
            lang: {
                "non_empty": diagnostics["non_empty_per_language"][lang],
                "empty": diagnostics["empty_per_language"][lang],
            }
            for lang in LANGUAGE_CODES
        },
        "translated_counts": {
            lang: diagnostics["non_empty_per_language"][lang]
            for lang in LANGUAGE_CODES
            if lang != PIVOT_LANGUAGE
        },
        "pending_counts": {
            lang: diagnostics["empty_per_language"][lang]
            for lang in LANGUAGE_CODES
            if lang != PIVOT_LANGUAGE
        },
        "canonical_source_count": diagnostics["non_empty_per_language"][PIVOT_LANGUAGE],
        "suppressed_error_responses_per_language": diagnostics[
            "suppressed_error_responses_per_language"
        ],
        "base_columns": list(BASE_COLUMNS),
        "decision_columns": list(DECISION_COLUMNS),
        "review_columns": list(REVIEW_COLUMNS),
        "n_decisions": diagnostics["n_decisions"],
        "decision_counts_per_language": diagnostics["decision_counts_per_language"],
        "decisions_source": display_path(args.decisions),
        "review_rows": diagnostics["review_rows"],
        "review_required_rows": sum(
            1 for r in diagnostics["review_rows"] if r["review_required"]
        ),
        "human_reviewed_rows": sum(
            1 for r in diagnostics["review_rows"] if r["human_reviewed"]
        ),
        "language_row_counts_ok": not diagnostics["behaviors_with_wrong_language_row_count"],
        "inconsistent_metadata_groups": diagnostics["inconsistent_metadata_groups"],
        "applied_vs_effective_disagreements": diagnostics["applied_vs_effective_disagreements"],
        "response_log": retry,
        "source_hashes": {
            "outputs_jsonl": file_sha256(args.outputs),
            "responses_jsonl": file_sha256(args.responses) if args.responses.exists() else None,
            "csv_sha256": file_sha256(args.csv),
            "decisions_jsonl": (
                file_sha256(args.decisions) if args.decisions.exists() else None
            ),
        },
        "row_hashes": {
            record["behavior_id"]: text_sha256(
                "\x1f".join(record[column] for column in COLUMNS)
            )
            for record in records
        },
        "notes": [
            "English is canonical source text, not a translation.",
            "Non-English cells come only from the latest effective OK response; "
            "ERROR responses are suppressed to empty.",
            "An empty cell means no translation exists -- distinct from an empty "
            "translation, which would be a defect.",
            "No row is marked HUMAN_REVIEWED. Canonical manifests are untouched.",
            "A <lang>_decision cell is populated only when that language actually has "
            "a translation; an accepted verdict on an empty cell would be meaningless.",
            "reviewer_competence machine_assisted does NOT clear review_required, so "
            "review_required stays true even where the verdict is ACCEPT.",
            "The original 12 columns are preserved as an exact prefix.",
        ],
    }
    args.metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nwrote {display_path(args.csv)}  ({args.csv.stat().st_size} bytes)")
    print(f"wrote {display_path(args.metadata)}")
    print(f"csv sha256: {metadata['source_hashes']['csv_sha256'][:24]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
