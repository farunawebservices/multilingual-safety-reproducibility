#!/usr/bin/env python3
"""Stage 1.2 -- validate the translation pilot rows.

Runs the schema invariants and the heuristic language checks over
``data/interim/translation_pilot_outputs.jsonl`` and writes
``results/translation_pilot_validation.json``.

What is checked
---------------
* Schema: required fields, valid statuses/methods, ``prompt_hash`` consistency.
* Group: ``behavior_id`` invariant across all seven languages; harm label,
  category, and source text constant within a behavior; no missing language.
* Content: empty text, English leakage, wrong script, duplicate translations,
  suspiciously short (or long) output, missing expected diacritics.
* Policy: ``review_required`` cleared only by a named human reviewer;
  ``split_placeholder`` still unassigned everywhere.

What this is NOT
----------------
Passing every check does **not** mean a translation is correct. The checks are
heuristics; they catch mechanical failures, not meaning. No row is marked
validated by this script, and no `review_required` flag is ever cleared here.
Native-speaker validation is claimed only where a named reviewer is recorded.

The models under evaluation are not used anywhere in this script -- judging
translations with a system under test would confound every downstream result.

Usage
-----
    python scripts/05_validate_translation_pilot.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multilingual_safety.hashing import deterministic_digest
from multilingual_safety.langcheck import ALL_ISSUE_CODES, check_translation, thresholds
from multilingual_safety.languages import LANGUAGE_CODES, PIVOT_LANGUAGE
from multilingual_safety.paths import INTERIM_DIR, RESULTS_DIR, display_path
from multilingual_safety.run_context import stage_report_header
from multilingual_safety.translation import (
    SPLIT_UNASSIGNED,
    TranslationRow,
    TranslationSchemaError,
    TranslationStatus,
    find_duplicate_prompt_hashes,
    validate_row_group,
)

OUTPUTS_PATH = INTERIM_DIR / "translation_pilot_outputs.jsonl"
VALIDATION_PATH = RESULTS_DIR / "translation_pilot_validation.json"


def load_rows(path: Path) -> tuple[list[TranslationRow], list[str]]:
    """Parse rows. Malformed lines are reported, never skipped (CLAUDE.md §2)."""
    rows: list[TranslationRow] = []
    errors: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(TranslationRow.from_dict(json.loads(line)))
        except (json.JSONDecodeError, TranslationSchemaError) as exc:
            errors.append(f"{path.name}:{lineno}: {exc}")
    return rows, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs", type=Path, default=OUTPUTS_PATH)
    parser.add_argument("--out", type=Path, default=VALIDATION_PATH)
    args = parser.parse_args()

    if not args.outputs.exists():
        print(f"missing {args.outputs}; run scripts/04_translate_pilot.py first", file=sys.stderr)
        return 1

    header = stage_report_header("stage1.2_validate_pilot", {"outputs": str(args.outputs)})
    print(f"run_id:     {header['run_id']}")
    print(f"git commit: {header['git_commit']}  (dirty={header['git_dirty']})")

    rows, parse_errors = load_rows(args.outputs)
    if parse_errors:
        for err in parse_errors:
            print(f"  SCHEMA FAIL: {err}", file=sys.stderr)

    # --- duplicate translations across behaviors, within a language --------
    duplicates = find_duplicate_prompt_hashes(rows)
    dup_index: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key, bids in duplicates.items():
        language, prompt_hash = key.split(":", 1)
        dup_index[(language, prompt_hash)] = bids

    # --- per-row heuristics ------------------------------------------------
    row_reports: list[dict[str, Any]] = []
    issue_counter: Counter[str] = Counter()
    issues_by_language: dict[str, Counter[str]] = {lang: Counter() for lang in LANGUAGE_CODES}

    for row in sorted(rows, key=lambda r: (r.harm_label, r.category, r.behavior_id, r.language)):
        duplicate_of = (
            [b for b in dup_index.get((row.language, row.prompt_hash or ""), []) if b != row.behavior_id]
            if row.prompt_hash
            else []
        )
        check = check_translation(
            translated_text=row.translated_text,
            source_text=row.source_text,
            language=row.language,
            duplicate_of=duplicate_of,
        )
        for issue in check.issues:
            issue_counter[issue] += 1
            issues_by_language[row.language][issue] += 1

        row_reports.append({
            "behavior_id": row.behavior_id,
            "language": row.language,
            "harm_label": row.harm_label,
            "category": row.category,
            "translation_status": row.translation_status,
            "translation_method": row.translation_method,
            "review_required": row.review_required,
            "reviewer_id": row.reviewer_id,
            "prompt_hash": row.prompt_hash,
            "issues": sorted(check.issues),
            "metrics": check.metrics,
        })

    # --- group invariants --------------------------------------------------
    by_behavior: dict[str, list[TranslationRow]] = defaultdict(list)
    for row in rows:
        by_behavior[row.behavior_id].append(row)
    group_violations = {
        bid: v for bid, group in sorted(by_behavior.items()) if (v := validate_row_group(group))
    }

    # --- policy invariants -------------------------------------------------
    policy_violations: list[str] = []
    for row in rows:
        if row.split_placeholder != SPLIT_UNASSIGNED:
            policy_violations.append(f"{row.behavior_id}/{row.language}: split already assigned")
        if row.language != PIVOT_LANGUAGE and not row.review_required and not row.reviewer_id:
            policy_violations.append(
                f"{row.behavior_id}/{row.language}: review cleared with no named reviewer"
            )

    reviewed = [r for r in rows if r.translation_status == TranslationStatus.HUMAN_REVIEWED.value]
    native_claimed = sorted({r.reviewer_id for r in reviewed if r.reviewer_id})

    # --- assemble ----------------------------------------------------------
    n_rows = len(rows)
    pending = sum(1 for r in rows if r.translated_text is None)
    report: dict[str, Any] = {
        **header,
        "n_rows": n_rows,
        "n_behaviors": len(by_behavior),
        "n_parse_errors": len(parse_errors),
        "parse_errors": parse_errors,
        "thresholds": thresholds(),
        "counts_by_language": dict(sorted(Counter(r.language for r in rows).items())),
        "counts_by_harm_label": dict(sorted(Counter(r.harm_label for r in rows).items())),
        "counts_by_category": dict(sorted(Counter(r.category for r in rows).items())),
        "counts_by_status": dict(sorted(Counter(r.translation_status for r in rows).items())),
        "counts_by_validation_status": {
            "pending_translation": pending,
            "canonical_source": sum(
                1 for r in rows if r.translation_status == TranslationStatus.CANONICAL_SOURCE.value
            ),
            "heuristics_passed": sum(
                1 for rep in row_reports
                if not rep["issues"] and rep["translation_status"] not in (
                    TranslationStatus.PENDING_TRANSLATION.value,
                    TranslationStatus.CANONICAL_SOURCE.value,
                )
            ),
            "heuristics_failed": sum(
                1 for rep in row_reports
                if rep["issues"] and rep["translation_status"]
                != TranslationStatus.PENDING_TRANSLATION.value
            ),
            "human_reviewed": len(reviewed),
        },
        "issue_counts": {code: issue_counter.get(code, 0) for code in ALL_ISSUE_CODES},
        "issue_counts_by_language": {
            lang: {code: issues_by_language[lang].get(code, 0) for code in ALL_ISSUE_CODES}
            for lang in LANGUAGE_CODES
        },
        "duplicate_prompt_hashes": duplicates,
        "n_duplicate_prompt_hashes": len(duplicates),
        "group_violations": group_violations,
        "policy_violations": sorted(policy_violations),
        "n_review_required": sum(1 for r in rows if r.review_required),
        "native_speaker_reviewed_rows": len(reviewed),
        "native_speaker_reviewers": native_claimed,
        "native_speaker_validation_claimed": bool(reviewed),
        "judged_by_model_under_evaluation": False,
        "contains_prompt_text": False,
        "rows": row_reports,
        "caveats": [
            "Heuristics catch mechanical failures, not meaning. Passing is not validation.",
            "No row is marked validated and no review_required flag is cleared here.",
            "Latin-script targets (ha, yo, sw) are the least reliably checkable; "
            "uncertainty there resolves to human review, never acceptance.",
            "No model under evaluation was used to produce or judge these rows.",
        ],
    }

    blocking = bool(parse_errors or group_violations or policy_violations)
    report["ok"] = not blocking
    report["deterministic_digest"] = deterministic_digest(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # --- console summary ---------------------------------------------------
    print(f"\nrows: {n_rows}   behaviors: {len(by_behavior)}   parse errors: {len(parse_errors)}")
    print(f"  by language: {report['counts_by_language']}")
    print(f"  by label:    {report['counts_by_harm_label']}")
    print(f"  by status:   {report['counts_by_status']}")
    print(f"  validation:  {report['counts_by_validation_status']}")
    print(f"\n  review_required: {report['n_review_required']} / {n_rows}")
    print(f"  native-speaker reviewed: {report['native_speaker_reviewed_rows']} "
          f"(claimed: {report['native_speaker_validation_claimed']})")

    active = {k: v for k, v in report["issue_counts"].items() if v}
    print(f"\n  issues detected: {active or 'none (nothing translated yet)'}")
    print(f"  duplicate prompt hashes: {report['n_duplicate_prompt_hashes']}")
    print(f"  group violations: {len(group_violations)}   policy violations: {len(policy_violations)}")

    for bid, violations in sorted(group_violations.items()):
        print(f"  FAIL {bid}: {violations}", file=sys.stderr)
    for violation in report["policy_violations"]:
        print(f"  FAIL {violation}", file=sys.stderr)

    print(f"\nwrote {display_path(args.out)}")
    print(f"deterministic_digest: {report['deterministic_digest'][:16]}...")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
