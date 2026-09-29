#!/usr/bin/env python3
"""Stage 1.5 -- import manually reviewed pilot decisions.

Reads ``data/review/pilot_decisions.tsv`` (the verdicts) and joins it to
``data/review/pilot_review.tsv`` (the sheet the reviewer worked from, which maps
the ordinal ``review_id`` to a behavior and language and carries the exact text
they saw).

Validation performed
--------------------
* every ``review_id`` resolves to one of the currently translated rows;
* no duplicate and no missing decisions across those rows;
* ``source_text_sha256`` recomputed from the review sheet matches the request;
* ``response_sha256`` recomputed from the review sheet matches the **effective**
  response in the append-only log.

The hash checks are the substantive ones: they prove the reviewer judged the
exact text this repository recorded, rather than a stale or edited copy.

Competence gate
---------------
Whether a decision clears ``review_required`` depends on the reviewer's language
competence, not on the decision (``multilingual_safety.review``).
``machine_assisted`` does **not** clear it, so ACCEPT decisions from that
competence are recorded as a separate review layer and ``review_required``
stays true. Nothing here writes ``HUMAN_REVIEWED``.

Outputs
-------
    data/review/pilot_review_decisions.jsonl        per-row decisions (git-ignored)
    results/translation_pilot_review_summary.json   metadata-only, tracked

The response log and the pilot outputs are read-only here: no ERROR record is
altered, nothing is deleted, and no translation status is changed.

Usage
-----
    python scripts/08_import_review_decisions.py --dry-run
    python scripts/08_import_review_decisions.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multilingual_safety.hashing import deterministic_digest, file_sha256, text_sha256
from multilingual_safety.paths import DATA_DIR, INTERIM_DIR, REPO_ROOT, RESULTS_DIR
from multilingual_safety.providers import STATUS_OK, ResponseLog, make_request_id
from multilingual_safety.review import (
    COMPETENCE_CAN_CLEAR_REVIEW,
    ReviewDecision,
    ReviewerCompetence,
    ReviewError,
    competence_rationale,
    summarize,
    validate_against_rows,
)
from multilingual_safety.run_context import stage_report_header

REVIEW_DIR = DATA_DIR / "review"
DECISIONS_TSV = REVIEW_DIR / "pilot_decisions.tsv"
REVIEW_SHEET_TSV = REVIEW_DIR / "pilot_review.tsv"
RESPONSES_PATH = INTERIM_DIR / "translation_pilot_responses.jsonl"
OUTPUTS_PATH = INTERIM_DIR / "translation_pilot_outputs.jsonl"
DECISIONS_JSONL = REVIEW_DIR / "pilot_review_decisions.jsonl"
SUMMARY_PATH = RESULTS_DIR / "translation_pilot_review_summary.json"

DEFAULT_REVIEWER = "Faruna_Godwin"
DEFAULT_COMPETENCE = ReviewerCompetence.MACHINE_ASSISTED.value


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh, delimiter="\t")]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--decisions", type=Path, default=DECISIONS_TSV)
    parser.add_argument("--sheet", type=Path, default=REVIEW_SHEET_TSV)
    parser.add_argument("--responses", type=Path, default=RESPONSES_PATH)
    parser.add_argument("--reviewer-id", default=DEFAULT_REVIEWER)
    parser.add_argument("--competence", default=DEFAULT_COMPETENCE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for path in (args.decisions, args.sheet):
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 1

    header = stage_report_header(
        "stage1.5_import_review_decisions",
        {"reviewer_id": args.reviewer_id, "competence": args.competence},
    )
    print(f"run_id:     {header['run_id']}")
    print(f"git commit: {header['git_commit']}  (dirty={header['git_dirty']})")

    # --- competence gate, decided before anything is imported -------------
    competence = ReviewerCompetence(args.competence)
    clears = competence in COMPETENCE_CAN_CLEAR_REVIEW
    print("\nCOMPETENCE GATE")
    print(f"  reviewer_id  : {args.reviewer_id}")
    print(f"  competence   : {competence.value}")
    print(f"  clears review: {clears}")
    print(f"  rationale    : {competence_rationale(competence)}")

    # --- join decisions to the sheet the reviewer worked from -------------
    verdicts = read_tsv(args.decisions)
    sheet = {row["review_id"]: row for row in read_tsv(args.sheet)}
    log = ResponseLog(args.responses)
    effective = log.effective()
    translated = {rid: row for rid, row in effective.items() if row.get("status") == STATUS_OK}

    print(f"\ndecisions in sheet   : {len(verdicts)}")
    print(f"review sheet rows    : {len(sheet)}")
    print(f"currently translated : {len(translated)}")

    decisions: list[ReviewDecision] = []
    failures: list[str] = []

    for verdict in verdicts:
        review_id = verdict["review_id"]
        mapped = sheet.get(review_id)
        if mapped is None:
            failures.append(f"review_id {review_id!r} is absent from {args.sheet.name}")
            continue

        behavior_id, language = mapped["behavior_id"], mapped["language"]
        request_id = make_request_id(behavior_id, language)
        response = translated.get(request_id)

        # Recompute both hashes from the text the reviewer actually saw.
        seen_source_sha = text_sha256(mapped["source_text"])
        seen_response_sha = text_sha256(mapped["translated_text"])
        source_ok = bool(response) and seen_source_sha == response.get("source_text_sha256")
        response_ok = bool(response) and seen_response_sha == response.get("response_sha256")

        if response is None:
            failures.append(f"{review_id}: {request_id} has no effective OK response")
        else:
            if not source_ok:
                failures.append(f"{review_id}: source_text_sha256 mismatch for {request_id}")
            if not response_ok:
                failures.append(f"{review_id}: response_sha256 mismatch for {request_id}")

        reviewer = verdict.get("reviewer_id") or args.reviewer_id
        row_competence = verdict.get("reviewer_language_competence") or args.competence
        if reviewer != args.reviewer_id:
            failures.append(f"{review_id}: reviewer_id {reviewer!r} != {args.reviewer_id!r}")
        if row_competence != args.competence:
            failures.append(f"{review_id}: competence {row_competence!r} != {args.competence!r}")

        notes = verdict.get("notes") or ""
        try:
            decisions.append(
                ReviewDecision(
                    review_id=review_id,
                    request_id=request_id,
                    behavior_id=behavior_id,
                    language=language,
                    decision=verdict["decision"].strip().upper(),
                    reviewer_id=reviewer,
                    reviewer_language_competence=row_competence,
                    source_text_sha256=seen_source_sha,
                    response_sha256=seen_response_sha,
                    source_hash_verified=source_ok,
                    response_hash_verified=response_ok,
                    notes_present=bool(notes.strip()),
                    notes_sha256=text_sha256(notes) if notes.strip() else None,
                    decided_at=None,
                )
            )
        except ReviewError as exc:
            failures.append(str(exc))

    failures.extend(validate_against_rows(decisions, translated))

    summary_counts = summarize(decisions)
    print("\nVALIDATION")
    print(f"  decisions parsed        : {len(decisions)}")
    print(f"  source hash verified    : {sum(1 for d in decisions if d.source_hash_verified)}"
          f" / {len(decisions)}")
    print(f"  response hash verified  : {sum(1 for d in decisions if d.response_hash_verified)}"
          f" / {len(decisions)}")
    print(f"  rows cleared for review : {summary_counts['n_clearing_review_required']}")
    print(f"  review_required retained: {summary_counts['n_review_required_retained']}")
    print(f"\n  by decision : {summary_counts['counts_by_decision']}")
    print(f"  by competence: {summary_counts['counts_by_competence']}")
    print("  by language :")
    for language, counts in summary_counts["counts_by_language"].items():
        active = {k: v for k, v in counts.items() if v}
        print(f"    {language}: {active}")

    if failures:
        for failure in sorted(set(failures)):
            print(f"  FAIL: {failure}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\nDry run: nothing written.")
        return 0

    # --- write ------------------------------------------------------------
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    with DECISIONS_JSONL.open("w", encoding="utf-8") as fh:
        for decision in sorted(decisions, key=lambda d: (d.language, d.behavior_id)):
            fh.write(json.dumps(decision.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")

    summary: dict[str, Any] = {
        **header,
        **summary_counts,
        "reviewer_id": args.reviewer_id,
        "reviewer_language_competence": competence.value,
        "competence_clears_review_required": clears,
        "competence_rationale": competence_rationale(competence),
        "human_reviewed_rows_written": 0,
        "review_required_cleared": 0,
        "translation_statuses_modified": 0,
        "decision_layer": "separate",
        "decision_layer_note": (
            "ACCEPT decisions are recorded as a distinct review layer. They do NOT set "
            "HUMAN_REVIEWED and do NOT clear review_required, because machine_assisted "
            "competence does not establish that meaning was preserved. A reviewer with "
            "native_speaker or professional_translator competence is still required."
        ),
        "inputs": {
            "decisions_tsv": str(args.decisions.relative_to(REPO_ROOT)),
            "review_sheet_tsv": str(args.sheet.relative_to(REPO_ROOT)),
            "responses_jsonl": str(args.responses.relative_to(REPO_ROOT)),
        },
        "source_hashes": {
            "decisions_tsv": file_sha256(args.decisions),
            "review_sheet_tsv": file_sha256(args.sheet),
            "responses_jsonl": file_sha256(args.responses),
            "outputs_jsonl": file_sha256(OUTPUTS_PATH) if OUTPUTS_PATH.exists() else None,
        },
        "response_log": log.retry_summary(),
        "decisions_file": str(DECISIONS_JSONL.relative_to(REPO_ROOT)),
        "decisions_file_git_ignored": True,
    }
    summary["deterministic_digest"] = deterministic_digest(summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"\nwrote {DECISIONS_JSONL.relative_to(REPO_ROOT)}  (git-ignored)")
    print(f"wrote {SUMMARY_PATH.relative_to(REPO_ROOT)}  (tracked, metadata-only)")
    print("\nNo row was marked HUMAN_REVIEWED. review_required is unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
