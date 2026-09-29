#!/usr/bin/env python3
"""Stage 1.6 -- VALIDATE a review decisions file. Imports nothing, writes nothing.

Read-only by construction: it opens the decisions TSV, the pilot outputs, and the
response log, and writes no file and mutates no state. Use
``scripts/08_import_review_decisions.py`` to actually import.

Checks
------
* exactly ``--expect-rows`` rows;
* exactly 20 behavior IDs x 6 non-English languages;
* no duplicate ``(behavior_id, language)`` pair;
* every row corresponds to a current **successful** translation response;
* source and response hashes match the canonical artifacts;
* every decision is in the permitted vocabulary;
* no blank reviewer IDs;
* no row asserts ``HUMAN_REVIEWED``;
* ``machine_assisted`` competence is not treated as native-speaker validation;
* known source anomalies are reported separately rather than folded into the
  pass/fail count.

Output is metadata only: counts, identifiers, and hashes. No prompt or
translation text is ever printed.

Usage
-----
    python scripts/09_validate_review_decisions.py
    python scripts/09_validate_review_decisions.py --decisions <path>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multilingual_safety.hashing import file_sha256, text_sha256
from multilingual_safety.languages import LANGUAGE_CODES, PIVOT_LANGUAGE
from multilingual_safety.paths import DATA_DIR, INTERIM_DIR, REPO_ROOT
from multilingual_safety.providers import STATUS_OK, ResponseLog, make_request_id
from multilingual_safety.review import (
    COMPETENCE_CAN_CLEAR_REVIEW,
    ReviewDecisionType,
    ReviewerCompetence,
    can_clear_human_review,
    competence_rationale,
)

REVIEW_DIR = DATA_DIR / "review"
DECISIONS_TSV = REVIEW_DIR / "full_review_decisions.tsv"
OUTPUTS_PATH = INTERIM_DIR / "translation_pilot_outputs.jsonl"
RESPONSES_PATH = INTERIM_DIR / "translation_pilot_responses.jsonl"

NON_PIVOT_LANGUAGES = tuple(c for c in LANGUAGE_CODES if c != PIVOT_LANGUAGE)
VALID_DECISIONS = {d.value for d in ReviewDecisionType}

#: Known upstream source anomalies, pinned by the SHA-256 of the English source
#: so the flag follows the content rather than a line number. Reported
#: separately: these are upstream labelling/quality observations, not defects in
#: the review file, and must not silently pass or silently fail it.
SOURCE_ANOMALIES: dict[str, dict[str, str]] = {
    "beh_04bf1788c1aa": {
        "code": "BENIGN_LABEL_ON_VIOLENCE_ADJACENT_TOPIC",
        "summary": (
            "upstream JailbreakBench labels this behavior benign while its topic is "
            "historical torture instruments; the English source also contains a "
            "spelling error that may degrade machine translation quality"
        ),
        "action": "confirm the benign label and the source spelling before use",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--decisions", type=Path, default=DECISIONS_TSV)
    parser.add_argument("--outputs", type=Path, default=OUTPUTS_PATH)
    parser.add_argument("--responses", type=Path, default=RESPONSES_PATH)
    parser.add_argument("--expect-rows", type=int, default=120)
    parser.add_argument("--expect-behaviors", type=int, default=20)
    args = parser.parse_args()

    for path in (args.decisions, args.outputs, args.responses):
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 1

    print("VALIDATE ONLY -- nothing is imported, written, or modified")
    print(f"file   : {args.decisions.relative_to(REPO_ROOT)}")
    print(f"sha256 : {file_sha256(args.decisions)}")

    with args.decisions.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        columns = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]

    failures: list[str] = []
    warnings: list[str] = []

    # --- canonical state --------------------------------------------------
    effective = ResponseLog(args.responses).effective()
    successful = {rid: r for rid, r in effective.items() if r.get("status") == STATUS_OK}
    canonical: dict[str, dict[str, Any]] = {}
    statuses: Counter[str] = Counter()
    for line in args.outputs.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            statuses[row["translation_status"]] += 1
            if row["language"] == PIVOT_LANGUAGE:
                canonical[row["behavior_id"]] = row

    # --- 1. shape ---------------------------------------------------------
    behaviors = {r["behavior_id"] for r in rows}
    languages = Counter(r["language"] for r in rows)
    pairs = Counter((r["behavior_id"], r["language"]) for r in rows)
    duplicates = sorted(p for p, n in pairs.items() if n > 1)

    print("\nSHAPE")
    print(f"  columns              : {len(columns)}  {columns}")
    print(f"  rows                 : {len(rows)}  (expected {args.expect_rows})")
    print(f"  distinct behavior_ids: {len(behaviors)}  (expected {args.expect_behaviors})")
    print(f"  rows per language    : {dict(sorted(languages.items()))}")
    print(f"  duplicate (behavior, language) pairs: {len(duplicates)}")

    if len(rows) != args.expect_rows:
        failures.append(f"expected {args.expect_rows} rows, got {len(rows)}")
    if len(behaviors) != args.expect_behaviors:
        failures.append(f"expected {args.expect_behaviors} behavior_ids, got {len(behaviors)}")
    if set(languages) != set(NON_PIVOT_LANGUAGES):
        failures.append(
            f"languages must be exactly {sorted(NON_PIVOT_LANGUAGES)}, got {sorted(languages)}"
        )
    if PIVOT_LANGUAGE in languages:
        failures.append("English rows must not appear: they are canonical source, not translations")
    off = {k: v for k, v in languages.items() if v != args.expect_behaviors}
    if off:
        failures.append(f"languages without {args.expect_behaviors} rows: {off}")
    if duplicates:
        failures.append(f"{len(duplicates)} duplicate (behavior_id, language) pair(s)")

    # --- 2. correspondence + hashes ---------------------------------------
    no_response, not_ok, source_mismatch, response_mismatch = [], [], [], []
    label_mismatch = []
    for row in rows:
        request_id = make_request_id(row["behavior_id"], row["language"])
        response = effective.get(request_id)
        if response is None:
            no_response.append(request_id)
            continue
        if response.get("status") != STATUS_OK:
            not_ok.append(request_id)
            continue

        anchor = canonical.get(row["behavior_id"])
        if anchor is None:
            no_response.append(f"{request_id} (no canonical English row)")
            continue
        if text_sha256(anchor["source_text"]) != response.get("source_text_sha256"):
            source_mismatch.append(request_id)
        if text_sha256(response.get("response_text") or "") != response.get("response_sha256"):
            response_mismatch.append(request_id)
        if row.get("harm_label") and row["harm_label"] != anchor["harm_label"]:
            label_mismatch.append(request_id)

    print("\nCORRESPONDENCE AND HASHES")
    print(f"  successful responses available : {len(successful)}")
    print(f"  rows with no response          : {len(no_response)}")
    print(f"  rows whose response is not OK  : {len(not_ok)}")
    print(f"  source_text_sha256 mismatches  : {len(source_mismatch)}")
    print(f"  response_sha256 mismatches     : {len(response_mismatch)}")
    print(f"  harm_label disagreements       : {len(label_mismatch)}")
    for label, bucket in (
        ("rows with no effective response", no_response),
        ("rows whose effective response is not OK", not_ok),
        ("source hash mismatches", source_mismatch),
        ("response hash mismatches", response_mismatch),
        ("harm_label disagreements", label_mismatch),
    ):
        if bucket:
            failures.append(f"{len(bucket)} {label}: {sorted(bucket)[:5]}")

    # --- 3. decision vocabulary, reviewers, competence ---------------------
    decisions = Counter(r["decision"] for r in rows)
    reviewers = Counter(r["reviewer_id"] for r in rows)
    competences = Counter(r["reviewer_language_competence"] for r in rows)
    blank_reviewers = [r["review_id"] for r in rows if not r["reviewer_id"].strip()]
    invalid = {d: n for d, n in decisions.items() if d not in VALID_DECISIONS}
    unknown_competence = {
        c: n for c, n in competences.items()
        if c not in {x.value for x in ReviewerCompetence}
    }

    print("\nDECISIONS AND REVIEWERS")
    print(f"  permitted vocabulary : {sorted(VALID_DECISIONS)}")
    print(f"  counts by decision   : {dict(sorted(decisions.items()))}")
    print(f"  invalid decisions    : {invalid or '{}'}")
    print(f"  reviewers            : {dict(sorted(reviewers.items()))}")
    print(f"  blank reviewer ids   : {len(blank_reviewers)}")
    print(f"  competences          : {dict(sorted(competences.items()))}")

    if invalid:
        failures.append(f"decisions outside the permitted vocabulary: {invalid}")
    if blank_reviewers:
        failures.append(f"{len(blank_reviewers)} row(s) have a blank reviewer_id")
    if unknown_competence:
        failures.append(f"unknown reviewer competence(s): {unknown_competence}")

    # --- 4. HUMAN_REVIEWED must not be asserted ---------------------------
    asserted = [
        r["review_id"] for r in rows
        if any("HUMAN_REVIEWED" in str(v).upper() for v in r.values())
    ]
    print("\nHUMAN_REVIEWED GUARD")
    print(f"  rows asserting HUMAN_REVIEWED in the file : {len(asserted)}")
    print(f"  rows currently HUMAN_REVIEWED in outputs  : {statuses.get('HUMAN_REVIEWED', 0)}")
    if asserted:
        failures.append(f"{len(asserted)} row(s) assert HUMAN_REVIEWED")
    if statuses.get("HUMAN_REVIEWED"):
        failures.append(f"{statuses['HUMAN_REVIEWED']} output row(s) are HUMAN_REVIEWED")

    # --- 5. competence gate -----------------------------------------------
    would_clear = sum(
        1 for r in rows
        if r["reviewer_language_competence"] in {c.value for c in ReviewerCompetence}
        and r["decision"] in VALID_DECISIONS
        and can_clear_human_review(r["reviewer_language_competence"], r["decision"])
    )
    machine_assisted = competences.get(ReviewerCompetence.MACHINE_ASSISTED.value, 0)
    print("\nCOMPETENCE GATE")
    print(f"  machine_assisted rows        : {machine_assisted}")
    print(f"  rows that WOULD clear review : {would_clear}")
    print(f"  clearing competences         : {sorted(c.value for c in COMPETENCE_CAN_CLEAR_REVIEW)}")
    print(f"  rationale                    : {competence_rationale('machine_assisted')}")
    if machine_assisted and would_clear:
        failures.append("machine_assisted rows would clear review; the gate is not holding")

    # --- 6. source anomalies, reported separately -------------------------
    print("\nSOURCE ANOMALIES (reported separately; not counted as failures)")
    for behavior_id, anomaly in sorted(SOURCE_ANOMALIES.items()):
        affected = sorted(r["language"] for r in rows if r["behavior_id"] == behavior_id)
        anchor = canonical.get(behavior_id)
        print(f"  {behavior_id}  [{anomaly['code']}]")
        print(f"    harm_label      : {anchor['harm_label'] if anchor else '?'}")
        print(f"    category        : {anchor['category'] if anchor else '?'}")
        print(f"    source_sha256   : "
              f"{text_sha256(anchor['source_text'])[:24] if anchor else '?'}")
        print(f"    affected rows   : {len(affected)}  languages={affected}")
        print(f"    summary         : {anomaly['summary']}")
        print(f"    action          : {anomaly['action']}")
        warnings.append(f"{behavior_id}: {anomaly['code']} ({len(affected)} rows)")

    # --- verdict ----------------------------------------------------------
    print("\nRESULT")
    print(f"  checks failed : {len(failures)}")
    print(f"  warnings      : {len(warnings)}")
    for warning in warnings:
        print(f"    WARN: {warning}")
    for failure in failures:
        print(f"    FAIL: {failure}")
    if failures:
        print(f"\n{len(failures)} validation failure(s). Nothing was imported.", file=sys.stderr)
        return 1
    print("\n  all validation checks passed. Nothing was imported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
