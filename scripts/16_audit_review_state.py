#!/usr/bin/env python3
"""Stage 3.1 -- audit the translation-review state. READ-ONLY.

Opens the frozen dataset, the split, the response log, and every review-decision
source, and reports counts. It writes nothing unless ``--write-report`` is given,
and it never touches the frozen dataset or the split under any flag.

What it answers
---------------
* how many non-English rows exist, and how many are MACHINE_TRANSLATED_UNCHECKED;
* how many carry a review decision, broken down by language and verdict;
* whether any decision references a ``response_sha256`` that no longer matches
  the corpus -- i.e. the reviewer judged text that has since been regenerated;
* whether every non-English row has a review record;
* which rows are length outliers, and which split they fall in.

Output is metadata only: counts, IDs, ratios and hashes. No prompt or
translated text is printed.

Usage
-----
    python scripts/16_audit_review_state.py
    python scripts/16_audit_review_state.py --write-report
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multilingual_safety.hashing import deterministic_digest, file_sha256
from multilingual_safety.langcheck import LONG_RATIO, SHORT_RATIO
from multilingual_safety.languages import LANGUAGE_CODES, PIVOT_LANGUAGE
from multilingual_safety.paths import DATA_DIR, INTERIM_DIR, PROCESSED_DIR, REPO_ROOT, RESULTS_DIR
from multilingual_safety.providers import STATUS_OK, ResponseLog, make_request_id
from multilingual_safety.review import ReviewDecisionType
from multilingual_safety.run_context import stage_report_header

FROZEN_PATH = PROCESSED_DIR / "frozen_dataset_v1.jsonl"
SPLITS_PATH = PROCESSED_DIR / "splits_v1.json"
RESPONSES_PATH = INTERIM_DIR / "full" / "full_translation_responses.jsonl"
REVIEW_DIR = DATA_DIR / "review"
IMPORTED_DECISIONS = REVIEW_DIR / "pilot_review_decisions.jsonl"
PENDING_TSV = REVIEW_DIR / "full_review_decisions.tsv"
REPORT_PATH = RESULTS_DIR / "review_state_audit.json"

NON_PIVOT_LANGUAGES = tuple(c for c in LANGUAGE_CODES if c != PIVOT_LANGUAGE)
DECISION_VALUES = tuple(d.value for d in ReviewDecisionType)

#: Robust outlier rule, in addition to the langcheck ratio thresholds: a row is
#: an outlier if its length ratio is more than MAD_K median-absolute-deviations
#: from its language's median. Median/MAD rather than mean/sd because a single
#: extreme value would otherwise inflate the threshold and hide itself.
MAD_K = 8.0


def load_frozen() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in FROZEN_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_imported_decisions() -> list[dict[str, Any]]:
    if not IMPORTED_DECISIONS.exists():
        return []
    return [
        json.loads(line)
        for line in IMPORTED_DECISIONS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_pending_tsv() -> list[dict[str, str]]:
    if not PENDING_TSV.exists():
        return []
    with PENDING_TSV.open(encoding="utf-8", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh, delimiter="\t")]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    for path in (FROZEN_PATH, SPLITS_PATH, RESPONSES_PATH):
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 1

    header = stage_report_header("stage3.1_audit_review_state", {"read_only": True})
    print("READ-ONLY AUDIT -- the frozen dataset and split are not modified")
    print(f"run_id:     {header['run_id']}")
    print(f"git commit: {header['git_commit']}  (dirty={header['git_dirty']})")

    frozen = load_frozen()
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    assignments = splits["assignments"]
    effective = ResponseLog(RESPONSES_PATH).effective()

    non_english = [r for r in frozen if r["language"] != PIVOT_LANGUAGE]
    unchecked = [r for r in non_english if r["translation_status"] == "MACHINE_TRANSLATED_UNCHECKED"]

    print("\n1. ROW COUNTS")
    print(f"  total rows                     : {len(frozen)}")
    print(f"  non-English rows               : {len(non_english)}")
    print(f"  MACHINE_TRANSLATED_UNCHECKED   : {len(unchecked)}")
    print(f"  other statuses (non-English)   : "
          f"{dict(sorted(Counter(r['translation_status'] for r in non_english if r not in unchecked).items())) or '{}'}")
    print(f"  HUMAN_REVIEWED                 : "
          f"{sum(1 for r in frozen if r['translation_status'] == 'HUMAN_REVIEWED')}")
    print(f"  review_required                : {sum(1 for r in non_english if r['review_required'])}"
          f" / {len(non_english)}")

    # --- 2/3. decisions ----------------------------------------------------
    imported = load_imported_decisions()
    pending_rows = load_pending_tsv()

    by_request: dict[str, dict[str, Any]] = {d["request_id"]: d for d in imported}
    print("\n2. REVIEW DECISIONS")
    print(f"  imported decision records      : {len(imported)}  ({IMPORTED_DECISIONS.name})")
    print(f"  un-imported PENDING sheet rows : {len(pending_rows)}  ({PENDING_TSV.name})")
    print(f"  distinct request_ids covered   : {len(by_request)}")

    imported_by_language = Counter(d["language"] for d in imported)
    imported_by_decision = Counter(d["decision"] for d in imported)
    pending_by_language = Counter(r["language"] for r in pending_rows)
    pending_by_decision = Counter(r["decision"] for r in pending_rows)

    print("\n3. DECISIONS BY LANGUAGE")
    print(f"  {'lang':<6}{'imported':>10}{'pending-sheet':>16}{'total':>8}{'unreviewed':>12}")
    per_language_rows = Counter(r["language"] for r in non_english)
    for lang in NON_PIVOT_LANGUAGES:
        total = imported_by_language[lang] + pending_by_language[lang]
        print(f"  {lang:<6}{imported_by_language[lang]:>10}{pending_by_language[lang]:>16}"
              f"{total:>8}{per_language_rows[lang] - imported_by_language[lang]:>12}")

    print("\n4. DECISIONS BY VERDICT")
    print(f"  {'verdict':<12}{'imported':>10}{'pending-sheet':>16}")
    for verdict in DECISION_VALUES:
        print(f"  {verdict:<12}{imported_by_decision.get(verdict, 0):>10}"
              f"{pending_by_decision.get(verdict, 0):>16}")

    # --- 5. hash linkage ---------------------------------------------------
    stale: list[dict[str, str]] = []
    verified = 0
    orphaned: list[str] = []
    for decision in imported:
        request_id = decision["request_id"]
        response = effective.get(request_id)
        if response is None or response.get("status") != STATUS_OK:
            orphaned.append(request_id)
            continue
        if decision.get("response_sha256") != response.get("response_sha256"):
            stale.append({
                "request_id": request_id,
                "language": decision["language"],
                "reviewed_sha256": (decision.get("response_sha256") or "")[:16],
                "corpus_sha256": (response.get("response_sha256") or "")[:16],
            })
        else:
            verified += 1

    print("\n5. DECISION -> RESPONSE HASH LINKAGE")
    print(f"  decisions whose response_sha256 matches the corpus : {verified}")
    print(f"  decisions linked to a STALE response_sha256        : {len(stale)}")
    print(f"  decisions with no corpus response                  : {len(orphaned)}")
    if stale:
        print("  stale (reviewed text is no longer the corpus text):")
        for row in stale[:12]:
            print(f"    {row['request_id']:<24} reviewed={row['reviewed_sha256']}  "
                  f"corpus={row['corpus_sha256']}")
        if len(stale) > 12:
            print(f"    ... and {len(stale) - 12} more")

    # --- 6. coverage -------------------------------------------------------
    all_request_ids = {
        make_request_id(r["behavior_id"], r["language"]) for r in non_english
    }
    reviewed_ids = set(by_request)
    uncovered = all_request_ids - reviewed_ids
    print("\n6. REVIEW COVERAGE")
    print(f"  non-English rows                 : {len(all_request_ids)}")
    print(f"  rows with a review record        : {len(reviewed_ids & all_request_ids)}")
    print(f"  rows with NO review record       : {len(uncovered)}")
    print(f"  every row reviewed               : {not uncovered}")

    # --- 7/8. length outliers ---------------------------------------------
    source_len = {
        r["behavior_id"]: len(r["source_text"]) for r in frozen if r["language"] == PIVOT_LANGUAGE
    }
    ratios: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in non_english:
        translated = row.get("translated_text") or ""
        base = max(source_len.get(row["behavior_id"], 1), 1)
        ratios[row["language"]].append(
            (make_request_id(row["behavior_id"], row["language"]), len(translated) / base)
        )

    outliers: list[dict[str, Any]] = []
    language_stats: dict[str, dict[str, float]] = {}
    for lang in NON_PIVOT_LANGUAGES:
        values = [r for _, r in ratios[lang]]
        median = statistics.median(values)
        mad = statistics.median([abs(v - median) for v in values]) or 1e-9
        language_stats[lang] = {
            "median_ratio": round(median, 4),
            "mad": round(mad, 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }
        for request_id, ratio in ratios[lang]:
            reasons = []
            if ratio > LONG_RATIO:
                reasons.append(f"ratio>{LONG_RATIO}")
            if ratio < SHORT_RATIO:
                reasons.append(f"ratio<{SHORT_RATIO}")
            if abs(ratio - median) > MAD_K * mad:
                reasons.append(f"|ratio-median|>{MAD_K}*MAD")
            if reasons:
                behavior = request_id.split(":")[0]
                outliers.append({
                    "request_id": request_id,
                    "behavior_id": behavior,
                    "language": lang,
                    "length_ratio": round(ratio, 3),
                    "language_median_ratio": round(median, 3),
                    "reasons": reasons,
                    "split": assignments.get(behavior),
                })
    outliers.sort(key=lambda o: -o["length_ratio"])

    print("\n7. LENGTH OUTLIERS")
    print(f"  thresholds: ratio>{LONG_RATIO} | ratio<{SHORT_RATIO} | "
          f"|ratio-median|>{MAD_K}*MAD (per language)")
    print(f"  {'lang':<6}{'median':>9}{'mad':>8}{'min':>8}{'max':>8}")
    for lang in NON_PIVOT_LANGUAGES:
        s = language_stats[lang]
        print(f"  {lang:<6}{s['median_ratio']:>9}{s['mad']:>8}{s['min']:>8}{s['max']:>8}")
    print(f"\n  outlier rows: {len(outliers)}")
    for row in outliers[:20]:
        print(f"    {row['request_id']:<24} ratio={row['length_ratio']:>7}  "
              f"median={row['language_median_ratio']:<6} split={row['split']:<11} "
              f"{','.join(row['reasons'])}")
    if len(outliers) > 20:
        print(f"    ... and {len(outliers) - 20} more")

    target = "beh_b117279f5262:yo"
    target_rows = [o for o in outliers if o["request_id"] == target]
    print("\n8. beh_b117279f5262 YORUBA")
    print(f"  flagged as an outlier : {bool(target_rows)}")
    if target_rows:
        row = target_rows[0]
        print(f"  length ratio          : {row['length_ratio']} "
              f"(language median {row['language_median_ratio']})")
        print(f"  reasons               : {row['reasons']}")
        print(f"  SPLIT                 : {row['split']}")
    print(f"  is it the ONLY outlier: {len(outliers) == 1 and bool(target_rows)}")
    if len(outliers) > 1:
        others = Counter(o["language"] for o in outliers if o["request_id"] != target)
        print(f"  other outliers by language: {dict(sorted(others.items()))}")
        print(f"  other outliers by split   : "
              f"{dict(sorted(Counter(o['split'] for o in outliers if o['request_id'] != target).items()))}")

    report = {
        **header,
        "read_only": True,
        "rows": {
            "total": len(frozen),
            "non_english": len(non_english),
            "machine_translated_unchecked": len(unchecked),
            "human_reviewed": sum(
                1 for r in frozen if r["translation_status"] == "HUMAN_REVIEWED"
            ),
            "review_required": sum(1 for r in non_english if r["review_required"]),
        },
        "decisions": {
            "imported": len(imported),
            "pending_sheet_rows": len(pending_rows),
            "imported_by_language": dict(sorted(imported_by_language.items())),
            "imported_by_decision": {v: imported_by_decision.get(v, 0) for v in DECISION_VALUES},
            "pending_sheet_by_language": dict(sorted(pending_by_language.items())),
            "pending_sheet_by_decision": {v: pending_by_decision.get(v, 0) for v in DECISION_VALUES},
        },
        "hash_linkage": {
            "verified": verified,
            "stale": len(stale),
            "orphaned": len(orphaned),
            "stale_request_ids": [s["request_id"] for s in stale],
            "stale_detail": stale,
        },
        "coverage": {
            "non_english_rows": len(all_request_ids),
            "rows_with_a_review_record": len(reviewed_ids & all_request_ids),
            "rows_without_a_review_record": len(uncovered),
            "every_row_reviewed": not uncovered,
        },
        "length_outliers": {
            "thresholds": {"long_ratio": LONG_RATIO, "short_ratio": SHORT_RATIO, "mad_k": MAD_K},
            "language_stats": language_stats,
            "n_outliers": len(outliers),
            "outliers": outliers,
            "target": target,
            "target_is_outlier": bool(target_rows),
            "target_is_only_outlier": len(outliers) == 1 and bool(target_rows),
            "target_split": target_rows[0]["split"] if target_rows else None,
        },
        "split_hash": splits["split_hash"],
        "frozen_dataset_unmodified": True,
        "contains_prompt_text": False,
        "source_hashes": {
            "frozen_dataset": file_sha256(FROZEN_PATH),
            "splits": file_sha256(SPLITS_PATH),
            "responses": file_sha256(RESPONSES_PATH),
        },
    }
    report["deterministic_digest"] = deterministic_digest(report)

    if args.write_report:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {REPORT_PATH.relative_to(REPO_ROOT)}  (tracked, metadata-only)")
    else:
        print("\n(no report written; pass --write-report to persist)")

    print(f"\nsplit_hash: {splits['split_hash'][:24]}  (unchanged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
