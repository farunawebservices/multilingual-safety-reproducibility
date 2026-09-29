#!/usr/bin/env python3
"""Stage 3.2 -- record the reviewed Yoruba length outlier as metadata only.

One row (``beh_b117279f5262:yo``, the 11.6x length outlier declared in
``configs/experiment_preregistration_v1.yaml``) was inspected out of band, with
external native-speaker checking, and accepted as semantically valid. That
happened outside this repository's review tooling, so no structured reviewer
identity, timestamp, or reviewed-text hash was captured.

This script records that outcome, and records the fact that it is unstructured.

What it does NOT do
-------------------
* It does not write ``data/processed/frozen_dataset_v1.jsonl`` or
  ``data/processed/splits_v1.json``; both are opened read-only.
* It does not change a split assignment, a source hash, or a translation hash.
* It does not set ``translation_status: HUMAN_REVIEWED`` and does not clear
  ``review_required``. Under ``multilingual_safety.review`` only a
  ``native_speaker`` or ``professional_translator`` decision *with a named
  reviewer* may clear review, and this record has no named reviewer -- see
  ``review_decision_schema_representable`` in the output.
* It does not print or store prompt text, translated text, or an exact
  character count. The row's length reaches the report only as a coarse bucket
  (``langcheck.character_length_bucket``).

The target row is in ``final_test``. Its text is measured in memory to derive a
bucket and is never printed, logged, or written -- the same read the stage-3.1
audit already performs to compute length ratios.

Usage
-----
    python scripts/17_record_length_outlier_review.py            # dry run
    python scripts/17_record_length_outlier_review.py --write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multilingual_safety.hashing import deterministic_digest, file_sha256
from multilingual_safety.langcheck import LONG_RATIO, character_length_bucket
from multilingual_safety.paths import PROCESSED_DIR, REPO_ROOT, RESULTS_DIR
from multilingual_safety.providers import make_request_id
from multilingual_safety.review import (
    ReviewDecisionType,
    ReviewerCompetence,
    competence_rationale,
)
from multilingual_safety.run_context import stage_report_header

FROZEN_PATH = PROCESSED_DIR / "frozen_dataset_v1.jsonl"
SPLITS_PATH = PROCESSED_DIR / "splits_v1.json"
REPORT_PATH = RESULTS_DIR / "length_outlier_review.json"

TARGET_BEHAVIOR = "beh_b117279f5262"
TARGET_LANGUAGE = "yo"
EXPECTED_SPLIT = "final_test"

OUTLIER_LABEL = "valid_translation_length_outlier"
REVIEW_PROVENANCE = "external_native_check_unstructured"

#: What the out-of-band review asserted. Recorded as assertions, under a
#: provenance label that says they are unstructured, because nothing in this
#: repository can verify them.
ASSERTED_FACTS = {
    "translation_manually_inspected": True,
    "external_native_speaker_check_performed": True,
    "accepted_as_semantically_valid": True,
}


def load_target_row() -> dict[str, Any]:
    """The one frozen row under review, plus its English source. Read-only."""
    target: dict[str, Any] | None = None
    source: dict[str, Any] | None = None
    with FROZEN_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["behavior_id"] != TARGET_BEHAVIOR:
                continue
            if row["language"] == TARGET_LANGUAGE:
                target = row
            elif row["language"] == "en":
                source = row
    if target is None:
        raise SystemExit(f"{TARGET_BEHAVIOR}:{TARGET_LANGUAGE} is not in {FROZEN_PATH.name}")
    if source is None:
        raise SystemExit(f"{TARGET_BEHAVIOR}:en is not in {FROZEN_PATH.name}")
    return {"target": target, "source": source}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--write", action="store_true", help="Persist the report.")
    args = parser.parse_args()

    for path in (FROZEN_PATH, SPLITS_PATH):
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 1

    # Hash the inputs BEFORE and AFTER, so "we did not touch them" is a checked
    # claim rather than a promise in a docstring.
    frozen_sha_before = file_sha256(FROZEN_PATH)
    splits_sha_before = file_sha256(SPLITS_PATH)

    rows = load_target_row()
    target, source = rows["target"], rows["source"]
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))

    # --- verify the target is the row we think it is -----------------------
    split_from_frozen = target.get("split")
    split_from_splits = splits["assignments"].get(TARGET_BEHAVIOR)
    problems: list[str] = []
    if split_from_frozen != EXPECTED_SPLIT:
        problems.append(f"frozen row split is {split_from_frozen!r}, expected {EXPECTED_SPLIT!r}")
    if split_from_splits != EXPECTED_SPLIT:
        problems.append(f"split file says {split_from_splits!r}, expected {EXPECTED_SPLIT!r}")

    # Measured in memory only: the bucket goes to the report, the count does not.
    n_chars = len(target.get("translated_text") or "")
    source_chars = max(len(source.get("source_text") or ""), 1)
    length_ratio = n_chars / source_chars
    bucket = character_length_bucket(n_chars)
    if length_ratio <= LONG_RATIO:
        problems.append(
            f"row is not a long-length outlier (ratio {length_ratio:.3f} <= {LONG_RATIO})"
        )

    if problems:
        for problem in problems:
            print(f"  FAIL: {problem}", file=sys.stderr)
        print("\nrefusing to record a review for a row that does not match the target",
              file=sys.stderr)
        return 1

    header = stage_report_header(
        "stage3.2_record_length_outlier_review",
        {
            "behavior_id": TARGET_BEHAVIOR,
            "language": TARGET_LANGUAGE,
            "outlier_label": OUTLIER_LABEL,
            "review_provenance": REVIEW_PROVENANCE,
        },
    )

    report = {
        **header,
        "behavior_id": TARGET_BEHAVIOR,
        "language": TARGET_LANGUAGE,
        "request_id": make_request_id(TARGET_BEHAVIOR, TARGET_LANGUAGE),
        "split": EXPECTED_SPLIT,
        "character_length_bucket": bucket,
        "length_ratio_bucketed_note": (
            "Exact character counts are withheld deliberately; the bucket is the "
            "only length detail recorded. The 11.6x length ratio for this row is "
            "already declared in configs/experiment_preregistration_v1.yaml."
        ),
        "decision": ReviewDecisionType.ACCEPT.value,
        "outlier_label": OUTLIER_LABEL,
        "review_provenance": REVIEW_PROVENANCE,
        "asserted_facts": dict(sorted(ASSERTED_FACTS.items())),
        # --- the limits of this record ------------------------------------
        "structured_reviewer_identity_captured": False,
        "structured_reviewer_identity_note": (
            "No structured reviewer identity was captured: this record has no "
            "reviewer_id, no reviewer_language_competence, no decided_at, and no "
            "hash of the text the reviewer actually saw. The external "
            "native-speaker check is asserted, not evidenced, and cannot be "
            "attributed, audited, or reproduced from this repository."
        ),
        "review_decision_schema_representable": False,
        "review_decision_schema_note": (
            "multilingual_safety.review.ReviewDecision rejects this record: it "
            "requires a named reviewer_id. That is why this is a standalone "
            "metadata report and not an entry in the review-decision layer."
        ),
        "clears_review_required": False,
        "clears_review_required_rationale": competence_rationale(ReviewerCompetence.NONE),
        "native_speaker_validation_of_record": False,
        "row_state_unchanged": {
            "translation_status": target["translation_status"],
            "review_required": target["review_required"],
            "reviewer_id": target["reviewer_id"],
            "translation_method": target["translation_method"],
        },
        "frozen": False,
        "frozen_note": "frozen=true is deliberately not set at this stage.",
        # --- what this stage did not touch --------------------------------
        "frozen_dataset_unmodified": True,
        "splits_unmodified": True,
        "split_assignment_unmodified": True,
        "source_hashes_unmodified": True,
        "translation_hashes_unmodified": True,
        "prompt_hash": target["prompt_hash"],
        "split_hash": target["split_hash"],
        "contains_prompt_text": False,
        "final_test_text_loaded_into_report": False,
        "source_hashes": {
            "frozen_dataset": frozen_sha_before,
            "splits": splits_sha_before,
        },
    }

    # The inputs must be byte-identical to how we found them.
    frozen_sha_after = file_sha256(FROZEN_PATH)
    splits_sha_after = file_sha256(SPLITS_PATH)
    if (frozen_sha_after, splits_sha_after) != (frozen_sha_before, splits_sha_before):
        print("input files changed during the run; refusing to write", file=sys.stderr)
        return 1
    if splits["split_hash"] != target["split_hash"]:
        print("split_hash mismatch between the split file and the frozen row",
              file=sys.stderr)
        return 1

    report["deterministic_digest"] = deterministic_digest(report)

    print("METADATA-ONLY OUTLIER REVIEW RECORD")
    print(f"run_id:     {header['run_id']}")
    print(f"git commit: {header['git_commit']}  (dirty={header['git_dirty']})")
    for key in (
        "behavior_id",
        "language",
        "split",
        "character_length_bucket",
        "decision",
        "outlier_label",
        "review_provenance",
        "structured_reviewer_identity_captured",
        "clears_review_required",
        "frozen",
    ):
        print(f"  {key:<40}: {report[key]}")
    print(f"  {'translation_status (unchanged)':<40}: "
          f"{report['row_state_unchanged']['translation_status']}")
    print(f"  {'review_required (unchanged)':<40}: "
          f"{report['row_state_unchanged']['review_required']}")
    print(f"\n  frozen dataset sha256 unchanged : {frozen_sha_after == frozen_sha_before}")
    print(f"  splits sha256 unchanged         : {splits_sha_after == splits_sha_before}")

    if args.write:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {REPORT_PATH.relative_to(REPO_ROOT)}  (tracked, metadata-only)")
    else:
        print("\n(no report written; pass --write to persist)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
