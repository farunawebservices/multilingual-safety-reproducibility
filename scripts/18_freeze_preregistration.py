#!/usr/bin/env python3
"""Stage 3.3 -- freeze the pre-registration, but only if the gate is clear.

Sets ``frozen: true`` and ``frozen_at`` in
``configs/experiment_preregistration_v1.yaml`` after
``multilingual_safety.freeze.check_freeze_preconditions`` reports no failures.
If any precondition fails, nothing is written and the failures are printed.

The edit is TEXTUAL, on three anchored lines. The config is not round-tripped
through a YAML dumper, because dumping would discard every comment in the file --
and in a pre-registration the comments carry the reasoning.

What this does NOT do
---------------------
* It does not touch ``data/processed/frozen_dataset_v1.jsonl`` or
  ``data/processed/splits_v1.json``; both are opened read-only and their hashes
  are compared before and after.
* It does not run a model, read ``final_test``, or log a holdout read event.
  Freezing makes the holdout *eligible* to be read; it does not read it.
* It does not set ``frozen_by``: no reviewer identity was supplied, and inventing
  one would put an unearned name on the freeze record.

Usage
-----
    python scripts/18_freeze_preregistration.py            # check the gate only
    python scripts/18_freeze_preregistration.py --write     # freeze
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multilingual_safety.freeze import (
    FREEZE_PRECONDITIONS,
    check_freeze_preconditions,
    freeze_is_consistent,
)
from multilingual_safety.hashing import file_sha256
from multilingual_safety.paths import CONFIGS_DIR, PROCESSED_DIR, REPO_ROOT, RESULTS_DIR

PREREG_PATH = CONFIGS_DIR / "experiment_preregistration_v1.yaml"
FROZEN_PATH = PROCESSED_DIR / "frozen_dataset_v1.jsonl"
SPLITS_PATH = PROCESSED_DIR / "splits_v1.json"
OUTLIER_REPORT_PATH = RESULTS_DIR / "length_outlier_review.json"

_FROZEN_LINE = re.compile(r"^frozen: false$", re.MULTILINE)
_FROZEN_AT_LINE = re.compile(r"^frozen_at: null$", re.MULTILINE)
_STATUS_LINE = re.compile(
    r"^# STATUS: NOT FROZEN\. `final_test` may not be read until `frozen: true` and\n"
    r"# `frozen_at` are set\. See `holdout_policy` below\.$",
    re.MULTILINE,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--write", action="store_true", help="Apply the freeze.")
    args = parser.parse_args()

    if not PREREG_PATH.exists():
        print(f"missing {PREREG_PATH}", file=sys.stderr)
        return 1

    text = PREREG_PATH.read_text(encoding="utf-8")
    prereg = yaml.safe_load(text)
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8")) if SPLITS_PATH.exists() else None
    outlier = (
        json.loads(OUTLIER_REPORT_PATH.read_text(encoding="utf-8"))
        if OUTLIER_REPORT_PATH.exists()
        else None
    )

    frozen_sha_before = file_sha256(FROZEN_PATH) if FROZEN_PATH.exists() else None
    splits_sha_before = file_sha256(SPLITS_PATH) if SPLITS_PATH.exists() else None

    print("FREEZE GATE")
    print(f"  config       : {PREREG_PATH.relative_to(REPO_ROOT)}")
    print(f"  split file   : {'present' if splits else 'ABSENT'}")
    print(f"  outlier report: {'present' if outlier else 'ABSENT'}")
    print(f"\n  preconditions checked ({len(FREEZE_PRECONDITIONS)}):")
    for name in FREEZE_PRECONDITIONS:
        print(f"    - {name}")

    failures = check_freeze_preconditions(prereg, splits, outlier)
    if failures:
        print(f"\n  {len(failures)} precondition failure(s):")
        for failure in failures:
            print(f"    FAIL: {failure}")
        print("\nnothing written; the pre-registration remains unfrozen", file=sys.stderr)
        return 1
    print("\n  all preconditions hold")

    if prereg.get("frozen") is True:
        print(f"\nalready frozen at {prereg.get('frozen_at')}; nothing to do")
        return 0

    frozen_at = datetime.now(timezone.utc).isoformat()

    new_text, n_frozen = _FROZEN_LINE.subn("frozen: true", text, count=1)
    new_text, n_at = _FROZEN_AT_LINE.subn(f"frozen_at: '{frozen_at}'", new_text, count=1)
    new_text, n_status = _STATUS_LINE.subn(
        "# STATUS: FROZEN. The commitments below are binding. `final_test` is now\n"
        "# eligible to be read ONCE per finalised configuration, and only with a\n"
        "# logged read event appended to `holdout_policy.final_test_reads`.",
        new_text,
        count=1,
    )
    if (n_frozen, n_at, n_status) != (1, 1, 1):
        print(
            f"anchored edit did not match exactly once "
            f"(frozen={n_frozen}, frozen_at={n_at}, status={n_status}); nothing written",
            file=sys.stderr,
        )
        return 1

    # Re-parse the edited text before trusting it, and re-check the gate on the
    # parsed result so a freeze can never be written on a file that no longer
    # satisfies its own preconditions.
    reparsed = yaml.safe_load(new_text)
    post_failures = check_freeze_preconditions(reparsed, splits, outlier)
    post_failures += freeze_is_consistent(reparsed)
    if post_failures:
        for failure in post_failures:
            print(f"  FAIL (post-edit): {failure}", file=sys.stderr)
        return 1
    if reparsed.get("frozen") is not True or reparsed.get("frozen_at") != frozen_at:
        print("post-edit parse does not show the freeze; nothing written", file=sys.stderr)
        return 1
    # The freeze must not have disturbed anything else in the file.
    for key in ("splits", "conditions", "dataset", "holdout_policy", "models"):
        if reparsed[key] != prereg[key]:
            print(f"post-edit parse changed {key!r}; nothing written", file=sys.stderr)
            return 1

    print("\n  frozen            : true")
    print(f"  frozen_at         : {frozen_at}")
    print(f"  frozen_by         : {reparsed.get('frozen_by')}  (not supplied; left unset)")
    print(f"  split_hash        : {reparsed['splits']['split_hash'][:24]}  (unchanged)")
    print(f"  final_test_readable: {reparsed['holdout_policy']['final_test_readable']}")
    print(f"  final_test_reads   : {reparsed['holdout_policy']['final_test_reads']}")
    print(f"  declared deviations: {len(reparsed['holdout_policy']['pre_freeze_deviations'])}")

    if args.write:
        PREREG_PATH.write_text(new_text, encoding="utf-8")
        print(f"\nwrote {PREREG_PATH.relative_to(REPO_ROOT)}")
    else:
        print("\n(gate only; pass --write to apply the freeze)")

    frozen_sha_after = file_sha256(FROZEN_PATH) if FROZEN_PATH.exists() else None
    splits_sha_after = file_sha256(SPLITS_PATH) if SPLITS_PATH.exists() else None
    if (frozen_sha_before, splits_sha_before) != (frozen_sha_after, splits_sha_after):
        print("dataset or split file changed during the freeze", file=sys.stderr)
        return 1
    print(f"  frozen dataset sha256 unchanged : {frozen_sha_after == frozen_sha_before}")
    print(f"  splits sha256 unchanged         : {splits_sha_after == splits_sha_before}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
