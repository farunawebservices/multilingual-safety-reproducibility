#!/usr/bin/env python3
"""Verify a local corpus against the tracked manifest.

Two independent checks:

1. **Structural** -- the manifest parses, IDs are unique, no forbidden field
   smuggles prompt text or credentials into a tracked file, all seven languages
   are present, both arms are present.
2. **Content** (optional, ``--data``) -- re-hash the locally materialised text
   and compare against the manifest's ``content_sha256``.

Check 2 is what lets two people confirm they built the same corpus without
either sending the text to the other -- the property that makes a text-free
tracked manifest workable at all.

Every mismatch is reported. Nothing is repaired, and nothing is dropped
(CLAUDE.md §2).

Usage
-----
    python scripts/verify_manifest.py
    python scripts/verify_manifest.py --data data/processed/corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multilingual_safety.ids import content_hash
from multilingual_safety.languages import LANGUAGE_CODES
from multilingual_safety.manifest import ManifestError, read_manifest, summarize
from multilingual_safety.paths import BEHAVIORS_MANIFEST, SPLITS_FILE
from multilingual_safety.splits import SplitError, load_splits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=BEHAVIORS_MANIFEST)
    parser.add_argument("--splits", type=Path, default=SPLITS_FILE)
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional local JSONL with record_id + text, for content hash verification. "
        "Git-ignored; never committed.",
    )
    args = parser.parse_args()

    failures: list[str] = []

    print(f"manifest: {args.manifest}")
    if not args.manifest.exists():
        print("\nNo manifest yet. Nothing to verify -- build it first.")
        return 0

    try:
        records = read_manifest(args.manifest)
    except ManifestError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1

    stats = summarize(records)
    print("\nSTRUCTURAL CHECKS")
    print(f"  rows:               {stats['n_rows']}")
    print(f"  distinct behaviors: {stats['n_behaviors']}")
    print(f"  rows per language:  {stats['rows_per_language']}")
    print(f"  rows per arm:       {stats['rows_per_arm']}")

    ids = [r.record_id for r in records]
    if len(ids) != len(set(ids)):
        failures.append(f"duplicate record_id values: {len(ids) - len(set(ids))} duplicates")
    if stats["languages_with_zero_rows"]:
        failures.append(f"languages with zero rows: {stats['languages_with_zero_rows']}")
    if stats["arms_with_zero_rows"]:
        failures.append(f"arms with zero rows: {stats['arms_with_zero_rows']}")
    if stats["n_behaviors_incomplete"]:
        failures.append(
            f"{stats['n_behaviors_incomplete']} behaviors missing at least one of "
            f"{len(LANGUAGE_CODES)} languages (breaks the paired design)"
        )

    print("\nSPLIT CHECKS")
    if args.splits.exists():
        try:
            payload = load_splits(args.splits)
            print(f"  split_hash: {payload['split_hash']}")
            print(f"  counts:     {payload['counts']}")
            unassigned = {r.behavior_id for r in records} - set(payload["assignments"])
            if unassigned:
                failures.append(f"{len(unassigned)} behaviors have no split assignment")
        except SplitError as exc:
            failures.append(str(exc))
    else:
        print(f"  no split file at {args.splits} (not yet frozen)")

    if args.data:
        print("\nCONTENT HASH CHECKS")
        if not args.data.exists():
            failures.append(f"data file not found: {args.data}")
        else:
            by_id = {r.record_id: r for r in records}
            checked = mismatched = unknown = 0
            for line in args.data.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                rid = row.get("record_id")
                rec = by_id.get(rid)
                if rec is None:
                    unknown += 1
                    continue
                checked += 1
                if content_hash(row["text"], normalize_form=rec.normalization) != rec.content_sha256:
                    mismatched += 1
                    # Report the ID, never the text.
                    print(f"  MISMATCH {rid} ({rec.language})")
            missing = len(by_id) - checked
            print(f"  checked: {checked}  mismatched: {mismatched}  "
                  f"missing from data: {missing}  unknown ids in data: {unknown}")
            if mismatched:
                failures.append(f"{mismatched} content hash mismatches")
            if missing:
                failures.append(f"{missing} manifest records absent from local data")
            if unknown:
                failures.append(f"{unknown} local records not in the manifest")

    print("\nRESULT")
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        print(f"\n{len(failures)} check(s) failed.", file=sys.stderr)
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
