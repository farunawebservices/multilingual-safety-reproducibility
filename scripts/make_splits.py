#!/usr/bin/env python3
"""Freeze the dev/test split. Run ONCE, before any model is run.

Splits are assigned at the *behavior* level, so all seven language renderings of
a behavior share a split. Record-level splitting would put the English rendering
in dev and the Hausa rendering in test -- leaking the item and destroying the
paired cross-lingual design.

The output is content-addressed. Its ``split_hash`` is recorded in every run's
metadata; if it ever changes, every prior test result is invalidated
(CLAUDE.md §1).

This script refuses to overwrite an existing split file.

Usage
-----
    python scripts/make_splits.py --dry-run
    python scripts/make_splits.py --salt "<recorded salt>" --test-fraction 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multilingual_safety.manifest import read_manifest
from multilingual_safety.paths import BEHAVIORS_MANIFEST, SPLITS_FILE
from multilingual_safety.splits import build_splits, write_splits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=BEHAVIORS_MANIFEST)
    parser.add_argument("--out", type=Path, default=SPLITS_FILE)
    parser.add_argument(
        "--salt",
        default="mls-split-v1",
        help="Recorded in the split file. Changing it re-splits the corpus and "
        "invalidates every prior test result.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true", help="Report the split; write nothing.")
    args = parser.parse_args()

    print(f"manifest: {args.manifest}")
    print(f"output:   {args.out}")
    print(f"salt:     {args.salt}")
    print(f"mode:     {'DRY-RUN' if args.dry_run else 'WRITE'}")
    print()

    if not args.manifest.exists():
        print(f"No manifest at {args.manifest}. Build it first (scripts/build_manifest.py).")
        return 0

    if args.out.exists() and not args.dry_run:
        print(
            f"STOP: {args.out} already exists. The split is frozen before any model run; "
            "re-splitting invalidates prior test results (CLAUDE.md §1).",
            file=sys.stderr,
        )
        return 1

    records = read_manifest(args.manifest)
    behaviors = sorted({r.behavior_id for r in records})
    payload = build_splits(behaviors, salt=args.salt, test_fraction=args.test_fraction)

    print("SUMMARY")
    print(f"  behaviors:  {len(behaviors)}")
    print(f"  counts:     {payload['counts']}")
    print(f"  split_hash: {payload['split_hash']}")

    if args.dry_run:
        print("\nDry run: nothing written.")
        return 0

    path = write_splits(payload, args.out)
    print(f"\nwrote -> {path}")
    print("Record this split_hash. Development uses dev ONLY (CLAUDE.md §1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
