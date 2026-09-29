#!/usr/bin/env python3
"""Build the behavior manifest from locally materialised data.

Reads text from git-ignored ``data/raw/`` (or ``data/interim/``), computes IDs
and content hashes, and writes a text-free JSONL manifest to
``data/manifests/behaviors.jsonl``.

**The manifest never contains prompt text.** It carries IDs, SHA-256 hashes,
language codes, provenance, and translation metadata; the hashes let a
collaborator verify they built the identical corpus without either party sending
the text (THIRD_PARTY_DATASETS.md).

Nothing has been fetched yet, so this script currently has no input to read and
reports that rather than writing an empty manifest.

Usage
-----
    python scripts/build_manifest.py --dry-run
    python scripts/build_manifest.py --input data/interim/aligned.jsonl --out data/manifests/behaviors.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multilingual_safety.ids import behavior_id, content_hash, record_id
from multilingual_safety.languages import (
    load_languages,
    normalize,
    orthography_fingerprint,
)
from multilingual_safety.manifest import (
    BehaviorRecord,
    summarize,
    write_manifest,
)
from multilingual_safety.paths import (
    BEHAVIORS_MANIFEST,
    INTERIM_DIR,
    LANGUAGES_CONFIG,
)


def build_records(rows: list[dict]) -> list[BehaviorRecord]:
    """Convert aligned input rows into text-free manifest records.

    Each input row must supply: ``source_id``, ``source_item_id``, ``arm``,
    ``behavior_family``, ``language``, ``text``, and translation provenance.
    ``text`` is hashed and discarded -- it never reaches the output.
    """
    records: list[BehaviorRecord] = []
    for i, row in enumerate(rows):
        try:
            text = row["text"]
            language = row["language"]
            arm = row["arm"]
            bid = behavior_id(
                source_id=row["source_id"],
                source_item_id=str(row["source_item_id"]),
                arm=arm,
            )
            normalized = normalize(text, "NFC")
            records.append(
                BehaviorRecord(
                    record_id=record_id(
                        behavior=bid, language=language, variant=row.get("variant", "base")
                    ),
                    behavior_id=bid,
                    language=language,
                    arm=arm,
                    behavior_family=row["behavior_family"],
                    source_id=row["source_id"],
                    source_item_id=str(row["source_item_id"]),
                    content_sha256=content_hash(normalized),
                    normalization="NFC",
                    variant=row.get("variant", "base"),
                    translation_method=row.get(
                        "translation_method", "original" if language == "en" else "machine_unreviewed"
                    ),
                    translation_model=row.get("translation_model"),
                    review_status=row.get("review_status", "none"),
                    backtranslation_similarity=row.get("backtranslation_similarity"),
                    char_length=len(normalized),
                    orthography=orthography_fingerprint(normalized),
                    license_id=row.get("license_id"),
                    redistribution_allowed=row.get("redistribution_allowed"),
                    notes=row.get("notes", ""),
                )
            )
        except (KeyError, ValueError) as exc:
            # Loud, not skipped: a row that cannot be built is reported with its
            # index and halts the build (CLAUDE.md §2).
            raise SystemExit(f"row {i}: cannot build manifest record: {exc}") from exc
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        type=Path,
        default=INTERIM_DIR / "aligned.jsonl",
        help="Aligned JSONL with text (git-ignored). Text is hashed, never copied out.",
    )
    parser.add_argument("--out", type=Path, default=BEHAVIORS_MANIFEST)
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarise; write nothing.")
    args = parser.parse_args()

    languages = load_languages(LANGUAGES_CONFIG)
    print(f"languages: {', '.join(languages)}")
    print(f"input:     {args.input}")
    print(f"output:    {args.out}")
    print(f"mode:      {'DRY-RUN' if args.dry_run else 'WRITE'}")
    print()

    if not args.input.exists():
        print(f"No input at {args.input}.")
        print("Nothing has been fetched or aligned yet, and an empty manifest is a")
        print("stop condition rather than a valid output (CLAUDE.md §10).")
        print("\nRun scripts/fetch_sources.py first, once sources are registered.")
        return 0

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        print(f"ERROR: {args.input} is empty (stop condition).", file=sys.stderr)
        return 1

    records = build_records(rows)
    stats = summarize(records)

    print("SUMMARY")
    print(f"  rows in:            {len(rows)}")
    print(f"  records built:      {stats['n_rows']}")
    print(f"  distinct behaviors: {stats['n_behaviors']}")
    print(f"  rows per language:  {stats['rows_per_language']}")
    print(f"  rows per arm:       {stats['rows_per_arm']}")
    print(f"  incomplete behaviors (missing a language): {stats['n_behaviors_incomplete']}")

    if stats["languages_with_zero_rows"]:
        print(f"\nSTOP: languages with zero rows: {stats['languages_with_zero_rows']}", file=sys.stderr)
        return 1
    if stats["arms_with_zero_rows"]:
        print(f"\nSTOP: arms with zero rows: {stats['arms_with_zero_rows']}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\nDry run: nothing written.")
        return 0

    n = write_manifest(records, args.out)
    print(f"\nwrote {n} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
