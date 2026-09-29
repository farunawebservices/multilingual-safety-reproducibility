#!/usr/bin/env python3
"""Materialise registered third-party sources into git-ignored ``data/raw/``.

**Nothing is downloaded at this stage of the project.** ``configs/data_sources.yaml``
registers no sources, so this script currently reports an empty plan and exits.
Downloading is gated behind ``--execute``; the default is ``--dry-run``.

Guarantees enforced here:

* A source with unresolved license metadata is never fetched.
* ``redistribution_allowed: unknown`` is treated as ``false``.
* A gated source is never fetched without ``requires_agreement`` acknowledged.
* Credentials come from the environment by NAME; no key is read from config,
  printed, or written to disk.
* Everything lands in ``data/raw/``, which is git-ignored.

Usage
-----
    python scripts/fetch_sources.py --dry-run     # default: show the plan
    python scripts/fetch_sources.py --execute     # perform fetches (not yet implemented)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from multilingual_safety.paths import RAW_DIR, SOURCES_CONFIG

REQUIRED_FIELDS = (
    "id",
    "name",
    "version",
    "url",
    "license",
    "license_url",
    "redistribution_allowed",
    "requires_agreement",
    "contains_harmful_text",
    "arm",
    "citation",
)


def load_sources(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(raw.get("sources") or [])


def check_source(source: dict[str, Any]) -> list[str]:
    """Return a list of blocking problems for one source (empty == fetchable)."""
    problems: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in source or source[field] in (None, ""):
            problems.append(f"missing required field: {field}")

    if str(source.get("version", "")).lower() in {"latest", "main", "head"}:
        problems.append("version must be pinned, not 'latest'/'main'/'head'")

    redistribution = source.get("redistribution_allowed", "unknown")
    if redistribution == "unknown":
        # Documented policy: unknown is treated as forbidden, not as permitted.
        problems.append(
            "redistribution_allowed is 'unknown'; treated as false. Resolve the license "
            "before fetching (THIRD_PARTY_DATASETS.md)."
        )

    if source.get("requires_agreement") and not source.get("agreement_accepted"):
        problems.append(
            "source is access-gated; complete the upstream agreement and set "
            "agreement_accepted: true. Do not route around the gate."
        )

    key_env = source.get("api_key_env")
    if key_env and not os.environ.get(key_env):
        problems.append(f"credential env var {key_env} is not set (name only; never store the value)")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=SOURCES_CONFIG)
    parser.add_argument("--out", type=Path, default=RAW_DIR)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", default=True, help="Show the plan only (default).")
    group.add_argument("--execute", action="store_true", help="Actually fetch. Not yet implemented.")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"ERROR: source registry not found: {args.config}", file=sys.stderr)
        return 2

    sources = load_sources(args.config)

    print("=" * 72)
    print("FETCH PLAN")
    print("=" * 72)
    print(f"registry:    {args.config}")
    print(f"destination: {args.out}  (git-ignored)")
    print(f"mode:        {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print(f"sources registered: {len(sources)}")
    print()

    if not sources:
        print("No sources registered. Nothing to fetch.")
        print()
        print("Next step: register each dataset in configs/data_sources.yaml with full")
        print("license metadata BEFORE fetching. Third-party terms are authoritative and")
        print("are NOT MIT. See THIRD_PARTY_DATASETS.md.")
        return 0

    blocked = 0
    for source in sources:
        sid = source.get("id", "<no id>")
        problems = check_source(source)
        status = "OK" if not problems else "BLOCKED"
        print(f"[{status}] {sid}  ({source.get('license', '?')}, arm={source.get('arm', '?')})")
        for problem in problems:
            print(f"         - {problem}")
        blocked += bool(problems)

    print()
    print(f"fetchable: {len(sources) - blocked}   blocked: {blocked}")

    if blocked:
        print("\nRefusing to fetch: resolve the blocking problems above first.", file=sys.stderr)
        return 1

    if args.execute:
        print(
            "\nERROR: fetching is not implemented yet. No data is downloaded at this "
            "stage of the project.",
            file=sys.stderr,
        )
        return 3

    print("\nDry run complete. No data was downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
