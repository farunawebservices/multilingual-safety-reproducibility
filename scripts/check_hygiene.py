#!/usr/bin/env python3
"""Repository hygiene: no secrets, no corpus text, no run artifacts in git.

Intended as a pre-commit / CI gate. Checks that:

* no file tracked by git lives under ``data/raw``, ``data/interim``,
  ``data/processed``, ``results``, ``logs``, or the dataset-v2 payload
  directories (``data/v2/{interim,cache,tmp,raw_downloads,credentials}``);
* no tracked manifest row carries a forbidden field (prompt text, credentials);
* no tracked file matches a credential-shaped pattern;
* ``.gitignore`` still covers the untracked-content directories.

Usage
-----
    python scripts/check_hygiene.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multilingual_safety.manifest import FORBIDDEN_FIELDS
from multilingual_safety.paths import REPO_ROOT
from multilingual_safety.secrets import scan_paths

BLOCKED_TRACKED_PREFIXES = (
    "data/raw/",
    "data/interim/",
    "data/processed/",
    "data/review/",
    "results/",
    "logs/",
    # Dataset v2 (configs/dataset_v2_rebuild.yaml). data/v2/manifests/ and the
    # LFS-tracked data/v2/processed/frozen_dataset_v2.jsonl are deliberately
    # absent from this list: both are intended to be tracked.
    "data/v2/interim/",
    "data/v2/cache/",
    "data/v2/tmp/",
    "data/v2/raw_downloads/",
    "data/v2/credentials/",
)

ALLOWED_PLACEHOLDERS = {".gitkeep", "README.md"}

#: Metadata-only stage reports explicitly whitelisted in .gitignore. Kept in
#: sync with tests/test_repo_hygiene.py, which additionally verifies they are
#: text-free. Everything else under results/ stays blocked.
ALLOWED_RESULT_REPORTS = {
    "results/environment_report.json",
    "results/source_inventory.json",
    "results/translation_pilot_inventory.json",
    "results/translation_pilot_validation.json",
    "results/translation_pilot_review_summary.json",
    "results/full_request_set_inventory.json",
    "results/cpu_pilot_archive_report.json",
    "results/full_apply_report.json",
    "results/split_report.json",
    "results/length_outlier_review.json",
}

TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".jsonl", ".toml", ".txt", ".cfg", ".ini"}


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if out.returncode != 0:
        print("WARNING: not a git repository; skipping tracked-file checks", file=sys.stderr)
        return []
    return [line for line in out.stdout.splitlines() if line.strip()]


def main() -> int:
    failures: list[str] = []
    files = tracked_files()
    print(f"tracked files: {len(files)}")

    # 1. No corpus/run artifacts tracked (except the whitelisted metadata reports).
    for path in files:
        if (
            path.startswith(BLOCKED_TRACKED_PREFIXES)
            and Path(path).name not in ALLOWED_PLACEHOLDERS
            and path not in ALLOWED_RESULT_REPORTS
        ):
            failures.append(f"tracked file in a git-ignored content directory: {path}")

    # 2. Credential-shaped literals, via the library scanner so the detection
    #    logic has exactly one implementation.
    scannable = [
        REPO_ROOT / path
        for path in files
        if (REPO_ROOT / path).suffix in TEXT_SUFFIXES and (REPO_ROOT / path).exists()
    ]
    for finding in scan_paths(scannable):
        # Location only. The finding deliberately carries no matched value.
        failures.append(f"possible {finding.pattern_name} in tracked file: {finding}")

    # 3. Forbidden manifest fields.
    for path in files:
        full = REPO_ROOT / path
        if full.suffix not in TEXT_SUFFIXES or not full.exists():
            continue
        try:
            text = full.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        if path.startswith("data/manifests/") and full.suffix == ".jsonl":
            for lineno, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    failures.append(f"malformed JSON in tracked manifest {path}:{lineno}")
                    continue
                present = FORBIDDEN_FIELDS.intersection({k.lower() for k in row})
                if present:
                    failures.append(f"forbidden field(s) {sorted(present)} in {path}:{lineno}")

    # 4. .gitignore still covers the content directories.
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8") if (REPO_ROOT / ".gitignore").exists() else ""
    for directory in (
        "data/raw/", "data/interim/", "data/processed/", "data/review/", "results/", "logs/",
        "data/v2/interim/", "data/v2/cache/", "data/v2/tmp/",
        "data/v2/raw_downloads/", "data/v2/credentials/",
    ):
        if directory not in gitignore:
            failures.append(f".gitignore no longer covers {directory}")

    print("\nRESULT")
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        print(f"\n{len(failures)} hygiene check(s) failed.", file=sys.stderr)
        return 1
    print("  all hygiene checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
