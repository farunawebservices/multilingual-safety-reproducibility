#!/usr/bin/env python3
"""Stage 2.2 -- archive the CPU-generated pilot artifacts and record their hashes.

The pilot's 120 successful translations were produced on **CPU**. The full
corpus is being generated on **CUDA**, and beam search is not bit-identical
across devices, so those CPU rows are being superseded by CUDA regenerations
rather than mixed in.

Superseding is not deleting. This script copies the CPU artifacts to
``data/interim/archive/cpu_pilot/`` and records their SHA-256 hashes and run
metadata, so the original provenance stays verifiable after the CUDA corpus
becomes canonical.

Copies, never moves: the originals stay exactly where they are, unchanged. The
archive is verified byte-for-byte against them.

Outputs
-------
    data/interim/archive/cpu_pilot/*            byte-identical copies (git-ignored)
    results/cpu_pilot_archive_report.json       metadata-only (tracked)

Usage
-----
    python scripts/12_archive_cpu_pilot.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multilingual_safety.hashing import deterministic_digest, file_sha256
from multilingual_safety.paths import INTERIM_DIR, REPO_ROOT, RESULTS_DIR
from multilingual_safety.providers import STATUS_OK, ResponseLog
from multilingual_safety.run_context import stage_report_header

ARCHIVE_DIR = INTERIM_DIR / "archive" / "cpu_pilot"
REPORT_PATH = RESULTS_DIR / "cpu_pilot_archive_report.json"

SOURCES = (
    INTERIM_DIR / "translation_pilot_responses.jsonl",
    INTERIM_DIR / "translation_pilot_outputs.jsonl",
    INTERIM_DIR / "translation_pilot_requests.jsonl",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    missing = [p for p in SOURCES if not p.exists()]
    if missing:
        print(f"missing source(s): {missing}", file=sys.stderr)
        return 1

    header = stage_report_header("stage2.2_archive_cpu_pilot", {"archive_dir": str(ARCHIVE_DIR)})
    print(f"run_id:     {header['run_id']}")
    print(f"git commit: {header['git_commit']}  (dirty={header['git_dirty']})")

    if not args.dry_run:
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    files: list[dict[str, Any]] = []
    failures: list[str] = []
    for source in SOURCES:
        target = ARCHIVE_DIR / source.name
        source_hash = file_sha256(source)
        if not args.dry_run:
            # copy2 preserves mtime, so the archive keeps the original timestamps.
            shutil.copy2(source, target)
        archived_hash = file_sha256(target) if target.exists() else None
        identical = archived_hash == source_hash
        if not identical and not args.dry_run:
            failures.append(f"{source.name}: archive hash differs from source")
        files.append({
            "name": source.name,
            "source_path": str(source.relative_to(REPO_ROOT)),
            "archive_path": str(target.relative_to(REPO_ROOT)),
            "sha256": source_hash,
            "archived_sha256": archived_hash,
            "bytes": source.stat().st_size,
            "identical": identical,
        })
        print(f"  {source.name:<40} {source_hash[:16]}  "
              f"{'IDENTICAL' if identical else 'MISMATCH'}")

    # --- provenance of the archived responses -----------------------------
    log = ResponseLog(SOURCES[0])
    records = [r for rows in log.history().values() for r in rows]
    effective = log.effective()
    devices = Counter(r.get("device") for r in records if r.get("status") == STATUS_OK)
    runs = Counter(r.get("run_id") for r in records)

    provenance = {
        "n_records_total": len(records),
        "n_effective": len(effective),
        "n_effective_ok": sum(1 for r in effective.values() if r.get("status") == STATUS_OK),
        "records_by_status": dict(sorted(Counter(r.get("status") for r in records).items())),
        "successful_records_by_device": dict(sorted(devices.items())),
        "records_by_run_id": dict(sorted(runs.items())),
        "model_ids": sorted({r.get("model_id") for r in records if r.get("model_id")}),
        "model_revisions": sorted({r.get("model_revision") for r in records if r.get("model_revision")}),
        "dtypes": sorted({r.get("dtype") for r in records if r.get("dtype")}),
    }
    print(f"\n  archived response records : {provenance['n_records_total']}")
    print(f"  effective OK              : {provenance['n_effective_ok']}")
    print(f"  successful by device      : {provenance['successful_records_by_device']}")
    print(f"  run_ids                   : {list(provenance['records_by_run_id'])}")

    report = {
        **header,
        "purpose": (
            "Preserve the CPU-generated pilot artifacts before the CUDA corpus supersedes "
            "them. Superseding is not deletion: these hashes keep the original provenance "
            "verifiable."
        ),
        "archive_dir": str(ARCHIVE_DIR.relative_to(REPO_ROOT)),
        "archive_dir_git_ignored": True,
        "files": files,
        "provenance": provenance,
        "originals_modified": False,
        "contains_prompt_text": False,
        "superseded_by": "CUDA regeneration written to data/interim/full/",
        "notes": [
            "Copies, not moves: the originals remain in data/interim/ unchanged.",
            "Beam search is not bit-identical across CPU and CUDA, so these rows are "
            "not reproducible under the CUDA configuration and are retained as a record.",
        ],
    }
    report["deterministic_digest"] = deterministic_digest(report)

    if failures:
        for failure in failures:
            print(f"  FAIL: {failure}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\nDry run: nothing written.")
        return 0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {REPORT_PATH.relative_to(REPO_ROOT)}  (tracked, metadata-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
