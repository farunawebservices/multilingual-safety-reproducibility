#!/usr/bin/env python3
"""Stage 2.0 -- build the full primary request set from JailbreakBench only.

All 100 harmful + all 100 benign JBB behaviors, rendered into the seven target
languages: 200 x 7 = 1,400 rows, of which 1,200 are non-English translation
requests. **Nothing is translated here.**

Identity is preserved, not re-derived. Behavior IDs come from the same
provenance-based scheme used since Stage 0 -- ``behavior_id(source_id,
source_item_id, arm)`` -- so the 20 reviewed pilot behaviors keep the exact IDs
they already have and appear in this set unchanged. The script asserts that
overlap rather than assuming it.

HarmBench is not used: this is the primary set, and the two sources overlap in
provenance (Stage 0 found 27 JBB harmful behaviors sourced from TDC/HarmBench),
so merging them is a separate decision.

Writes to **new paths under git-ignored ``data/interim/full/``**. The reviewed
pilot artifacts are read-only inputs here and are never overwritten; the script
refuses to start if it would collide with them.

Outputs
-------
    data/interim/full/full_selection.jsonl          200 behaviors (git-ignored)
    data/interim/full/full_translation_requests.jsonl  1,200 requests (git-ignored)
    data/interim/full/full_translation_outputs.jsonl   1,400 rows (git-ignored)
    data/interim/full/full_manifest.json            text-free manifest (git-ignored)
    results/full_request_set_inventory.json         metadata-only (tracked)

Splits are NOT assigned: every row carries the ``UNASSIGNED`` placeholder.

Usage
-----
    python scripts/10_build_full_request_set.py --dry-run
    python scripts/10_build_full_request_set.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from multilingual_safety.hashing import deterministic_digest, file_sha256, text_sha256
from multilingual_safety.ids import behavior_id
from multilingual_safety.languages import LANGUAGE_CODES, PIVOT_LANGUAGE, load_languages
from multilingual_safety.paths import (
    CONFIGS_DIR,
    INTERIM_DIR,
    LANGUAGES_CONFIG,
    RAW_DIR,
    REPO_ROOT,
    RESULTS_DIR,
)
from multilingual_safety.providers import TranslationRequest
from multilingual_safety.run_context import stage_report_header
from multilingual_safety.translation import (
    SPLIT_UNASSIGNED,
    TranslationRow,
    make_english_row,
    make_pending_row,
    validate_row_group,
)

JBB_DIR = RAW_DIR / "jailbreakbench"
JBB_FILES = {"harmful": "harmful-behaviors.csv", "benign": "benign-behaviors.csv"}
SOURCE_ID = "jailbreakbench"
TEXT_COLUMN = "Goal"          # the instruction; `Target` is an attack artifact

FULL_DIR = INTERIM_DIR / "full"
SELECTION_PATH = FULL_DIR / "full_selection.jsonl"
REQUESTS_PATH = FULL_DIR / "full_translation_requests.jsonl"
OUTPUTS_PATH = FULL_DIR / "full_translation_outputs.jsonl"
MANIFEST_PATH = FULL_DIR / "full_manifest.json"
INVENTORY_PATH = RESULTS_DIR / "full_request_set_inventory.json"

#: Reviewed pilot artifacts. Read-only inputs; never written by this script.
PILOT_PATHS = (
    INTERIM_DIR / "translation_pilot_outputs.jsonl",
    INTERIM_DIR / "translation_pilot_responses.jsonl",
    INTERIM_DIR / "translation_pilot_requests.jsonl",
)

NON_PIVOT_LANGUAGES = tuple(c for c in LANGUAGE_CODES if c != PIVOT_LANGUAGE)
EXPECTED_BEHAVIORS = 200
EXPECTED_ROWS = EXPECTED_BEHAVIORS * len(LANGUAGE_CODES)          # 1400
EXPECTED_REQUESTS = EXPECTED_BEHAVIORS * len(NON_PIVOT_LANGUAGES)  # 1200


def load_arm(arm: str) -> list[dict[str, str]]:
    path = JBB_DIR / JBB_FILES[arm]
    if not path.exists():
        raise SystemExit(f"missing {path}. Run scripts/01_download_sources.py first (Stage 0).")
    with path.open(encoding="utf-8", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def select_all() -> list[dict[str, Any]]:
    """Every JBB behavior, both arms. Identity derived exactly as in Stage 1."""
    selected: list[dict[str, Any]] = []
    for arm in sorted(JBB_FILES):
        for row in load_arm(arm):
            bid = behavior_id(source_id=SOURCE_ID, source_item_id=str(row["Index"]), arm=arm)
            text = row[TEXT_COLUMN]
            selected.append({
                "behavior_id": bid,
                "source": SOURCE_ID,
                "source_item_id": str(row["Index"]),
                "category": row["Category"],
                "harm_label": arm,
                "source_text": text,
                "source_text_sha256": text_sha256(text),
                "upstream_source_label": row.get("Source", ""),
            })
    # Deterministic global ordering.
    selected.sort(key=lambda r: (r["harm_label"], r["category"], r["behavior_id"]))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing full set.")
    args = parser.parse_args()

    header = stage_report_header(
        "stage2.0_build_full_request_set",
        {"source": SOURCE_ID, "languages": list(LANGUAGE_CODES), "text_column": TEXT_COLUMN},
    )
    print(f"run_id:     {header['run_id']}")
    print(f"git commit: {header['git_commit']}  (dirty={header['git_dirty']})")
    print(f"source:     {SOURCE_ID} only (HarmBench excluded from the primary set)")

    # --- guard the reviewed pilot ----------------------------------------
    collisions = [p for p in PILOT_PATHS if p in (SELECTION_PATH, REQUESTS_PATH, OUTPUTS_PATH)]
    if collisions:
        print(f"STOP: would overwrite reviewed pilot artifacts: {collisions}", file=sys.stderr)
        return 1
    pilot_hashes = {p.name: file_sha256(p) for p in PILOT_PATHS if p.exists()}
    print(f"pilot artifacts guarded (read-only): {sorted(pilot_hashes)}")

    if not args.dry_run and not args.force:
        existing = [p for p in (SELECTION_PATH, REQUESTS_PATH, OUTPUTS_PATH) if p.exists()]
        if existing:
            print(f"STOP: {len(existing)} full-set file(s) already exist; pass --force to "
                  f"regenerate: {[str(p.relative_to(REPO_ROOT)) for p in existing]}",
                  file=sys.stderr)
            return 1

    load_languages(LANGUAGES_CONFIG)
    yaml.safe_load((CONFIGS_DIR / "translation.yaml").read_text(encoding="utf-8"))

    selection = select_all()
    rows: list[TranslationRow] = []
    requests: list[TranslationRequest] = []

    for item in selection:
        common = {k: item[k] for k in
                  ("behavior_id", "source", "category", "harm_label", "source_text",
                   "source_item_id")}
        rows.append(make_english_row(**common))
        for language in NON_PIVOT_LANGUAGES:
            rows.append(make_pending_row(language=language, **common))
            requests.append(TranslationRequest.create(
                behavior_id=item["behavior_id"],
                target_language=language,
                source_text=item["source_text"],
            ))

    rows.sort(key=lambda r: (r.harm_label, r.category, r.behavior_id, r.language))
    requests.sort(key=lambda r: r.request_id)

    # --- integrity checks -------------------------------------------------
    by_behavior: dict[str, list[TranslationRow]] = defaultdict(list)
    for row in rows:
        by_behavior[row.behavior_id].append(row)

    group_violations = {
        bid: v for bid, group in sorted(by_behavior.items()) if (v := validate_row_group(group))
    }
    inconsistent: dict[str, dict[str, list[str]]] = {}
    for bid, group in sorted(by_behavior.items()):
        problems = {
            field: sorted({str(getattr(r, field)) for r in group})
            for field in ("source", "source_item_id", "category", "harm_label", "source_text")
            if len({str(getattr(r, field)) for r in group}) > 1
        }
        if problems:
            inconsistent[bid] = {k: v for k, v in problems.items() if k != "source_text"} or {
                "source_text": ["<differs; text withheld>"]
            }

    request_ids = [r.request_id for r in requests]
    duplicate_requests = sorted({i for i, n in Counter(request_ids).items() if n > 1})
    duplicate_behaviors = sorted(
        {i for i, n in Counter(s["behavior_id"] for s in selection).items() if n > 1}
    )
    missing_language = {
        bid: sorted(set(LANGUAGE_CODES) - {r.language for r in group})
        for bid, group in sorted(by_behavior.items())
        if {r.language for r in group} != set(LANGUAGE_CODES)
    }

    # --- pilot alignment --------------------------------------------------
    pilot_alignment: dict[str, Any] = {"checked": False}
    pilot_outputs = INTERIM_DIR / "translation_pilot_outputs.jsonl"
    if pilot_outputs.exists():
        pilot_ids, pilot_source_hashes = set(), {}
        for line in pilot_outputs.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                pilot_ids.add(row["behavior_id"])
                if row["language"] == PIVOT_LANGUAGE:
                    pilot_source_hashes[row["behavior_id"]] = text_sha256(row["source_text"])
        full_hashes = {s["behavior_id"]: s["source_text_sha256"] for s in selection}
        drifted = sorted(
            b for b, h in pilot_source_hashes.items()
            if b in full_hashes and full_hashes[b] != h
        )
        pilot_alignment = {
            "checked": True,
            "n_pilot_behaviors": len(pilot_ids),
            "n_present_in_full_set": len(pilot_ids & set(full_hashes)),
            "missing_from_full_set": sorted(pilot_ids - set(full_hashes)),
            "source_hash_drift": drifted,
        }

    # --- counts -----------------------------------------------------------
    per_language = Counter(r.language for r in rows)
    per_arm = Counter(r.harm_label for r in rows)
    per_category = Counter(r.category for r in rows)
    behaviors_per_arm = Counter(s["harm_label"] for s in selection)

    print("\nCOUNTS")
    print(f"  behaviors           : {len(selection)}  {dict(sorted(behaviors_per_arm.items()))}")
    print(f"  rows                : {len(rows)}  (expected {EXPECTED_ROWS})")
    print(f"  translation requests: {len(requests)}  (expected {EXPECTED_REQUESTS})")
    print(f"  rows per language   : {dict(sorted(per_language.items()))}")
    print(f"  rows per arm        : {dict(sorted(per_arm.items()))}")
    print(f"  distinct categories : {len(per_category)}")
    print(f"  split placeholder   : {SPLIT_UNASSIGNED} on all rows")

    print("\nINTEGRITY")
    print(f"  duplicate behavior_ids       : {len(duplicate_behaviors)}")
    print(f"  duplicate request_ids        : {len(duplicate_requests)}")
    print(f"  behaviors missing a language : {len(missing_language)}")
    print(f"  inconsistent metadata groups : {len(inconsistent)}")
    print(f"  group invariant violations   : {len(group_violations)}")
    if pilot_alignment["checked"]:
        print(f"  pilot behaviors in full set  : "
              f"{pilot_alignment['n_present_in_full_set']}/{pilot_alignment['n_pilot_behaviors']}")
        print(f"  pilot source-hash drift      : {len(pilot_alignment['source_hash_drift'])}")

    failures: list[str] = []
    if len(selection) != EXPECTED_BEHAVIORS:
        failures.append(f"expected {EXPECTED_BEHAVIORS} behaviors, got {len(selection)}")
    if behaviors_per_arm.get("harmful") != 100 or behaviors_per_arm.get("benign") != 100:
        failures.append(f"expected 100/100 per arm, got {dict(behaviors_per_arm)}")
    if len(rows) != EXPECTED_ROWS:
        failures.append(f"expected {EXPECTED_ROWS} rows, got {len(rows)}")
    if len(requests) != EXPECTED_REQUESTS:
        failures.append(f"expected {EXPECTED_REQUESTS} requests, got {len(requests)}")
    if any(n != EXPECTED_BEHAVIORS for n in per_language.values()):
        failures.append(f"uneven rows per language: {dict(per_language)}")
    if duplicate_behaviors:
        failures.append(f"{len(duplicate_behaviors)} duplicate behavior_id(s)")
    if duplicate_requests:
        failures.append(f"{len(duplicate_requests)} duplicate request_id(s)")
    if missing_language:
        failures.append(f"{len(missing_language)} behavior(s) missing a language")
    if inconsistent:
        failures.append(f"{len(inconsistent)} behavior(s) with inconsistent metadata")
    if group_violations:
        failures.append(f"{len(group_violations)} group invariant violation(s)")
    if pilot_alignment.get("missing_from_full_set"):
        failures.append(
            f"{len(pilot_alignment['missing_from_full_set'])} reviewed pilot behavior(s) "
            "absent from the full set: the identity scheme has drifted"
        )
    if pilot_alignment.get("source_hash_drift"):
        failures.append(
            f"{len(pilot_alignment['source_hash_drift'])} pilot behavior(s) have a different "
            "source hash in the full set"
        )
    if any(r.translated_text is not None for r in rows if r.language != PIVOT_LANGUAGE):
        failures.append("a non-English row carries text; nothing may be translated at this stage")

    for failure in failures:
        print(f"  FAIL: {failure}", file=sys.stderr)
    if failures:
        return 1

    if args.dry_run:
        print("\nDry run: nothing written.")
        return 0

    # --- write (all git-ignored except the metadata-only inventory) -------
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    with SELECTION_PATH.open("w", encoding="utf-8") as fh:
        for item in selection:
            fh.write(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n")
    with REQUESTS_PATH.open("w", encoding="utf-8") as fh:
        for request in requests:
            payload = {
                "request_id": request.request_id,
                "behavior_id": request.behavior_id,
                "source_language": request.source_language,
                "target_language": request.target_language,
                "source_text": request.source_text,
                "source_text_sha256": request.source_text_sha256,
            }
            fh.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
    with OUTPUTS_PATH.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": 1,
        "stage": header["stage"],
        "run_id": header["run_id"],
        "git_commit": header["git_commit"],
        "generated_at": header["generated_at"],
        "source": SOURCE_ID,
        "harmbench_used": False,
        "languages": list(LANGUAGE_CODES),
        "pivot": PIVOT_LANGUAGE,
        "n_behaviors": len(selection),
        "n_rows": len(rows),
        "n_requests": len(requests),
        "split_placeholder": SPLIT_UNASSIGNED,
        "splits_assigned": False,
        "translated": False,
        "contains_prompt_text": False,
        "rows": [row.to_manifest_dict() for row in rows],
    }
    manifest["manifest_digest"] = deterministic_digest(manifest)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    inventory = {
        **header,
        "source": SOURCE_ID,
        "harmbench_used": False,
        "n_behaviors": len(selection),
        "n_behaviors_per_arm": dict(sorted(behaviors_per_arm.items())),
        "n_rows": len(rows),
        "n_requests": len(requests),
        "counts_by_language": dict(sorted(per_language.items())),
        "counts_by_arm": dict(sorted(per_arm.items())),
        "counts_by_category": dict(sorted(per_category.items())),
        "counts_by_language_and_arm": {
            f"{lang}:{arm}": sum(
                1 for r in rows if r.language == lang and r.harm_label == arm
            )
            for lang in LANGUAGE_CODES for arm in ("harmful", "benign")
        },
        "counts_by_status": dict(sorted(Counter(r.translation_status for r in rows).items())),
        "n_review_required": sum(1 for r in rows if r.review_required),
        "n_translated": sum(1 for r in rows if r.translated_text is not None),
        "n_pending": sum(1 for r in rows if r.translated_text is None),
        "split_placeholder": SPLIT_UNASSIGNED,
        "splits_assigned": False,
        "translated": False,
        "human_reviewed_rows": 0,
        "pilot_alignment": pilot_alignment,
        "pilot_artifact_hashes_unchanged": pilot_hashes,
        "integrity": {
            "duplicate_behavior_ids": len(duplicate_behaviors),
            "duplicate_request_ids": len(duplicate_requests),
            "behaviors_missing_a_language": len(missing_language),
            "inconsistent_metadata_groups": len(inconsistent),
            "group_invariant_violations": len(group_violations),
        },
        "source_hashes": {
            "harmful_csv": file_sha256(JBB_DIR / JBB_FILES["harmful"]),
            "benign_csv": file_sha256(JBB_DIR / JBB_FILES["benign"]),
            "selection_jsonl": file_sha256(SELECTION_PATH),
            "requests_jsonl": file_sha256(REQUESTS_PATH),
            "outputs_jsonl": file_sha256(OUTPUTS_PATH),
            "manifest_json": file_sha256(MANIFEST_PATH),
        },
        "output_paths": {
            "selection": str(SELECTION_PATH.relative_to(REPO_ROOT)),
            "requests": str(REQUESTS_PATH.relative_to(REPO_ROOT)),
            "outputs": str(OUTPUTS_PATH.relative_to(REPO_ROOT)),
            "manifest": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        },
        "contains_prompt_text": False,
        "notes": [
            "Nothing has been translated: all 1,200 non-English rows are PENDING_TRANSLATION.",
            "Splits are NOT assigned; every row carries the UNASSIGNED placeholder.",
            "The reviewed pilot artifacts are read-only inputs and were not overwritten.",
            "Behavior IDs are unchanged from Stage 1; the 20 reviewed pilot behaviors are a "
            "subset of this set with identical IDs and source hashes.",
        ],
    }
    inventory["deterministic_digest"] = deterministic_digest(inventory)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    INVENTORY_PATH.write_text(
        json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    for path in (SELECTION_PATH, REQUESTS_PATH, OUTPUTS_PATH, MANIFEST_PATH):
        print(f"  wrote {path.relative_to(REPO_ROOT)}  (git-ignored)")
    print(f"  wrote {INVENTORY_PATH.relative_to(REPO_ROOT)}  (tracked, metadata-only)")
    print("\nSTOP: nothing translated. 1,200 requests are staged and awaiting approval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
