#!/usr/bin/env python3
"""Stage 2.3 -- apply the effective successful responses to the full output manifest.

Reads ``data/interim/full/full_translation_responses.jsonl``, takes the **latest
successful** response per ``request_id`` (``ResponseLog.effective()``), and folds
it into ``data/interim/full/full_translation_outputs.jsonl``.

Correlation is on ``request_id`` **and** ``source_text_sha256``. The request ID
alone proves the response is filed under the right key; the source hash proves
it was produced from the text this row actually holds. A response whose source
hash disagrees is not applied -- it was generated from something else.

Invariants, all enforced rather than trusted
--------------------------------------------
* English ``CANONICAL_SOURCE`` rows are passed through byte-identically.
* Applied rows become ``MACHINE_TRANSLATED_UNCHECKED`` -- never ``HUMAN_REVIEWED``.
* ``review_required`` stays ``True`` on every non-English row; no machine path
  clears it.
* Splits stay ``UNASSIGNED``.
* Any missing, duplicated, orphaned, or hash-mismatched request aborts the run
  before a single byte is written.

Provenance beyond what a row can hold (``response_sha256``, model ID and
revision, device, dtype, generation config, run ID) is written to a **local
sidecar**, ``full_translation_provenance.jsonl``, keyed by request ID. The row
schema deliberately has no room for it, and a tracked manifest must not carry it.

Pilot artifacts are never opened for writing.

Outputs
-------
    data/interim/full/full_translation_outputs.jsonl      updated (git-ignored)
    data/interim/full/full_translation_provenance.jsonl   sidecar (git-ignored)
    results/full_apply_report.json                        metadata-only (tracked)

Usage
-----
    python scripts/13_apply_full_responses.py --dry-run
    python scripts/13_apply_full_responses.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multilingual_safety.hashing import deterministic_digest, file_sha256
from multilingual_safety.languages import LANGUAGE_CODES, PIVOT_LANGUAGE
from multilingual_safety.paths import (
    INTERIM_DIR,
    RESULTS_DIR,
    PathSafetyError,
    assert_output_path_is_safe,
    display_path,
)
from multilingual_safety.providers import STATUS_OK, ResponseLog, make_request_id
from multilingual_safety.run_context import stage_report_header
from multilingual_safety.translation import (
    SPLIT_UNASSIGNED,
    TranslationMethod,
    TranslationRow,
    TranslationStatus,
    compute_prompt_hash,
)

FULL_DIR = INTERIM_DIR / "full"
OUTPUTS_PATH = FULL_DIR / "full_translation_outputs.jsonl"
RESPONSES_PATH = FULL_DIR / "full_translation_responses.jsonl"
PROVENANCE_PATH = FULL_DIR / "full_translation_provenance.jsonl"
REPORT_PATH = RESULTS_DIR / "full_apply_report.json"

#: Never opened for writing by this script.
PILOT_PATHS = tuple(INTERIM_DIR.glob("translation_pilot_*.jsonl"))

NON_PIVOT_LANGUAGES = tuple(c for c in LANGUAGE_CODES if c != PIVOT_LANGUAGE)

#: Provenance fields copied verbatim into the sidecar.
PROVENANCE_FIELDS = (
    "request_id", "behavior_id", "source_language", "target_language",
    "provider", "model_id", "model_revision", "tokenizer_id",
    "device", "dtype", "generation_config",
    "response_sha256", "source_text_sha256",
    "timestamp", "latency_seconds", "status", "run_id",
)


def load_rows(path: Path) -> list[TranslationRow]:
    rows: list[TranslationRow] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(TranslationRow.from_dict(json.loads(line)))
        except Exception as exc:
            raise SystemExit(f"{path.name}:{lineno}: cannot parse row: {exc}") from exc
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--outputs", type=Path, default=OUTPUTS_PATH)
    parser.add_argument("--responses", type=Path, default=RESPONSES_PATH)
    # Parameterised for the same reason as stage 2.1: a dataset-v2 apply must
    # write under data/v2/interim/ and must not overwrite the tracked v1 report
    # at results/full_apply_report.json. Defaults are the v1 paths, unchanged.
    parser.add_argument("--provenance", type=Path, default=PROVENANCE_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for path in (args.outputs, args.responses):
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 1
    if args.outputs.resolve() in {p.resolve() for p in PILOT_PATHS}:
        print("refusing to write a pilot artifact", file=sys.stderr)
        return 1

    # Same gate as stage 2.1, so the two stages cannot drift on what a safe
    # output path is. Runs before anything is opened for writing.
    read_only = (args.responses,)
    try:
        assert_output_path_is_safe(args.outputs, inputs=read_only, what="applied rows")
        assert_output_path_is_safe(args.provenance, inputs=read_only, what="provenance sidecar")
        assert_output_path_is_safe(
            args.report, kind="metadata", inputs=read_only, what="report"
        )
    except PathSafetyError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("PATHS")
    print(f"  responses  : {display_path(args.responses)}  (READ-ONLY)")
    print(f"  outputs    : {display_path(args.outputs)}")
    print(f"  provenance : {display_path(args.provenance)}")
    print(f"  report     : {display_path(args.report)}")

    header = stage_report_header("stage2.3_apply_full_responses", {"outputs": str(args.outputs)})
    print(f"run_id:     {header['run_id']}")
    print(f"git commit: {header['git_commit']}  (dirty={header['git_dirty']})")

    pilot_before = {p.name: file_sha256(p) for p in PILOT_PATHS}

    rows = load_rows(args.outputs)
    log = ResponseLog(args.responses)
    effective = log.effective()
    successful = {rid: r for rid, r in effective.items() if r.get("status") == STATUS_OK}

    print(f"\nrows in            : {len(rows)}")
    print(f"effective responses: {len(effective)}  (successful: {len(successful)})")

    # --- correlate --------------------------------------------------------
    expected_ids = {
        make_request_id(r.behavior_id, r.language)
        for r in rows if r.language != PIVOT_LANGUAGE
    }
    missing = sorted(expected_ids - set(successful))
    orphaned = sorted(set(successful) - expected_ids)
    duplicates = sorted(
        rid for rid, n in Counter(
            make_request_id(r.behavior_id, r.language)
            for r in rows if r.language != PIVOT_LANGUAGE
        ).items() if n > 1
    )

    applied: list[TranslationRow] = []
    provenance: list[dict[str, Any]] = []
    hash_mismatch: list[str] = []
    empty_text: list[str] = []
    counts = Counter()

    for row in rows:
        if row.language == PIVOT_LANGUAGE:
            applied.append(row)                      # untouched, byte-identical
            counts["canonical_source_passthrough"] += 1
            continue

        request_id = make_request_id(row.behavior_id, row.language)
        response = successful.get(request_id)
        if response is None:
            applied.append(row)                      # stays PENDING; reported below
            counts["left_pending"] += 1
            continue

        # Correlation requires BOTH keys to agree.
        row_source_hash = compute_prompt_hash(row.source_text)
        if response.get("source_text_sha256") != row_source_hash:
            hash_mismatch.append(request_id)
            applied.append(row)
            continue

        text = response.get("response_text") or ""
        if not text.strip():
            empty_text.append(request_id)
            applied.append(row)
            continue

        prompt_hash = compute_prompt_hash(text)
        if prompt_hash != response.get("response_sha256"):
            hash_mismatch.append(f"{request_id} (response_sha256)")
            applied.append(row)
            continue

        applied.append(TranslationRow(
            behavior_id=row.behavior_id,
            source=row.source,
            category=row.category,
            harm_label=row.harm_label,
            source_text=row.source_text,
            language=row.language,
            translated_text=text,
            translation_method=TranslationMethod.MACHINE_UNREVIEWED.value,
            translation_model=response.get("model_id"),
            translation_timestamp=response.get("timestamp"),
            translation_status=TranslationStatus.MACHINE_TRANSLATED_UNCHECKED.value,
            review_required=True,        # never cleared by a machine
            prompt_hash=prompt_hash,
            source_item_id=row.source_item_id,
            reviewer_id=None,            # never set here
            issues=row.issues,
        ))
        provenance.append({k: response.get(k) for k in PROVENANCE_FIELDS})
        counts["applied"] += 1

    applied.sort(key=lambda r: (r.harm_label, r.category, r.behavior_id, r.language))
    provenance.sort(key=lambda p: p["request_id"])

    # --- post-conditions --------------------------------------------------
    by_status = Counter(r.translation_status for r in applied)
    non_english = [r for r in applied if r.language != PIVOT_LANGUAGE]
    english = [r for r in applied if r.language == PIVOT_LANGUAGE]

    print("\nAPPLICATION")
    print(f"  applied                    : {counts['applied']}")
    print(f"  canonical source untouched : {counts['canonical_source_passthrough']}")
    print(f"  left pending               : {counts['left_pending']}")
    print(f"  missing responses          : {len(missing)}")
    print(f"  orphaned responses         : {len(orphaned)}")
    print(f"  duplicate request keys     : {len(duplicates)}")
    print(f"  hash mismatches            : {len(hash_mismatch)}")
    print(f"  empty response text        : {len(empty_text)}")

    failures: list[str] = []
    if missing:
        failures.append(f"{len(missing)} request(s) have no successful response: {missing[:5]}")
    if orphaned:
        failures.append(f"{len(orphaned)} orphaned response(s): {orphaned[:5]}")
    if duplicates:
        failures.append(f"{len(duplicates)} duplicate request key(s): {duplicates[:5]}")
    if hash_mismatch:
        failures.append(f"{len(hash_mismatch)} hash mismatch(es): {hash_mismatch[:5]}")
    if empty_text:
        failures.append(f"{len(empty_text)} empty response(s): {empty_text[:5]}")
    if counts["left_pending"]:
        failures.append(f"{counts['left_pending']} row(s) left pending")
    if len(applied) != len(rows):
        failures.append(f"row count changed: {len(rows)} -> {len(applied)}")
    if len(english) != 200:
        failures.append(f"expected 200 canonical English rows, got {len(english)}")
    if any(not r.review_required for r in non_english):
        failures.append("a non-English row has review_required=False")
    if any(r.translation_status == TranslationStatus.HUMAN_REVIEWED.value for r in applied):
        failures.append("a row is HUMAN_REVIEWED")
    if any(r.reviewer_id for r in applied):
        failures.append("a row has a reviewer_id")
    if {r.split_placeholder for r in applied} != {SPLIT_UNASSIGNED}:
        failures.append("split placeholder changed")

    # English rows must be byte-identical to what we read.
    original_english = {
        r.behavior_id: r.to_dict() for r in rows if r.language == PIVOT_LANGUAGE
    }
    changed_english = sorted(
        r.behavior_id for r in english if original_english[r.behavior_id] != r.to_dict()
    )
    if changed_english:
        failures.append(f"{len(changed_english)} English row(s) were modified")

    for failure in failures:
        print(f"  FAIL: {failure}", file=sys.stderr)
    if failures:
        print("\nNothing written.", file=sys.stderr)
        return 1

    print("\nRESULT COUNTS")
    print(f"  by status   : {dict(sorted(by_status.items()))}")
    print(f"  by language : {dict(sorted(Counter(r.language for r in applied).items()))}")
    print(f"  by arm      : {dict(sorted(Counter(r.harm_label for r in applied).items()))}")
    print(f"  translated  : {sum(1 for r in applied if r.translated_text is not None)}")
    print(f"  review_required: {sum(1 for r in applied if r.review_required)} / {len(applied)}")

    if args.dry_run:
        print("\nDry run: nothing written.")
        return 0

    with args.outputs.open("w", encoding="utf-8") as fh:
        for row in applied:
            fh.write(json.dumps(row.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    with args.provenance.open("w", encoding="utf-8") as fh:
        for record in provenance:
            fh.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

    # --- pilot artifacts must be untouched --------------------------------
    pilot_after = {p.name: file_sha256(p) for p in PILOT_PATHS}
    drifted = sorted(k for k in pilot_before if pilot_before[k] != pilot_after.get(k))
    if drifted:
        print(f"  FAIL: pilot artifact(s) changed: {drifted}", file=sys.stderr)
        return 1

    report = {
        **header,
        "n_rows": len(applied),
        "n_applied": counts["applied"],
        "n_canonical_source_untouched": counts["canonical_source_passthrough"],
        "n_left_pending": counts["left_pending"],
        "counts_by_status": dict(sorted(by_status.items())),
        "counts_by_language": dict(sorted(Counter(r.language for r in applied).items())),
        "counts_by_arm": dict(sorted(Counter(r.harm_label for r in applied).items())),
        "counts_by_language_and_status": {
            f"{lang}:{status}": sum(
                1 for r in applied if r.language == lang and r.translation_status == status
            )
            for lang in LANGUAGE_CODES
            for status in sorted(by_status)
        },
        "n_translated": sum(1 for r in applied if r.translated_text is not None),
        "n_review_required": sum(1 for r in applied if r.review_required),
        "n_human_reviewed": 0,
        "split_placeholder": SPLIT_UNASSIGNED,
        "splits_assigned": False,
        "correlation": {
            "keys": ["request_id", "source_text_sha256"],
            "missing": len(missing),
            "orphaned": len(orphaned),
            "duplicates": len(duplicates),
            "hash_mismatches": len(hash_mismatch),
            "empty_responses": len(empty_text),
            "fully_correlated": True,
        },
        "provenance": {
            "sidecar": display_path(args.provenance),
            "n_records": len(provenance),
            "fields": list(PROVENANCE_FIELDS),
            "providers": dict(sorted(Counter(p["provider"] for p in provenance).items())),
            "devices": dict(sorted(Counter(p["device"] for p in provenance).items())),
            "dtypes": dict(sorted(Counter(p["dtype"] for p in provenance).items())),
            "model_ids": sorted({p["model_id"] for p in provenance}),
            "model_revisions": sorted({p["model_revision"] for p in provenance}),
            "run_ids": dict(sorted(Counter(p["run_id"] for p in provenance).items())),
            "generation_configs": [
                json.loads(g) for g in sorted(
                    {json.dumps(p["generation_config"], sort_keys=True) for p in provenance}
                )
            ],
        },
        "pilot_artifacts_unchanged": pilot_before == pilot_after,
        "pilot_artifact_hashes": dict(sorted(pilot_after.items())),
        "source_hashes": {
            "outputs_jsonl": file_sha256(args.outputs),
            "responses_jsonl": file_sha256(args.responses),
            "provenance_jsonl": file_sha256(args.provenance),
        },
        "contains_prompt_text": False,
        "notes": [
            "Correlated on request_id AND source_text_sha256; a response generated from "
            "different source text is not applied.",
            "English rows are passed through byte-identically.",
            "review_required stays true on every non-English row; no row is HUMAN_REVIEWED.",
            "Device/dtype/generation provenance lives in the git-ignored sidecar, not in "
            "the row schema and not in any tracked report.",
        ],
    }
    report["deterministic_digest"] = deterministic_digest(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"\nwrote {display_path(args.outputs)}  (git-ignored)")
    print(f"wrote {display_path(args.provenance)}  ({len(provenance)} records)")
    print(f"wrote {display_path(args.report)}  (metadata-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
