#!/usr/bin/env python3
"""Stage 0.2 -- inspect the acquired sources and write a machine-readable inventory.

Reports, per source and per file: column names and dtypes, row counts, missing
and empty-string counts, duplicate IDs, duplicate behavior *text* (by hash),
category distributions, and source labels. Also checks the cross-file structural
claims that later stages will depend on -- HarmBench's val/test partition, and
the JBB/HarmBench provenance overlap.

**The inventory contains no prompt text.** Only column names, counts, category
labels, source labels, and hashes. That is what makes it safe to commit
alongside the code while the corpus itself stays git-ignored
(THIRD_PARTY_DATASETS.md).

Missing values are *reported, never imputed or dropped* (CLAUDE.md §2). Some
emptiness is expected by design and is labelled as such rather than flagged as a
defect: HarmBench's ``ContextString`` is empty for all non-contextual rows.

Writes ``results/source_inventory.json``.

Usage
-----
    python scripts/02_inspect_sources.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from multilingual_safety.duplicates import duplicate_summary
from multilingual_safety.hashing import deterministic_digest, file_sha256, table_digest
from multilingual_safety.paths import MANIFESTS_DIR, REPO_ROOT, RESULTS_DIR
from multilingual_safety.run_context import stage_report_header
from multilingual_safety.schemas import EXPECTED_SCHEMAS, get_part, validate_part
from multilingual_safety.secrets import scan_file

SOURCE_MANIFEST = MANIFESTS_DIR / "source_manifest.yaml"

#: Columns whose emptiness is expected by upstream design, not a data defect.
EXPECTED_EMPTY = {
    ("harmbench", "ContextString"): "empty for all non-contextual rows (standard/copyright)",
    ("harmbench", "Tags"): "empty for standard rows; populated for contextual/copyright",
}


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        columns = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return columns, rows


def infer_dtype(values: list[str]) -> str:
    """Coarse dtype for a CSV column, deterministic and dependency-free."""
    non_empty = [v for v in values if v != ""]
    if not non_empty:
        return "empty"
    if all(v.lstrip("-").isdigit() for v in non_empty):
        return "integer"
    try:
        for v in non_empty:
            float(v)
        return "float"
    except ValueError:
        return "string"


def profile_column(name: str, values: list[str], source_id: str) -> dict[str, Any]:
    n_missing = sum(1 for v in values if v is None)
    n_empty = sum(1 for v in values if v is not None and v.strip() == "")
    distinct = sorted({v for v in values})
    profile: dict[str, Any] = {
        "name": name,
        "dtype": infer_dtype(values),
        "n_missing": n_missing,
        "n_empty_string": n_empty,
        "n_distinct": len(distinct),
        "min_char_length": min((len(v) for v in values if v), default=0),
        "max_char_length": max((len(v) for v in values if v), default=0),
    }
    if (source_id, name) in EXPECTED_EMPTY:
        profile["emptiness_expected"] = True
        profile["emptiness_note"] = EXPECTED_EMPTY[(source_id, name)]
    return profile


def profile_file(source_id: str, part_id: str, path: Path) -> dict[str, Any]:
    columns, rows = read_csv_rows(path)
    schema = get_part(source_id, part_id)
    by_column = {col: [r.get(col, "") for r in rows] for col in columns}

    # Three-level duplicate detection (see multilingual_safety.duplicates):
    # IDs and composite duplicates are defects; a repeated primary text column
    # is not, when another text column distinguishes the rows.
    dups = duplicate_summary(rows, id_column=schema.id_column, text_columns=schema.text_columns)

    # Categorical distributions: labels are metadata, not prompt content.
    categories = {
        col: dict(sorted(Counter(by_column.get(col, [])).items()))
        for col in schema.categorical_columns
        if col in by_column
    }

    violations = validate_part(source_id, part_id, columns=columns, n_rows=len(rows))

    # Missing values split into upstream-by-design vs genuinely unexpected.
    # Lumping them together makes 500 expected empties look like 500 defects.
    column_profiles = [profile_column(c, by_column[c], source_id) for c in columns]
    expected_missing = sum(
        p["n_missing"] + p["n_empty_string"] for p in column_profiles if p.get("emptiness_expected")
    )
    total_missing = sum(p["n_missing"] + p["n_empty_string"] for p in column_profiles)
    missing_summary = {
        "n_total": total_missing,
        "n_expected_by_design": expected_missing,
        "n_unexpected": total_missing - expected_missing,
        "by_column": {
            p["name"]: p["n_missing"] + p["n_empty_string"]
            for p in column_profiles
            if p["n_missing"] + p["n_empty_string"]
        },
    }

    return {
        "part_id": part_id,
        "local_path": str(path.relative_to(REPO_ROOT)),
        "sha256": file_sha256(path),
        "table_digest": table_digest(rows, columns=columns),
        "n_rows": len(rows),
        "n_columns": len(columns),
        "columns": columns,
        "column_profiles": column_profiles,
        "id_column": schema.id_column,
        "duplicates": dups,
        "n_duplicate_ids": dups["ids"]["n_duplicate_keys"],
        "n_duplicate_behavior_text": dups["primary_text"]["n_duplicate_keys"],
        "n_rows_sharing_duplicate_text": dups["primary_text"]["n_rows_affected"],
        "n_duplicate_composite_key": dups["composite"]["n_duplicate_keys"],
        "duplicates_are_defective": dups["is_defective"],
        "missing_values": missing_summary,
        "n_total_missing": missing_summary["n_total"],
        "n_unexpected_missing": missing_summary["n_unexpected"],
        "categories": categories,
        "arm": schema.arm,
        "schema_violations": violations,
        "schema_ok": not violations,
        "notes": schema.notes,
    }


def cross_file_checks(profiles: dict[str, dict[str, dict[str, Any]]], raw_rows: dict) -> dict[str, Any]:
    """Structural claims later stages will rely on. Verified, not assumed."""
    checks: dict[str, Any] = {}

    # 1. HarmBench: does val + test reconstruct text_all, disjointly?
    hb = raw_rows.get("harmbench", {})
    if {"text_all", "text_val", "text_test"} <= set(hb):
        ids = {p: {r["BehaviorID"] for r in hb[p]} for p in ("text_all", "text_val", "text_test")}
        checks["harmbench_partition"] = {
            "n_all": len(ids["text_all"]),
            "n_val": len(ids["text_val"]),
            "n_test": len(ids["text_test"]),
            "val_test_overlap": len(ids["text_val"] & ids["text_test"]),
            "val_union_test_equals_all": (ids["text_val"] | ids["text_test"]) == ids["text_all"],
            "partition_mechanism": "separate FILES; there is no split column in any HarmBench CSV",
            "official_val_file": "harmbench_behaviors_text_val.csv",
            "official_test_file": "harmbench_behaviors_text_test.csv",
        }

        # 2. Which HarmBench rows are self-contained text behaviors?
        func = Counter(r["FunctionalCategory"] for r in hb["text_all"])
        n_ctx = sum(1 for r in hb["text_all"] if r["ContextString"].strip())
        checks["harmbench_text_only"] = {
            "functional_categories": dict(sorted(func.items())),
            "n_rows_with_context_string": n_ctx,
            "n_self_contained_text_behaviors": func.get("standard", 0),
            "interpretation": (
                "harmbench_behaviors_text_all.csv is the TEXT (non-multimodal) set, but it is "
                "not uniformly self-contained: 'contextual' rows require ContextString, and "
                "'copyright' rows are hash-checked generation requests. Only 'standard' rows "
                "are stand-alone text behaviors."
            ),
        }

    # 3. JBB provenance -- the reason the two sources must not be merged blindly.
    jbb = raw_rows.get("jailbreakbench", {})
    if "harmful" in jbb:
        src = Counter(r["Source"] for r in jbb["harmful"])
        overlap = sum(n for label, n in src.items() if "harmbench" in label.lower() or "tdc" in label.lower())
        checks["jbb_harmbench_provenance_overlap"] = {
            "jbb_harmful_source_labels": dict(sorted(src.items())),
            "n_jbb_harmful_derived_from_tdc_or_harmbench": overlap,
            "risk": (
                "Concatenating JBB and HarmBench would double-count these behaviors and "
                "inflate any rate computed over the union. The sources are kept separate "
                "at Stage 0; any later merge must deduplicate on behavior semantics first."
            ),
        }

    # 4. JBB Index uniqueness is per-split, not global.
    if {"harmful", "benign"} <= set(jbb):
        h_idx = {r["Index"] for r in jbb["harmful"]}
        b_idx = {r["Index"] for r in jbb["benign"]}
        checks["jbb_index_scope"] = {
            "harmful_index_overlaps_benign_index": len(h_idx & b_idx),
            "warning": (
                "JBB 'Index' restarts at 0 in each split, so it is unique only WITHIN a "
                "split. It must never be used as a corpus-wide key; behavior IDs are "
                "derived from (source_id, source_item_id, arm) instead."
            ),
        }
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "source_inventory.json")
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"no source manifest at {args.manifest}; run scripts/01_download_sources.py first",
              file=sys.stderr)
        return 1

    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    report = stage_report_header("stage0.2_inspect_sources", {"manifest": str(args.manifest)})
    report["source_manifest_run_id"] = manifest.get("run_id")

    profiles: dict[str, dict[str, dict[str, Any]]] = {}
    raw_rows: dict[str, dict[str, list[dict[str, str]]]] = {}
    sources_out: list[dict[str, Any]] = []
    failures: list[str] = []

    for source in sorted(manifest.get("sources", []), key=lambda s: s["source_id"]):
        sid = source["source_id"]
        profiles[sid], raw_rows[sid] = {}, {}
        files_out: list[dict[str, Any]] = []

        for info in sorted(source.get("files", []), key=lambda f: f["part_id"]):
            path = REPO_ROOT / info["local_path"]
            if not path.exists():
                failures.append(f"{sid}/{info['part_id']}: local file missing ({info['local_path']})")
                continue
            if file_sha256(path) != info["sha256"]:
                failures.append(f"{sid}/{info['part_id']}: sha256 differs from the manifest")

            profile = profile_file(sid, info["part_id"], path)
            _, rows = read_csv_rows(path)
            raw_rows[sid][info["part_id"]] = rows
            profiles[sid][info["part_id"]] = profile
            files_out.append(profile)

            if not profile["schema_ok"]:
                failures.append(f"{sid}/{info['part_id']}: {profile['schema_violations']}")
            if profile["n_duplicate_ids"]:
                failures.append(f"{sid}/{info['part_id']}: {profile['n_duplicate_ids']} duplicate IDs")
            if profile["n_duplicate_composite_key"]:
                failures.append(
                    f"{sid}/{info['part_id']}: {profile['n_duplicate_composite_key']} "
                    "genuinely duplicated items (identical across ALL text columns)"
                )
            if profile["n_unexpected_missing"]:
                failures.append(
                    f"{sid}/{info['part_id']}: {profile['n_unexpected_missing']} unexpected "
                    f"missing values in {profile['missing_values']['by_column']}"
                )

        expected = EXPECTED_SCHEMAS.get(sid)
        sources_out.append({
            "source_id": sid,
            "name": source.get("name"),
            "url": source.get("url"),
            "revision": source.get("revision"),
            "revision_type": source.get("revision_type"),
            "license_id": source.get("license_id"),
            "license_url": source.get("license_url"),
            "retrieved_at": source.get("retrieved_at"),
            "acquisition_method": source.get("acquisition_method"),
            "local_dir": source.get("local_dir"),
            "n_files": len(files_out),
            "n_rows_total": sum(f["n_rows"] for f in files_out),
            "expected_schema_notes": expected.notes if expected else None,
            "files": files_out,
        })

    report["sources"] = sources_out
    report["cross_file_checks"] = cross_file_checks(profiles, raw_rows)
    report["merge_policy"] = {
        "sources_merged": False,
        "reason": "Stage 0 keeps JBB and HarmBench strictly separate; they overlap in "
                  "provenance and merging them now would double-count behaviors.",
    }
    report["contains_prompt_text"] = False
    report["failures"] = sorted(failures)
    report["ok"] = not failures
    report["deterministic_digest"] = deterministic_digest(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Self-check: the report we just wrote must carry no credentials. Cheap, and
    # it runs at the moment the file becomes a commit candidate.
    leaked = scan_file(args.out)
    if leaked:
        report["failures"] = sorted([*report["failures"], *(str(f) for f in leaked)])
        failures.extend(str(f) for f in leaked)

    # --- console summary (CLAUDE.md §9) -----------------------------------
    print(f"run_id:     {report['run_id']}")
    print(f"git commit: {report['git_commit']}  (dirty={report['git_dirty']})")
    for source in sources_out:
        print(f"\n=== {source['source_id']}  ({source['license_id']}, rev {str(source['revision'])[:12]}) ===")
        for f in source["files"]:
            print(f"  {f['part_id']:10} rows={f['n_rows']:>4} cols={f['n_columns']} "
                  f"dup_ids={f['n_duplicate_ids']} dup_text={f['n_duplicate_behavior_text']} "
                  f"dup_item={f['n_duplicate_composite_key']} "
                  f"missing={f['n_total_missing']}(expected {f['missing_values']['n_expected_by_design']}, "
                  f"unexpected {f['n_unexpected_missing']}) "
                  f"schema={'OK' if f['schema_ok'] else 'DRIFT'}")
    print("\ncross-file checks:")
    for name, payload in sorted(report["cross_file_checks"].items()):
        print(f"  {name}: {json.dumps({k: v for k, v in payload.items() if not isinstance(v, str) or len(v) < 60}, sort_keys=True)}")
    print(f"\nwrote {args.out}")
    print(f"deterministic_digest: {report['deterministic_digest'][:16]}...")
    for failure in report["failures"]:
        print(f"  FAIL: {failure}", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
