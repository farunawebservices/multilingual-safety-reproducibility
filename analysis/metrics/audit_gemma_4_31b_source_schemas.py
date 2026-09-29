from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path("paper_artifacts/metrics/gemma_4_31b_source_schema_audit.csv")

SOURCES = [
    Path(
        "data/raw/phase1/gemma_4_31b/originals/"
        "gemma_4_31b_baseline_results (1).jsonl"
    ),
    Path(
        "data/raw/phase1/gemma_4_31b/originals/"
        "gemma_4_31b_steered_results (1).jsonl"
    ),
]

MAX_RECORDS_TO_INSPECT = 25
MAX_DEPTH = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def type_name(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def flatten_types(
    value: object,
    prefix: str = "",
    depth: int = 0,
) -> list[tuple[str, str]]:
    name = prefix or "<root>"
    pairs = [(name, type_name(value))]

    if depth >= MAX_DEPTH:
        return pairs

    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            pairs.extend(flatten_types(child, child_prefix, depth + 1))

    elif isinstance(value, list) and value:
        pairs.extend(flatten_types(value[0], f"{prefix}[]", depth + 1))

    return pairs


rows = []

for source in SOURCES:
    if not source.is_file():
        raise FileNotFoundError(f"Missing source: {source}")

    total_nonempty_lines = 0
    valid_json_records = 0
    invalid_json_lines = 0
    top_level_types = Counter()
    field_present = Counter()
    field_types = defaultdict(Counter)
    field_null_n = Counter()

    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            total_nonempty_lines += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_json_lines += 1
                continue

            valid_json_records += 1
            top_level_types[type_name(record)] += 1

            if valid_json_records > MAX_RECORDS_TO_INSPECT:
                continue

            for field, dtype in flatten_types(record):
                field_present[field] += 1
                field_types[field][dtype] += 1
                if dtype == "null":
                    field_null_n[field] += 1

    source_hash = sha256(source)

    for field in sorted(field_present):
        rows.append({
            "source_path": source.as_posix(),
            "source_sha256": source_hash,
            "source_bytes": source.stat().st_size,
            "total_nonempty_lines": total_nonempty_lines,
            "valid_json_records": valid_json_records,
            "invalid_json_lines": invalid_json_lines,
            "sample_records_structurally_inspected": min(
                valid_json_records,
                MAX_RECORDS_TO_INSPECT,
            ),
            "top_level_json_types": ";".join(
                f"{name}:{count}"
                for name, count in sorted(top_level_types.items())
            ),
            "field_path": field,
            "sample_records_with_field": field_present[field],
            "sample_null_n": field_null_n[field],
            "observed_types": ";".join(
                f"{name}:{count}"
                for name, count in sorted(field_types[field].items())
            ),
        })

OUT.parent.mkdir(parents=True, exist_ok=True)

with OUT.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote: {OUT}")
print(f"Sources audited: {len(SOURCES)}")

for source in SOURCES:
    source_rows = [row for row in rows if row["source_path"] == source.as_posix()]
    first = source_rows[0]
    print(
        f"{source.name}: lines={first['total_nonempty_lines']}, "
        f"valid_json={first['valid_json_records']}, "
        f"invalid_json={first['invalid_json_lines']}, "
        f"schema_fields={len(source_rows)}"
    )
