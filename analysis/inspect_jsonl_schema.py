from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

RAW = Path("data/raw/phase1")
OUT = Path("registry/schema_reports")
OUT.mkdir(parents=True, exist_ok=True)

def type_name(value: Any) -> str:
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

def safe_scalar(value: Any) -> str:
    if value is None or isinstance(value, (bool, int, float)):
        return str(value)
    if isinstance(value, str):
        return value[:100]
    return f"<{type_name(value)}>"

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

reports = []
for path in sorted(RAW.rglob("*.jsonl")):
    top_keys = Counter()
    key_types = defaultdict(Counter)
    scalar_values = defaultdict(Counter)
    nested_keys = defaultdict(Counter)
    total = parsed = invalid = blank = 0

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            total += 1
            if not line.strip():
                blank += 1
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            if not isinstance(row, dict):
                invalid += 1
                continue

            parsed += 1
            for key, value in row.items():
                top_keys[key] += 1
                key_types[key][type_name(value)] += 1
                if isinstance(value, dict):
                    for subkey in value:
                        nested_keys[key][subkey] += 1
                elif isinstance(value, (str, int, float, bool)) or value is None:
                    scalar_values[key][safe_scalar(value)] += 1

    report = {
        "path": str(path),
        "sha256": file_hash(path),
        "total_lines": total,
        "parsed_object_rows": parsed,
        "invalid_or_nonobject_rows": invalid,
        "blank_rows": blank,
        "top_level_keys": dict(top_keys),
        "value_types_by_key": {k: dict(v) for k, v in key_types.items()},
        "nested_keys_by_parent": {k: dict(v) for k, v in nested_keys.items()},
        "scalar_value_samples_and_counts": {
            k: dict(v.most_common(25)) for k, v in scalar_values.items()
        },
    }
    out_path = OUT / f"{path.stem.replace(' ', '_')}.schema.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    reports.append({
        "path": str(path),
        "sha256": report["sha256"],
        "total_lines": total,
        "parsed_rows": parsed,
        "invalid_rows": invalid,
        "top_keys": ",".join(sorted(top_keys)),
    })

(OUT / "schema_inventory.json").write_text(
    json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8"
)

print("path\tlines\tparsed\tinvalid\tsha256")
for r in reports:
    print(f"{r['path']}\t{r['total_lines']}\t{r['parsed_rows']}\t{r['invalid_rows']}\t{r['sha256']}")
