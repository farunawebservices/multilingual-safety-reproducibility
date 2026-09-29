#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

REGISTRY = Path("registry/source_artifacts.csv")
OUT = Path("paper_artifacts/metrics/schema_audit.csv")

def inspect_jsonl(path: Path):
    keys = Counter()
    rows = 0
    valid = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            valid += 1
            if isinstance(record, dict):
                keys.update(record.keys())
    return rows, valid, ";".join(sorted(keys))

results = []
with REGISTRY.open(encoding="utf-8") as handle:
    for record in csv.DictReader(handle):
        path = Path(record["path"])
        suffix = record["suffix"]
        row = {
            **record,
            "rows_seen": "",
            "rows_valid_json": "",
            "top_level_keys": "",
            "status": "not_jsonl",
        }
        if suffix == ".jsonl":
            try:
                seen, valid, keys = inspect_jsonl(path)
                row.update(
                    rows_seen=seen,
                    rows_valid_json=valid,
                    top_level_keys=keys,
                    status="audited",
                )
            except Exception as exc:
                row["status"] = f"error:{type(exc).__name__}"
        results.append(row)

with OUT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"Wrote {OUT}")
print(f"JSONL files audited: {sum(x['status'] == 'audited' for x in results)}")
