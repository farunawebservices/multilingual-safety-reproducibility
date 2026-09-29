from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

OUT = Path(
    "paper_artifacts/metrics/"
    "gemma_4_31b_missing_outcome_impact_audit.csv"
)

SOURCES = [
    (
        "baseline",
        Path(
            "data/raw/phase1/gemma_4_31b/originals/"
            "gemma_4_31b_baseline_results (1).jsonl"
        ),
    ),
    (
        "steered",
        Path(
            "data/raw/phase1/gemma_4_31b/originals/"
            "gemma_4_31b_steered_results (1).jsonl"
        ),
    ),
]

FIELDS = ["strength", "config", "language", "harmfulness"]
SCALAR = r'("(?:\\.|[^"\\])*"|true|false|null|-?\d+(?:\.\d+)?)'


def extract_scalar(text: str, field: str):
    match = re.search(
        rf'"{re.escape(field)}"\s*:\s*{SCALAR}',
        text,
    )
    return json.loads(match.group(1)) if match else None


rows = []

for source_group, source in SOURCES:
    valid_records = []
    truncated_text = None

    with source.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                valid_records.append(json.loads(line))
            except json.JSONDecodeError:
                truncated_text = line.rstrip("\n")

    if truncated_text is None:
        raise ValueError(f"No truncated row found in {source}")

    missing = {
        field: extract_scalar(truncated_text, field)
        for field in ["id", *FIELDS]
    }

    for field in FIELDS:
        counts = Counter(
            str(record.get(field, "<MISSING>"))
            for record in valid_records
        )
        missing_value = str(missing[field])
        rows.append({
            "source_group": source_group,
            "source_path": source.as_posix(),
            "valid_record_n": len(valid_records),
            "missing_outcome_record_n": 1,
            "field": field,
            "truncated_record_value": missing_value,
            "valid_records_with_same_value_n": counts[missing_value],
            "valid_records_with_same_value_rate_percentage": (
                100 * counts[missing_value] / len(valid_records)
            ),
            "truncated_record_id": missing["id"],
            "outcome_available": False,
            "analysis_policy": (
                "Do not impute outcome; exclude only in a documented "
                "complete-case sensitivity analysis unless clean exports "
                "are recovered."
            ),
        })

OUT.parent.mkdir(parents=True, exist_ok=True)

with OUT.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote: {OUT}")
for row in rows:
    print(
        f"{row['source_group']:8} "
        f"{row['field']:12} "
        f"truncated_value={row['truncated_record_value']!r:20} "
        f"valid_same_value_n={row['valid_records_with_same_value_n']}/"
        f"{row['valid_record_n']}"
    )
