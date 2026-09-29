from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

OUT = Path(
    "registry/field_reports/"
    "gemma_4_31b_truncated_prefix_recovery_audit.json"
)

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

SCALAR_FIELDS = [
    "strength",
    "config",
    "id",
    "language",
    "harmfulness",
    "is_vetoed",
]

reports = []

for source in SOURCES:
    last_nonempty = None
    last_number = None

    with source.open("r", encoding="utf-8", errors="replace") as handle:
        for number, line in enumerate(handle, start=1):
            if line.strip():
                last_nonempty = line.rstrip("\n")
                last_number = number

    if last_nonempty is None:
        raise ValueError(f"No nonempty lines found: {source}")

    extracted = {}
    field_status = {}

    for field in SCALAR_FIELDS:
        pattern = re.compile(
            rf'"{re.escape(field)}"\s*:\s*'
            r'(?P<value>"(?:\\.|[^"\\])*"|true|false|null|-?\d+(?:\.\d+)?)'
        )
        match = pattern.search(last_nonempty)

        if not match:
            field_status[field] = "not_found_before_truncation"
            continue

        raw_value = match.group("value")

        try:
            extracted[field] = json.loads(raw_value)
            field_status[field] = "parsed"
        except json.JSONDecodeError:
            field_status[field] = "found_but_unparseable"

    reports.append({
        "source_path": source.as_posix(),
        "truncated_line_number": last_number,
        "truncated_line_sha256": hashlib.sha256(
            last_nonempty.encode("utf-8")
        ).hexdigest(),
        "truncated_line_bytes": len(last_nonempty.encode("utf-8")),
        "scalar_field_status": field_status,
        "recovered_scalar_fields": extracted,
        "all_required_analysis_fields_recovered": all(
            field_status[field] == "parsed" for field in SCALAR_FIELDS
        ),
        "recovery_policy": (
            "No source is modified. This report does not create a repaired "
            "JSONL or infer omitted fields. A truncated row is usable only "
            "if every required analysis scalar is present before truncation."
        ),
    })

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"Wrote: {OUT}")

for report in reports:
    print(f"\n--- {Path(report['source_path']).name} ---")
    print(f"Truncated line: {report['truncated_line_number']}")
    print(
        "All required analysis fields recovered: "
        f"{report['all_required_analysis_fields_recovered']}"
    )
    for field, status in report["scalar_field_status"].items():
        print(f"{field}: {status}")
