from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import math

import polars as pl

INVENTORY = Path(
    "paper_artifacts/metrics/closed_model_source_inventory.csv"
)

OUT = Path(
    "paper_artifacts/metrics/"
    "claude_sonnet_4_phase1_schema_profile.csv"
)

TEXT_FIELDS = {
    "prompt", "response", "completion", "content", "message",
    "messages", "output", "text", "reasoning", "analysis",
}

inventory = pl.read_csv(INVENTORY)

candidates = (
    inventory
    .filter(
        (pl.col("model") == "claude_sonnet_4")
        & (pl.col("file_type") == "jsonl")
        & (pl.col("location") == "raw")
    )
    .select("path")
    .to_series()
    .to_list()
)

if len(candidates) != 1:
    raise ValueError(
        "Expected exactly one raw Claude Sonnet 4 JSONL file; "
        f"found {len(candidates)}."
    )

source = Path(candidates[0])

stats = defaultdict(lambda: {
    "present_n": 0,
    "null_n": 0,
    "types": Counter(),
    "bool_true_n": 0,
    "bool_false_n": 0,
    "numeric_n": 0,
    "numeric_min": None,
    "numeric_max": None,
    "distinct_scalar_values": set(),
})

record_n = 0
parse_error_n = 0

with source.open("r", encoding="utf-8", errors="replace") as handle:
    for line in handle:
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            parse_error_n += 1
            continue

        if not isinstance(record, dict):
            raise TypeError(
                f"Expected object records; received {type(record).__name__}."
            )

        record_n += 1

        for key, value in record.items():
            key = str(key)
            item = stats[key]
            item["present_n"] += 1

            if value is None:
                item["null_n"] += 1
                item["types"]["null"] += 1
                continue

            type_name = type(value).__name__
            item["types"][type_name] += 1

            if isinstance(value, bool):
                item["bool_true_n"] += int(value)
                item["bool_false_n"] += int(not value)

            elif (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                item["numeric_n"] += 1
                item["numeric_min"] = (
                    value
                    if item["numeric_min"] is None
                    else min(item["numeric_min"], value)
                )
                item["numeric_max"] = (
                    value
                    if item["numeric_max"] is None
                    else max(item["numeric_max"], value)
                )

            if (
                key.lower() not in TEXT_FIELDS
                and isinstance(value, (str, int, float, bool))
                and len(item["distinct_scalar_values"]) <= 50
            ):
                item["distinct_scalar_values"].add(str(value))

rows = []
for key in sorted(stats):
    item = stats[key]
    distinct_values = sorted(item["distinct_scalar_values"])

    rows.append({
        "field": key,
        "present_n": item["present_n"],
        "missing_n": record_n - item["present_n"],
        "null_n": item["null_n"],
        "observed_types": "|".join(sorted(item["types"])),
        "bool_true_n": item["bool_true_n"],
        "bool_false_n": item["bool_false_n"],
        "numeric_n": item["numeric_n"],
        "numeric_min": item["numeric_min"],
        "numeric_max": item["numeric_max"],
        "distinct_scalar_value_n_capped": len(distinct_values),
        "distinct_scalar_values_if_le_50": (
            "|".join(distinct_values)
            if len(distinct_values) <= 50
            else "SUPPRESSED_OVER_50"
        ),
    })

OUT.parent.mkdir(parents=True, exist_ok=True)

with OUT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(rows)

result = pl.DataFrame(rows)

print(f"Wrote: {OUT}")
print(f"Source: {source}")
print(f"Valid object records: {record_n}")
print(f"JSON parse errors: {parse_error_n}")
print(f"Top-level fields: {result.height}")

with pl.Config(tbl_cols=-1, tbl_width_chars=240):
    print(
        result.select([
            "field",
            "present_n",
            "missing_n",
            "null_n",
            "observed_types",
            "bool_true_n",
            "bool_false_n",
            "numeric_n",
            "numeric_min",
            "numeric_max",
            "distinct_scalar_value_n_capped",
        ])
    )
