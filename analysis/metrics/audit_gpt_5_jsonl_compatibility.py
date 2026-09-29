from collections import Counter, defaultdict
from pathlib import Path
import csv
import json

import polars as pl

RECONCILIATION = Path(
    "paper_artifacts/metrics/gpt_5_source_reconciliation.csv"
)

OUT = Path(
    "paper_artifacts/metrics/gpt_5_jsonl_compatibility_audit.csv"
)

TEXT_FIELDS = {
    "prompt", "response", "completion", "content", "message",
    "messages", "output", "text", "reasoning", "analysis",
}

reconciliation = pl.read_csv(RECONCILIATION)

paths = (
    reconciliation
    .filter(
        (pl.col("location") == "raw")
        & (pl.col("file_type") == "jsonl")
    )
    .select("path")
    .to_series()
    .to_list()
)

if len(paths) != 2:
    raise ValueError(f"Expected 2 raw GPT-5 JSONL files; found {len(paths)}.")

def profile_jsonl(path: Path):
    record_n = 0
    parse_error_n = 0
    field_present_n = Counter()
    field_types = defaultdict(Counter)
    scalar_values = defaultdict(set)

    with path.open("r", encoding="utf-8", errors="replace") as handle:
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
                    f"{path} contains a non-object JSON record."
                )

            record_n += 1

            for key, value in record.items():
                key = str(key)
                field_present_n[key] += 1
                field_types[key][type(value).__name__] += 1

                if (
                    key.lower() not in TEXT_FIELDS
                    and isinstance(value, (str, int, float, bool))
                    and len(scalar_values[key]) <= 100
                ):
                    scalar_values[key].add(str(value))

    return {
        "record_n": record_n,
        "parse_error_n": parse_error_n,
        "field_present_n": field_present_n,
        "field_types": field_types,
        "scalar_values": scalar_values,
    }

profiles = {Path(path): profile_jsonl(Path(path)) for path in paths}
all_fields = sorted(
    set().union(*[
        set(profile["field_present_n"])
        for profile in profiles.values()
    ])
)

rows = []
for field in all_fields:
    row = {"field": field}

    for index, path in enumerate(sorted(profiles), start=1):
        profile = profiles[path]
        prefix = f"file_{index}"

        row[f"{prefix}_path"] = str(path)
        row[f"{prefix}_present_n"] = profile["field_present_n"][field]
        row[f"{prefix}_types"] = "|".join(
            sorted(profile["field_types"][field])
        )

        values = sorted(profile["scalar_values"][field])
        row[f"{prefix}_distinct_scalar_n_capped"] = len(values)
        row[f"{prefix}_scalar_values_if_le_100"] = (
            "|".join(values)
            if len(values) <= 100
            else "SUPPRESSED_OVER_100"
        )

    rows.append(row)

OUT.parent.mkdir(parents=True, exist_ok=True)

with OUT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(rows)

ordered_paths = sorted(profiles)

print(f"Wrote: {OUT}")
for index, path in enumerate(ordered_paths, start=1):
    profile = profiles[path]
    print(
        f"File {index}: {path} | "
        f"valid records={profile['record_n']} | "
        f"parse errors={profile['parse_error_n']} | "
        f"fields={len(profile['field_present_n'])}"
    )

print(f"Union fields: {len(all_fields)}")
print(
    "Identical field sets: "
    f"{set(profiles[ordered_paths[0]]['field_present_n']) == set(profiles[ordered_paths[1]]['field_present_n'])}"
)

with pl.Config(tbl_cols=-1, tbl_width_chars=280):
    print(pl.DataFrame(rows))
