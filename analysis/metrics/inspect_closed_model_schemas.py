from pathlib import Path
import csv
import json

import polars as pl

INVENTORY = Path(
    "paper_artifacts/metrics/closed_model_source_inventory.csv"
)

OUT = Path(
    "paper_artifacts/metrics/closed_model_schema_inventory.csv"
)

TEXT_KEYS = {
    "prompt", "response", "completion", "content", "message",
    "messages", "output", "text", "reasoning", "analysis",
}

ID_HINTS = {"id", "record_id", "prompt_id", "trial_id", "run_id"}
OUTCOME_HINTS = {
    "is_vetoed", "veto", "decision", "allow", "operational_failure",
    "success", "failure", "status",
}
CONFIG_HINTS = {
    "alpha", "config", "condition", "intervention",
    "steered", "baseline", "language", "prompt_class",
}

def flatten_keys(value, prefix=""):
    keys = set()

    if isinstance(value, dict):
        for key, child in value.items():
            key = str(key)
            full = f"{prefix}.{key}" if prefix else key
            keys.add(full)
            keys.update(flatten_keys(child, full))

    elif isinstance(value, list) and value:
        keys.update(flatten_keys(value[0], f"{prefix}[]"))

    return keys

def classify_keys(keys, hints):
    lower_keys = {key.lower() for key in keys}
    return any(
        hint in key or key.endswith(f".{hint}")
        for hint in hints
        for key in lower_keys
    )

def inspect_jsonl(path):
    record_count = 0
    parse_error_count = 0
    union_keys = set()

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                parse_error_count += 1
                continue

            record_count += 1
            union_keys.update(flatten_keys(record))

    return record_count, parse_error_count, union_keys

def inspect_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return 0, 1, set()

    if isinstance(value, list):
        union_keys = set()
        for record in value[:100]:
            union_keys.update(flatten_keys(record))
        return len(value), 0, union_keys

    return 1, 0, flatten_keys(value)

def inspect_tabular(path):
    if path.suffix.lower() == ".parquet":
        frame = pl.read_parquet_schema(path)
        keys = set(frame.names())
        return None, 0, keys

    frame = pl.read_csv_schema(path)
    keys = set(frame.names())
    return None, 0, keys

inventory = pl.read_csv(INVENTORY).filter(
    pl.col("path").is_not_null()
)

rows = []
for item in inventory.iter_rows(named=True):
    path = Path(item["path"])
    suffix = path.suffix.lower()

    try:
        if suffix == ".jsonl":
            record_n, parse_error_n, keys = inspect_jsonl(path)
        elif suffix == ".json":
            record_n, parse_error_n, keys = inspect_json(path)
        elif suffix in {".csv", ".parquet"}:
            record_n, parse_error_n, keys = inspect_tabular(path)
        else:
            record_n, parse_error_n, keys = None, 0, set()

        safe_keys = sorted(
            key for key in keys
            if key.lower().split(".")[-1].replace("[]", "") not in TEXT_KEYS
        )

        rows.append({
            "model": item["model"],
            "path": str(path),
            "file_type": suffix.lstrip("."),
            "record_n_if_json": record_n,
            "parse_error_n_if_json": parse_error_n,
            "key_n": len(safe_keys),
            "likely_id_field_present": classify_keys(safe_keys, ID_HINTS),
            "likely_outcome_field_present": classify_keys(
                safe_keys, OUTCOME_HINTS
            ),
            "likely_config_field_present": classify_keys(
                safe_keys, CONFIG_HINTS
            ),
            "safe_schema_keys": "|".join(safe_keys),
        })

    except Exception as exc:
        rows.append({
            "model": item["model"],
            "path": str(path),
            "file_type": suffix.lstrip("."),
            "record_n_if_json": None,
            "parse_error_n_if_json": None,
            "key_n": None,
            "likely_id_field_present": False,
            "likely_outcome_field_present": False,
            "likely_config_field_present": False,
            "safe_schema_keys": f"INSPECTION_ERROR:{type(exc).__name__}",
        })

OUT.parent.mkdir(parents=True, exist_ok=True)

with OUT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "model",
            "path",
            "file_type",
            "record_n_if_json",
            "parse_error_n_if_json",
            "key_n",
            "likely_id_field_present",
            "likely_outcome_field_present",
            "likely_config_field_present",
            "safe_schema_keys",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

result = pl.DataFrame(rows).sort(["model", "path"])

print(f"Wrote: {OUT}")
print(f"Files inspected: {result.height}")
print(
    "Files with likely ID fields: "
    f"{result.filter(pl.col('likely_id_field_present')).height}"
)
print(
    "Files with likely outcome fields: "
    f"{result.filter(pl.col('likely_outcome_field_present')).height}"
)
print(
    "Files with likely configuration fields: "
    f"{result.filter(pl.col('likely_config_field_present')).height}"
)

with pl.Config(tbl_cols=9, tbl_width_chars=240):
    print(
        result.select([
            "model",
            "file_type",
            "record_n_if_json",
            "parse_error_n_if_json",
            "key_n",
            "likely_id_field_present",
            "likely_outcome_field_present",
            "likely_config_field_present",
            "path",
        ])
    )
