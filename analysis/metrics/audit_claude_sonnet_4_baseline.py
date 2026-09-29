from pathlib import Path
import json

import polars as pl

INVENTORY = Path(
    "paper_artifacts/metrics/closed_model_source_inventory.csv"
)

OUT = Path(
    "paper_artifacts/metrics/"
    "claude_sonnet_4_phase1_baseline_audit.csv"
)

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
        "Expected exactly one Claude Sonnet 4 raw JSONL; "
        f"found {len(candidates)}."
    )

source = Path(candidates[0])

frame = pl.read_ndjson(source)

required = ["id", "language", "harmfulness", "model", "is_vetoed"]
missing = sorted(set(required) - set(frame.columns))
if missing:
    raise ValueError(f"Missing required fields: {missing}")

if frame.height != 1695:
    raise ValueError(f"Expected 1695 records; found {frame.height}")

if frame.select(pl.col("id").is_null().any()).item():
    raise ValueError("Found null IDs.")

if frame.select(pl.col("language").is_null().any()).item():
    raise ValueError("Found null languages.")

if frame.select(pl.col("harmfulness").is_null().any()).item():
    raise ValueError("Found null harmfulness labels.")

if frame.select(pl.col("is_vetoed").is_null().any()).item():
    raise ValueError("Found null veto outcomes.")

result = (
    frame
    .group_by(["language", "harmfulness"])
    .agg([
        pl.len().alias("record_n"),
        pl.col("id").n_unique().alias("unique_id_n"),
        pl.col("id").is_duplicated().sum().alias("duplicate_id_row_n"),
        pl.col("is_vetoed").sum().alias("veto_n"),
        (~pl.col("is_vetoed")).sum().alias("non_veto_n"),
    ])
    .with_columns([
        (
            pl.col("record_n") - pl.col("unique_id_n")
        ).alias("duplicate_id_excess_n"),
        (
            100.0 * pl.col("veto_n")
            / pl.col("record_n")
        ).alias("veto_rate_percentage"),
        (
            100.0 * pl.col("non_veto_n")
            / pl.col("record_n")
        ).alias("non_veto_failure_rate_percentage"),
    ])
    .sort(["harmfulness", "language"])
)

if not (result["record_n"] == result["unique_id_n"]).all():
    raise ValueError("IDs are not unique within every language/harmfulness cell.")

if not (
    result["veto_n"] + result["non_veto_n"] == result["record_n"]
).all():
    raise ValueError("Decision counts do not sum to cell totals.")

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

print(f"Wrote: {OUT}")
print(f"Source records: {frame.height}")
print(f"Language/harmfulness cells: {result.height}")
print(f"Total unique IDs: {frame['id'].n_unique()}")
print(
    "All cell IDs unique: "
    f"{(result['record_n'] == result['unique_id_n']).all()}"
)

with pl.Config(tbl_cols=-1, tbl_width_chars=220):
    print(result)
