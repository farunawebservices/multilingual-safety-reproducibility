from pathlib import Path

import polars as pl

RECONCILIATION = Path(
    "paper_artifacts/metrics/gpt_5_source_reconciliation.csv"
)

OUT = Path(
    "paper_artifacts/metrics/"
    "gpt_5_clean_frontier_baseline_audit.csv"
)

reconciliation = pl.read_csv(RECONCILIATION)

candidates = (
    reconciliation
    .filter(
        (pl.col("location") == "raw")
        & (pl.col("file_type") == "jsonl")
        & (pl.col("parse_error_n_if_json") == 0)
        & pl.col("eligible_for_schema_level_comparison")
    )
    .select("path")
    .to_series()
    .to_list()
)

if len(candidates) != 1:
    raise ValueError(
        "Expected exactly one clean eligible GPT-5 JSONL; "
        f"found {len(candidates)}."
    )

source = Path(candidates[0])
frame = pl.read_ndjson(source)

required = [
    "id",
    "composite_run_id",
    "language",
    "harmfulness",
    "model",
    "config",
    "strength",
    "is_vetoed",
]

missing = sorted(set(required) - set(frame.columns))
if missing:
    raise ValueError(f"Missing required fields: {missing}")

if frame.select(pl.col("id").is_null().any()).item():
    raise ValueError("Found null IDs.")

if frame.select(pl.col("is_vetoed").is_null().any()).item():
    raise ValueError("Found null veto outcomes.")

if frame.select(pl.col("id").n_unique() != pl.len()).item():
    raise ValueError("Found duplicate IDs in the clean GPT-5 file.")

if frame.select(pl.col("config").n_unique()).item() != 1:
    raise ValueError("Expected one configuration in this baseline file.")

if frame.select(pl.col("strength").n_unique()).item() != 1:
    raise ValueError("Expected one strength value in this baseline file.")

result = (
    frame
    .group_by(["language", "harmfulness"])
    .agg([
        pl.len().alias("record_n"),
        pl.col("id").n_unique().alias("unique_id_n"),
        pl.col("is_vetoed").sum().alias("veto_n"),
        (~pl.col("is_vetoed")).sum().alias("non_veto_n"),
        pl.col("composite_run_id").n_unique().alias("run_id_n"),
    ])
    .with_columns([
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
    raise ValueError("IDs are not unique within every cell.")

if not (
    result["veto_n"] + result["non_veto_n"] == result["record_n"]
).all():
    raise ValueError("Cell decision counts do not sum to totals.")

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

print(f"Wrote: {OUT}")
print(f"Source: {source}")
print(f"Source records: {frame.height}")
print(f"Language/harmfulness cells: {result.height}")
print(f"Total unique IDs: {frame['id'].n_unique()}")
print(f"Configuration: {frame['config'][0]}")
print(f"Strength: {frame['strength'][0]}")
print(
    "All cell IDs unique: "
    f"{(result['record_n'] == result['unique_id_n']).all()}"
)

with pl.Config(tbl_cols=-1, tbl_width_chars=220):
    print(result)
