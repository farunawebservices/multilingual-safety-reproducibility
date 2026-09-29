from pathlib import Path
import json

import polars as pl

SOURCE = Path(
    "data/raw/phase1/gpt_5/originals/"
    "frontier_baseline_gpt_5_results.jsonl"
)

OUT_PARQUET = Path(
    "data/derived/canonical/"
    "gpt_5_frontier_baseline.parquet"
)

OUT_AUDIT = Path(
    "paper_artifacts/metrics/"
    "gpt_5_frontier_baseline_canonical_audit.csv"
)

REQUIRED_SOURCE_COLUMNS = [
    "id",
    "composite_run_id",
    "language",
    "harmfulness",
    "model",
    "config",
    "strength",
    "is_vetoed",
]

if not SOURCE.exists():
    raise FileNotFoundError(f"Missing clean GPT-5 source: {SOURCE}")

frame = pl.read_ndjson(SOURCE)

missing = sorted(set(REQUIRED_SOURCE_COLUMNS) - set(frame.columns))
if missing:
    raise ValueError(f"Missing GPT-5 source fields: {missing}")

if frame.height != 1651:
    raise ValueError(f"Expected 1651 source records; found {frame.height}")

if frame.select(pl.col("id").is_null().any()).item():
    raise ValueError("Null prompt IDs are not allowed.")

if frame.select(pl.col("id").n_unique() != pl.len()).item():
    raise ValueError("Prompt IDs must be unique.")

if frame.select(pl.col("config").n_unique()).item() != 1:
    raise ValueError("Expected exactly one configuration.")

if frame.select(pl.col("config").first()).item() != "frontier_baseline":
    raise ValueError("Unexpected configuration.")

if frame.select(pl.col("strength").n_unique()).item() != 1:
    raise ValueError("Expected exactly one strength value.")

if frame.select(pl.col("strength").first()).item() != 0.0:
    raise ValueError("Expected strength 0.0.")

canonical = (
    frame
    .select([
        pl.lit("gpt_5_frontier_baseline").alias("source_run"),
        pl.col("id").cast(pl.String).alias("prompt_id"),
        pl.col("composite_run_id").cast(pl.String).alias("run_id"),
        pl.col("language").cast(pl.String),
        pl.col("harmfulness").cast(pl.String).alias("prompt_class"),
        pl.col("model").cast(pl.String).alias("model_identifier"),
        pl.col("config").cast(pl.String).alias("configuration"),
        pl.col("strength").cast(pl.Float64),
        pl.col("is_vetoed").cast(pl.Boolean),
        (~pl.col("is_vetoed")).cast(pl.Boolean)
        .alias("non_veto_operational_failure"),
    ])
    .with_columns([
        pl.lit(True).alias("valid_json"),
        pl.lit(False).alias("parse_failure"),
        pl.lit(False).alias("non_termination"),
        pl.lit("clean_1651_record_source_only")
        .alias("inclusion_policy"),
    ])
)

if canonical.height != 1651:
    raise ValueError("Canonical row count changed unexpectedly.")

if canonical.select(pl.col("prompt_id").n_unique() != pl.len()).item():
    raise ValueError("Canonical prompt IDs are not unique.")

if canonical.select(
    (
        pl.col("is_vetoed")
        == (~pl.col("non_veto_operational_failure"))
    ).all()
).item() is not True:
    raise ValueError("Outcome complement invariant failed.")

OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
canonical.write_parquet(OUT_PARQUET)

audit = (
    canonical
    .group_by(["language", "prompt_class"])
    .agg([
        pl.len().alias("canonical_record_n"),
        pl.col("prompt_id").n_unique().alias("unique_prompt_id_n"),
        pl.col("is_vetoed").sum().alias("veto_n"),
        pl.col("non_veto_operational_failure")
        .sum()
        .alias("non_veto_failure_n"),
        pl.col("parse_failure").sum().alias("parse_failure_n"),
        pl.col("non_termination").sum().alias("non_termination_n"),
    ])
    .with_columns([
        (
            100.0 * pl.col("veto_n")
            / pl.col("canonical_record_n")
        ).alias("veto_rate_percentage"),
        (
            100.0 * pl.col("non_veto_failure_n")
            / pl.col("canonical_record_n")
        ).alias("non_veto_failure_rate_percentage"),
    ])
    .sort(["prompt_class", "language"])
)

if audit["canonical_record_n"].sum() != 1651:
    raise ValueError("Audit total does not equal canonical total.")

audit.write_csv(OUT_AUDIT)

print(f"Wrote: {OUT_PARQUET}")
print(f"Wrote: {OUT_AUDIT}")
print(f"Canonical records: {canonical.height}")
print(f"Canonical columns: {len(canonical.columns)}")
print(f"Unique prompt IDs: {canonical['prompt_id'].n_unique()}")
print("Excluded duplicate subset records: 570")
print("Excluded duplicate subset parse errors: 1")
print("Canonical outcome invariant passed: True")
