from pathlib import Path

import polars as pl

SOURCE_INVENTORY = Path(
    "paper_artifacts/metrics/closed_model_source_inventory.csv"
)

SCHEMA_INVENTORY = Path(
    "paper_artifacts/metrics/closed_model_schema_inventory.csv"
)

OUT = Path(
    "paper_artifacts/metrics/gpt_5_source_reconciliation.csv"
)

sources = (
    pl.read_csv(SOURCE_INVENTORY)
    .filter(
        (pl.col("model") == "gpt_5")
        & pl.col("path").is_not_null()
    )
)

schemas = (
    pl.read_csv(SCHEMA_INVENTORY)
    .filter(pl.col("model") == "gpt_5")
    .select([
        "path",
        "file_type",
        "record_n_if_json",
        "parse_error_n_if_json",
        "key_n",
        "likely_id_field_present",
        "likely_outcome_field_present",
        "likely_config_field_present",
    ])
)

result = (
    sources
    .join(
        schemas,
        on=["path", "file_type"],
        how="left",
        validate="1:1",
    )
    .with_columns([
        pl.when(pl.col("location") == "raw")
        .then(pl.lit("candidate_raw_source"))
        .when(pl.col("location") == "derived")
        .then(pl.lit("candidate_derived_source"))
        .when(pl.col("location") == "registry")
        .then(pl.lit("metadata_only"))
        .otherwise(pl.lit("other"))
        .alias("role"),
        (
            (pl.col("location") == "raw")
            & (pl.col("file_type") == "jsonl")
            & (pl.col("parse_error_n_if_json").fill_null(0) == 0)
            & pl.col("likely_id_field_present").fill_null(False)
            & pl.col("likely_outcome_field_present").fill_null(False)
        ).alias("eligible_for_schema_level_comparison"),
    ])
    .sort(["role", "path"])
)

if result.height != 7:
    raise ValueError(f"Expected 7 GPT-5 inventory rows; found {result.height}")

raw_jsonl = result.filter(
    (pl.col("location") == "raw")
    & (pl.col("file_type") == "jsonl")
)

if raw_jsonl.height == 0:
    raise ValueError("No raw GPT-5 JSONL sources found.")

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

print(f"Wrote: {OUT}")
print(f"GPT-5 candidate files: {result.height}")
print(f"Raw JSONL candidates: {raw_jsonl.height}")
print(
    "Clean raw JSONL candidates: "
    f"{raw_jsonl.filter(pl.col('parse_error_n_if_json') == 0).height}"
)
print(
    "Raw JSONL files with parse errors: "
    f"{raw_jsonl.filter(pl.col('parse_error_n_if_json') > 0).height}"
)
print(
    "Eligible schema-level candidates: "
    f"{result.filter(pl.col('eligible_for_schema_level_comparison')).height}"
)

with pl.Config(tbl_cols=-1, tbl_width_chars=260):
    print(
        result.select([
            "role",
            "location",
            "file_type",
            "record_n_if_json",
            "parse_error_n_if_json",
            "key_n",
            "eligible_for_schema_level_comparison",
            "path",
        ])
    )
