from pathlib import Path

import polars as pl

INFILE = Path(
    "data/derived/canonical/qwen25_72b_primary_role_strength.parquet"
)

OUT = Path(
    "paper_artifacts/metrics/qwen25_72b_primary_decision_composition.csv"
)

frame = pl.read_parquet(INFILE)

required = {
    "model",
    "source_file_group",
    "record_id",
    "language",
    "prompt_class",
    "config",
    "alpha",
    "is_vetoed",
    "parser_outcome",
    "operational_failure",
    "critic_response_contains_veto_token",
    "critic_response_contains_allow_token",
}

missing = required - set(frame.columns)
if missing:
    raise ValueError(f"Missing canonical columns: {sorted(missing)}")

result = (
    frame
    .group_by([
        "model",
        "source_file_group",
        "config",
        "language",
        "prompt_class",
        "alpha",
    ])
    .agg([
        pl.len().alias("record_n"),
        pl.col("record_id").n_unique().alias("unique_record_id_n"),
        pl.col("is_vetoed").sum().alias("veto_explicit_n"),
        (~pl.col("is_vetoed")).sum().alias("non_veto_unresolved_n"),
        pl.col("operational_failure").sum().alias("operational_failure_n"),
        pl.col("critic_response_contains_veto_token")
        .sum()
        .alias("critic_response_veto_token_n"),
        pl.col("critic_response_contains_allow_token")
        .sum()
        .alias("critic_response_allow_token_n"),
    ])
    .with_columns([
        (
            100 * pl.col("veto_explicit_n") / pl.col("record_n")
        ).alias("veto_explicit_rate_percentage"),
        (
            100 * pl.col("non_veto_unresolved_n") / pl.col("record_n")
        ).alias("non_veto_unresolved_rate_percentage"),
        (
            100 * pl.col("operational_failure_n") / pl.col("record_n")
        ).alias("operational_failure_rate_percentage"),
        (
            pl.col("record_n") == pl.col("unique_record_id_n")
        ).alias("record_ids_unique_within_cell"),
        (
            pl.col("veto_explicit_n")
            == pl.col("critic_response_veto_token_n")
        ).alias("veto_boolean_matches_critic_veto_token"),
        (
            pl.col("operational_failure_n")
            == pl.col("non_veto_unresolved_n")
        ).alias("failure_matches_non_veto_unresolved"),
    ])
    .sort([
        "source_file_group",
        "prompt_class",
        "language",
        "config",
        "alpha",
    ])
)

for column in [
    "record_ids_unique_within_cell",
    "veto_boolean_matches_critic_veto_token",
    "failure_matches_non_veto_unresolved",
]:
    if not result[column].all():
        raise ValueError(f"Validation failed for: {column}")

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

print(f"Wrote: {OUT}")
print(f"Decision-composition rows: {result.height}")
print(
    "All validation flags true: "
    f"{all(result[column].all() for column in [
        'record_ids_unique_within_cell',
        'veto_boolean_matches_critic_veto_token',
        'failure_matches_non_veto_unresolved',
    ])}"
)

with pl.Config(tbl_cols=-1, tbl_width_chars=240):
    print(result)
