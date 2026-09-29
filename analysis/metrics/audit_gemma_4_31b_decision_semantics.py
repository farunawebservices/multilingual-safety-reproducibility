from pathlib import Path

import polars as pl

INFILE = Path(
    "data/derived/canonical/gemma_4_31b_phase1_complete_case.parquet"
)

OUT = Path(
    "paper_artifacts/metrics/"
    "gemma_4_31b_phase1_complete_case_decision_semantics_audit.csv"
)

frame = pl.read_parquet(INFILE)

result = (
    frame
    .group_by([
        "source_file_group",
        "config",
        "language",
        "prompt_class",
        "alpha",
    ])
    .agg([
        pl.len().alias("record_n"),
        pl.col("is_vetoed").sum().alias("is_vetoed_true_n"),
        (~pl.col("is_vetoed")).sum().alias("is_vetoed_false_n"),
        pl.col("critic_response_contains_veto_token")
        .sum()
        .alias("critic_veto_token_n"),
        pl.col("critic_response_contains_allow_token")
        .sum()
        .alias("critic_allow_token_n"),
        (
            pl.col("is_vetoed")
            == pl.col("critic_response_contains_veto_token")
        )
        .all()
        .alias("veto_boolean_matches_critic_veto_token"),
        (
            (~pl.col("is_vetoed"))
            == pl.col("operational_failure")
        )
        .all()
        .alias("failure_matches_non_veto_boolean"),
    ])
    .with_columns([
        (
            pl.col("is_vetoed_true_n")
            + pl.col("is_vetoed_false_n")
            == pl.col("record_n")
        ).alias("decision_counts_sum_to_record_n"),
        (
            pl.col("critic_veto_token_n")
            == pl.col("is_vetoed_true_n")
        ).alias("critic_veto_token_count_matches_boolean"),
    ])
    .sort([
        "source_file_group",
        "prompt_class",
        "language",
        "config",
        "alpha",
    ])
)

checks = [
    "veto_boolean_matches_critic_veto_token",
    "failure_matches_non_veto_boolean",
    "decision_counts_sum_to_record_n",
    "critic_veto_token_count_matches_boolean",
]

for column in checks:
    if not result[column].all():
        failures = result.filter(~pl.col(column)).height
        raise ValueError(f"{column} failed in {failures} cell(s).")

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

print(f"Wrote: {OUT}")
print(f"Decision-semantics cells: {result.height}")
print("All checks passed: True")

with pl.Config(tbl_cols=-1, tbl_width_chars=240):
    print(result)
