from pathlib import Path

import polars as pl

INFERENCE = Path(
    "paper_artifacts/metrics/"
    "qwen25_72b_primary_paired_inference_holm.csv"
)

COMPOSITION = Path(
    "paper_artifacts/metrics/"
    "qwen25_72b_primary_decision_composition.csv"
)

OUT = Path(
    "paper_artifacts/metrics/"
    "qwen25_72b_primary_unified_results.csv"
)

EXPECTED_ROWS = 33

inference = pl.read_csv(INFERENCE)
composition = pl.read_csv(COMPOSITION)

required_inference = {
    "source_file_group",
    "baseline_config",
    "steered_config",
    "language",
    "prompt_class",
    "alpha",
    "paired_record_n",
    "partial_baseline_coverage",
    "all_steered_ids_match_baseline",
    "baseline_failure_n",
    "steered_failure_n",
    "baseline_failure_rate_percentage",
    "steered_failure_rate_percentage",
    "delta_steered_minus_baseline_percentage_points",
    "worsened_safe_to_failure_n",
    "improved_failure_to_safe_n",
    "both_safe_n",
    "both_failure_n",
    "discordant_n",
    "mcnemar_exact_two_sided_p",
    "mcnemar_holm_adjusted_p",
    "mcnemar_holm_reject_alpha_0_05",
    "ci95_low_percentage_points",
    "ci95_high_percentage_points",
    "ci_excludes_zero",
    "correction_family",
    "correction_family_size",
    "familywise_alpha",
}

required_composition = {
    "source_file_group",
    "config",
    "language",
    "prompt_class",
    "alpha",
    "record_n",
    "unique_record_id_n",
    "veto_explicit_n",
    "non_veto_unresolved_n",
    "operational_failure_n",
    "critic_response_veto_token_n",
    "critic_response_allow_token_n",
    "veto_explicit_rate_percentage",
    "non_veto_unresolved_rate_percentage",
    "operational_failure_rate_percentage",
    "record_ids_unique_within_cell",
    "veto_boolean_matches_critic_veto_token",
    "failure_matches_non_veto_unresolved",
}

missing = required_inference - set(inference.columns)
if missing:
    raise ValueError(f"Inference is missing columns: {sorted(missing)}")

missing = required_composition - set(composition.columns)
if missing:
    raise ValueError(f"Composition is missing columns: {sorted(missing)}")

if inference.height != EXPECTED_ROWS:
    raise ValueError(
        f"Expected {EXPECTED_ROWS} inference rows; found {inference.height}."
    )

baseline = (
    composition
    .filter(
        (pl.col("source_file_group") == "baseline")
        & (pl.col("config") == "baseline")
        & (pl.col("alpha") == 0.0)
    )
    .select([
        "language",
        "prompt_class",
        pl.col("record_n").alias("baseline_composition_record_n"),
        pl.col("veto_explicit_n").alias("baseline_veto_explicit_n"),
        pl.col("non_veto_unresolved_n")
        .alias("baseline_non_veto_unresolved_n"),
        pl.col("operational_failure_n")
        .alias("baseline_operational_failure_n"),
        pl.col("critic_response_veto_token_n")
        .alias("baseline_critic_veto_token_n"),
        pl.col("critic_response_allow_token_n")
        .alias("baseline_critic_allow_token_n"),
        pl.col("veto_explicit_rate_percentage")
        .alias("baseline_veto_explicit_rate_percentage"),
        pl.col("non_veto_unresolved_rate_percentage")
        .alias("baseline_non_veto_unresolved_rate_percentage"),
        pl.col("operational_failure_rate_percentage")
        .alias("baseline_operational_failure_rate_percentage"),
        pl.col("record_ids_unique_within_cell")
        .alias("baseline_record_ids_unique"),
        pl.col("veto_boolean_matches_critic_veto_token")
        .alias("baseline_veto_marker_valid"),
        pl.col("failure_matches_non_veto_unresolved")
        .alias("baseline_failure_definition_valid"),
    ])
)

steered = (
    composition
    .filter(
        pl.col("source_file_group").is_in([
            "critic_and_both_steered",
            "planner_steered",
        ])
        & (pl.col("alpha") > 0.0)
    )
    .select([
        "source_file_group",
        pl.col("config").alias("steered_config"),
        "language",
        "prompt_class",
        "alpha",
        pl.col("record_n").alias("steered_composition_record_n"),
        pl.col("veto_explicit_n").alias("steered_veto_explicit_n"),
        pl.col("non_veto_unresolved_n")
        .alias("steered_non_veto_unresolved_n"),
        pl.col("operational_failure_n")
        .alias("steered_operational_failure_n"),
        pl.col("critic_response_veto_token_n")
        .alias("steered_critic_veto_token_n"),
        pl.col("critic_response_allow_token_n")
        .alias("steered_critic_allow_token_n"),
        pl.col("veto_explicit_rate_percentage")
        .alias("steered_veto_explicit_rate_percentage"),
        pl.col("non_veto_unresolved_rate_percentage")
        .alias("steered_non_veto_unresolved_rate_percentage"),
        pl.col("operational_failure_rate_percentage")
        .alias("steered_operational_failure_rate_percentage"),
        pl.col("record_ids_unique_within_cell")
        .alias("steered_record_ids_unique"),
        pl.col("veto_boolean_matches_critic_veto_token")
        .alias("steered_veto_marker_valid"),
        pl.col("failure_matches_non_veto_unresolved")
        .alias("steered_failure_definition_valid"),
    ])
)

if baseline.select(["language", "prompt_class"]).unique().height != baseline.height:
    raise ValueError("Baseline composition keys are not unique.")

if steered.select([
    "source_file_group",
    "steered_config",
    "language",
    "prompt_class",
    "alpha",
]).unique().height != steered.height:
    raise ValueError("Steered composition keys are not unique.")

result = (
    inference
    .join(
        baseline,
        on=["language", "prompt_class"],
        how="inner",
        validate="m:1",
    )
    .join(
        steered,
        on=[
            "source_file_group",
            "steered_config",
            "language",
            "prompt_class",
            "alpha",
        ],
        how="inner",
        validate="1:1",
    )
    .with_columns([
        pl.lit("qwen25_72b").alias("model"),
        pl.lit(
            "Operational failure means is_vetoed=false; non-veto records "
            "are not assumed to be explicit ALLOW."
        ).alias("outcome_caveat"),
        pl.when(pl.col("partial_baseline_coverage"))
        .then(pl.lit(
            "Matched subset only: steered IDs are contained in baseline, "
            "but baseline has additional IDs."
        ))
        .otherwise(pl.lit(
            "Complete matched baseline and steered ID sets."
        ))
        .alias("pairing_coverage_note"),
    ])
    .sort([
        "prompt_class",
        "language",
        "steered_config",
        "alpha",
        "source_file_group",
    ])
)

if result.height != EXPECTED_ROWS:
    raise ValueError(
        f"Expected {EXPECTED_ROWS} unified rows; found {result.height}."
    )

if not result["all_steered_ids_match_baseline"].all():
    raise ValueError("A steered row lacks its baseline ID match.")

if not (
    result["paired_record_n"] == result["steered_composition_record_n"]
).all():
    raise ValueError("Paired n does not equal steered composition n.")

for column in [
    "baseline_record_ids_unique",
    "baseline_veto_marker_valid",
    "baseline_failure_definition_valid",
    "steered_record_ids_unique",
    "steered_veto_marker_valid",
    "steered_failure_definition_valid",
]:
    if not result[column].all():
        raise ValueError(f"Validation failed: {column}")

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

print(f"Wrote: {OUT}")
print(f"Unified rows: {result.height}")
print(
    "Partial-coverage rows: "
    f"{result.filter(pl.col('partial_baseline_coverage')).height}"
)
print(
    "Holm-significant rows: "
    f"{result.filter(pl.col('mcnemar_holm_reject_alpha_0_05')).height}"
)

with pl.Config(tbl_cols=-1, tbl_width_chars=280):
    print(result)
