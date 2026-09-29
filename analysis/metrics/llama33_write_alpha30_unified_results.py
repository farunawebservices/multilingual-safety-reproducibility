from pathlib import Path

import polars as pl

INFERENCE = Path(
    "paper_artifacts/metrics/"
    "llama33_turbo_alpha30_role_strength_paired_inference.csv"
)

COMPOSITION = Path(
    "paper_artifacts/metrics/"
    "llama33_turbo_role_strength_decision_composition.csv"
)

OUT = Path(
    "paper_artifacts/metrics/"
    "llama33_turbo_alpha30_role_strength_unified_results.csv"
)

ALPHA = 30.0
BASELINE_ALPHA = 0.0
EXPECTED_ROWS = 6

inference = pl.read_csv(INFERENCE)
composition = pl.read_csv(COMPOSITION)

required_inference = {
    "config",
    "language",
    "alpha",
    "paired_harmful_n",
    "baseline_failure_n",
    "steered_failure_n",
    "baseline_failure_rate",
    "steered_failure_rate",
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
}

required_composition = {
    "source_phase",
    "condition_type",
    "config",
    "language",
    "alpha",
    "harmful_n",
    "explicit_veto_n",
    "explicit_allow_n",
    "no_decision_n",
    "operational_failure_n",
    "explicit_veto_rate_percentage",
    "explicit_allow_rate_percentage",
    "no_decision_rate_percentage",
    "operational_failure_rate_percentage",
    "parser_outcomes_sum_to_harmful_n",
    "record_ids_are_unique",
    "operational_failure_matches_allow_plus_no_decision",
}

missing_inference = required_inference - set(inference.columns)
if missing_inference:
    raise ValueError(
        f"Inference file is missing columns: {sorted(missing_inference)}"
    )

missing_composition = required_composition - set(composition.columns)
if missing_composition:
    raise ValueError(
        f"Composition file is missing columns: {sorted(missing_composition)}"
    )

inference = (
    inference
    .filter(pl.col("alpha") == ALPHA)
    .select([
        "config",
        "language",
        "alpha",
        "paired_harmful_n",
        "baseline_failure_n",
        "steered_failure_n",
        "baseline_failure_rate",
        "steered_failure_rate",
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
    ])
)

if inference.height != EXPECTED_ROWS:
    raise ValueError(
        f"Expected {EXPECTED_ROWS} alpha-30 inference rows; "
        f"found {inference.height}."
    )

if inference.select(["config", "language"]).unique().height != EXPECTED_ROWS:
    raise ValueError("Inference rows are not unique by config and language.")

baseline = (
    composition
    .filter(
        (pl.col("alpha") == BASELINE_ALPHA)
        & (pl.col("condition_type") == "alpha0_baseline")
        & (pl.col("source_phase") == "phase1_main")
    )
    .select([
        "config",
        "language",
        pl.col("harmful_n").alias("baseline_harmful_n"),
        pl.col("explicit_veto_n").alias("baseline_explicit_veto_n"),
        pl.col("explicit_allow_n").alias("baseline_explicit_allow_n"),
        pl.col("no_decision_n").alias("baseline_no_decision_n"),
        pl.col("operational_failure_n").alias("baseline_operational_failure_n"),
        pl.col("explicit_veto_rate_percentage")
        .alias("baseline_explicit_veto_rate_percentage"),
        pl.col("explicit_allow_rate_percentage")
        .alias("baseline_explicit_allow_rate_percentage"),
        pl.col("no_decision_rate_percentage")
        .alias("baseline_no_decision_rate_percentage"),
        pl.col("operational_failure_rate_percentage")
        .alias("baseline_operational_failure_rate_percentage"),
        pl.col("parser_outcomes_sum_to_harmful_n")
        .alias("baseline_parser_outcomes_valid"),
        pl.col("record_ids_are_unique")
        .alias("baseline_record_ids_unique"),
        pl.col("operational_failure_matches_allow_plus_no_decision")
        .alias("baseline_failure_definition_valid"),
    ])
)

steered = (
    composition
    .filter(
        (pl.col("alpha") == ALPHA)
        & (pl.col("condition_type") == "turbo_steered")
        & (pl.col("source_phase") == "phase2_turbo")
    )
    .select([
        "config",
        "language",
        pl.col("harmful_n").alias("steered_harmful_n"),
        pl.col("explicit_veto_n").alias("steered_explicit_veto_n"),
        pl.col("explicit_allow_n").alias("steered_explicit_allow_n"),
        pl.col("no_decision_n").alias("steered_no_decision_n"),
        pl.col("operational_failure_n").alias("steered_operational_failure_n"),
        pl.col("explicit_veto_rate_percentage")
        .alias("steered_explicit_veto_rate_percentage"),
        pl.col("explicit_allow_rate_percentage")
        .alias("steered_explicit_allow_rate_percentage"),
        pl.col("no_decision_rate_percentage")
        .alias("steered_no_decision_rate_percentage"),
        pl.col("operational_failure_rate_percentage")
        .alias("steered_operational_failure_rate_percentage"),
        pl.col("parser_outcomes_sum_to_harmful_n")
        .alias("steered_parser_outcomes_valid"),
        pl.col("record_ids_are_unique")
        .alias("steered_record_ids_unique"),
        pl.col("operational_failure_matches_allow_plus_no_decision")
        .alias("steered_failure_definition_valid"),
    ])
)

for name, frame in [("baseline", baseline), ("steered", steered)]:
    if frame.height != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} {name} composition rows; "
            f"found {frame.height}."
        )

    if frame.select(["config", "language"]).unique().height != EXPECTED_ROWS:
        raise ValueError(
            f"{name} composition rows are not unique by config and language."
        )

result = (
    inference
    .join(
        baseline,
        on=["config", "language"],
        how="inner",
        validate="1:1",
    )
    .join(
        steered,
        on=["config", "language"],
        how="inner",
        validate="1:1",
    )
    .with_columns([
        pl.lit("phase1_main").alias("baseline_source_phase"),
        pl.lit("phase2_turbo").alias("steered_source_phase"),
        pl.lit(True).alias("cross_phase_comparison"),
        pl.lit(
            "Matched by record_id; phases differ, so non-prompt "
            "run-condition equivalence is not established by pairing alone."
        ).alias("cross_phase_caveat"),
    ])
    .sort(["language", "config"])
)

if result.height != EXPECTED_ROWS:
    raise ValueError(
        f"Expected {EXPECTED_ROWS} unified rows; found {result.height}."
    )

if not (result["paired_harmful_n"] == result["baseline_harmful_n"]).all():
    raise ValueError("Paired n differs from baseline composition harmful_n.")

if not (result["paired_harmful_n"] == result["steered_harmful_n"]).all():
    raise ValueError("Paired n differs from steered composition harmful_n.")

if not result["baseline_parser_outcomes_valid"].all():
    raise ValueError("Baseline parser-outcome validation failed.")

if not result["steered_parser_outcomes_valid"].all():
    raise ValueError("Steered parser-outcome validation failed.")

if not result["baseline_record_ids_unique"].all():
    raise ValueError("Baseline record-ID uniqueness validation failed.")

if not result["steered_record_ids_unique"].all():
    raise ValueError("Steered record-ID uniqueness validation failed.")

if not result["baseline_failure_definition_valid"].all():
    raise ValueError("Baseline failure-definition validation failed.")

if not result["steered_failure_definition_valid"].all():
    raise ValueError("Steered failure-definition validation failed.")

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

with pl.Config(tbl_cols=-1, tbl_width_chars=300):
    print(result)

print(f"\nWrote: {OUT}")
