from pathlib import Path

import polars as pl

MAIN = "data/derived/canonical/llama33_phase1_main.parquet"
TURBO = "data/derived/canonical/llama33_phase2_turbo.parquet"

PLANNER_INF = "paper_artifacts/metrics/llama33_planner20_paired_inference.csv"
CRITIC_INF = "paper_artifacts/metrics/llama33_critic20_paired_inference.csv"
BOTH_INF = "paper_artifacts/metrics/llama33_both_turbo_paired_inference.csv"

OUT = Path("paper_artifacts/metrics/llama33_turbo_role_strength_summary.csv")

LANGUAGES = ["Hausa", "Yoruba"]
CONFIGS = ["planner_only", "critic_only", "both"]
ALPHAS = [20.0, 30.0]

main = pl.read_parquet(MAIN)
turbo = pl.read_parquet(TURBO)

baseline = (
    main.filter(
        (pl.col("alpha") == 0.0)
        & (pl.col("prompt_class") == "harmful")
        & pl.col("language").is_in(LANGUAGES)
        & pl.col("config").is_in(CONFIGS)
    )
    .group_by(["config", "language"])
    .agg([
        pl.len().alias("baseline_n"),
        pl.col("legacy_operational_failure").sum().alias("baseline_failure_n"),
        (pl.col("parser_outcome") == "VETO_EXPLICIT").sum().alias("baseline_veto_n"),
        (pl.col("parser_outcome") == "ALLOW_EXPLICIT").sum().alias("baseline_allow_n"),
        (pl.col("parser_outcome") == "NO_DECISION").sum().alias("baseline_no_decision_n"),
        (pl.col("parser_outcome") == "BOTH_LABELS").sum().alias("baseline_both_labels_n"),
    ])
    .with_columns([
        (100 * pl.col("baseline_failure_n") / pl.col("baseline_n"))
        .alias("baseline_failure_rate_percentage"),
        (100 * pl.col("baseline_veto_n") / pl.col("baseline_n"))
        .alias("baseline_veto_rate_percentage"),
        (100 * pl.col("baseline_allow_n") / pl.col("baseline_n"))
        .alias("baseline_allow_rate_percentage"),
        (100 * pl.col("baseline_no_decision_n") / pl.col("baseline_n"))
        .alias("baseline_no_decision_rate_percentage"),
        (100 * pl.col("baseline_both_labels_n") / pl.col("baseline_n"))
        .alias("baseline_both_labels_rate_percentage"),
    ])
)

summary = (
    turbo.filter(
        (pl.col("prompt_class") == "harmful")
        & pl.col("language").is_in(LANGUAGES)
        & pl.col("config").is_in(CONFIGS)
        & pl.col("alpha").is_in(ALPHAS)
    )
    .group_by(["config", "alpha", "language"])
    .agg([
        pl.len().alias("steered_n"),
        pl.col("legacy_operational_failure").sum().alias("steered_failure_n"),
        (pl.col("parser_outcome") == "VETO_EXPLICIT").sum().alias("steered_veto_n"),
        (pl.col("parser_outcome") == "ALLOW_EXPLICIT").sum().alias("steered_allow_n"),
        (pl.col("parser_outcome") == "NO_DECISION").sum().alias("steered_no_decision_n"),
        (pl.col("parser_outcome") == "BOTH_LABELS").sum().alias("steered_both_labels_n"),
    ])
    .join(baseline, on=["config", "language"], how="left", validate="m:1")
    .with_columns([
        (100 * pl.col("steered_failure_n") / pl.col("steered_n"))
        .alias("steered_failure_rate_percentage"),
        (
            100 * (
                pl.col("steered_failure_n") / pl.col("steered_n")
                - pl.col("baseline_failure_n") / pl.col("baseline_n")
            )
        ).alias("delta_failure_percentage_points"),
        (100 * pl.col("steered_veto_n") / pl.col("steered_n"))
        .alias("steered_veto_rate_percentage"),
        (100 * pl.col("steered_allow_n") / pl.col("steered_n"))
        .alias("steered_allow_rate_percentage"),
        (100 * pl.col("steered_no_decision_n") / pl.col("steered_n"))
        .alias("steered_no_decision_rate_percentage"),
        (100 * pl.col("steered_both_labels_n") / pl.col("steered_n"))
        .alias("steered_both_labels_rate_percentage"),
    ])
)

planner = (
    pl.read_csv(PLANNER_INF)
    .filter(pl.col("language").is_in(LANGUAGES))
    .select([
        "language",
        "mcnemar_exact_two_sided_p",
        "mcnemar_holm_adjusted_p",
        "mcnemar_holm_reject_alpha_0_05",
        "ci95_low_percentage_points",
        "ci95_high_percentage_points",
    ])
    .with_columns([
        pl.lit("planner_only").alias("config"),
        pl.lit(20.0).alias("alpha"),
        pl.lit("planner_main_language_family").alias("inference_family"),
    ])
)

critic = (
    pl.read_csv(CRITIC_INF)
    .filter(pl.col("language").is_in(LANGUAGES))
    .select([
        "language",
        "mcnemar_exact_two_sided_p",
        "mcnemar_holm_adjusted_p",
        "mcnemar_holm_reject_alpha_0_05",
        "ci95_low_percentage_points",
        "ci95_high_percentage_points",
    ])
    .with_columns([
        pl.lit("critic_only").alias("config"),
        pl.lit(20.0).alias("alpha"),
        pl.lit("critic_main_language_family").alias("inference_family"),
    ])
)

both = (
    pl.read_csv(BOTH_INF)
    .select([
        "alpha",
        "language",
        "mcnemar_exact_two_sided_p",
        "mcnemar_holm_adjusted_p",
        "mcnemar_holm_reject_alpha_0_05",
        "ci95_low_percentage_points",
        "ci95_high_percentage_points",
    ])
    .with_columns([
        pl.lit("both").alias("config"),
        pl.lit("both_turbo_alpha_language_family").alias("inference_family"),
    ])
)

inference = pl.concat([planner, critic, both], how="diagonal")

result = (
    summary.join(
        inference,
        on=["config", "alpha", "language"],
        how="left",
        validate="1:1",
    )
    .with_columns([
        pl.when(pl.col("inference_family").is_null())
        .then(pl.lit("not_previously_tested_descriptive_only"))
        .otherwise(pl.col("inference_family"))
        .alias("inference_family"),
        pl.lit(
            "Harmful prompts only; main alpha-0 baseline matched by config and language. "
            "Turbo alpha-20/30 results are descriptive unless inference columns are populated."
        ).alias("scope_note"),
    ])
    .sort(["config", "alpha", "language"])
)

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

print(
    result.select([
        "config",
        "alpha",
        "language",
        "baseline_failure_rate_percentage",
        "steered_failure_rate_percentage",
        "delta_failure_percentage_points",
        "steered_veto_rate_percentage",
        "steered_allow_rate_percentage",
        "steered_no_decision_rate_percentage",
        "baseline_no_decision_rate_percentage",
        "inference_family",
        "mcnemar_holm_adjusted_p",
        "mcnemar_holm_reject_alpha_0_05",
    ])
)
print(f"\nWrote: {OUT}")
