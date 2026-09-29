from pathlib import Path

import polars as pl

INFILE = Path(
    "paper_artifacts/metrics/"
    "qwen25_72b_primary_paired_inference_holm.csv"
)

OUT = Path(
    "paper_artifacts/metrics/"
    "qwen25_72b_planner_alpha1_primary_paper_table.csv"
)

frame = pl.read_csv(INFILE)

required = [
    "source_file_group",
    "baseline_config",
    "steered_config",
    "language",
    "prompt_class",
    "alpha",
    "paired_record_n",
    "baseline_failure_n",
    "steered_failure_n",
    "baseline_failure_rate_percentage",
    "steered_failure_rate_percentage",
    "delta_steered_minus_baseline_percentage_points",
    "worsened_safe_to_failure_n",
    "improved_failure_to_safe_n",
    "discordant_n",
    "mcnemar_exact_two_sided_p",
    "mcnemar_holm_adjusted_p",
    "mcnemar_holm_reject_alpha_0_05",
    "ci95_low_percentage_points",
    "ci95_high_percentage_points",
    "all_steered_ids_match_baseline",
    "partial_baseline_coverage",
]

missing = sorted(set(required) - set(frame.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

primary = (
    frame
    .filter(
        (pl.col("source_file_group") == "planner_steered")
        & (pl.col("baseline_config") == "baseline")
        & (pl.col("steered_config") == "planner_only")
        & (pl.col("alpha") == 1.0)
    )
    .sort(["prompt_class", "language"])
)

if primary.height != 8:
    raise ValueError(f"Expected 8 primary rows; found {primary.height}")

if primary["paired_record_n"].sum() != 1695:
    raise ValueError(
        "Expected 1695 paired records; "
        f"found {primary['paired_record_n'].sum()}."
    )

if not primary["all_steered_ids_match_baseline"].all():
    raise ValueError("Primary rows contain steered IDs absent from baseline.")

if primary["partial_baseline_coverage"].any():
    raise ValueError("Primary rows must have complete baseline coverage.")

if primary["language"].n_unique() != 4:
    raise ValueError("Expected four languages.")

if primary["prompt_class"].n_unique() != 2:
    raise ValueError("Expected benign and harmful prompt classes.")

result = (
    primary
    .select([
        pl.lit("Qwen 2.5 72B: planner-only steering, alpha 1.0")
        .alias("Comparison"),
        pl.col("language").alias("Language"),
        pl.col("prompt_class").alias("Prompt class"),
        pl.col("paired_record_n").alias("Paired n"),
        pl.col("baseline_failure_n").alias("Baseline failure n"),
        pl.col("baseline_failure_rate_percentage")
        .round(2)
        .alias("Baseline failure (%)"),
        pl.col("steered_failure_n").alias("Steered failure n"),
        pl.col("steered_failure_rate_percentage")
        .round(2)
        .alias("Steered failure (%)"),
        pl.col("delta_steered_minus_baseline_percentage_points")
        .round(2)
        .alias("Steered minus baseline (pp)"),
        pl.col("worsened_safe_to_failure_n")
        .alias("Worsened n"),
        pl.col("improved_failure_to_safe_n")
        .alias("Improved n"),
        pl.col("discordant_n").alias("Discordant n"),
        pl.col("mcnemar_exact_two_sided_p")
        .round(6)
        .alias("McNemar exact p"),
        pl.col("mcnemar_holm_adjusted_p")
        .round(6)
        .alias("Holm adjusted p"),
        pl.col("mcnemar_holm_reject_alpha_0_05")
        .alias("Holm reject 0.05"),
        pl.col("ci95_low_percentage_points")
        .round(2)
        .alias("Bootstrap 95% CI low (pp)"),
        pl.col("ci95_high_percentage_points")
        .round(2)
        .alias("Bootstrap 95% CI high (pp)"),
        pl.lit("Complete matched coverage").alias("Coverage"),
    ])
)

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

print(f"Wrote: {OUT}")
print(f"Primary rows: {result.height}")
print(f"Paired prompts: {result['Paired n'].sum()}")
print(
    "Holm-significant rows: "
    f"{result.filter(pl.col('Holm reject 0.05')).height}"
)
print(
    "Negative failure-rate-difference rows: "
    f"{result.filter(pl.col('Steered minus baseline (pp)') < 0).height}"
)
print("Partial-coverage rows: 0")

with pl.Config(tbl_cols=-1, tbl_width_chars=260):
    print(result)
