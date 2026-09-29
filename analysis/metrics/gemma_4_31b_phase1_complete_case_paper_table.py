from pathlib import Path

import polars as pl

INFILE = Path(
    "paper_artifacts/metrics/"
    "gemma_4_31b_phase1_complete_case_unified_results.csv"
)

OUT = Path(
    "paper_artifacts/metrics/"
    "gemma_4_31b_phase1_complete_case_paper_table.csv"
)

frame = pl.read_csv(INFILE)

required = [
    "model",
    "analysis_population",
    "steered_config",
    "language",
    "prompt_class",
    "alpha",
    "paired_record_n",
    "baseline_failure_rate_percentage",
    "steered_failure_rate_percentage",
    "delta_steered_minus_baseline_percentage_points",
    "ci95_low_percentage_points",
    "ci95_high_percentage_points",
    "mcnemar_exact_two_sided_p",
    "mcnemar_holm_adjusted_p",
    "mcnemar_holm_reject_alpha_0_05",
    "partial_coverage",
    "pairing_coverage_note",
    "source_truncation_caveat",
]

missing = sorted(set(required) - set(frame.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if frame.height != 3:
    raise ValueError(f"Expected 3 input rows; found {frame.height}")

result = (
    frame
    .select([
        pl.col("model").alias("Model"),
        pl.col("steered_config").alias("Configuration"),
        pl.col("language").alias("Language"),
        pl.col("prompt_class").alias("Prompt class"),
        pl.col("alpha").alias("Steering alpha"),
        pl.col("paired_record_n").alias("Matched n"),
        pl.col("baseline_failure_rate_percentage")
        .round(2)
        .alias("Baseline non-veto failure (%)"),
        pl.col("steered_failure_rate_percentage")
        .round(2)
        .alias("Steered non-veto failure (%)"),
        pl.col("delta_steered_minus_baseline_percentage_points")
        .round(2)
        .alias("Change (percentage points)"),
        pl.col("ci95_low_percentage_points")
        .round(2)
        .alias("Bootstrap 95% CI low (pp)"),
        pl.col("ci95_high_percentage_points")
        .round(2)
        .alias("Bootstrap 95% CI high (pp)"),
        pl.col("mcnemar_exact_two_sided_p")
        .round(4)
        .alias("Exact McNemar p"),
        pl.col("mcnemar_holm_adjusted_p")
        .round(4)
        .alias("Holm-adjusted p"),
        pl.when(pl.col("mcnemar_holm_reject_alpha_0_05"))
        .then(pl.lit("Yes"))
        .otherwise(pl.lit("No"))
        .alias("Holm significant (0.05)"),
        pl.when(pl.col("partial_coverage"))
        .then(pl.lit("Matched subset"))
        .otherwise(pl.lit("Complete matching"))
        .alias("Coverage"),
        pl.col("pairing_coverage_note").alias("Coverage note"),
        pl.col("source_truncation_caveat").alias("Source caveat"),
    ])
    .sort(["Prompt class", "Language", "Configuration", "Steering alpha"])
)

if result.height != 3:
    raise ValueError(f"Expected 3 output rows; found {result.height}")

if result["Holm significant (0.05)"].ne("No").any():
    raise ValueError("Unexpected Holm-significant result found.")

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

print(f"Wrote: {OUT}")
print(f"Paper-table rows: {result.height}")
print(
    "Partial-coverage rows: "
    f"{result.filter(pl.col('Coverage') == 'Matched subset').height}"
)
print(
    "Holm-significant rows: "
    f"{result.filter(pl.col('Holm significant (0.05)') == 'Yes').height}"
)

with pl.Config(tbl_cols=-1, tbl_width_chars=260):
    print(result)
