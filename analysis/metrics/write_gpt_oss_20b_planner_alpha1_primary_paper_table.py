from pathlib import Path

import polars as pl

INFILE = Path(
    "paper_artifacts/metrics/"
    "gpt_oss_20b_planner_alpha1_paired_effect_holm.csv"
)

OUT = Path(
    "paper_artifacts/metrics/"
    "gpt_oss_20b_planner_alpha1_primary_harmful_paper_table.csv"
)

required = [
    "language",
    "paired_harmful_n",
    "baseline_failure_rate",
    "steered_failure_rate",
    "absolute_change_percentage_points",
    "worsened_safe_to_failure_n",
    "improved_failure_to_safe_n",
    "mcnemar_holm_adjusted_p",
    "mcnemar_holm_reject_alpha_0_05",
]

frame = pl.read_csv(INFILE)

missing = [column for column in required if column not in frame.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if frame.height != 4:
    raise ValueError(f"Expected 4 language rows; found {frame.height}.")

if frame.get_column("paired_harmful_n").sum() != 1200:
    raise ValueError(
        "Expected 1200 paired harmful prompts; "
        f"found {frame.get_column('paired_harmful_n').sum()}."
    )

if not (frame.get_column("paired_harmful_n") == 300).all():
    raise ValueError("Expected 300 paired harmful prompts in each language.")

table = (
    frame
    .select(
        [
            pl.col("language").alias("Language"),
            pl.lit("Harmful").alias("Prompt class"),
            pl.col("paired_harmful_n").alias("Paired n"),
            pl.col("baseline_failure_rate")
            .mul(100)
            .round(1)
            .alias("Baseline failure (%)"),
            pl.col("steered_failure_rate")
            .mul(100)
            .round(1)
            .alias("Planner-only alpha=1 failure (%)"),
            pl.col("absolute_change_percentage_points")
            .round(1)
            .alias("Steered minus baseline (pp)"),
            pl.col("worsened_safe_to_failure_n").alias("Worsened n"),
            pl.col("improved_failure_to_safe_n").alias("Improved n"),
            pl.col("mcnemar_holm_adjusted_p")
            .round(6)
            .alias("Holm-adjusted p"),
            pl.col("mcnemar_holm_reject_alpha_0_05")
            .alias("Holm reject 0.05"),
            pl.lit("Complete matched coverage").alias("Coverage"),
        ]
    )
    .sort("Language")
)

holm_significant = table.filter(pl.col("Holm reject 0.05")).height
negative_effects = table.filter(
    pl.col("Steered minus baseline (pp)") < 0
).height
positive_effects = table.filter(
    pl.col("Steered minus baseline (pp)") > 0
).height

if holm_significant != 2:
    raise ValueError(
        f"Expected 2 Holm-significant rows; found {holm_significant}."
    )

if negative_effects != 2:
    raise ValueError(
        f"Expected 2 negative-effect rows; found {negative_effects}."
    )

if positive_effects != 1:
    raise ValueError(
        f"Expected 1 positive-effect row; found {positive_effects}."
    )

OUT.parent.mkdir(parents=True, exist_ok=True)
table.write_csv(OUT)

print(f"Wrote: {OUT}")
print(f"Paper-table rows: {table.height}")
print(f"Paired harmful prompts: {table['Paired n'].sum()}")
print(f"Holm-significant rows: {holm_significant}")
print(f"Improvement rows (negative pp): {negative_effects}")
print(f"Worsening rows (positive pp): {positive_effects}")
print("Complete matched coverage rows: 4")
