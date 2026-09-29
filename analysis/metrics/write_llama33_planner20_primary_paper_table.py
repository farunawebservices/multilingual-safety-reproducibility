from pathlib import Path

import polars as pl

HARMFUL = Path(
    "paper_artifacts/metrics/llama33_planner20_paired_inference.csv"
)
BENIGN = Path(
    "paper_artifacts/metrics/llama33_planner20_benign_refusal_inference.csv"
)
OUT = Path(
    "paper_artifacts/metrics/"
    "llama33_planner20_primary_paper_table.csv"
)

harmful = pl.read_csv(HARMFUL)
benign = pl.read_csv(BENIGN)

harmful_required = [
    "language",
    "paired_harmful_n",
    "baseline_failure_rate",
    "steered_failure_rate",
    "delta_steered_minus_baseline_percentage_points",
    "ci95_low_percentage_points",
    "ci95_high_percentage_points",
    "mcnemar_holm_adjusted_p",
    "mcnemar_holm_reject_alpha_0_05",
]

benign_required = [
    "language",
    "paired_benign_n",
    "baseline_refusal_rate",
    "steered_refusal_rate",
    "delta_refusal_percentage_points",
    "ci95_low_percentage_points",
    "ci95_high_percentage_points",
    "mcnemar_holm_adjusted_p",
    "mcnemar_holm_reject_alpha_0_05",
]

for name, frame, columns in [
    ("harmful", harmful, harmful_required),
    ("benign", benign, benign_required),
]:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} inference is missing columns: {missing}")

if harmful.height != 4:
    raise ValueError(f"Expected four harmful rows; found {harmful.height}.")

if benign.height != 4:
    raise ValueError(f"Expected four benign rows; found {benign.height}.")

if set(harmful["language"]) != set(benign["language"]):
    raise ValueError("Harmful and benign language sets do not match.")

if harmful["paired_harmful_n"].sum() != 1200:
    raise ValueError("Expected 1200 paired harmful prompts.")

if benign["paired_benign_n"].sum() != 495:
    raise ValueError("Expected 495 paired benign prompts.")

harmful_table = harmful.select(
    [
        pl.col("language").alias("Language"),
        pl.col("paired_harmful_n").alias("Harmful paired n"),
        pl.col("baseline_failure_rate")
        .mul(100)
        .round(1)
        .alias("Harmful baseline failure (%)"),
        pl.col("steered_failure_rate")
        .mul(100)
        .round(1)
        .alias("Harmful steered failure (%)"),
        pl.col("delta_steered_minus_baseline_percentage_points")
        .round(1)
        .alias("Harmful delta (pp)"),
        pl.col("ci95_low_percentage_points")
        .round(1)
        .alias("Harmful CI low (pp)"),
        pl.col("ci95_high_percentage_points")
        .round(1)
        .alias("Harmful CI high (pp)"),
        pl.col("mcnemar_holm_adjusted_p")
        .round(6)
        .alias("Harmful Holm p"),
        pl.col("mcnemar_holm_reject_alpha_0_05")
        .alias("Harmful Holm reject 0.05"),
    ]
)

benign_table = benign.select(
    [
        pl.col("language").alias("Language"),
        pl.col("paired_benign_n").alias("Benign paired n"),
        pl.col("baseline_refusal_rate")
        .mul(100)
        .round(1)
        .alias("Benign baseline refusal (%)"),
        pl.col("steered_refusal_rate")
        .mul(100)
        .round(1)
        .alias("Benign steered refusal (%)"),
        pl.col("delta_refusal_percentage_points")
        .round(1)
        .alias("Benign delta refusal (pp)"),
        pl.col("ci95_low_percentage_points")
        .round(1)
        .alias("Benign CI low (pp)"),
        pl.col("ci95_high_percentage_points")
        .round(1)
        .alias("Benign CI high (pp)"),
        pl.col("mcnemar_holm_adjusted_p")
        .round(6)
        .alias("Benign Holm p"),
        pl.col("mcnemar_holm_reject_alpha_0_05")
        .alias("Benign Holm reject 0.05"),
    ]
)

table = harmful_table.join(benign_table, on="Language", how="inner").sort(
    "Language"
)

harmful_sig = table.filter(pl.col("Harmful Holm reject 0.05")).height
benign_sig = table.filter(pl.col("Benign Holm reject 0.05")).height

if harmful_sig != 3:
    raise ValueError(f"Expected three harmful Holm rejections; found {harmful_sig}.")

if benign_sig != 0:
    raise ValueError(f"Expected zero benign Holm rejections; found {benign_sig}.")

OUT.parent.mkdir(parents=True, exist_ok=True)
table.write_csv(OUT)

print(f"Wrote: {OUT}")
print(f"Paper-table rows: {table.height}")
print(f"Paired harmful prompts: {table['Harmful paired n'].sum()}")
print(f"Paired benign prompts: {table['Benign paired n'].sum()}")
print(f"Harmful Holm-significant rows: {harmful_sig}")
print(f"Benign Holm-significant rows: {benign_sig}")
print("Harmful and benign outcomes kept separate: True")
