from pathlib import Path

import polars as pl
from statsmodels.stats.multitest import multipletests

INFILE = Path(
    "paper_artifacts/metrics/qwen25_72b_primary_paired_inference.csv"
)

OUT = Path(
    "paper_artifacts/metrics/"
    "qwen25_72b_primary_paired_inference_holm.csv"
)

FAMILY_NAME = "qwen25_72b_primary_33_steered_cells"
ALPHA = 0.05
EXPECTED_ROWS = 33

frame = pl.read_csv(INFILE)

required = {
    "source_file_group",
    "steered_config",
    "language",
    "prompt_class",
    "alpha",
    "paired_record_n",
    "partial_baseline_coverage",
    "mcnemar_exact_two_sided_p",
}

missing = required - set(frame.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

if frame.height != EXPECTED_ROWS:
    raise ValueError(
        f"Expected {EXPECTED_ROWS} inference rows; found {frame.height}."
    )

if frame.select([
    "source_file_group",
    "steered_config",
    "language",
    "prompt_class",
    "alpha",
]).unique().height != EXPECTED_ROWS:
    raise ValueError("Inference comparison keys are not unique.")

p_values = frame["mcnemar_exact_two_sided_p"].to_list()

if any(value is None for value in p_values):
    raise ValueError("Cannot apply Holm correction with null p-values.")

reject, adjusted_p, _, _ = multipletests(
    p_values,
    alpha=ALPHA,
    method="holm",
)

result = (
    frame
    .with_columns([
        pl.Series("mcnemar_holm_adjusted_p", adjusted_p),
        pl.Series("mcnemar_holm_reject_alpha_0_05", reject),
        pl.lit(FAMILY_NAME).alias("correction_family"),
        pl.lit(EXPECTED_ROWS).alias("correction_family_size"),
        pl.lit(ALPHA).alias("familywise_alpha"),
    ])
    .sort([
        "prompt_class",
        "language",
        "steered_config",
        "alpha",
        "source_file_group",
    ])
)

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

print(f"Wrote: {OUT}")
print(f"Inference rows: {result.height}")
print(f"Holm family: {FAMILY_NAME}")
print(
    "Holm rejections: "
    f"{result.filter(pl.col('mcnemar_holm_reject_alpha_0_05')).height}/"
    f"{result.height}"
)
print(
    "Partial-coverage rows: "
    f"{result.filter(pl.col('partial_baseline_coverage')).height}/"
    f"{result.height}"
)

with pl.Config(tbl_cols=-1, tbl_width_chars=260):
    print(
        result.select([
            "steered_config",
            "language",
            "prompt_class",
            "alpha",
            "paired_record_n",
            "partial_baseline_coverage",
            "delta_steered_minus_baseline_percentage_points",
            "ci95_low_percentage_points",
            "ci95_high_percentage_points",
            "mcnemar_exact_two_sided_p",
            "mcnemar_holm_adjusted_p",
            "mcnemar_holm_reject_alpha_0_05",
        ])
    )
