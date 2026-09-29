from pathlib import Path

import polars as pl
from statsmodels.stats.multitest import multipletests

INFILE = Path(
    "paper_artifacts/metrics/"
    "gemma_4_31b_phase1_complete_case_paired_inference.csv"
)

OUT = Path(
    "paper_artifacts/metrics/"
    "gemma_4_31b_phase1_complete_case_paired_inference_holm.csv"
)

FAMILYWISE_ALPHA = 0.05

frame = pl.read_csv(INFILE)

if frame.height != 3:
    raise ValueError(f"Expected 3 inference rows; found {frame.height}")

p_values = frame["mcnemar_exact_two_sided_p"].to_list()

reject, adjusted_p, _, _ = multipletests(
    p_values,
    alpha=FAMILYWISE_ALPHA,
    method="holm",
)

result = (
    frame
    .with_columns([
        pl.Series("mcnemar_holm_adjusted_p", adjusted_p),
        pl.Series(
            "mcnemar_holm_reject_alpha_0_05",
            reject.tolist(),
        ),
        pl.lit("gemma_phase1_three_cell_family")
        .alias("correction_family"),
        pl.lit(frame.height).alias("correction_family_size"),
        pl.lit(FAMILYWISE_ALPHA).alias("familywise_alpha"),
    ])
    .sort(["prompt_class", "language", "steered_config", "alpha"])
)

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

print(f"Wrote: {OUT}")
print(f"Correction family size: {result.height}")
print(
    "Holm-significant rows: "
    f"{result.filter(pl.col('mcnemar_holm_reject_alpha_0_05')).height}"
)

with pl.Config(tbl_cols=-1, tbl_width_chars=240):
    print(
        result.select([
            "language",
            "prompt_class",
            "paired_record_n",
            "partial_coverage",
            "delta_steered_minus_baseline_percentage_points",
            "mcnemar_exact_two_sided_p",
            "mcnemar_holm_adjusted_p",
            "mcnemar_holm_reject_alpha_0_05",
        ])
    )
