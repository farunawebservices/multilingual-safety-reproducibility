from pathlib import Path
import polars as pl
from statsmodels.stats.multitest import multipletests

SOURCE = Path("paper_artifacts/metrics/gpt_oss_20b_planner_alpha1_paired_effect.csv")
OUT = Path("paper_artifacts/metrics/gpt_oss_20b_planner_alpha1_paired_effect_holm.csv")

df = pl.read_csv(SOURCE).sort("language")
p_values = df["mcnemar_exact_two_sided_p"].to_list()

reject, p_holm, _, _ = multipletests(
    p_values,
    alpha=0.05,
    method="holm",
)

result = df.with_columns([
    pl.Series("mcnemar_holm_adjusted_p", p_holm),
    pl.Series("mcnemar_holm_reject_alpha_0_05", reject),
])

result.write_csv(OUT)

print(
    result.select([
        "language",
        "baseline_failure_n",
        "steered_failure_n",
        "worsened_safe_to_failure_n",
        "improved_failure_to_safe_n",
        "absolute_change_percentage_points",
        "mcnemar_exact_two_sided_p",
        "mcnemar_holm_adjusted_p",
        "mcnemar_holm_reject_alpha_0_05",
    ])
)
print(f"\nWrote: {OUT}")
