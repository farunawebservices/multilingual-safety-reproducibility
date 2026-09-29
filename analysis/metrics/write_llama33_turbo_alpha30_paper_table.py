#!/usr/bin/env python3
"""
Write Addition 2 (Llama 3.3 Turbo Alpha 30) paper table.
"""
from pathlib import Path
import polars as pl

INFILE = Path("paper_artifacts/metrics/llama33_turbo_alpha30_role_strength_unified_results.csv")
OUT_CSV = Path("paper_artifacts/metrics/llama33_turbo_alpha30_paper_table.csv")
OUT_TEX = Path("paper_artifacts/tables/llama33_turbo_alpha30_cross_phase_comparison.tex")

frame = pl.read_csv(INFILE).sort(["config", "language"])

if frame.height != 6:
    raise ValueError(f"Expected 6 Alpha 30 rows; found {frame.height}.")

if frame["paired_harmful_n"].sum() != 1800:
    raise ValueError("Expected 1800 paired harmful prompts.")

table = frame.select([
    pl.col("config").alias("Configuration"),
    pl.col("language").alias("Language"),
    pl.col("paired_harmful_n").alias("Matched n"),
    pl.col("baseline_failure_rate").mul(100).round(1).alias("Baseline failure rate (%)"),
    pl.col("steered_failure_rate").mul(100).round(1).alias("Alpha 30 failure rate (%)"),
    pl.col("delta_steered_minus_baseline_percentage_points").round(1).alias("Change (pp)"),
    pl.col("ci95_low_percentage_points").round(1).alias("95% CI low (pp)"),
    pl.col("ci95_high_percentage_points").round(1).alias("95% CI high (pp)"),
    pl.col("mcnemar_holm_adjusted_p").round(6).alias("Holm-adjusted p"),
    pl.col("mcnemar_holm_reject_alpha_0_05").alias("Holm reject at 0.05"),
])

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
table.write_csv(OUT_CSV)

tex_content = r"""\begin{table}[t]
\centering
\caption{Archived cross-phase comparison of harmful operational-failure rates at Alpha 30.}
\label{tab:llama33_turbo_alpha30_cross_phase}
\begin{tabular}{llrrrrrrr}
\toprule
Configuration & Language & Matched $n$ & Baseline (\%) & Alpha 30 (\%) & Change (pp) & 95\% CI (pp) & Holm $p$ & Reject? \\
\midrule
"""

for row in table.iter_rows(named=True):
    ci_str = f"[{row['95% CI low (pp)']:.1f}, {row['95% CI high (pp)']:.1f}]"
    reject_str = "Yes" if row["Holm reject at 0.05"] else "No"
    tex_content += f"{row['Configuration']} & {row['Language']} & {row['Matched n']} & {row['Baseline failure rate (%)']:.1f} & {row['Alpha 30 failure rate (%)']:.1f} & {row['Change (pp)']:+.1f} & {ci_str} & {row['Holm-adjusted p']:.6f} & {reject_str} \\\\\n"

tex_content += r"""\bottomrule
\end{tabular}
\end{table}
"""

OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
OUT_TEX.write_text(tex_content, encoding="utf-8")

print(f"Wrote: {OUT_CSV}")
print(f"Wrote: {OUT_TEX}")
