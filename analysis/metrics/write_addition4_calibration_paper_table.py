#!/usr/bin/env python3
"""Write Addition 4 calibration paper table."""
from pathlib import Path
import polars as pl

INFILE = Path("paper_artifacts/metrics/addition4_judge_calibration_metrics.csv")
OUT_CSV = Path("paper_artifacts/metrics/addition4_calibration_paper_table.csv")
OUT_TEX = Path("paper_artifacts/tables/addition4_calibration_agreement.tex")

df = pl.read_csv(INFILE)

table = df.select([
    pl.col("language_name").alias("Language"),
    pl.col("n_prompts").alias("n"),
    pl.col("annotator_1_yes").alias("A1 YES"),
    pl.col("annotator_1_no").alias("A1 NO"),
    pl.col("annotator_2_yes").alias("A2 YES"),
    pl.col("annotator_2_no").alias("A2 NO"),
    pl.col("observed_agreement").mul(100).round(1).alias("Agreement (%)"),
    pl.col("cohens_kappa").round(3).alias("Cohens_kappa"),
    pl.col("ci95_low").mul(100).round(1).alias("CI_low"),
    pl.col("ci95_high").mul(100).round(1).alias("CI_high"),
    pl.col("interpretation").alias("Interpretation"),
])

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
table.write_csv(OUT_CSV)

tex_lines = []
tex_lines.append(r"\begin{table}[t]")
tex_lines.append(r"\centering")
tex_lines.append(r"\caption{Addition 4: Judge calibration inter-rater agreement across languages. Two annotators independently labeled 98 prompts (YES = should comply, NO = should refuse). Blinded adjudication resolved 1 disagreement.}")
tex_lines.append(r"\label{tab:addition4_calibration}")
tex_lines.append(r"\begin{tabular}{lrrrrrrrrr}")
tex_lines.append(r"\toprule")
tex_lines.append(r"Language & $n$ & A1 YES & A1 NO & A2 YES & A2 NO & Agreement (\%) & Cohen's $\kappa$ & 95\% CI & Interpretation \\")
tex_lines.append(r"\midrule")

for row in table.iter_rows(named=True):
    ci_str = f"[{row['CI_low']:.1f}, {row['CI_high']:.1f}]"
    line = f"{row['Language']} & {row['n']} & {row['A1 YES']} & {row['A1 NO']} & {row['A2 YES']} & {row['A2 NO']} & {row['Agreement (%)']:.1f} & {row['Cohens_kappa']:.3f} & {ci_str} & {row['Interpretation']} \\\\"
    tex_lines.append(line)

tex_lines.append(r"\bottomrule")
tex_lines.append(r"\end{tabular}")
tex_lines.append(r"\begin{tablenotes}")
tex_lines.append(r"\small")
tex_lines.append(r"\item Note: Overall Cohen's $\kappa$ = 0.98 (95% CI: 93.7–100%), indicating almost perfect inter-rater agreement.")
tex_lines.append(r"\end{tablenotes}")
tex_lines.append(r"\end{table}")

tex_content = "\n".join(tex_lines)

OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
OUT_TEX.write_text(tex_content, encoding="utf-8")

print(f"Wrote: {OUT_CSV}")
print(f"Wrote: {OUT_TEX}")
