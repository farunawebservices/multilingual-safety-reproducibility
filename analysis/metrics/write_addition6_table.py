#!/usr/bin/env python3
"""Write Addition 6 paper table."""
from pathlib import Path
import polars as pl

INFILE = Path("paper_artifacts/metrics/addition6_prompt_filter_baseline_metrics.csv")
OUT_CSV = Path("paper_artifacts/metrics/addition6_prompt_filter_baseline_table.csv")
OUT_TEX = Path("paper_artifacts/tables/addition6_prompt_filter_baseline.tex")

df = pl.read_csv(INFILE)

table = df.select([
    pl.col("condition").alias("Condition"),
    pl.col("harmful_total").alias("Harmful n"),
    pl.col("harmful_vetoes").alias("Harmful vetoes"),
    pl.col("harmful_veto_rate").mul(100).round(1).alias("Harmful veto rate (%)"),
    pl.col("benign_total").alias("Benign n"),
    pl.col("benign_vetoes").alias("Benign vetoes"),
    pl.col("benign_veto_rate").mul(100).round(1).alias("Benign FPR (%)"),
    pl.col("missing_labels").alias("Missing labels"),
])

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
table.write_csv(OUT_CSV)

tex_lines = []
tex_lines.append(r"\begin{table}[t]")
tex_lines.append(r"\centering")
tex_lines.append(r"\caption{Addition 6: Prompt filter baseline vs SAE steering. Three conditions compared on 460 prompts (400 harmful, 60 benign) across 4 languages.}")
tex_lines.append(r"\label{tab:addition6_baseline}")
tex_lines.append(r"\begin{tabular}{lrrrrrrr}")
tex_lines.append(r"\toprule")
tex_lines.append(r"Condition & Harmful $n$ & Vetoes & Rate (\%) & Benign $n$ & FPR (\%) & Missing \\")
tex_lines.append(r"\midrule")

for row in table.iter_rows(named=True):
    cond_short = row['Condition'].replace('planner_steering_l50_s20_historical_all_steps', 'SAE steering').replace('prompt_filter', 'Prompt filter').replace('no_intervention', 'No intervention')
    line = f"{cond_short} & {row['Harmful n']} & {row['Harmful vetoes']} & {row['Harmful veto rate (%)']:.1f} & {row['Benign n']} & {row['Benign FPR (%)']:.1f} & {row['Missing labels']} \\\\"
    tex_lines.append(line)

tex_lines.append(r"\bottomrule")
tex_lines.append(r"\end{tabular}")
tex_lines.append(r"\begin{tablenotes}")
tex_lines.append(r"\small")
tex_lines.append(r"\item Note: SAE steering increased harmful veto rate (+5.0 pp vs baseline) but introduced 3 benign false positives and 25 missing labels.")
tex_lines.append(r"\end{tablenotes}")
tex_lines.append(r"\end{table}")

tex_content = "\n".join(tex_lines)
OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
OUT_TEX.write_text(tex_content, encoding="utf-8")

print(f"Wrote: {OUT_CSV}")
print(f"Wrote: {OUT_TEX}")
