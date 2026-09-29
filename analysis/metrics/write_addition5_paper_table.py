#!/usr/bin/env python3
"""Write Addition 5 paper table."""
from pathlib import Path
import polars as pl

INFILE = Path("paper_artifacts/metrics/addition5_adversarial_robustness_metrics.csv")
OUT_CSV = Path("paper_artifacts/metrics/addition5_adversarial_robustness_table.csv")
OUT_TEX = Path("paper_artifacts/tables/addition5_adversarial_robustness.tex")

df = pl.read_csv(INFILE)

table = df.select([
    pl.col("attack_family").alias("Attack Family"),
    pl.col("n_prompts").alias("n"),
    pl.col("baseline_jailbreak_n").alias("Baseline JB"),
    pl.col("steered_jailbreak_n").alias("Steered JB"),
    pl.col("baseline_jailbreak_rate").mul(100).alias("Baseline (%)"),
    pl.col("steered_jailbreak_rate").mul(100).alias("Steered (%)"),
    pl.col("interpretation").alias("Result"),
])

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
table.write_csv(OUT_CSV)

tex_lines = []
tex_lines.append(r"\begin{table}[t]")
tex_lines.append(r"\centering")
tex_lines.append(r"\caption{Addition 5: Adversarial robustness check. 50 jailbreak prompts across 6 attack families tested on Llama-3.3-70B-Instruct with random steering (layer 52, strength 20).}")
tex_lines.append(r"\label{tab:addition5_robustness}")
tex_lines.append(r"\begin{tabular}{lrrrrrr}")
tex_lines.append(r"\toprule")
tex_lines.append(r"Attack Family & $n$ & Baseline JB & Steered JB & Baseline (\%) & Steered (\%) & Result \\")
tex_lines.append(r"\midrule")

for row in table.iter_rows(named=True):
    line = f"{row['Attack Family']} & {row['n']} & {row['Baseline JB']} & {row['Steered JB']} & {row['Baseline (%)']:.0f} & {row['Steered (%)']:.0f} & {row['Result']} \\\\"
    tex_lines.append(line)

tex_lines.append(r"\bottomrule")
tex_lines.append(r"\end{tabular}")
tex_lines.append(r"\begin{tablenotes}")
tex_lines.append(r"\small")
tex_lines.append(r"\item Note: All 50 prompts were evaluated as NEITHER (no jailbreak success in baseline or steered conditions).")
tex_lines.append(r"\item Steering used a random vector (not a refusal direction).")
tex_lines.append(r"\end{tablenotes}")
tex_lines.append(r"\end{table}")

tex_content = "\n".join(tex_lines)
OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
OUT_TEX.write_text(tex_content, encoding="utf-8")

print(f"Wrote: {OUT_CSV}")
print(f"Wrote: {OUT_TEX}")
